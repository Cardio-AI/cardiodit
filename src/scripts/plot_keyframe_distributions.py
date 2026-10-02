"""Plot disease-wise cardiac keyframe distributions.

This script intentionally avoids pandas/matplotlib so it can run in the lean
project environment. It joins:

* data/dataset_information.csv: SUBJECT_CODE -> DISEASE
* data/preprocessed/MNM2_native_grid/manifest.csv: subject_id -> T_native
* data/keyframes/*.json: ED/MS/ES/PF/MD frame indices

The keyframe positions are represented relative to ED and normalized by each
subject's native temporal length.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import os
from typing import Iterable


PHASES = ("ED", "MS", "ES", "PF", "MD")
SEGMENTS = ("ED-MS", "MS-ES", "ES-PF", "PF-MD", "MD-ED")
PHASE_COLORS = {
    "ED": "#374151",
    "MS": "#1f77b4",
    "ES": "#d62728",
    "PF": "#2ca02c",
    "MD": "#9467bd",
}
SEGMENT_COLORS = {
    "ED-MS": "#6aaed6",
    "MS-ES": "#f08b86",
    "ES-PF": "#8bcf88",
    "PF-MD": "#b89bd6",
    "MD-ED": "#d8b365",
}
CURRENT_ANCHORS_T32 = {
    "ED": 0.0,
    "MS": 6.0,
    "ES": 12.0,
    "PF": 19.0,
    "MD": 28.0,
}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = load_records(
        dataset_info=args.dataset_info,
        manifest=args.manifest,
        keyframe_dir=args.keyframe_dir,
    )
    if not records:
        raise SystemExit("No joined records found.")

    disease_order = sorted(
        {r["disease"] for r in records},
        key=lambda disease: (-sum(r["disease"] == disease for r in records), disease),
    )
    write_subject_table(out_dir / "subject_keyframes_joined.csv", records)
    write_summary_tables(out_dir, records, disease_order, args.target_frames, skipped)
    write_timeline_svg(out_dir / "phase_positions_by_disease.svg", records, disease_order)
    write_segments_svg(
        out_dir / "segment_durations_by_disease.svg",
        records,
        disease_order,
    )
    write_facet_svg(
        out_dir / "keyframe_distributions_by_disease.svg",
        records,
        disease_order,
    )
    write_heatmap_svg(
        out_dir / "anchors_t32_heatmap.svg",
        records,
        disease_order,
        args.target_frames,
    )
    write_readme(out_dir / "README.md", records, skipped, disease_order, args.target_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_info",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--keyframe_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "temporal_alignment" / "keyframe_distributions",
    )
    parser.add_argument("--target_frames", type=int, default=32)
    return parser.parse_args()


def load_records(
    dataset_info: Path,
    manifest: Path,
    keyframe_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    metadata = load_dataset_info(dataset_info)
    native_t = load_manifest_t(manifest)
    keyframes = load_keyframes(keyframe_dir)

    records: list[dict[str, object]] = []
    skipped = {
        "missing_metadata": [],
        "missing_manifest_t": [],
        "invalid_keyframes": [],
    }

    for subject_id, frames in sorted(keyframes.items()):
        if subject_id not in metadata:
            skipped["missing_metadata"].append(subject_id)
            continue
        if subject_id not in native_t:
            skipped["missing_manifest_t"].append(subject_id)
            continue

        t_native = native_t[subject_id]
        try:
            phase_pos, segment_pos = normalize_keyframes(frames, t_native)
        except ValueError:
            skipped["invalid_keyframes"].append(subject_id)
            continue

        record: dict[str, object] = {
            "subject_id": subject_id,
            "disease": metadata[subject_id],
            "T_native": t_native,
        }
        for phase in PHASES:
            record[f"{phase}_frame"] = frames[phase]
            record[f"{phase}_phase"] = phase_pos[phase]
        for segment in SEGMENTS:
            record[f"{segment}_duration"] = segment_pos[segment]
        records.append(record)

    return records, skipped


def load_dataset_info(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    header_idx = raw.find(b"SUBJECT_CODE")
    if header_idx < 0:
        raise ValueError(f"Could not find SUBJECT_CODE header in {path}")
    text = raw[header_idx:].replace(b"\0", b"").decode("utf-8", errors="replace")
    mapping: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(text)):
        code = row.get("SUBJECT_CODE", "").strip()
        disease = row.get("DISEASE", "").strip()
        if not code or not disease:
            continue
        mapping[f"{int(code):03d}_SA_CINE"] = disease
    return mapping


def load_manifest_t(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id") and row.get("T_native"):
                mapping[row["subject_id"]] = int(row["T_native"])
    return mapping


def load_keyframes(path: Path) -> dict[str, dict[str, int]]:
    mapping: dict[str, dict[str, int]] = {}
    for json_path in sorted(path.glob("*.json")):
        frames = json.loads(json_path.read_text())
        if all(phase in frames for phase in PHASES):
            mapping[json_path.stem] = {phase: int(frames[phase]) for phase in PHASES}
    return mapping


def normalize_keyframes(
    frames: dict[str, int],
    t_native: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if t_native < 2:
        raise ValueError("T_native must be >= 2")

    ed = frames["ED"]
    if any(frames[p] < 0 or frames[p] >= t_native for p in PHASES):
        raise ValueError("Keyframe outside native frame range")

    unwrapped = {"ED": float(ed)}
    previous = float(ed)
    for phase in PHASES[1:]:
        frame = float(frames[phase])
        while frame <= previous:
            frame += t_native
        unwrapped[phase] = frame
        previous = frame

    phase_pos = {
        phase: (unwrapped[phase] - ed) / t_native
        for phase in PHASES
    }
    segment_pos = {
        "ED-MS": phase_pos["MS"] - phase_pos["ED"],
        "MS-ES": phase_pos["ES"] - phase_pos["MS"],
        "ES-PF": phase_pos["PF"] - phase_pos["ES"],
        "PF-MD": phase_pos["MD"] - phase_pos["PF"],
        "MD-ED": 1.0 - phase_pos["MD"],
    }

    if any(v <= 0 for v in segment_pos.values()):
        raise ValueError("Non-positive segment duration")
    return phase_pos, segment_pos


def write_subject_table(path: Path, records: list[dict[str, object]]) -> None:
    fields = ["subject_id", "disease", "T_native"]
    fields += [f"{phase}_frame" for phase in PHASES]
    fields += [f"{phase}_phase" for phase in PHASES]
    fields += [f"{segment}_duration" for segment in SEGMENTS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: format_value(record.get(field, "")) for field in fields})


def write_summary_tables(
    out_dir: Path,
    records: list[dict[str, object]],
    disease_order: list[str],
    target_frames: int,
    skipped: dict[str, list[str]],
) -> None:
    with (out_dir / "summary_by_disease.csv").open("w", newline="") as f:
        fields = ["disease", "n"]
        for phase in PHASES:
            fields += [
                f"{phase}_phase_mean",
                f"{phase}_phase_std",
                f"{phase}_t{target_frames}_mean",
                f"{phase}_t{target_frames}_round",
            ]
        for segment in SEGMENTS:
            fields += [f"{segment}_mean", f"{segment}_std"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for disease in disease_order:
            subset = [r for r in records if r["disease"] == disease]
            row: dict[str, object] = {"disease": disease, "n": len(subset)}
            for phase in PHASES:
                values = [float(r[f"{phase}_phase"]) for r in subset]
                mean = statistics.fmean(values)
                row[f"{phase}_phase_mean"] = mean
                row[f"{phase}_phase_std"] = sample_std(values)
                row[f"{phase}_t{target_frames}_mean"] = mean * target_frames
                row[f"{phase}_t{target_frames}_round"] = round(mean * target_frames)
            for segment in SEGMENTS:
                values = [float(r[f"{segment}_duration"]) for r in subset]
                row[f"{segment}_mean"] = statistics.fmean(values)
                row[f"{segment}_std"] = sample_std(values)
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})

    with (out_dir / "anchors_t32_by_disease.csv").open("w", newline="") as f:
        fields = ["disease", "n"] + [f"{phase}_mean_frame" for phase in PHASES]
        fields += [f"{phase}_rounded_frame" for phase in PHASES]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for disease in disease_order:
            subset = [r for r in records if r["disease"] == disease]
            row: dict[str, object] = {"disease": disease, "n": len(subset)}
            for phase in PHASES:
                mean = statistics.fmean(float(r[f"{phase}_phase"]) for r in subset)
                row[f"{phase}_mean_frame"] = mean * target_frames
                row[f"{phase}_rounded_frame"] = round(mean * target_frames)
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})

    with (out_dir / "excluded_subjects.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reason", "subject_id"])
        for reason, subjects in skipped.items():
            for subject_id in subjects:
                writer.writerow([reason, subject_id])


def write_timeline_svg(
    path: Path,
    records: list[dict[str, object]],
    disease_order: list[str],
) -> None:
    width = 1500
    row_h = 78
    top = 116
    left = 150
    right = 80
    bottom = 90
    plot_w = width - left - right
    height = top + bottom + row_h * len(disease_order)
    elements = base_styles()
    elements.append(text(36, 44, "ED-relative keyframe positions by disease", "title"))
    elements.append(
        text(
            36,
            75,
            "Dots are subjects; large markers and whiskers show mean +/- 1 SD. Dashed lines show current global T=32 anchors.",
            "subtitle",
        )
    )

    y_min = top - 20
    y_max = top + row_h * len(disease_order) - 18
    for phase in PHASES:
        anchor = CURRENT_ANCHORS_T32[phase] / 32.0
        x = left + anchor * plot_w
        elements.append(
            line(x, y_min, x, y_max, stroke=PHASE_COLORS[phase], width=1.4, dash="6 8", opacity=0.42)
        )

    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_w
        elements.append(line(x, y_min, x, y_max, stroke="#e5e7eb", width=1.0))
        elements.append(text(x, height - 52, f"{tick:.2f}", "axis", anchor="middle"))
    elements.append(text(left + plot_w / 2, height - 20, "fraction of cardiac cycle after ED", "axis-label", anchor="middle"))

    for row_idx, disease in enumerate(disease_order):
        y = top + row_idx * row_h
        subset = [r for r in records if r["disease"] == disease]
        bg = "#f8fafc" if row_idx % 2 == 0 else "#ffffff"
        elements.append(rect(26, y - 30, width - 52, row_h - 8, bg, stroke="none", rx=10))
        elements.append(text(36, y + 6, f"{disease}  n={len(subset)}", "row-label"))
        elements.append(line(left, y, left + plot_w, y, stroke="#cbd5e1", width=1.2))

        for phase in PHASES:
            vals = [float(r[f"{phase}_phase"]) for r in subset]
            color = PHASE_COLORS[phase]
            for record in subset:
                x = left + float(record[f"{phase}_phase"]) * plot_w
                jy = stable_jitter(f"{record['subject_id']}-{phase}", scale=18.0)
                elements.append(circle(x, y + jy, 3.0, color, opacity=0.24))

            mean = statistics.fmean(vals)
            sd = sample_std(vals)
            x_mean = left + mean * plot_w
            x_l = left + max(0.0, mean - sd) * plot_w
            x_r = left + min(1.0, mean + sd) * plot_w
            elements.append(line(x_l, y, x_r, y, stroke=color, width=4.0, opacity=0.78))
            elements.append(circle(x_mean, y, 7.5, color, opacity=0.96, stroke="#ffffff", stroke_width=1.4))

    legend_x = width - 520
    legend_y = 42
    for idx, phase in enumerate(PHASES):
        x = legend_x + idx * 92
        elements.append(circle(x, legend_y, 6, PHASE_COLORS[phase]))
        elements.append(text(x + 12, legend_y + 5, phase, "legend"))

    write_svg(path, width, height, elements)


def write_segments_svg(
    path: Path,
    records: list[dict[str, object]],
    disease_order: list[str],
) -> None:
    width = 1450
    row_h = 58
    top = 122
    left = 160
    right = 70
    bottom = 86
    plot_w = width - left - right
    rows = ["Global T32"] + disease_order
    height = top + bottom + row_h * len(rows)
    elements = base_styles()
    elements.append(text(36, 44, "Mean cardiac phase durations by disease", "title"))
    elements.append(
        text(
            36,
            75,
            "Stacked bars show average ED-to-MS, MS-to-ES, ES-to-PF, PF-to-MD, and MD-to-next-ED fractions.",
            "subtitle",
        )
    )

    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_w
        elements.append(line(x, top - 28, x, height - bottom + 8, stroke="#e5e7eb", width=1.0))
        elements.append(text(x, height - 48, f"{tick:.2f}", "axis", anchor="middle"))
    elements.append(text(left + plot_w / 2, height - 18, "fraction of cardiac cycle", "axis-label", anchor="middle"))

    global_phase = {phase: CURRENT_ANCHORS_T32[phase] / 32.0 for phase in PHASES}
    global_segments = {
        "ED-MS": global_phase["MS"] - global_phase["ED"],
        "MS-ES": global_phase["ES"] - global_phase["MS"],
        "ES-PF": global_phase["PF"] - global_phase["ES"],
        "PF-MD": global_phase["MD"] - global_phase["PF"],
        "MD-ED": 1.0 - global_phase["MD"],
    }

    for row_idx, label in enumerate(rows):
        y = top + row_idx * row_h
        elements.append(text(36, y + 10, label, "row-label"))
        if label == "Global T32":
            means = global_segments
        else:
            subset = [r for r in records if r["disease"] == label]
            means = {
                segment: statistics.fmean(float(r[f"{segment}_duration"]) for r in subset)
                for segment in SEGMENTS
            }
        x = left
        for segment in SEGMENTS:
            w = means[segment] * plot_w
            elements.append(rect(x, y - 20, w, 34, SEGMENT_COLORS[segment], stroke="#ffffff", stroke_width=1.0, rx=4))
            if w > 82:
                elements.append(
                    text(
                        x + w / 2,
                        y + 2,
                        f"{means[segment] * 100:.0f}%",
                        "bar-label",
                        anchor="middle",
                    )
                )
            x += w

    legend_x = width - 720
    legend_y = 42
    for idx, segment in enumerate(SEGMENTS):
        x = legend_x + idx * 135
        elements.append(rect(x, legend_y - 10, 16, 16, SEGMENT_COLORS[segment], stroke="none", rx=3))
        elements.append(text(x + 24, legend_y + 4, segment, "legend"))

    write_svg(path, width, height, elements)


def write_facet_svg(
    path: Path,
    records: list[dict[str, object]],
    disease_order: list[str],
) -> None:
    width = 1600
    height = 1100
    left = 95
    right = 56
    top = 124
    bottom = 72
    panel_gap_x = 70
    panel_gap_y = 86
    panel_w = (width - left - right - panel_gap_x) / 2
    panel_h = (height - top - bottom - panel_gap_y) / 2
    phases = ("MS", "ES", "PF", "MD")
    elements = base_styles()
    elements.append(text(36, 44, "Disease-wise keyframe distributions", "title"))
    elements.append(
        text(
            36,
            75,
            "Each panel shows subject-level normalized phase positions, with IQR boxes, medians, and current global-anchor reference lines.",
            "subtitle",
        )
    )

    for phase_idx, phase in enumerate(phases):
        col = phase_idx % 2
        row = phase_idx // 2
        x0 = left + col * (panel_w + panel_gap_x)
        y0 = top + row * (panel_h + panel_gap_y)
        color = PHASE_COLORS[phase]
        elements.append(text(x0, y0 - 22, phase, "panel-title"))
        elements.append(rect(x0, y0, panel_w, panel_h, "#ffffff", stroke="#cbd5e1", stroke_width=1.0, rx=10))

        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - tick * panel_h
            elements.append(line(x0, y, x0 + panel_w, y, stroke="#edf2f7", width=1.0))
            elements.append(text(x0 - 12, y + 4, f"{tick:.2f}", "axis", anchor="end"))

        anchor_y = y0 + panel_h - (CURRENT_ANCHORS_T32[phase] / 32.0) * panel_h
        elements.append(line(x0, anchor_y, x0 + panel_w, anchor_y, stroke=color, width=1.8, dash="6 7", opacity=0.52))

        band_w = panel_w / len(disease_order)
        for idx, disease in enumerate(disease_order):
            cx = x0 + band_w * (idx + 0.5)
            subset = [r for r in records if r["disease"] == disease]
            vals = sorted(float(r[f"{phase}_phase"]) for r in subset)
            q1, med, q3 = quantiles(vals)
            mean = statistics.fmean(vals)
            y_q1 = y0 + panel_h - q1 * panel_h
            y_med = y0 + panel_h - med * panel_h
            y_q3 = y0 + panel_h - q3 * panel_h
            y_mean = y0 + panel_h - mean * panel_h

            for record in subset:
                value = float(record[f"{phase}_phase"])
                y = y0 + panel_h - value * panel_h
                x = cx + stable_jitter(f"{record['subject_id']}-{phase}", scale=min(18.0, band_w * 0.28))
                elements.append(circle(x, y, 3.0, color, opacity=0.25))

            box_w = min(42, band_w * 0.62)
            elements.append(rect(cx - box_w / 2, y_q3, box_w, y_q1 - y_q3, "#ffffff", stroke=color, stroke_width=2.0, rx=3))
            elements.append(line(cx - box_w / 2, y_med, cx + box_w / 2, y_med, stroke=color, width=2.5))
            elements.append(diamond(cx, y_mean, 8, color, opacity=0.95))
            elements.append(text(cx, y0 + panel_h + 26, disease, "axis", anchor="middle"))

        if col == 0:
            elements.append(text(x0 - 58, y0 + panel_h / 2, "cycle fraction after ED", "axis-label", anchor="middle", rotate=-90))

    write_svg(path, width, height, elements)


def write_heatmap_svg(
    path: Path,
    records: list[dict[str, object]],
    disease_order: list[str],
    target_frames: int,
) -> None:
    width = 1180
    cell_w = 158
    cell_h = 54
    left = 170
    top = 128
    height = top + cell_h * (len(disease_order) + 1) + 80
    elements = base_styles()
    elements.append(text(36, 44, f"Disease-specific mean keyframe anchors for T={target_frames}", "title"))
    elements.append(
        text(
            36,
            75,
            "Cell values are mean normalized keyframe positions mapped to output frame coordinates; parentheses show rounded anchors.",
            "subtitle",
        )
    )

    for idx, phase in enumerate(PHASES):
        x = left + idx * cell_w + cell_w / 2
        elements.append(text(x, top - 22, phase, "col-label", anchor="middle"))

    for row_idx, disease in enumerate(disease_order):
        y = top + row_idx * cell_h
        subset = [r for r in records if r["disease"] == disease]
        elements.append(text(36, y + 34, f"{disease}  n={len(subset)}", "row-label"))
        for col_idx, phase in enumerate(PHASES):
            mean = statistics.fmean(float(r[f"{phase}_phase"]) for r in subset) * target_frames
            rounded = round(mean)
            x = left + col_idx * cell_w
            fill = heat_color(mean / target_frames)
            elements.append(rect(x, y, cell_w - 8, cell_h - 8, fill, stroke="#ffffff", stroke_width=1.2, rx=8))
            elements.append(text(x + (cell_w - 8) / 2, y + 30, f"{mean:.1f} ({rounded})", "heat-label", anchor="middle"))

    y = top + len(disease_order) * cell_h + 16
    elements.append(text(36, y + 22, "Current global", "row-label"))
    for col_idx, phase in enumerate(PHASES):
        mean = CURRENT_ANCHORS_T32[phase]
        x = left + col_idx * cell_w
        elements.append(rect(x, y, cell_w - 8, cell_h - 8, "#f3f4f6", stroke="#d1d5db", stroke_width=1.2, rx=8))
        elements.append(text(x + (cell_w - 8) / 2, y + 30, f"{mean:.1f} ({round(mean)})", "heat-label", anchor="middle"))

    write_svg(path, width, height, elements)


def write_readme(
    path: Path,
    records: list[dict[str, object]],
    skipped: dict[str, list[str]],
    disease_order: list[str],
    target_frames: int,
) -> None:
    counts = Counter(r["disease"] for r in records)
    t_counts = Counter(int(r["T_native"]) for r in records)
    lines = [
        "# Keyframe Distributions by Disease",
        "",
        f"Generated from {len(records)} subjects with metadata, native T, and keyframes.",
        "",
        "## Plots",
        "",
        "- `phase_positions_by_disease.svg`: subject-level keyframe positions and disease-wise mean +/- SD.",
        "- `segment_durations_by_disease.svg`: mean cardiac phase-duration fractions by disease.",
        "- `keyframe_distributions_by_disease.svg`: per-phase distribution panels by disease.",
        f"- `anchors_t32_heatmap.svg`: disease-specific mean anchors mapped to T={target_frames}.",
        "",
        "## Included Subjects",
        "",
        "| Disease | n |",
        "|---|---:|",
    ]
    for disease in disease_order:
        lines.append(f"| {disease} | {counts[disease]} |")
    lines += [
        "",
        "## Native Temporal Lengths",
        "",
        "| T_native | n |",
        "|---:|---:|",
    ]
    for t_native, count in sorted(t_counts.items()):
        lines.append(f"| {t_native} | {count} |")
    lines += [
        "",
        "## Exclusions",
        "",
    ]
    for reason, subjects in skipped.items():
        lines.append(f"- {reason}: {len(subjects)}")
    lines += [
        "",
        "CSV outputs:",
        "",
        "- `subject_keyframes_joined.csv`",
        "- `summary_by_disease.csv`",
        f"- `anchors_t32_by_disease.csv`",
        "- `excluded_subjects.csv`",
        "",
    ]
    path.write_text("\n".join(lines))


def base_styles() -> list[str]:
    return [
        "<style>",
        "text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; }",
        ".title { font-size: 30px; font-weight: 760; }",
        ".subtitle { font-size: 15px; fill: #4b5563; }",
        ".row-label { font-size: 16px; font-weight: 700; fill: #1f2937; }",
        ".axis { font-size: 13px; fill: #64748b; }",
        ".axis-label { font-size: 14px; font-weight: 650; fill: #475569; }",
        ".legend { font-size: 14px; fill: #374151; }",
        ".bar-label { font-size: 13px; font-weight: 720; fill: #111827; }",
        ".panel-title { font-size: 22px; font-weight: 760; fill: #111827; }",
        ".col-label { font-size: 16px; font-weight: 760; fill: #1f2937; }",
        ".heat-label { font-size: 15px; font-weight: 740; fill: #111827; }",
        "</style>",
    ]


def write_svg(path: Path, width: float, height: float, elements: Iterable[str]) -> None:
    body = "\n".join(elements)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">\n'
        f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def text(
    x: float,
    y: float,
    value: str,
    cls: str,
    anchor: str = "start",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.0f} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{transform}>'
        f"{html.escape(value)}</text>"
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
    opacity: float | None = None,
) -> str:
    attrs = [
        f'x1="{x1:.2f}"',
        f'y1="{y1:.2f}"',
        f'x2="{x2:.2f}"',
        f'y2="{y2:.2f}"',
        f'stroke="{stroke}"',
        f'stroke-width="{width:.2f}"',
    ]
    if dash is not None:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity:.3f}"')
    return f"<line {' '.join(attrs)}/>"


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = "#cbd5e1",
    stroke_width: float = 0.0,
    rx: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}"/>'
    )


def circle(
    x: float,
    y: float,
    r: float,
    fill: str,
    opacity: float = 1.0,
    stroke: str | None = None,
    stroke_width: float = 0.0,
) -> str:
    attrs = [
        f'cx="{x:.2f}"',
        f'cy="{y:.2f}"',
        f'r="{r:.2f}"',
        f'fill="{fill}"',
        f'opacity="{opacity:.3f}"',
    ]
    if stroke is not None:
        attrs.append(f'stroke="{stroke}"')
        attrs.append(f'stroke-width="{stroke_width:.2f}"')
    return f"<circle {' '.join(attrs)}/>"


def diamond(x: float, y: float, r: float, fill: str, opacity: float = 1.0) -> str:
    points = [
        (x, y - r),
        (x + r, y),
        (x, y + r),
        (x - r, y),
    ]
    point_str = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return f'<polygon points="{point_str}" fill="{fill}" opacity="{opacity:.3f}" stroke="#ffffff" stroke-width="1.2"/>'


def stable_jitter(key: str, scale: float) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    return (value - 0.5) * 2.0 * scale


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return percentile(values, 0.25), percentile(values, 0.5), percentile(values, 0.75)


def percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = q * (len(values) - 1)
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return values[low]
    frac = idx - low
    return values[low] * (1 - frac) + values[high] * frac


def heat_color(value: float) -> str:
    value = min(max(value, 0.0), 1.0)
    stops = [
        (0.0, (239, 246, 255)),
        (0.5, (186, 230, 253)),
        (1.0, (251, 191, 36)),
    ]
    for (left_v, left_rgb), (right_v, right_rgb) in zip(stops, stops[1:]):
        if left_v <= value <= right_v:
            frac = (value - left_v) / (right_v - left_v)
            rgb = tuple(
                round(left_rgb[i] * (1 - frac) + right_rgb[i] * frac)
                for i in range(3)
            )
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#fbbf24"


def format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


if __name__ == "__main__":
    main()

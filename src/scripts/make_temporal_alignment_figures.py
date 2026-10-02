"""
Generate one overview figure per CardioDiT-TempAlign temporal alignment method.

The figures are documentation artifacts, not model outputs.  They mirror the
current implementation split:

- image-space methods in src/utils/temporal_align.py
- native-T temporal-PE methods in src/models/dit.py

Outputs default to CARDIODIT_RUNS_DIR/temporal_alignment/figures.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import os
from textwrap import fill

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patheffects import withStroke


PALETTE = {
    "ink": "#172033",
    "muted": "#5f6b7a",
    "line": "#c9d2df",
    "panel": "#f6f8fb",
    "blue": "#2f6fed",
    "teal": "#1e9c8a",
    "green": "#45a857",
    "gold": "#d79a22",
    "red": "#d9544d",
    "purple": "#7d5fff",
    "gray": "#8291a7",
    "cyan": "#2ca9bc",
}


METHODS = [
    {
        "slug": "exp01_cyclic_repeat",
        "exp": "Exp 1",
        "title": "Cyclic repetition baseline",
        "subtitle": "Repeat native cine frames and crop to fixed T=32 before VQ-GAN encoding.",
        "family": "Image-space preprocessing",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/cyclic/",
        "code": "src/utils/temporal_align.py::cyclic_repeat",
        "draw": "cyclic",
        "notes": [
            "Uses the original CardioDiT fixed-T assumption.",
            "Fast and deterministic, but repeats early-cycle frames when T_native < 32.",
            "DiT uses temporal_pe='absolute' with fixed latent Tg=8.",
        ],
    },
    {
        "slug": "exp02_linear_interp",
        "exp": "Exp 2",
        "title": "Linear interpolation",
        "subtitle": "Resample the temporal axis to T=32 with per-voxel linear blending.",
        "family": "Image-space preprocessing",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/interp_linear/",
        "code": "src/utils/temporal_align.py::linear_interp",
        "draw": "linear",
        "notes": [
            "Calls torch.nn.functional.interpolate(..., mode='linear', align_corners=False).",
            "Produces a uniform fixed-T cine for the unchanged VQ-GAN and DiT.",
            "Can smooth fast cardiac motion because intermediate frames are intensity blends.",
        ],
    },
    {
        "slug": "exp03_piecewise_keyframes",
        "exp": "Exp 3",
        "title": "Piecewise keyframe alignment",
        "subtitle": "Warp the time axis so ED, MS, ES, PF, and MD land on shared output anchors.",
        "family": "Image-space preprocessing",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/interp_piecewise/",
        "code": "src/utils/temporal_align.py::piecewise_interp",
        "draw": "piecewise",
        "notes": [
            "Reads per-subject keyframes from sidecar JSON files.",
            "Default T=32 anchors are ED/MS/ES/PF/MD = 0/6/12/19/28.",
            "Preserves clinically meaningful phase landmarks before VQ-GAN encoding.",
        ],
    },
    {
        "slug": "exp04_fourier_resample",
        "exp": "Exp 4",
        "title": "Fourier resampling",
        "subtitle": "Resample periodic cardiac motion by padding or truncating the temporal spectrum.",
        "family": "Image-space preprocessing",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/interp_fourier/",
        "code": "src/utils/temporal_align.py::fourier_resample",
        "draw": "fourier",
        "notes": [
            "Uses rFFT along T, keeps low frequencies, then irFFT to T=32.",
            "Amplitude is rescaled by T_out / T_in to match scipy.signal.resample behavior.",
            "Best matched to approximately periodic, band-limited cine motion.",
        ],
    },
    {
        "slug": "exp05_dtw_template",
        "exp": "Exp 5",
        "title": "Descriptor-DTW alignment",
        "subtitle": "Warp each subject's descriptor to a canonical template, then resample images.",
        "family": "Image-space preprocessing",
        "status": "implemented; needs sidecars",
        "status_color": "gold",
        "config": "configs/transformer/test_configs/interp_dtw/",
        "code": "src/utils/temporal_align.py::dtw_align",
        "draw": "dtw",
        "notes": [
            "Builds a 32-frame training-set descriptor template before latent encoding.",
            "Uses constrained DTW to map each target frame to a subject-specific source time.",
            "The DiT still sees fixed-T latents and keeps temporal_pe='absolute'.",
        ],
    },
    {
        "slug": "exp06_dtw_pathology",
        "exp": "Exp 6",
        "title": "Pathology-specific DTW",
        "subtitle": "Build one descriptor template per DISEASE label and select it at encoding time.",
        "family": "Image-space preprocessing",
        "status": "implemented; needs sidecars",
        "status_color": "gold",
        "config": "configs/transformer/test_configs/interp_dtw_pathology/",
        "code": "src/scripts/build_dtw_template.py --group_by_label",
        "draw": "dtw_pathology",
        "notes": [
            "Creates label-specific templates plus a global fallback template.",
            "Uses dataset_information.csv to map each subject to its DISEASE label.",
            "Tests whether disease-specific motion timing is a better fixed preprocessing target.",
        ],
    },
    {
        "slug": "exp07_motionfield_warp",
        "exp": "Exp 7",
        "title": "Motion-field interpolation",
        "subtitle": "Use dense deformation fields phi_t to warp frames to target timestamps.",
        "family": "Image-space preprocessing",
        "status": "blocked on phi_t sidecars",
        "status_color": "red",
        "config": "configs/transformer/test_configs/interp_motionfield/",
        "code": "src/utils/temporal_align.py::motionfield_warp",
        "draw": "motionfield",
        "notes": [
            "Planned interpolation is I_i warped by alpha * phi_i for each fractional target time.",
            "Designed to preserve sharp structures better than intensity blending.",
            "Current implementation intentionally raises NotImplementedError until the sidecar format is confirmed.",
        ],
    },
    {
        "slug": "exp08_varivit_pe",
        "exp": "Exp 8",
        "title": "Temporal PE interpolation",
        "subtitle": "Keep native-T latents and recompute the T-axis sincos PE for the actual Tg.",
        "family": "Architecture-side native-T",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/varivit_pe/",
        "code": "src/models/dit.py::DiT4D._build_pos_embed",
        "draw": "varivit",
        "notes": [
            "Encoding uses align(method='native') to pad image T to a valid multiple.",
            "DiT keeps the trained temporal span fixed and samples Tg positions inside it.",
            "Bucket-by-T batching keeps every training batch at one latent length.",
        ],
    },
    {
        "slug": "exp09_phase_pe",
        "exp": "Exp 9",
        "title": "Continuous phase encoding",
        "subtitle": "Replace integer temporal positions with sincos(alpha_t), a physiologic phase signal.",
        "family": "Architecture-side native-T",
        "status": "blocked on alpha_t sidecars",
        "status_color": "red",
        "config": "configs/transformer/test_configs/phase_pe/",
        "code": "src/models/dit.py::temporal_pe='phase'",
        "draw": "phase",
        "notes": [
            "LatentDataset loads 1D alpha_t and average-pools image T to latent T by factor 4.",
            "DiT receives alpha_t through forward(..., alpha_t=...).",
            "The temporal PE becomes cardiac-cycle phase rather than frame index.",
        ],
    },
    {
        "slug": "exp10_rope_t",
        "exp": "Exp 10",
        "title": "RoPE on the temporal axis",
        "subtitle": "Keep native-T latents and encode relative temporal distance inside attention.",
        "family": "Architecture-side native-T",
        "status": "implemented",
        "status_color": "green",
        "config": "configs/transformer/test_configs/rope_t/",
        "code": "src/models/attention.py::apply_rope_t",
        "draw": "rope",
        "notes": [
            "Z/X/Y keep fixed sincos PE; the T slice is zeroed in the additive PE.",
            "Attention rotates a slice of Q and K according to each token's T coordinate.",
            "This removes the fixed-T PE table requirement and supports variable Tg.",
        ],
    },
]


def add_text(ax, x, y, text, size=11, color=None, weight="normal", ha="left", va="top", wrap=None):
    if wrap is not None:
        text = "\n".join(fill(line, wrap) for line in text.splitlines())
    return ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color or PALETTE["ink"],
        fontweight=weight,
        family="DejaVu Sans",
    )


def add_badge(ax, x, y, text, color_key, width=None):
    width = width or max(15, len(text) * 0.64)
    box = patches.FancyBboxPatch(
        (x, y - 2.1),
        width,
        3.4,
        boxstyle="round,pad=0.35,rounding_size=1.5",
        linewidth=0,
        facecolor=PALETTE[color_key],
    )
    ax.add_patch(box)
    add_text(ax, x + width / 2, y - 0.08, text, size=9.5, color="white", weight="bold", ha="center", va="center")


def add_round_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body=None,
    fc=None,
    ec=None,
    title_color=None,
    body_size=9.2,
):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.55,rounding_size=1.3",
        linewidth=1.2,
        edgecolor=ec or PALETTE["line"],
        facecolor=fc or "white",
    )
    ax.add_patch(patch)
    add_text(ax, x + 1.3, y + h - 1.3, title, size=10.5, weight="bold", color=title_color or PALETTE["ink"])
    if body:
        add_text(ax, x + 1.3, y + h - 4.5, body, size=body_size, color=PALETTE["muted"], wrap=max(18, int(w * 1.6)))
    return patch


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.8, style="-|>", mutation=12, alpha=1.0):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color or PALETTE["muted"],
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=mutation,
            alpha=alpha,
        ),
    )


def setup(method):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 56), 100, 8, facecolor="#eef3fa", edgecolor="none"))
    ax.add_patch(patches.Rectangle((0, 55.7), 100, 0.35, facecolor=PALETTE[method["status_color"]], edgecolor="none"))
    add_text(ax, 4, 61.6, f"{method['exp']} | {method['title']}", size=22, weight="bold", va="top")
    add_text(ax, 4, 57.9, method["subtitle"], size=11.5, color=PALETTE["muted"], va="top")
    add_badge(ax, 71, 61.1, method["family"], "blue", width=20.5)
    add_badge(ax, 71, 57.8, method["status"], method["status_color"], width=20.5)
    return fig, ax


def add_pipeline(ax, center_title, center_body, center_color):
    y = 46.5
    add_round_box(ax, 4, y, 18, 6.7, "Raw cine", "T_native varies, usually 20-30 frames", fc="#ffffff")
    add_round_box(ax, 29, y - 0.3, 22, 7.3, center_title, center_body, fc="#fbfdff", ec=PALETTE[center_color], title_color=PALETTE[center_color])
    add_round_box(ax, 58, y, 17, 6.7, "VQ-GAN", "Encode image-space frames to 4D latents", fc="#ffffff")
    add_round_box(ax, 82, y, 14, 6.7, "DiT4D", "Train or sample latent video", fc="#ffffff")
    arrow(ax, 22.7, y + 3.3, 28.2, y + 3.3)
    arrow(ax, 51.8, y + 3.3, 57.2, y + 3.3)
    arrow(ax, 75.7, y + 3.3, 81.2, y + 3.3)


def frame_strip(ax, x0, x1, y, n, color, label=None, every_label=8, height=3.0, edge=None, alpha=1.0):
    gap = (x1 - x0) / n
    w = gap * 0.72
    for i in range(n):
        fc = color(i) if callable(color) else color
        rect = patches.FancyBboxPatch(
            (x0 + i * gap, y),
            w,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.25",
            linewidth=0.75,
            edgecolor=edge or "white",
            facecolor=fc,
            alpha=alpha,
        )
        ax.add_patch(rect)
        if every_label and (i == 0 or (i + 1) % every_label == 0 or i == n - 1):
            add_text(ax, x0 + i * gap + w / 2, y - 0.7, str(i), size=6.8, color=PALETTE["muted"], ha="center", va="top")
    if label:
        add_text(ax, x0, y + height + 1.1, label, size=10.2, weight="bold")


def curve(ax, x0, x1, y0, amp, n=300, phase=0, color=None, lw=2.4):
    xs = np.linspace(x0, x1, n)
    t = np.linspace(0, 2 * np.pi, n) + phase
    ys = y0 + amp * (np.sin(t) + 0.25 * np.sin(2 * t + 0.9))
    ax.plot(xs, ys, color=color or PALETTE["blue"], lw=lw)
    return xs, ys


def sample_points(ax, xs, ys, count, color, marker="o", size=34, edge="white"):
    idx = np.linspace(0, len(xs) - 1, count).astype(int)
    ax.scatter(xs[idx], ys[idx], s=size, marker=marker, color=color, edgecolor=edge, linewidth=0.9, zorder=5)


def add_footer(ax, method):
    y = 1.6
    h = 12.4
    add_round_box(ax, 4, y, 28.5, h, "Implementation hook", method["code"], fc=PALETTE["panel"], body_size=8.8)
    add_round_box(ax, 36, y, 24.5, h, "Experiment config", method["config"], fc=PALETTE["panel"], body_size=8.8)
    notes = "\n".join([f"- {n}" for n in method["notes"]])
    add_round_box(ax, 64, y, 32, h, "Behavior summary", notes, fc=PALETTE["panel"], body_size=8.45)


def draw_cyclic(ax):
    add_pipeline(ax, "Repeat + crop", "out[t] = x[t mod T_native], then keep first 32", "blue")
    add_text(ax, 6, 43.0, "Native timeline", size=10.5, weight="bold")
    frame_strip(ax, 6, 72, 38.3, 24, PALETTE["blue"], every_label=6)
    add_text(ax, 6, 35.0, "Fixed output timeline T=32", size=10.5, weight="bold")

    def out_color(i):
        return PALETTE["blue"] if i < 24 else PALETTE["gold"]

    frame_strip(ax, 6, 92, 30.3, 32, out_color, every_label=8)
    add_text(ax, 74.5, 38.7, "repeat f0..f7", size=10, color=PALETTE["gold"], weight="bold")
    for i in range(8):
        x_src = 6 + i * ((72 - 6) / 24) + 0.8
        x_dst = 6 + (24 + i) * ((92 - 6) / 32) + 0.8
        arrow(ax, x_src, 38.0, x_dst, 33.8, color=PALETTE["gold"], lw=1.1, mutation=8, alpha=0.75)
    add_text(ax, 6, 24.8, "Result: every subject becomes T=32 before encoding, but the cycle boundary can appear inside the generated sequence.", size=11, color=PALETTE["muted"], wrap=112)


def draw_linear(ax):
    add_pipeline(ax, "Linear resample", "Uniform target grid; interpolate between adjacent source frames", "teal")
    xs, ys = curve(ax, 8, 92, 34.2, 5.8, color=PALETTE["teal"], lw=2.7)
    sample_points(ax, xs, ys, 20, PALETTE["blue"], size=42)
    sample_points(ax, xs, ys, 32, PALETTE["gold"], marker="s", size=25, edge="white")
    add_text(ax, 8, 43.0, "Same per-voxel signal sampled on two temporal grids", size=10.5, weight="bold")
    add_badge(ax, 8, 28.0, "source frames", "blue", width=13.5)
    add_badge(ax, 23, 28.0, "T=32 targets", "gold", width=13.5)
    for x in (39, 48, 57):
        arrow(ax, x - 3, 25.7, x, 27.4, color=PALETTE["teal"], lw=1.2, mutation=9)
    add_text(ax, 40, 27.2, "target frame = weighted average of neighboring frames", size=10.5, color=PALETTE["muted"])
    add_text(ax, 8, 22.2, "Result: a smooth fixed-length input to VQ-GAN and unchanged absolute temporal PE in DiT.", size=11, color=PALETTE["muted"], wrap=112)


def draw_piecewise(ax):
    add_pipeline(ax, "Keyframe warp", "Circular piecewise-linear mapping by ED/MS/ES/PF/MD", "green")
    phases = ["ED", "MS", "ES", "PF", "MD"]
    colors = [PALETTE["red"], PALETTE["gold"], PALETTE["blue"], PALETTE["teal"], PALETTE["purple"]]
    src = np.array([8, 28, 43, 62, 84], dtype=float)
    tgt = np.array([8, 24, 40, 58, 82], dtype=float)
    y_src, y_tgt = 39.5, 29.5
    ax.plot([6, 94], [y_src, y_src], color=PALETTE["line"], lw=2)
    ax.plot([6, 94], [y_tgt, y_tgt], color=PALETTE["line"], lw=2)
    add_text(ax, 6, 43.0, "Detected native keyframes, irregular per subject", size=10.5, weight="bold")
    add_text(ax, 6, 33.0, "Shared T=32 anchors: 0 / 6 / 12 / 19 / 28", size=10.5, weight="bold")
    for i, (p, c) in enumerate(zip(phases, colors)):
        ax.scatter([src[i]], [y_src], s=170, color=c, edgecolor="white", linewidth=1.8, zorder=4)
        ax.scatter([tgt[i]], [y_tgt], s=170, color=c, edgecolor="white", linewidth=1.8, zorder=4)
        arrow(ax, src[i], y_src - 1.3, tgt[i], y_tgt + 1.5, color=c, lw=1.3, mutation=8, alpha=0.8)
        add_text(ax, src[i], y_src + 2.6, p, size=9.4, color=c, weight="bold", ha="center")
        add_text(ax, tgt[i], y_tgt - 2.6, p, size=9.4, color=c, weight="bold", ha="center", va="top")
    for i in range(len(tgt) - 1):
        ax.plot([tgt[i], tgt[i + 1]], [y_tgt, y_tgt], color=colors[i], lw=5, alpha=0.28, solid_capstyle="round")
    add_text(ax, 6, 22.4, "Between anchor pairs, source time is linearly sampled. ED is treated as circular frame zero, so wrapped cycles are handled explicitly.", size=11, color=PALETTE["muted"], wrap=112)


def draw_dtw(ax):
    add_pipeline(ax, "Descriptor DTW", "Align alpha_t to a canonical template, then resample images", "cyan")
    xs, ys = curve(ax, 7, 47, 38.3, 4.7, color=PALETTE["blue"], lw=2.4)
    sample_points(ax, xs, ys, 24, PALETTE["blue"], size=32)
    add_text(ax, 7, 43.4, "Subject descriptor alpha_t", size=10.5, weight="bold")

    xt, yt = curve(ax, 56, 94, 38.3, 4.2, phase=0.35, color=PALETTE["gold"], lw=2.4)
    sample_points(ax, xt, yt, 32, PALETTE["gold"], marker="s", size=23)
    add_text(ax, 56, 43.4, "Canonical T=32 template", size=10.5, weight="bold")

    arrow(ax, 48.2, 38.3, 54.0, 38.3, color=PALETTE["cyan"])
    add_round_box(ax, 9, 18.5, 24, 9.0, "Template build", "Median keyframe intervals + interval-wise DTW refinement", fc="#fbfdff", ec=PALETTE["cyan"], title_color=PALETTE["cyan"], body_size=8.4)
    add_round_box(ax, 38, 18.5, 24, 9.0, "DTW path", "One source-time position per target frame", fc="#fbfdff", ec=PALETTE["cyan"], title_color=PALETTE["cyan"], body_size=8.4)
    add_round_box(ax, 67, 18.5, 24, 9.0, "Image resample", "Apply the mapping to raw cine frames before VQ-GAN", fc="#fbfdff", ec=PALETTE["cyan"], title_color=PALETTE["cyan"], body_size=8.4)
    arrow(ax, 33.8, 23.0, 37.0, 23.0, color=PALETTE["cyan"], mutation=9)
    arrow(ax, 62.8, 23.0, 66.0, 23.0, color=PALETTE["cyan"], mutation=9)
    add_text(ax, 7, 14.6, "Result: each subject is converted to fixed T=32 using a physiology-derived time warp while the DiT architecture remains unchanged.", size=11, color=PALETTE["muted"], wrap=112)


def draw_dtw_pathology(ax):
    add_pipeline(ax, "Label template", "Select a disease-specific template before DTW resampling", "gold")
    labels = [("NOR", PALETTE["green"]), ("HCM", PALETTE["purple"]), ("ARR", PALETTE["red"]), ("LV", PALETTE["blue"])]
    add_round_box(ax, 7, 34.5, 24, 8.4, "Metadata", "dataset_information.csv provides the DISEASE label", fc="#fbfdff", ec=PALETTE["gold"], title_color=PALETTE["gold"], body_size=8.4)
    for i, (label, color) in enumerate(labels):
        add_badge(ax, 39, 41.4 - i * 3.7, f"{label}.pt", "green" if label == "NOR" else ("purple" if label == "HCM" else ("red" if label == "ARR" else "blue")), width=11.5)
        arrow(ax, 31.8, 38.5, 38.0, 41.0 - i * 3.7, color=color, lw=1.1, mutation=8, alpha=0.75)
    add_round_box(ax, 60, 34.5, 30, 8.4, "Template selection", "Use subject label; fall back to global.pt for missing groups", fc="#fbfdff", ec=PALETTE["gold"], title_color=PALETTE["gold"], body_size=8.4)
    arrow(ax, 52, 38.5, 59, 38.5, color=PALETTE["gold"])

    xs, ys = curve(ax, 9, 44, 23.5, 3.6, color=PALETTE["blue"], lw=2.2)
    sample_points(ax, xs, ys, 24, PALETTE["blue"], size=27)
    xt, yt = curve(ax, 56, 92, 23.5, 3.1, phase=0.6, color=PALETTE["gold"], lw=2.2)
    sample_points(ax, xt, yt, 32, PALETTE["gold"], marker="s", size=22)
    arrow(ax, 46.5, 23.5, 54.0, 23.5, color=PALETTE["gold"])
    add_text(ax, 9, 29.1, "Subject descriptor", size=10.5, weight="bold")
    add_text(ax, 56, 29.1, "Selected disease template", size=10.5, weight="bold")
    add_text(ax, 7, 14.0, "Result: still a fixed preprocessing transform, but the DTW target can reflect disease-specific cardiac timing.", size=11, color=PALETTE["muted"], wrap=112)


def draw_varivit(ax):
    add_pipeline(ax, "Native T + PE", "Pad to valid multiple; do not resample frames", "purple")
    frame_strip(ax, 7, 56, 39.0, 24, PALETTE["blue"], label="Image frames kept at native cadence", every_label=6)
    arrow(ax, 58, 40.5, 68, 40.5, color=PALETTE["purple"])
    frame_strip(ax, 70, 92, 39.0, 6, PALETTE["teal"], label="Latent Tg = T / 4", every_label=2)

    add_text(ax, 7, 31.5, "Temporal positional embedding range", size=10.5, weight="bold")
    xs, ys = curve(ax, 11, 90, 26.8, 3.7, color=PALETTE["gray"], lw=2.1)
    sample_points(ax, xs, ys, 8, PALETTE["gray"], size=46)
    sample_points(ax, xs, ys, 6, PALETTE["purple"], marker="s", size=42)
    add_text(ax, 11, 21.7, "default Tg=8 samples", size=9.5, color=PALETTE["gray"])
    add_text(ax, 47, 21.7, "actual Tg samples placed inside the same trained span", size=9.5, color=PALETTE["purple"], weight="bold")
    add_text(ax, 7, 17.0, "Result: the model adapts its T-axis PE to the latent length while Z/X/Y PE tables stay fixed.", size=11, color=PALETTE["muted"], wrap=112)


def draw_motionfield(ax):
    add_pipeline(ax, "Warp by phi_t", "Use displacement fields instead of intensity blending", "red")
    add_frame_with_grid(ax, 9, 28, "I_i", PALETTE["blue"], offset=0.0)
    add_frame_with_grid(ax, 40, 28, "I_i warped by alpha * phi_i", PALETTE["gold"], offset=1.0)
    add_frame_with_grid(ax, 73, 28, "target frame", PALETTE["teal"], offset=0.55)
    arrow(ax, 27.5, 35.5, 38.3, 35.5, color=PALETTE["red"])
    arrow(ax, 60.5, 35.5, 71.2, 35.5, color=PALETTE["red"])
    for j in range(5):
        x = 33 + j * 3.1
        y = 24 + (j % 3) * 4.2
        arrow(ax, x, y, x + 2.2, y + 0.8, color=PALETTE["red"], lw=1.0, mutation=7)
    add_text(ax, 33, 23.0, "dense phi_t field", size=9.5, color=PALETTE["red"], weight="bold")
    add_text(ax, 9, 19.0, "Planned operation: for each target time t*, choose neighboring source frame i and fractional offset alpha, then spatially warp I_i by alpha * phi_i.", size=11, color=PALETTE["muted"], wrap=118)


def add_frame_with_grid(ax, x, y, label, color, offset=0.0):
    box = patches.FancyBboxPatch((x, y), 18, 15, boxstyle="round,pad=0.35,rounding_size=1.1", linewidth=1.2, edgecolor=color, facecolor="#fbfdff")
    ax.add_patch(box)
    for k in range(4):
        xx = x + 3.5 + k * 3.2 + offset * np.sin(k)
        ax.plot([xx, xx + offset * 0.8], [y + 2.6, y + 12.4], color=PALETTE["line"], lw=0.9)
        yy = y + 3.2 + k * 2.7 + offset * np.cos(k)
        ax.plot([x + 2.5, x + 15.8], [yy, yy + offset * 0.8], color=PALETTE["line"], lw=0.9)
    heart = patches.Ellipse((x + 9 + offset * 1.4, y + 7.5 + offset * 0.6), 5.0, 6.7, angle=8 + offset * 14, facecolor=color, edgecolor="white", alpha=0.8)
    ax.add_patch(heart)
    add_text(ax, x + 9, y + 13.5, label, size=8.4, color=color, weight="bold", ha="center", va="center", wrap=20)


def draw_phase(ax):
    add_pipeline(ax, "Native T + alpha_t", "Use physiologic phase signal as temporal position", "cyan")
    frame_strip(ax, 7, 55, 39, 24, PALETTE["blue"], label="Native image frames", every_label=6)
    arrow(ax, 57, 40.6, 68, 40.6, color=PALETTE["cyan"])
    frame_strip(ax, 70, 91, 39, 6, PALETTE["teal"], label="Latent Tg", every_label=2)
    xs = np.linspace(8, 92, 240)
    t = np.linspace(0, 2 * np.pi, 240)
    ys = 28.5 + 5.0 * np.cos(t - 0.8)
    ax.plot(xs, ys, color=PALETTE["cyan"], lw=2.6)
    ax.axhline(28.5, 8, 92, color=PALETTE["line"], lw=1.0)
    sample_points(ax, xs, ys, 24, PALETTE["blue"], size=30)
    sample_points(ax, xs, ys, 6, PALETTE["gold"], marker="s", size=44)
    add_text(ax, 8, 35.3, "alpha_t descriptor in [-1, 1]", size=10.5, weight="bold")
    add_text(ax, 10, 20.8, "LatentDataset average-pools alpha_t from image-T to latent-T, then DiT builds sincos((alpha_t + 1) * pi).", size=11, color=PALETTE["muted"], wrap=112)


def draw_rope(ax):
    add_pipeline(ax, "Native T + RoPE", "Temporal coordinates rotate Q and K inside attention", "purple")
    frame_strip(ax, 7, 54, 39.5, 24, PALETTE["blue"], label="Native frames, no T=32 resampling", every_label=6)
    add_round_box(ax, 62, 33.8, 29, 8.6, "Additive PE", "Z/X/Y sincos tables are kept; T slice is zeroed.", fc="#fbfdff", ec=PALETTE["purple"], title_color=PALETTE["purple"], body_size=8.6)
    add_text(ax, 8, 31.8, "Attention step", size=10.5, weight="bold")
    centers = [(20, 25.5), (38, 25.5), (56, 25.5), (74, 25.5)]
    for idx, (cx, cy) in enumerate(centers):
        circ = patches.Circle((cx, cy), 5.0, facecolor="#fbfdff", edgecolor=PALETTE["line"], lw=1.2)
        ax.add_patch(circ)
        angle = 0.45 + idx * 0.6
        ax.plot([cx, cx + 4.0 * np.cos(angle)], [cy, cy + 4.0 * np.sin(angle)], color=PALETTE["purple"], lw=2.0)
        ax.scatter([cx], [cy], s=16, color=PALETTE["purple"])
        add_text(ax, cx, cy - 6.4, f"t={idx}", size=9, color=PALETTE["muted"], ha="center", va="top")
    arrow(ax, 25, 25.5, 33, 25.5, color=PALETTE["gray"], lw=1.2, mutation=8)
    arrow(ax, 43, 25.5, 51, 25.5, color=PALETTE["gray"], lw=1.2, mutation=8)
    arrow(ax, 61, 25.5, 69, 25.5, color=PALETTE["gray"], lw=1.2, mutation=8)
    add_text(ax, 8, 17.2, "apply_rope_t rotates the first rope_t_dim slice of Q and K by t_coord * omega. Relative temporal offsets are represented in attention scores.", size=10.1, color=PALETTE["muted"], wrap=118)


def draw_fourier(ax):
    add_pipeline(ax, "Frequency resample", "rFFT -> truncate or zero-pad -> irFFT", "gold")
    xs, ys = curve(ax, 7, 44, 36.0, 5.1, color=PALETTE["blue"], lw=2.6)
    sample_points(ax, xs, ys, 24, PALETTE["blue"], size=34)
    add_text(ax, 7, 43.2, "Native periodic cine signal", size=10.5, weight="bold")
    arrow(ax, 46, 36.0, 55, 36.0, color=PALETTE["gold"])
    add_spectrum(ax, 57, 29, 24, 15)
    arrow(ax, 82, 36.0, 91, 36.0, color=PALETTE["gold"])
    xs2, ys2 = curve(ax, 7, 92, 22.7, 3.2, color=PALETTE["teal"], lw=2.3)
    sample_points(ax, xs2, ys2, 32, PALETTE["teal"], marker="s", size=24)
    add_text(ax, 57, 45.0, "Low temporal frequencies carry the cycle", size=10.5, weight="bold")
    add_text(ax, 7, 17.6, "Formula: out = irfft(pad_or_truncate(rfft(x)), n=T_out) * (T_out / T_in). The output is fixed T=32 before VQ-GAN.", size=11, color=PALETTE["muted"], wrap=112)


def add_spectrum(ax, x, y, w, h):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", linewidth=1.1, edgecolor=PALETTE["line"], facecolor="#fbfdff"))
    heights = [10, 7.3, 5.1, 3.2, 2.2, 1.3, 0.8, 0.5]
    for i, ht in enumerate(heights):
        bx = x + 2.5 + i * 2.4
        ax.add_patch(patches.Rectangle((bx, y + 2), 1.25, ht, facecolor=PALETTE["gold"] if i < 5 else PALETTE["line"], edgecolor="none"))
    add_text(ax, x + w / 2, y - 1.0, "rFFT coefficients", size=9.5, color=PALETTE["muted"], ha="center", va="top")


DRAWERS = {
    "cyclic": draw_cyclic,
    "linear": draw_linear,
    "piecewise": draw_piecewise,
    "dtw": draw_dtw,
    "dtw_pathology": draw_dtw_pathology,
    "varivit": draw_varivit,
    "motionfield": draw_motionfield,
    "phase": draw_phase,
    "rope": draw_rope,
    "fourier": draw_fourier,
}


def render_method(method, output_dir: Path, dpi: int):
    fig, ax = setup(method)
    DRAWERS[method["draw"]](ax)
    add_footer(ax, method)

    png = output_dir / f"{method['slug']}.png"
    svg = output_dir / f"{method['slug']}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def write_index(output_dir: Path, rendered: list[tuple[Path, Path]]):
    lines = [
        "# Temporal Alignment Figures",
        "",
        "Generated by `src/scripts/make_temporal_alignment_figures.py`.",
        "",
    ]
    for method, (png, svg) in zip(METHODS, rendered):
        lines.extend(
            [
                f"## {method['exp']} - {method['title']}",
                "",
                f"- PNG: `{png.name}`",
                f"- SVG: `{svg.name}`",
                f"- Status: {method['status']}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "temporal_alignment" / "figures",
        help="Directory for generated PNG/SVG figures.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG rendering DPI.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = [render_method(method, args.output_dir, args.dpi) for method in METHODS]
    write_index(args.output_dir, rendered)
    for png, svg in rendered:
        print(png)
        print(svg)


if __name__ == "__main__":
    main()

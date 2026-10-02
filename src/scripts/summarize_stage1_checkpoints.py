"""
Summarize stage1 VQ-GAN evaluation directories across best and last checkpoints.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_ORDER = [
    "S1_ds4_all_dims_fixed32",
    "S1_ds4_all_dims_paddiv",
    "S1_ds4xy_noT_native",
    "S1_ds8xy_noT_native",
]

DISPLAY_NAMES = {
    "S1_ds4_all_dims_fixed32": "ds4 fixed32",
    "S1_ds4_all_dims_paddiv": "ds4 paddiv",
    "S1_ds4xy_noT_native": "ds4xy noT",
    "S1_ds8xy_noT_native": "ds8xy noT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stage1 best and last checkpoint evaluations.")
    parser.add_argument("--best_dir", default=str(Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "outputs/stage1_eval_center"))
    parser.add_argument("--last_dir", default=str(Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "outputs/stage1_eval_center_last"))
    parser.add_argument("--output_dir", default=str(Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")) / "outputs/stage1_eval_center_comparison"))
    return parser.parse_args()


def display_name(run_name: str) -> str:
    return DISPLAY_NAMES.get(run_name, run_name)


def load_summary(eval_dir: Path, checkpoint_type: str) -> pd.DataFrame:
    path = eval_dir / "metrics_summary.csv"
    df = pd.read_csv(path)
    df["checkpoint_type"] = checkpoint_type
    df["display_name"] = df["run"].map(display_name)
    if "checkpoint_name" not in df.columns:
        df["checkpoint_name"] = df["checkpoint"].map(lambda p: Path(str(p)).name)
    df["run_order"] = df["run"].map({name: idx for idx, name in enumerate(RUN_ORDER)})
    return df


def rank_all(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["psnr_rank"] = ranked["psnr"].rank(ascending=False, method="min")
    ranked["ssim_rank"] = ranked["ssim"].rank(ascending=False, method="min")
    ranked["mae_rank"] = ranked["mae"].rank(ascending=True, method="min")
    ranked["mean_reconstruction_rank"] = ranked[["psnr_rank", "ssim_rank", "mae_rank"]].mean(axis=1)
    ranked = ranked.sort_values(["mean_reconstruction_rank", "mae", "run_order", "checkpoint_type"])
    ranked.insert(0, "overall_rank", np.arange(1, len(ranked) + 1))
    return ranked


def load_subject_metrics(eval_dir: Path, checkpoint_type: str) -> pd.DataFrame:
    frames = []
    for run_name in RUN_ORDER:
        path = eval_dir / run_name / "metrics_by_subject.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["checkpoint_type"] = checkpoint_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_best_last(summary: pd.DataFrame, subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_name in RUN_ORDER:
        pair = summary[summary["run"] == run_name].set_index("checkpoint_type")
        if not {"best", "last"}.issubset(pair.index):
            continue
        best = pair.loc["best"]
        last = pair.loc["last"]
        wins = {
            "psnr": "last" if last["psnr"] > best["psnr"] else "best",
            "ssim": "last" if last["ssim"] > best["ssim"] else "best",
            "mae": "last" if last["mae"] < best["mae"] else "best",
        }
        last_votes = sum(value == "last" for value in wins.values())

        subjects = subject_metrics[subject_metrics["run"] == run_name]
        subject_wide = subjects.pivot(index="subject", columns="checkpoint_type", values=["psnr", "ssim", "mae"])
        psnr_improved = float((subject_wide["psnr"]["last"] > subject_wide["psnr"]["best"]).mean())
        ssim_improved = float((subject_wide["ssim"]["last"] > subject_wide["ssim"]["best"]).mean())
        mae_improved = float((subject_wide["mae"]["last"] < subject_wide["mae"]["best"]).mean())

        rows.append(
            {
                "run": run_name,
                "display_name": display_name(run_name),
                "winner": "last" if last_votes >= 2 else "best",
                "last_metric_votes": last_votes,
                "delta_psnr_last_minus_best": last["psnr"] - best["psnr"],
                "delta_ssim_last_minus_best": last["ssim"] - best["ssim"],
                "delta_mae_last_minus_best": last["mae"] - best["mae"],
                "subjects_last_better_psnr_fraction": psnr_improved,
                "subjects_last_better_ssim_fraction": ssim_improved,
                "subjects_last_better_mae_fraction": mae_improved,
                "best_psnr": best["psnr"],
                "last_psnr": last["psnr"],
                "best_ssim": best["ssim"],
                "last_ssim": last["ssim"],
                "best_mae": best["mae"],
                "last_mae": last["mae"],
            }
        )
    return pd.DataFrame(rows)


def paired_subject_deltas(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_name in RUN_ORDER:
        subjects = subject_metrics[subject_metrics["run"] == run_name]
        if subjects.empty:
            continue
        wide = subjects.pivot(index="subject", columns="checkpoint_type", values=["psnr", "ssim", "mae"])
        if not {"best", "last"}.issubset(set(wide["psnr"].columns)):
            continue
        for subject in wide.index:
            rows.append(
                {
                    "run": run_name,
                    "display_name": display_name(run_name),
                    "subject": subject,
                    "delta_psnr_last_minus_best": wide.loc[subject, ("psnr", "last")] - wide.loc[subject, ("psnr", "best")],
                    "delta_ssim_last_minus_best": wide.loc[subject, ("ssim", "last")] - wide.loc[subject, ("ssim", "best")],
                    "delta_mae_last_minus_best": wide.loc[subject, ("mae", "last")] - wide.loc[subject, ("mae", "best")],
                }
            )
    return pd.DataFrame(rows)


def save_metric_comparison(ranked: pd.DataFrame, output_dir: Path) -> None:
    ordered = ranked.sort_values(["run_order", "checkpoint_type"])
    labels = [f"{display_name(row.run)}\n{row.checkpoint_type}" for row in ordered.itertuples()]
    x = np.arange(len(ordered))
    metrics = [
        ("psnr", "PSNR, dB\nhigher is better"),
        ("ssim", "SSIM\nhigher is better"),
        ("mae", "MAE\nlower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    colors = ordered["checkpoint_type"].map({"best": "#2f6f77", "last": "#8a5a44"}).tolist()
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(x, ordered[metric], color=colors, width=0.72)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "all_checkpoint_metric_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_best_last_delta_plot(delta: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(delta))
    labels = delta["display_name"].tolist()
    metrics = [
        ("delta_psnr_last_minus_best", "last - best PSNR\npositive favors last"),
        ("delta_ssim_last_minus_best", "last - best SSIM\npositive favors last"),
        ("delta_mae_last_minus_best", "last - best MAE\nnegative favors last"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (metric, title) in zip(axes, metrics):
        values = delta[metric].to_numpy()
        colors = ["#2f6f77" if v >= 0 else "#8a5a44" for v in values]
        if metric == "delta_mae_last_minus_best":
            colors = ["#2f6f77" if v <= 0 else "#8a5a44" for v in values]
        ax.axhline(0, color="#202020", linewidth=0.8)
        ax.bar(x, values, color=colors, width=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "best_vs_last_deltas.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_subject_delta_plot(subject_delta: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    data = [
        subject_delta[subject_delta["run"] == run]["delta_psnr_last_minus_best"].to_numpy()
        for run in RUN_ORDER
    ]
    ax.axhline(0, color="#202020", linewidth=0.8)
    ax.boxplot(data, tick_labels=[display_name(run) for run in RUN_ORDER], showfliers=False)
    ax.set_title("Subject-wise PSNR delta, last - best", fontsize=10)
    ax.set_ylabel("dB")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "subject_psnr_delta_boxplot.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_codebook_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values(["run_order", "checkpoint_type"])
    labels = [f"{display_name(row.run)}\n{row.checkpoint_type}" for row in ordered.itertuples()]
    x = np.arange(len(ordered))
    metrics = [
        ("normalized_perplexity", "effective code ratio\nhigher is better"),
        ("top10_code_fraction", "top-10 token share\nlower is better"),
        ("usage_ratio", "used code ratio\nhigher is better"),
        ("dead_codes", "dead codes\nlower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    colors = ordered["checkpoint_type"].map({"best": "#2f6f77", "last": "#8a5a44"}).tolist()
    for ax, (metric, title) in zip(axes.flat, metrics):
        ax.bar(x, ordered[metric], color=colors, width=0.72)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "all_checkpoint_codebook_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dir = Path(args.best_dir)
    last_dir = Path(args.last_dir)
    summary = pd.concat(
        [
            load_summary(best_dir, "best"),
            load_summary(last_dir, "last"),
        ],
        ignore_index=True,
    )
    subject_metrics = pd.concat(
        [
            load_subject_metrics(best_dir, "best"),
            load_subject_metrics(last_dir, "last"),
        ],
        ignore_index=True,
    )

    ranked = rank_all(summary)
    deltas = paired_best_last(summary, subject_metrics)
    subject_delta = paired_subject_deltas(subject_metrics)

    summary.to_csv(output_dir / "combined_metrics_summary.csv", index=False)
    ranked.to_csv(output_dir / "combined_ranking.csv", index=False)
    deltas.to_csv(output_dir / "best_vs_last_summary.csv", index=False)
    subject_delta.to_csv(output_dir / "per_subject_best_vs_last_deltas.csv", index=False)

    save_metric_comparison(ranked, output_dir)
    save_best_last_delta_plot(deltas, output_dir)
    save_subject_delta_plot(subject_delta, output_dir)
    save_codebook_comparison(summary, output_dir)

    print(f"Saved combined summaries and plots under: {output_dir}")


if __name__ == "__main__":
    main()

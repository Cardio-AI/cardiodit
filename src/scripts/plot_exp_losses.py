#!/usr/bin/env python3
"""Plot train and validation losses for temporal-alignment experiments 1-3."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()

# The repository has a local wandb/ run directory. When this script is executed
# from the repo root, make sure imports resolve to the installed wandb package.
sys.path = [
    path
    for path in sys.path
    if path not in ("", str(REPO_ROOT))
]

from wandb.proto import wandb_internal_pb2 as wandb_pb  # noqa: E402
from wandb.sdk.internal.datastore import DataStore  # noqa: E402


@dataclass(frozen=True)
class Experiment:
    exp_id: int
    name: str
    run_id: str
    run_dir: Path


@dataclass
class LossHistory:
    train_epoch_mean: list[tuple[float, float]]
    val_loss: list[tuple[float, float]]


EXPERIMENTS = [
    Experiment(
        exp_id=1,
        name="Cyclic repetition",
        run_id="w5luqzxu",
        run_dir=RUNS_ROOT / "wandb" / "run-20260528_111028-w5luqzxu",
    ),
    Experiment(
        exp_id=2,
        name="Linear interpolation",
        run_id="kskdtvyq",
        run_dir=RUNS_ROOT / "wandb" / "run-20260529_204716-kskdtvyq",
    ),
    Experiment(
        exp_id=3,
        name="Piecewise keyframe alignment",
        run_id="tnc92yln",
        run_dir=RUNS_ROOT / "wandb" / "run-20260609_161744-tnc92yln",
    ),
]

TRAIN_PLOT_START_EPOCH = 600.0


def _centered_rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values

    radius = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:stop]) / (stop - start))
    return smoothed


def _from_epoch(
    points: list[tuple[float, float]],
    start_epoch: float,
) -> list[tuple[float, float]]:
    return [(epoch, value) for epoch, value in points if epoch >= start_epoch]


def _history_item_key(item) -> str:
    if item.nested_key:
        return "/".join(item.nested_key)
    return item.key


def _read_history_records(path: Path) -> list[dict[str, float]]:
    datastore = DataStore()
    datastore.open_for_scan(str(path))
    rows: list[dict[str, float]] = []

    try:
        while True:
            data = datastore.scan_data()
            if data is None:
                break

            record = wandb_pb.Record()
            record.ParseFromString(data)
            if not record.HasField("history"):
                continue

            row = {}
            for item in record.history.item:
                key = _history_item_key(item)
                if not key:
                    continue
                row[key] = json.loads(item.value_json)

            if row:
                rows.append(row)
    finally:
        datastore.close()

    return rows


def _infer_steps_per_epoch(rows: list[dict[str, float]]) -> int:
    train_steps = [int(row["_step"]) for row in rows if "train/loss" in row and "_step" in row]
    trainer_epochs = [
        int(row["trainer/epoch"])
        for row in rows
        if "train/loss" in row and "trainer/epoch" in row
    ]
    if train_steps and trainer_epochs:
        max_epoch = max(trainer_epochs) + 1
        return max(1, round(len(train_steps) / max_epoch))

    val_steps = [int(row["_step"]) for row in rows if "val/loss" in row and "_step" in row]
    if val_steps:
        first_val_step = min(val_steps)
        if first_val_step > 0:
            # Validation is logged every 50 completed epochs in these runs.
            return max(1, round((first_val_step + 1) / 50))

    if train_steps:
        return max(1, round(len(train_steps) / 5000))

    raise ValueError("Cannot infer steps per epoch from W&B history")


def _train_epoch(row: dict[str, float], steps_per_epoch: int) -> float:
    if "trainer/epoch" in row:
        return float(row["trainer/epoch"]) + 1.0
    return float(int(row["_step"]) // steps_per_epoch + 1)


def _validation_epoch(row: dict[str, float], steps_per_epoch: int) -> float:
    if "trainer/epoch" in row:
        return float(row["trainer/epoch"]) + 1.0
    return (float(row["_step"]) + 1.0) / steps_per_epoch


def extract_loss_history(experiment: Experiment) -> LossHistory:
    wandb_file = experiment.run_dir / f"run-{experiment.run_id}.wandb"
    if not wandb_file.exists():
        raise FileNotFoundError(wandb_file)

    rows = _read_history_records(wandb_file)
    steps_per_epoch = _infer_steps_per_epoch(rows)

    train_by_epoch: dict[float, list[float]] = defaultdict(list)
    val_loss: list[tuple[float, float]] = []

    for row in rows:
        if "train/loss" in row:
            epoch = _train_epoch(row, steps_per_epoch)
            train_by_epoch[epoch].append(float(row["train/loss"]))
        if "val/loss" in row:
            epoch = _validation_epoch(row, steps_per_epoch)
            val_loss.append((epoch, float(row["val/loss"])))

    train_epoch_mean = [
        (epoch, sum(values) / len(values))
        for epoch, values in sorted(train_by_epoch.items())
        if values
    ]
    val_loss.sort()

    return LossHistory(train_epoch_mean=train_epoch_mean, val_loss=val_loss)


def save_epoch_csv(histories: dict[Experiment, LossHistory], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "name", "split", "epoch", "loss"],
        )
        writer.writeheader()
        for experiment, history in histories.items():
            for epoch, loss in history.train_epoch_mean:
                writer.writerow(
                    {
                        "experiment": experiment.exp_id,
                        "name": experiment.name,
                        "split": "train_epoch_mean",
                        "epoch": f"{epoch:.6g}",
                        "loss": f"{loss:.10g}",
                    }
                )
            for epoch, loss in history.val_loss:
                writer.writerow(
                    {
                        "experiment": experiment.exp_id,
                        "name": experiment.name,
                        "split": "val",
                        "epoch": f"{epoch:.6g}",
                        "loss": f"{loss:.10g}",
                    }
                )


def plot_train_val(histories: dict[Experiment, LossHistory], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.5), constrained_layout=True)
    train_color = "#2563eb"
    val_color = "#dc2626"

    for axis, (experiment, history) in zip(axes, histories.items(), strict=True):
        train_points = _from_epoch(history.train_epoch_mean, TRAIN_PLOT_START_EPOCH)
        val_points = _from_epoch(history.val_loss, TRAIN_PLOT_START_EPOCH)
        train_x, train_y = zip(*train_points)
        train_y_smooth = _centered_rolling_mean(list(train_y), window=25)
        val_x, val_y = zip(*val_points)

        axis.plot(
            train_x,
            train_y_smooth,
            color=train_color,
            linewidth=1.6,
            label="Train loss (25-epoch mean)",
        )
        axis.plot(
            val_x,
            val_y,
            color=val_color,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label="Validation loss",
        )

        axis.set_title(f"Experiment {experiment.exp_id}: {experiment.name}", loc="left")
        axis.set_ylabel("Loss")
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.margins(x=0.01)
        axis.set_xlim(left=TRAIN_PLOT_START_EPOCH)
        axis.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel("Completed epoch")
    fig.suptitle(
        "Train and Validation Losses for Experiments 1-3 "
        f"(Epoch {TRAIN_PLOT_START_EPOCH:.0f}+)",
        fontsize=14,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_comparison(
    histories: dict[Experiment, LossHistory],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    colors = ["#0f766e", "#7c3aed", "#c2410c"]
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    for color, (experiment, history) in zip(colors, histories.items(), strict=True):
        val_x, val_y = zip(*history.val_loss)
        axis.plot(
            val_x,
            val_y,
            marker="o",
            markersize=3.6,
            linewidth=1.6,
            color=color,
            label=f"Exp {experiment.exp_id}: {experiment.name}",
        )

    axis.set_title("Validation Loss Comparison", loc="left")
    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("Validation loss")
    axis.xaxis.set_major_locator(MaxNLocator(8))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.margins(x=0.01)
    axis.legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_comparison(
    histories: dict[Experiment, LossHistory],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    colors = ["#0f766e", "#7c3aed", "#c2410c"]
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    for color, (experiment, history) in zip(colors, histories.items(), strict=True):
        train_points = _from_epoch(history.train_epoch_mean, TRAIN_PLOT_START_EPOCH)
        train_x, train_y = zip(*train_points)
        train_y_smooth = _centered_rolling_mean(list(train_y), window=25)
        axis.plot(
            train_x,
            train_y_smooth,
            linewidth=1.6,
            color=color,
            label=f"Exp {experiment.exp_id}: {experiment.name}",
        )

    axis.set_title(
        f"Training Loss Comparison (Epoch {TRAIN_PLOT_START_EPOCH:.0f}+)",
        loc="left",
    )
    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("Training loss (25-epoch mean)")
    axis.xaxis.set_major_locator(MaxNLocator(8))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    axis.margins(x=0.01)
    axis.set_xlim(left=TRAIN_PLOT_START_EPOCH)
    axis.legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RUNS_ROOT / "temporal_alignment" / "loss_plots",
        help="Directory for generated plots and exported epoch CSV.",
    )
    parser.add_argument(
        "--experiment", action="append", nargs=3, metavar=("NAME", "RUN_ID", "RUN_DIR"),
        help="Repeat for each local W&B run. Omit to use the three historical run names.",
    )
    args = parser.parse_args()
    experiments = (
        [Experiment(index, name, run_id, Path(run_dir).expanduser())
         for index, (name, run_id, run_dir) in enumerate(args.experiment, start=1)]
        if args.experiment else EXPERIMENTS
    )

    histories = {
        experiment: extract_loss_history(experiment)
        for experiment in experiments
    }

    save_epoch_csv(histories, args.out_dir / "exp1-3_loss_history_epoch.csv")
    plot_train_val(histories, args.out_dir / "exp1-3_train_val_losses.png")
    plot_train_val(histories, args.out_dir / "exp1-3_train_val_losses.pdf")
    plot_training_comparison(histories, args.out_dir / "exp1-3_training_losses.png")
    plot_training_comparison(histories, args.out_dir / "exp1-3_training_losses.pdf")
    plot_validation_comparison(histories, args.out_dir / "exp1-3_validation_losses.png")
    plot_validation_comparison(histories, args.out_dir / "exp1-3_validation_losses.pdf")

    for experiment, history in histories.items():
        train_last = history.train_epoch_mean[-1]
        val_best = min(history.val_loss, key=lambda item: item[1])
        print(
            f"exp{experiment.exp_id}: "
            f"{len(history.train_epoch_mean)} train epochs, "
            f"{len(history.val_loss)} validation points, "
            f"last train={train_last[1]:.5f} @ epoch {train_last[0]:.0f}, "
            f"best val={val_best[1]:.5f} @ epoch {val_best[0]:.0f}"
        )


if __name__ == "__main__":
    main()

"""
Sample all saved DDPM ablation checkpoints from outputs/00* through outputs/10*.

For each run/checkpoint pair, this launches sample_dit.py once with EMA weights
and once with the raw non-EMA model weights. Temporary inference configs are
written under the output root so the checked-in configs can keep evolving
without breaking older checkpoints.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.environ.get("CARDIODIT_RUNS_DIR", Path.home() / "CardioDiT_runs")).expanduser()
OUTPUTS_DIR = RUNS_ROOT / "outputs"
DDPM_CFG_DIR = REPO_ROOT / "configs" / "transformer" / "test_configs" / "ddpm"
STAGE1_SOURCE_CFG = REPO_ROOT / "configs" / "stage1" / "best.yaml"
STAGE1_CKPT = RUNS_ROOT / "outputs" / "stage1" / "vqgan-best-2026-04-30_13-53" / "last_checkpoint.pth"
SAMPLE_SCRIPT = REPO_ROOT / "src" / "scripts" / "sample_dit.py"

LATENT_SHAPE = ["8", "10", "28", "28", "8"]
UMM_MODEL_SPACING = [10.0, 13.6, 13.6, 4.0]
UMM_INPUT_SIZE = [10, 28, 28, 8]

RUN_SPECS = [
    {"run": "00_baseline", "source": "00_baseline.yaml"},
    {"run": "01_dropout_zero", "source": "01_dropout_zero.yaml"},
    {"run": "02_aniso_pos_embed", "source": "02_aniso_pos_embed.yaml"},
    {"run": "03_zero_terminal_snr", "source": "03_zero_terminal_snr.yaml"},
    {"run": "03_zero_terminal_snr_small", "source": "03_zero_terminal_snr_small.yaml"},
    {"run": "03_zero_terminal_snr_tiny", "source": "03_zero_terminal_snr_tiny.yaml"},
    {"run": "04_offset_noise", "source": "04_offset_noise.yaml"},
    {"run": "05_normalize_latents", "source": "05_normalize_latents.yaml"},
    {
        "run": "06_self_conditioning",
        "source": "06_self_conditioning.yaml",
        "overrides": {
            "model.params.input_size": UMM_INPUT_SIZE,
            "model.params.spacing": UMM_MODEL_SPACING,
        },
    },
    {"run": "07_self_conditioning_no_norm", "source": "07_self_conditioning_no_norm.yaml"},
    {"run": "08_self_conditioning_compact", "source": "08_self_conditioning_compact.yaml"},
    {
        "run": "09_self_conditioning_wide",
        "source": "08_self_conditioning_compact.yaml",
        "overrides": {
            "model.params.hidden_size": 1152,
            "model.params.depth": 16,
            "model.params.num_heads": 12,
        },
    },
    {
        "run": "10_self_conditioning_p2",
        "source": "06_self_conditioning_p2.yaml",
        "overrides": {
            "model.params.input_size": UMM_INPUT_SIZE,
            "model.params.spacing": UMM_MODEL_SPACING,
        },
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(RUNS_ROOT / "samples" / "ema_vs_raw_00_to_10"),
    )
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--checkpoint_names", type=str, nargs="*", default=None)
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help=(
            "Sample best_model.pth, last_checkpoint.pth, and periodic "
            "checkpoint_epoch_*.pth/checkpoint_update_*.pth files for each run."
        ),
    )
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--weights", type=str, nargs="+", default=["both"], choices=["ema", "raw", "both"])
    parser.add_argument("--only", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=4,
        default=[10.0, 1.7, 1.7, 1.0],
        metavar=("D", "H", "W", "T"),
    )
    parser.add_argument("--scheduler", type=str, default="ddpm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def checkpoint_names_for_run(run_name: str, args) -> list[str]:
    if args.checkpoint_names:
        return args.checkpoint_names
    if args.checkpoint_name:
        return [args.checkpoint_name]
    if not args.all_checkpoints:
        return ["last_checkpoint.pth"]

    run_dir = OUTPUTS_DIR / run_name
    names = []
    if (run_dir / "best_model.pth").is_file():
        names.append("best_model.pth")
    names.extend(p.name for p in sorted(run_dir.glob("checkpoint_epoch_*.pth")))
    names.extend(p.name for p in sorted(run_dir.glob("checkpoint_update_*.pth")))
    if (run_dir / "last_checkpoint.pth").is_file():
        names.append("last_checkpoint.pth")
    return names


def save_stage1_inference_cfg(cfg_dir: Path) -> Path:
    cfg = OmegaConf.load(STAGE1_SOURCE_CFG)
    OmegaConf.update(cfg, "model.params.num_embeddings", 8192, merge=False)
    OmegaConf.update(cfg, "training.roi_size", [224, 224, 32], merge=False, force_add=True)
    out_path = cfg_dir / "stage1_vqgan_04_30_8192.yaml"
    OmegaConf.save(cfg, out_path)
    return out_path


def save_dit_inference_cfg(spec: dict, cfg_dir: Path, stage1_cfg: Path) -> Path:
    source = DDPM_CFG_DIR / spec["source"]
    cfg = OmegaConf.load(source)
    for key, value in spec.get("overrides", {}).items():
        OmegaConf.update(cfg, key, value, merge=False)

    OmegaConf.update(cfg, "training.sample_log.stage1_cfg", str(stage1_cfg), merge=False, force_add=True)
    OmegaConf.update(cfg, "training.sample_log.stage1_ckpt", str(STAGE1_CKPT), merge=False, force_add=True)
    OmegaConf.update(cfg, "training.sample_log.latent_shape", [int(v) for v in LATENT_SHAPE], merge=False, force_add=True)

    out_path = cfg_dir / f"{spec['run']}.yaml"
    OmegaConf.save(cfg, out_path)
    return out_path


def completed_samples(out_dir: Path) -> int:
    return len(list(out_dir.glob("sample_*.nii.gz")))


def run_job(job: dict, device: str, args, failures: list[tuple[str, int]], lock: threading.Lock):
    run_name = job["run"]
    checkpoint_name = job["checkpoint_name"]
    checkpoint_stem = Path(checkpoint_name).stem
    weights = job["weights"]
    label = f"{run_name}/{checkpoint_stem}/{weights}"
    out_dir = (
        Path(args.output_root)
        / run_name
        / checkpoint_stem
        / ("non_ema" if weights == "raw" else "ema")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = OUTPUTS_DIR / run_name / checkpoint_name
    if not ckpt.is_file():
        with lock:
            print(f"[fail] {label}: missing checkpoint {ckpt}")
            failures.append((label, 2))
        return

    # Comparable runs/checkpoints/weights must start from identical noise.
    seed = args.seed
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_{checkpoint_stem}_{weights}.log"

    cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        "--stage1_cfg",
        str(job["stage1_cfg"]),
        "--stage1_ckpt",
        str(STAGE1_CKPT),
        "--diff_cfg",
        str(job["diff_cfg"]),
        "--diff_ckpt",
        str(ckpt),
        "--output_dir",
        str(out_dir),
        "--n_samples",
        str(args.n_samples),
        "--timesteps",
        str(args.timesteps),
        "--scheduler",
        args.scheduler,
        "--scale_factor",
        str(job["scale_factor"]),
        "--weights",
        weights,
        "--decoder_mode",
        "both",
        "--seed",
        str(seed),
        "--latent_shape",
        *LATENT_SHAPE,
        "--spacing",
        *[str(s) for s in args.spacing],
        "--device",
        device,
    ]

    with lock:
        print(f"[start] {label} on {device} -> {out_dir}")
        print(f"        log: {log_path}")
    if args.dry_run:
        with lock:
            print("        " + " ".join(cmd))
        return

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/cardiodit-mpl")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        with lock:
            print(f"[done] {label}")
    else:
        with lock:
            print(f"[fail] {label}: exit {result.returncode}; see {log_path}")
            failures.append((label, result.returncode))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    cfg_dir = output_root / "_inference_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    stage1_cfg = save_stage1_inference_cfg(cfg_dir)

    selected = RUN_SPECS
    if args.only:
        wanted = set(args.only)
        selected = [spec for spec in RUN_SPECS if spec["run"] in wanted]
        missing = sorted(wanted - {spec["run"] for spec in selected})
        if missing:
            print(f"Unknown --only runs: {missing}", file=sys.stderr)
            return 2

    jobs = []
    for run_index, spec in enumerate(selected):
        diff_cfg = save_dit_inference_cfg(spec, cfg_dir, stage1_cfg)
        cfg = OmegaConf.load(diff_cfg)
        scale_factor = float(cfg.training.scale_factor)
        for checkpoint_index, checkpoint_name in enumerate(checkpoint_names_for_run(spec["run"], args)):
            for weights in args.weights:
                jobs.append({
                    "run": spec["run"],
                    "run_index": run_index,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_name": checkpoint_name,
                    "weights": weights,
                    "diff_cfg": diff_cfg,
                    "stage1_cfg": stage1_cfg,
                    "scale_factor": scale_factor,
                })

    print(f"Prepared {len(jobs)} jobs. Output root: {output_root}")
    print(f"Stage1 checkpoint: {STAGE1_CKPT}")
    print(f"Stage1 config: {stage1_cfg}")

    job_queue: queue.Queue[dict] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    failures: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(device: str):
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(job, device, args, failures, lock)
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        print("\nFailures:")
        for label, code in failures:
            print(f"  {label}: exit {code}")
        return 1

    print("\nAll sampling jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

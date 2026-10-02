"""
Orchestrator: take a DiT checkpoint + config, sample N volumes, run the four
metric scripts (FID, FVD, phase, temporal consistency), and write a combined
``eval_results.json`` to the run directory.

Status: skeleton — wires the existing eval_*.py scripts together via the
shared --samples_dir / --output_json contract once they are implemented.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    parser = argparse.ArgumentParser(description="DiT evaluation orchestrator")
    parser.add_argument("--diff_cfg", required=True)
    parser.add_argument("--diff_ckpt", required=True)
    parser.add_argument("--stage1_cfg", required=True)
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--out_dir", required=True,
                        help="Run directory; samples and eval_results.json land here.")
    parser.add_argument("--i3d_weights", default=None, help="Required for FVD")
    parser.add_argument("--mueller_weights", default=None, help="Required for phase eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Names of metrics to skip: fid fvd phase temporal")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    py = sys.executable

    run([
        py, str(here / "sample_dit.py"),
        "--stage1_cfg", args.stage1_cfg, "--stage1_ckpt", args.stage1_ckpt,
        "--diff_cfg", args.diff_cfg, "--diff_ckpt", args.diff_ckpt,
        "--output_dir", str(samples_dir),
        "--n_samples", str(args.n_samples),
        "--device", args.device,
    ])

    results: dict = {}
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if "fid" not in args.skip:
        out = metrics_dir / "fid.json"
        run([py, str(here / "eval_fid.py"),
             "--samples_dir", str(samples_dir),
             "--reference_dir", args.reference_dir,
             "--output_json", str(out), "--device", args.device])
        results["fid"] = json.loads(out.read_text())

    if "fvd" not in args.skip and args.i3d_weights:
        out = metrics_dir / "fvd.json"
        run([py, str(here / "eval_fvd.py"),
             "--samples_dir", str(samples_dir),
             "--reference_dir", args.reference_dir,
             "--i3d_weights", args.i3d_weights,
             "--output_json", str(out), "--device", args.device])
        results["fvd"] = json.loads(out.read_text())

    if "phase" not in args.skip and args.mueller_weights:
        out = metrics_dir / "phase.json"
        run([py, str(here / "eval_phase.py"),
             "--samples_dir", str(samples_dir),
             "--reference_dir", args.reference_dir,
             "--mueller_weights", args.mueller_weights,
             "--output_json", str(out), "--device", args.device])
        results["phase"] = json.loads(out.read_text())

    if "temporal" not in args.skip:
        out = metrics_dir / "temporal_consistency.json"
        run([py, str(here / "eval_temporal_consistency.py"),
             "--samples_dir", str(samples_dir),
             "--output_json", str(out), "--device", args.device])
        results["temporal_consistency"] = json.loads(out.read_text())

    combined = out_dir / "eval_results.json"
    combined.write_text(json.dumps(results, indent=2))
    print(f"\nWrote combined eval → {combined}")


if __name__ == "__main__":
    main()

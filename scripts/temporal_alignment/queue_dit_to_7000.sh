#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}"
export CARDIODIT_RUNS_DIR="${RUNS_ROOT}"
PY="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NPROC_PER_NODE
QUEUE_ROOT="${RUNS_ROOT}/outputs/training_queue"
QUEUE_ID="dit_to_7000_$(date +%Y%m%d_%H%M%S)"
QUEUE_DIR="${QUEUE_ROOT}/${QUEUE_ID}"
CURRENT_LINK="${QUEUE_ROOT}/current_dit_to_7000"

mkdir -p "${QUEUE_DIR}"
ln -sfn "${QUEUE_DIR}" "${CURRENT_LINK}"

cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${QUEUE_DIR}/queue.log"
}

checkpoint_epoch() {
  local ckpt="$1"
  "${PY}" - "$ckpt" <<'PY'
import sys
from pathlib import Path
import torch

path = Path(sys.argv[1])
if not path.exists():
    print(-1)
else:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(int(ckpt.get("epoch", -1)))
PY
}

wait_for_cuda() {
  until "${PY}" - <<'PY'
import sys
import torch
required = int(__import__("os").environ.get("NPROC_PER_NODE", "2"))
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= required else 1)
PY
  do
    log "Fewer than ${NPROC_PER_NODE} CUDA devices are available; waiting 300 seconds before retrying."
    sleep 300
  done
}

run_one() {
  local run_name="$1"
  local config="$2"
  local train_csv="$3"
  local val_csv="$4"
  local wandb_step_floor="${5:-}"
  local ckpt="${RUNS_ROOT}/outputs/${run_name}/last_checkpoint.pth"
  local epoch

  epoch="$(checkpoint_epoch "${ckpt}")"
  if [[ "${epoch}" -ge 6999 ]]; then
    log "${run_name}: already at epoch ${epoch}; skipping."
    return 0
  fi

  log "${run_name}: starting/resuming from epoch ${epoch}; target final epoch 6999 using ${NPROC_PER_NODE} DDP ranks."
  if [[ -n "${wandb_step_floor}" ]]; then
    log "${run_name}: applying W&B step floor ${wandb_step_floor} for monotonic resumed logging."
  fi
  wait_for_cuda

  local env_prefix=()
  if [[ -n "${wandb_step_floor}" ]]; then
    env_prefix=(env "CARDIODIT_WANDB_STEP_FLOOR=${wandb_step_floor}")
  fi

  "${env_prefix[@]}" "${PY}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --max_restarts=0 \
    src/scripts/train_dit.py \
    --config "${config}" \
    --training_ids "${train_csv}" \
    --validation_ids "${val_csv}" \
    --output_dir "${RUNS_ROOT}/outputs" \
    --run_name "${run_name}" \
    2>&1 | tee -a "${QUEUE_DIR}/${run_name}.log"

  epoch="$(checkpoint_epoch "${ckpt}")"
  if [[ "${epoch}" -lt 6999 ]]; then
    log "${run_name}: stopped at epoch ${epoch}, below target."
    return 1
  fi
  log "${run_name}: completed at epoch ${epoch}."
}

log "Queue ${QUEUE_ID} started in ${ROOT}"
log "Configs are expected to have training.n_epochs=7000."
log "Each run will use ${NPROC_PER_NODE} DDP ranks."

run_one \
  "dit-cyclic-mnm2" \
  "configs/temporal_alignment/cyclic/00_baseline.yaml" \
  "${RUNS_ROOT}/latents/MNM2/train/latents.csv" \
  "${RUNS_ROOT}/latents/MNM2/val/latents.csv" \
  "1000000"

run_one \
  "dit-linear-hwdt-mnm2" \
  "configs/temporal_alignment/interp_linear/00_baseline.yaml" \
  "${RUNS_ROOT}/latents/linear_hwdt/train/latents.csv" \
  "${RUNS_ROOT}/latents/linear_hwdt/val/latents.csv"

run_one \
  "dit-piecewise-hwdt-mnm2" \
  "configs/temporal_alignment/interp_piecewise/00_baseline.yaml" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/train/latents.csv" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/val/latents.csv"

run_one \
  "dit-fourier-hwdt-256z12-mnm2" \
  "configs/temporal_alignment/interp_fourier/00_baseline.yaml" \
  "${RUNS_ROOT}/latents/fourier_hwdt_256z12/train/latents.csv" \
  "${RUNS_ROOT}/latents/fourier_hwdt_256z12/val/latents.csv"

log "Queue ${QUEUE_ID} finished."

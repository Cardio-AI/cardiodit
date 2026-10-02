#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}"
export CARDIODIT_RUNS_DIR="${RUNS_ROOT}"
PY="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NPROC_PER_NODE
MAX_USED_GPU_MEM_MIB="${MAX_USED_GPU_MEM_MIB:-2048}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_FOR_QUEUE_LOG="${WAIT_FOR_QUEUE_LOG:-}"
WAIT_FOR_FINISH="${WAIT_FOR_FINISH:-1}"

QUEUE_ROOT="${RUNS_ROOT}/outputs/training_queue"
QUEUE_ID="piecewise_loss_ablation_$(date +%Y%m%d_%H%M%S)"
QUEUE_DIR="${QUEUE_ROOT}/${QUEUE_ID}"
CURRENT_LINK="${QUEUE_ROOT}/current_piecewise_loss_ablation"

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

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

discover_prior_queue() {
  if [[ -z "${WAIT_FOR_PID}" && -f "${QUEUE_ROOT}/dit_to_7000.pid" ]]; then
    WAIT_FOR_PID="$(tr -d '[:space:]' < "${QUEUE_ROOT}/dit_to_7000.pid")"
  fi

  if [[ -z "${WAIT_FOR_QUEUE_LOG}" && -e "${QUEUE_ROOT}/current_dit_to_7000/queue.log" ]]; then
    WAIT_FOR_QUEUE_LOG="$(readlink -f "${QUEUE_ROOT}/current_dit_to_7000/queue.log")"
  fi
}

wait_for_prior_queue() {
  discover_prior_queue

  if [[ -n "${WAIT_FOR_PID}" ]]; then
    if pid_alive "${WAIT_FOR_PID}"; then
      log "Waiting for prior DiT queue PID ${WAIT_FOR_PID} to finish before starting loss ablation."
      while pid_alive "${WAIT_FOR_PID}"; do
        sleep 300
      done
      log "Prior DiT queue PID ${WAIT_FOR_PID} exited."
    else
      log "Prior DiT queue PID ${WAIT_FOR_PID} is not running."
    fi
  fi

  if [[ "${WAIT_FOR_FINISH}" == "1" && -n "${WAIT_FOR_QUEUE_LOG}" ]]; then
    if grep -Eq 'Queue .* finished\.' "${WAIT_FOR_QUEUE_LOG}"; then
      log "Prior DiT queue finish marker found in ${WAIT_FOR_QUEUE_LOG}."
    else
      log "Prior DiT queue log has no finish marker; exiting so ablation does not start before the base queue is complete."
      return 1
    fi
  fi
}

wait_for_cuda() {
  until "${PY}" - <<'PY'
import os
import sys
import torch

required = int(os.environ.get("NPROC_PER_NODE", "2"))
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= required else 1)
PY
  do
    log "Fewer than ${NPROC_PER_NODE} CUDA devices are available; waiting 300 seconds before retrying."
    sleep 300
  done
}

wait_for_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  while true; do
    mapfile -t used_mib < <(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
    )
    local ok=1
    for ((i = 0; i < NPROC_PER_NODE; i++)); do
      if [[ "${used_mib[$i]:-999999}" -gt "${MAX_USED_GPU_MEM_MIB}" ]]; then
        ok=0
      fi
    done
    if [[ "${ok}" -eq 1 ]]; then
      return 0
    fi
    log "GPU memory is still busy (${used_mib[*]} MiB used); waiting 300 seconds before retrying."
    sleep 300
  done
}

run_one() {
  local run_name="$1"
  local config="$2"
  local train_csv="$3"
  local val_csv="$4"
  local ckpt="${RUNS_ROOT}/outputs/${run_name}/last_checkpoint.pth"
  local epoch

  epoch="$(checkpoint_epoch "${ckpt}")"
  if [[ "${epoch}" -ge 6999 ]]; then
    log "${run_name}: already at epoch ${epoch}; skipping."
    return 0
  fi

  log "${run_name}: starting/resuming from epoch ${epoch}; target final epoch 6999 using ${NPROC_PER_NODE} DDP ranks."
  wait_for_cuda
  wait_for_gpu_memory

  "${PY}" -m torch.distributed.run \
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
log "Piecewise loss ablation configs are expected to have training.n_epochs=7000."
log "Each run will use ${NPROC_PER_NODE} DDP ranks."

wait_for_prior_queue

run_one \
  "dit-piecewise-l1-hwdt-mnm2" \
  "configs/temporal_alignment/interp_piecewise/01_l1_loss.yaml" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/train/latents.csv" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/val/latents.csv"

run_one \
  "dit-piecewise-l2-hwdt-mnm2" \
  "configs/temporal_alignment/interp_piecewise/02_l2_loss.yaml" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/train/latents.csv" \
  "${RUNS_ROOT}/latents/piecewise_hwdt/val/latents.csv"

log "Queue ${QUEUE_ID} finished."

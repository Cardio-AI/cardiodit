#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}"
export CARDIODIT_RUNS_DIR="${RUNS_ROOT}"
PY="${PYTHON:-python}"
GPU="${GPU:-0}"

cd "${ROOT}"

OUT_ROOT="${RUNS_ROOT}/outputs/dit-cyclic-mnm2/checkpoint_samples_20"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

STAGE1_CKPT="${RUNS_ROOT}/autoencoder/best_model.pth"
STAGE1_CFG="configs/stage1/vqgan_ds8xy_ds4t.yaml"
DIFF_CFG="configs/temporal_alignment/cyclic/00_baseline.yaml"
SCALE_FACTOR="3.762895"
N_SAMPLES="20"

run_sample() {
  local epoch="$1"
  local scheduler="$2"
  local steps="$3"
  local out_dir="${OUT_ROOT}/epoch_${epoch}/${scheduler}_${steps}"
  local log_file="${LOG_DIR}/epoch_${epoch}_${scheduler}_${steps}.log"

  mkdir -p "${out_dir}"
  printf '[%s] epoch=%s scheduler=%s steps=%s output=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${epoch}" "${scheduler}" "${steps}" "${out_dir}" \
    | tee -a "${LOG_DIR}/sampling_queue.log"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" src/scripts/sample_dit.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage1_cfg "${STAGE1_CFG}" \
    --diff_cfg "${DIFF_CFG}" \
    --diff_ckpt "${RUNS_ROOT}/outputs/dit-cyclic-mnm2/checkpoint_epoch_${epoch}.pth" \
    --output_dir "${out_dir}" \
    --n_samples "${N_SAMPLES}" \
    --scheduler "${scheduler}" \
    --timesteps "${steps}" \
    --scale_factor "${SCALE_FACTOR}" \
    --spacing 10 1.5 1.5 1 \
    --output_axes hwd \
    --flip_axes 2 \
    --foreground_crop \
    --foreground_threshold -0.95 \
    --foreground_min_fraction 0.005 \
    --amp_dtype bf16 \
    2>&1 | tee -a "${log_file}"
}

printf '[%s] Starting cyclic checkpoint sampling on CUDA_VISIBLE_DEVICES=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" "${GPU}" | tee -a "${LOG_DIR}/sampling_queue.log"

for epoch in 6899 6949 6999; do
  run_sample "${epoch}" "ddim" "150"
done

for epoch in 6899 6949 6999; do
  run_sample "${epoch}" "ddpm" "1000"
done

printf '[%s] Finished cyclic checkpoint sampling.\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_DIR}/sampling_queue.log"

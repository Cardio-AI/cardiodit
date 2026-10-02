#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}"
export CARDIODIT_RUNS_DIR="${RUNS_ROOT}"
PY="${PYTHON:-python}"
GPU="${GPU:-0}"
N_SAMPLES="20"
STAGE1_CKPT="${RUNS_ROOT}/autoencoder/best_model.pth"
STAGE1_CFG="configs/stage1/vqgan_ds8xy_ds4t.yaml"

cd "${ROOT}"

LOG_ROOT="${RUNS_ROOT}/outputs/checkpoint_sampling_followup_20"
mkdir -p "${LOG_ROOT}"
QUEUE_LOG="${LOG_ROOT}/sampling_queue.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${QUEUE_LOG}"
}

sample_count() {
  local out_dir="$1"
  if [[ -d "${out_dir}" ]]; then
    find "${out_dir}" -maxdepth 1 -name 'sample_*.nii.gz' | wc -l
  else
    printf '0\n'
  fi
}

wait_for_pid_file() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  while kill -0 "${pid}" 2>/dev/null; do
    log "Waiting for existing cyclic sampling process ${pid} to finish."
    sleep 120
  done
}

run_sample() {
  local run_name="$1"
  local label="$2"
  local config="$3"
  local scale_factor="$4"
  local epoch="$5"
  local scheduler="$6"
  local steps="$7"
  local out_root="$8"
  local out_dir="${out_root}/epoch_${epoch}/${scheduler}_${steps}"
  local log_file="${out_root}/logs/epoch_${epoch}_${scheduler}_${steps}.log"
  local count

  mkdir -p "${out_root}/logs" "${out_dir}"
  count="$(sample_count "${out_dir}")"
  if [[ "${count}" -ge "${N_SAMPLES}" ]]; then
    log "${label} epoch ${epoch} ${scheduler}_${steps}: already has ${count}/${N_SAMPLES}; skipping."
    return 0
  fi

  log "${label} epoch ${epoch} ${scheduler}_${steps}: generating ${N_SAMPLES} samples in ${out_dir}."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" src/scripts/sample_dit.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage1_cfg "${STAGE1_CFG}" \
    --diff_cfg "${config}" \
    --diff_ckpt "${RUNS_ROOT}/outputs/${run_name}/checkpoint_epoch_${epoch}.pth" \
    --output_dir "${out_dir}" \
    --n_samples "${N_SAMPLES}" \
    --scheduler "${scheduler}" \
    --timesteps "${steps}" \
    --scale_factor "${scale_factor}" \
    --spacing 10 1.5 1.5 1 \
    --output_axes hwd \
    --flip_axes 2 \
    --foreground_crop \
    --foreground_threshold -0.95 \
    --foreground_min_fraction 0.005 \
    --amp_dtype bf16 \
    2>&1 | tee -a "${log_file}"
}

log "Starting follow-up sampling queue on CUDA_VISIBLE_DEVICES=${GPU}."
wait_for_pid_file "${RUNS_ROOT}/outputs/dit-cyclic-mnm2/checkpoint_sampling_20.pid"

CYCLIC_OUT="${RUNS_ROOT}/outputs/dit-cyclic-mnm2/checkpoint_samples_20"
for epoch in 6899 6949 6999; do
  run_sample \
    "dit-cyclic-mnm2" \
    "cyclic" \
    "configs/temporal_alignment/cyclic/00_baseline.yaml" \
    "3.762895" \
    "${epoch}" \
    "ddim" \
    "150" \
    "${CYCLIC_OUT}"
done
for epoch in 6899 6949 6999; do
  run_sample \
    "dit-cyclic-mnm2" \
    "cyclic" \
    "configs/temporal_alignment/cyclic/00_baseline.yaml" \
    "3.762895" \
    "${epoch}" \
    "ddpm" \
    "1000" \
    "${CYCLIC_OUT}"
done

PIECEWISE_OUT="${RUNS_ROOT}/outputs/dit-piecewise-hwdt-mnm2/checkpoint_samples_20_ddpm"
for epoch in 4899 4949 4999; do
  run_sample \
    "dit-piecewise-hwdt-mnm2" \
    "piecewise" \
    "configs/temporal_alignment/interp_piecewise/00_baseline.yaml" \
    "3.777852" \
    "${epoch}" \
    "ddpm" \
    "1000" \
    "${PIECEWISE_OUT}"
done

LINEAR_OUT="${RUNS_ROOT}/outputs/dit-linear-hwdt-mnm2/checkpoint_samples_20_ddpm"
for epoch in 4899 4949 4999; do
  run_sample \
    "dit-linear-hwdt-mnm2" \
    "linear" \
    "configs/temporal_alignment/interp_linear/00_baseline.yaml" \
    "3.780683" \
    "${epoch}" \
    "ddpm" \
    "1000" \
    "${LINEAR_OUT}"
done

OKAYISCH_CYCLIC_OUT="${RUNS_ROOT}/outputs/okayisch/cyclic/checkpoint_samples_20_ddpm"
run_sample \
  "okayisch/cyclic" \
  "okayisch-cyclic" \
  "configs/temporal_alignment/cyclic/00_baseline.yaml" \
  "3.762895" \
  "6149" \
  "ddpm" \
  "1000" \
  "${OKAYISCH_CYCLIC_OUT}"

OKAYISCH_PIECEWISE_OUT="${RUNS_ROOT}/outputs/okayisch/piecewise/checkpoint_samples_20_ddpm"
run_sample \
  "okayisch/piecewise" \
  "okayisch-piecewise" \
  "configs/temporal_alignment/interp_piecewise/00_baseline.yaml" \
  "3.777852" \
  "2649" \
  "ddpm" \
  "1000" \
  "${OKAYISCH_PIECEWISE_OUT}"

log "Follow-up sampling queue finished."

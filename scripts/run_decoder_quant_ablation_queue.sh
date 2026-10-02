#!/usr/bin/env bash
# Wait for an optional upstream training marker, then run the paired decoder ablation.
set -uo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-$(command -v python3)}
CARDIODIT_RUNS_DIR=${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${CARDIODIT_RUNS_DIR}/samples/decoder_quant_ablation_00_to_10}
QUEUE_LOG=${QUEUE_LOG:-${OUTPUT_ROOT}/decoder_quant_ablation_queue.log}
UPSTREAM_LOG=${UPSTREAM_LOG:-}
UPSTREAM_MARKER=${UPSTREAM_MARKER:-"Phase 2 VQGAN queue complete."}
POLL_SECONDS=${POLL_SECONDS:-300}
STAGE1_CKPT=${STAGE1_CKPT:-${CARDIODIT_RUNS_DIR}/outputs/stage1/vqgan-best-2026-04-30_13-53/last_checkpoint.pth}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
DRY_RUN=${DRY_RUN:-0}

# Preserve caller-selected CUDA visibility. DEVICES uses the visible CUDA indices.
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-${CARDIODIT_RUNS_DIR}/cache/matplotlib}
mkdir -p "${OUTPUT_ROOT}" "$(dirname "${QUEUE_LOG}")" "${MPLCONFIGDIR}"

log_msg() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

if [[ -n "${UPSTREAM_LOG}" ]]; then
  log_msg "Waiting for upstream marker '${UPSTREAM_MARKER}' in ${UPSTREAM_LOG}."
  if [[ "${DRY_RUN}" != "1" ]]; then
    while [[ ! -f "${UPSTREAM_LOG}" ]] || ! grep -Fq "${UPSTREAM_MARKER}" "${UPSTREAM_LOG}"; do
      sleep "${POLL_SECONDS}"
    done
  fi
fi

read -r -a DEVICE_ARRAY <<< "${DEVICES}"
cmd=(
  "${PYTHON}" "${REPO_DIR}/src/scripts/sample_decoder_quant_ablation.py"
  --n_samples "${N_SAMPLES:-5}"
  --timesteps "${TIMESTEPS:-1000}"
  --output_root "${OUTPUT_ROOT}"
  --checkpoint_name "${CHECKPOINT_NAME:-last_checkpoint.pth}"
  --stage1_ckpt "${STAGE1_CKPT}"
  --weights "${WEIGHTS:-ema}"
  --devices "${DEVICE_ARRAY[@]}"
  --seed "${SEED:-6042}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

log_msg "Starting decoder quantization ablation with ${STAGE1_CKPT}."
cd "${REPO_DIR}" || exit 1
"${cmd[@]}" >> "${QUEUE_LOG}" 2>&1
status=$?
if [[ "${status}" -eq 0 ]]; then
  log_msg "Decoder quantization ablation queue complete."
else
  log_msg "Decoder quantization ablation queue failed with exit status ${status}."
fi
exit "${status}"

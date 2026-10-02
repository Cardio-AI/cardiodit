#!/usr/bin/env bash
#
# Synthesize 100 samples at every 50,000-update checkpoint in every Fxx and
# S1_xx evolution run. Two workers share the checkpoint queue across the local
# GPUs. Quantized NIfTI files are written directly under RUN/CHECKPOINT/ while
# hidden sample-level manifests preserve resumability.
#
# Configuration is checkpoint-specific: sample_evolution_dit_models.py reads
# the immutable resolved config embedded in each modern checkpoint and passes
# that through to sample_dit.py.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
CARDIODIT_RUNS_DIR=${CARDIODIT_RUNS_DIR:-${HOME}/CardioDiT_runs}
RUNS_ROOT=${RUNS_ROOT:-${CARDIODIT_RUNS_DIR}/outputs/dit}
OUTPUT_ROOT=${OUTPUT_ROOT:-${CARDIODIT_RUNS_DIR}/outputs/synthetic_samples_all_checkpoints}
LOG_ROOT=${LOG_ROOT:-${OUTPUT_ROOT}/_orchestration_logs}
RUN_ID=${RUN_ID:-$(date +%Y%m%dT%H%M%S)}
RUN_LOG_ROOT=${LOG_ROOT}/${RUN_ID}
N_SAMPLES=${N_SAMPLES:-100}
FLOW_MATCHING_TIMESTEPS=${FLOW_MATCHING_TIMESTEPS:-100}
SEED=${SEED:-1234}
GPU_IDS=${GPU_IDS:-"0 1"}
RECHECK_ALL=${RECHECK_ALL:-0}
DRY_RUN=${DRY_RUN:-0}

read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Missing Python executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [ "${DRY_RUN}" != "1" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for local CUDA synthesis." >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${RUN_LOG_ROOT}" /tmp/cardiodit-mpl /tmp/cardiodit-xdg-cache
exec 9>"${LOG_ROOT}/orchestrator.lock"
if ! flock -n 9; then
    echo "Another synthesis orchestrator already holds ${LOG_ROOT}/orchestrator.lock" >&2
    exit 2
fi
QUEUE_FILE=${RUN_LOG_ROOT}/checkpoint_queue.tsv
: > "${QUEUE_FILE}"

completion_count() {
    local run_name=$1
    local checkpoint_stem=$2
    local count=0
    local sample_index

    for ((sample_index = 0; sample_index < N_SAMPLES; sample_index++)); do
        if [ -f "${OUTPUT_ROOT}/${run_name}/${checkpoint_stem}/.manifests/sample_$(printf '%03d' "${sample_index}").json" ]; then
            count=$((count + 1))
        fi
    done
    printf '%s\n' "${count}"
}

batch_size_for_run() {
    local run_name=$1
    case "${run_name}" in
        F00_ddpm_abs_simple_test2_*|F07_*|S1_ds4xy_noT_native_*patch2x2*|S1_ds4xy_noT_native_*patch4x4*)
            printf '1\n'
            ;;
        S1_ds4_all_dims_paddiv_*|S1_ds8xy_noT_native_flow_matching_rope4d_selfcond_retrain_*)
            printf '4\n'
            ;;
        *)
            printf '2\n'
            ;;
    esac
}

for run_dir in "${RUNS_ROOT}"/F[0-9][0-9]* "${RUNS_ROOT}"/S1_*; do
    [ -d "${run_dir}" ] || continue
    run_name=${run_dir##*/}
    while IFS= read -r checkpoint_path; do
        checkpoint_name=${checkpoint_path##*/}
        checkpoint_stem=${checkpoint_name%.pth}
        update=${checkpoint_stem#checkpoint_update_}
        [[ "${update}" =~ ^[0-9]+$ ]] || continue
        [ $((update % 50000)) -eq 0 ] || continue
        complete=$(completion_count "${run_name}" "${checkpoint_stem}")
        if [ "${RECHECK_ALL}" = "1" ] || [ "${complete}" -lt "${N_SAMPLES}" ]; then
            printf '%s\t%s\t%s\n' \
                "${run_name}" "${checkpoint_name}" "$(batch_size_for_run "${run_name}")" \
                >> "${QUEUE_FILE}"
        fi
    done < <(
        find "${run_dir}" -maxdepth 1 -type f -name 'checkpoint_update_*.pth' -print | sort -V
    )
done

task_count=$(wc -l < "${QUEUE_FILE}")
echo "Queued ${task_count} checkpoint(s) for ${N_SAMPLES} sample(s) each."
echo "Queue: ${QUEUE_FILE}"
echo "Run logs: ${RUN_LOG_ROOT}"
if [ "${task_count}" -eq 0 ]; then
    echo "All target checkpoints already have completion manifests."
    exit 0
fi

if [ "${DRY_RUN}" != "1" ]; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/cardiodit-mpl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/cardiodit-xdg-cache}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

worker() {
    local worker_index=$1
    local gpu_id=$2
    local log_file=${RUN_LOG_ROOT}/gpu${gpu_id}.log
    local line_index=0
    local failures=0
    local run_name
    local checkpoint_name
    local batch_size

    : > "${log_file}"
    while IFS=$'\t' read -r run_name checkpoint_name batch_size; do
        if [ $((line_index % ${#GPU_ARRAY[@]})) -ne "${worker_index}" ]; then
            line_index=$((line_index + 1))
            continue
        fi
        line_index=$((line_index + 1))

        cmd=(
            "${PYTHON_BIN}" "${REPO_ROOT}/scripts/sample_evolution_dit_models.py"
            --runs_root "${RUNS_ROOT}"
            --output_root "${OUTPUT_ROOT}"
            --all_checkpoints
            --checkpoint_glob "${checkpoint_name}"
            --only "${run_name}"
            --samplers trained
            --n_samples "${N_SAMPLES}"
            --batch_size "${batch_size}"
            --per_sample_output_dirs
            --flat_output_layout
            --flow_matching_timesteps "${FLOW_MATCHING_TIMESTEPS}"
            --weights ema
            --decoder_modes quantized
            --seed "${SEED}"
            --device "cuda:${gpu_id}"
        )
        if [ "${DRY_RUN}" = "1" ]; then
            cmd+=(--dry_run)
        fi

        {
            echo
            echo "[$(date --iso-8601=seconds)] GPU ${gpu_id}: ${run_name}/${checkpoint_name} batch=${batch_size}"
            printf ' %q' "${cmd[@]}"
            echo
        } | tee -a "${log_file}"
        if ! "${cmd[@]}" 2>&1 | tee -a "${log_file}"; then
            failures=$((failures + 1))
        fi
    done < "${QUEUE_FILE}"

    if [ "${failures}" -ne 0 ]; then
        echo "GPU ${gpu_id} worker finished with ${failures} failed checkpoint(s)." | tee -a "${log_file}"
        return 1
    fi
    echo "GPU ${gpu_id} worker finished successfully." | tee -a "${log_file}"
}

pids=()
for worker_index in "${!GPU_ARRAY[@]}"; do
    worker "${worker_index}" "${GPU_ARRAY[worker_index]}" &
    pids+=("$!")
done
printf 'worker_pids=%s\n' "${pids[*]}" | tee "${RUN_LOG_ROOT}/worker_pids.txt"

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

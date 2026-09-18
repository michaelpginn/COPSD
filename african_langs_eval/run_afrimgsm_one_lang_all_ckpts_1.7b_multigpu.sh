#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parallel multi-GPU version of run_afrimgsm_one_lang_all_ckpts_1.7b.sh
#
# Each (language, checkpoint) evaluation is an independent vLLM job on a
# single GPU (tensor_parallel_size=1 -- a 1.7B model fits on one GPU).
# Jobs are dispatched across the GPU pool, keeping one running per GPU.
# ============================================================

# =========================
# Preamble
# =========================
BASE_MODEL="Qwen/Qwen3-1.7B"

LANGUAGES=(
  "AMH"
  "EWE"
  "HAU"
  "IBO"
  "KIN"
  "LIN"
  "LUG"
  "ORM"
  "SNA"
  "SOT"
  "SWA"
  "TWI"
  "VAI"
  "WOL"
  "XHO"
  "YOR"
  "ZUL"
)

CHECKPOINT_STEPS=(5 10 15 20 25 30 35 40 45 50)

# GPU pool: one eval job runs per GPU at a time.
GPUS=(0 1 2 3 4 5 6 7)

VAL_N=12
TEMPERATURE=1.0

BENCHMARK="afrimgsm"
AFRIMGSM_SPLIT="test"

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

MAX_NEW_TOKENS=4096

# =========================
# Paths
# =========================
PROJECT_ROOT="/home/ec2-user/COPSD"
MODEL_ROOT="${PROJECT_ROOT}/african_langs_models"

RUN_CONFIG_PREFIX="qwen3_1.7b"
DATA_TAG="opsd_500"
TRAIN_MAX_COMPLETION_LENGTH=2048

# =========================
# Single evaluation (runs on one GPU passed as $4)
# =========================
run_eval() {
    local language="$1"
    local output_file="$2"
    local checkpoint_dir="${3:-}"
    local gpu="${4:-0}"

    if [[ -f "$output_file" ]]; then
        echo "[SKIP gpu${gpu}] Existing output: $output_file"
        return
    fi

    mkdir -p "$(dirname "$output_file")"

    echo "[START gpu${gpu}] ${language} -> $(basename "$output_file")" \
         "${checkpoint_dir:+(ckpt: $(basename "$checkpoint_dir"))}"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpu" python evaluate_math.py \
        --benchmark "$BENCHMARK" \
        --base_model "$BASE_MODEL" \
        --val_n "$VAL_N" \
        --temperature "$TEMPERATURE" \
        --afrimgsm_split "$AFRIMGSM_SPLIT" \
        --enable_thinking \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --language "$language" \
        --output_file "$output_file" \
        ${checkpoint_dir:+--checkpoint_dir "$checkpoint_dir"} \
        > "${output_file}.log" 2>&1

    echo "[DONE gpu${gpu}] $(basename "$output_file")"
}

# =========================
# Build the flat job list: base model + each existing checkpoint per language.
# Each job encodes: language|output_file|checkpoint_dir  (checkpoint_dir empty = base)
# =========================
declare -a JOBS
for language in "${LANGUAGES[@]}"; do
    lang_lower="$(echo "$language" | tr '[:upper:]' '[:lower:]')"
    run_config="${RUN_CONFIG_PREFIX}_${lang_lower}_${DATA_TAG}_max${TRAIN_MAX_COMPLETION_LENGTH}"
    run_dir="${MODEL_ROOT}/${lang_lower}/${run_config}_${lang_lower}"
    eval_root="eval_results/afrimgsm/${lang_lower}/${AFRIMGSM_SPLIT}"

    JOBS+=("${language}|${eval_root}/qwen3_1.7b_base_${lang_lower}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json|")

    for step in "${CHECKPOINT_STEPS[@]}"; do
        ckpt="${run_dir}/checkpoint-${step}"
        if [[ ! -d "$ckpt" ]]; then
            echo "[WARN] Missing checkpoint, skipping: $ckpt"
            continue
        fi
        JOBS+=("${language}|${eval_root}/qwen3_1.7b_opsd_${lang_lower}_step${step}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json|${ckpt}")
    done
done

echo "============================================================"
echo "Total eval jobs : ${#JOBS[@]}"
echo "GPU pool        : ${GPUS[*]}  (${#GPUS[@]} concurrent)"
echo "Per-job stdout  : written to <output_file>.log"
echo "============================================================"

# =========================
# Dispatch in waves: each wave launches at most one job per GPU (so a GPU is
# never double-booked), then waits for the whole wave before starting the next.
# A wave runs at the speed of its slowest job -- simple and OOM-safe.
# =========================
gpu_count=${#GPUS[@]}
total=${#JOBS[@]}
launched=0

for (( start=0; start<total; start+=gpu_count )); do
    for (( slot=0; slot<gpu_count && start+slot<total; slot++ )); do
        IFS='|' read -r lang out ckpt <<< "${JOBS[$(( start + slot ))]}"
        gpu=${GPUS[$slot]}
        run_eval "$lang" "$out" "$ckpt" "$gpu" &
        launched=$(( launched + 1 ))
    done
    wait   # drain this wave before starting the next
done

echo "All AfriMGSM checkpoint evaluations finished (${launched} jobs dispatched)."

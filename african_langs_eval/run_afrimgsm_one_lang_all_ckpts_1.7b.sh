#!/usr/bin/env bash
set -euo pipefail

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

GPU_ID=1
VAL_N=12
TEMPERATURE=1.0

BENCHMARK="afrimgsm"
AFRIMGSM_SPLIT="test"

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9

MAX_NEW_TOKENS=1024

# =========================
# Paths
# =========================
PROJECT_ROOT="/home/ec2-user/COPSD"
MODEL_ROOT="${PROJECT_ROOT}/african_langs_models"

RUN_CONFIG_PREFIX="qwen3_1.7b"
DATA_TAG="opsd_500"
TRAIN_MAX_COMPLETION_LENGTH=2048

# =========================
# Function
# =========================
run_eval() {
    local language="$1"
    local output_file="$2"
    local checkpoint_dir="${3:-}"

    if [[ -f "$output_file" ]]; then
        echo "[SKIP] Existing output: $output_file"
        return
    fi

    mkdir -p "$(dirname "$output_file")"

    echo "============================================================"
    echo "Running AfriMGSM evaluation"
    echo "Base model      : $BASE_MODEL"
    echo "Benchmark       : $BENCHMARK"
    echo "Language        : $language"
    echo "Split           : $AFRIMGSM_SPLIT"
    echo "Val-N           : $VAL_N"
    echo "Temperature     : $TEMPERATURE"
    echo "Max new tokens  : $MAX_NEW_TOKENS"
    echo "Output file     : $output_file"

    if [[ -n "$checkpoint_dir" ]]; then
        echo "Checkpoint dir  : $checkpoint_dir"
    else
        echo "Checkpoint dir  : <base model>"
    fi
    echo "============================================================"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_math.py \
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
        ${checkpoint_dir:+--checkpoint_dir "$checkpoint_dir"}
}

eval_one_language() {
    local language="$1"

    local lang_lower
    lang_lower="$(echo "$language" | tr '[:upper:]' '[:lower:]')"

    local run_config
    run_config="${RUN_CONFIG_PREFIX}_${lang_lower}_${DATA_TAG}_max${TRAIN_MAX_COMPLETION_LENGTH}"

    local model_lang_root
    local run_dir
    local eval_root
    local base_output

    model_lang_root="${MODEL_ROOT}/${lang_lower}"
    run_dir="${model_lang_root}/${run_config}_${lang_lower}"

    eval_root="eval_results/afrimgsm/${lang_lower}/${AFRIMGSM_SPLIT}"
    base_output="${eval_root}/qwen3_1.7b_base_${lang_lower}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json"

    echo "============================================================"
    echo "Evaluate selected checkpoints for one AfriMGSM language"
    echo "Language        : $language"
    echo "Language lower  : $lang_lower"
    echo "Model root      : $MODEL_ROOT"
    echo "Run config      : $run_config"
    echo "Run dir         : $run_dir"
    echo "Eval root       : $eval_root"
    echo "Checkpoint steps: ${CHECKPOINT_STEPS[*]}"
    echo "============================================================"

    run_eval "$language" "$base_output"

    for step in "${CHECKPOINT_STEPS[@]}"; do
        local checkpoint_dir
        local ckpt_output

        checkpoint_dir="${run_dir}/checkpoint-${step}"
        ckpt_output="${eval_root}/qwen3_1.7b_opsd_${lang_lower}_step${step}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json"

        if [[ ! -d "$checkpoint_dir" ]]; then
            echo "[WARN] Missing checkpoint, skipping: $checkpoint_dir"
            continue
        fi

        run_eval "$language" "$ckpt_output" "$checkpoint_dir"
    done

    echo "[OK] Finished selected checkpoint evaluations for ${language}"
    echo
}

# =========================
# Run all configured languages
# =========================
for language in "${LANGUAGES[@]}"; do
    eval_one_language "$language"
done

echo "All selected AfriMGSM checkpoint evaluations finished."

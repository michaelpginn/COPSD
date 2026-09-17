#!/usr/bin/env bash
set -euo pipefail

# =========================
# Preamble
# =========================
BASE_MODEL="Qwen/Qwen3-4B"

LANGUAGES=(
  "BN"
  "ES"
  "JA"
  "RU"
  "SW"
  "TE"
  "TH"
  "ZH"
)

# Evaluate all checkpoints that exist under each language run directory.
# For example: checkpoint-5, checkpoint-10, ..., checkpoint-100
CHECKPOINT_STEPS=(5 10 15 20 25 30 35 40 45 50)

GPU_ID=3
VAL_N=12
TEMPERATURE=1.0

POLYMATH_SPLIT="low"

TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.95
MAX_NEW_TOKENS=8192

PROJECT_ROOT="/home/ec2-user/COPSD"
MODEL_ROOT="${PROJECT_ROOT}/multilingual_models_new_additional"

RUN_CONFIG_PREFIX="qwen3_4b"
DATA_TAG="opsd_3000"
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

    if [[ -n "$checkpoint_dir" && ! -d "$checkpoint_dir" ]]; then
        echo "[WARN] Missing checkpoint, skipping: $checkpoint_dir"
        return
    fi

    mkdir -p "$(dirname "$output_file")"

    echo "============================================================"
    echo "Running Polymath evaluation"
    echo "Base model      : $BASE_MODEL"
    echo "Language        : $language"
    echo "Split           : $POLYMATH_SPLIT"
    echo "Max new tokens  : $MAX_NEW_TOKENS"
    echo "Val-N           : $VAL_N"
    echo "Temperature     : $TEMPERATURE"
    echo "Output file     : $output_file"
    if [[ -n "$checkpoint_dir" ]]; then
        echo "Checkpoint dir  : $checkpoint_dir"
    else
        echo "Checkpoint dir  : <base model>"
    fi
    echo "============================================================"

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --val_n "$VAL_N" \
        --temperature "$TEMPERATURE" \
        --polymath_split "$POLYMATH_SPLIT" \
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

    local eval_root
    local model_tag
    local run_dir
    local base_output

    eval_root="eval_results/${lang_lower}/${POLYMATH_SPLIT}"

    model_tag="${RUN_CONFIG_PREFIX}_${lang_lower}_${DATA_TAG}_max${TRAIN_MAX_COMPLETION_LENGTH}"
    run_dir="${MODEL_ROOT}/${lang_lower}/${model_tag}_${lang_lower}"

    base_output="${eval_root}/qwen3_4b_base_${lang_lower}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json"

    echo "============================================================"
    echo "Evaluate base model and existing OPSD checkpoints"
    echo "Language        : $language"
    echo "Language lower  : $lang_lower"
    echo "Eval root       : $eval_root"
    echo "Run dir         : $run_dir"
    echo "Checkpoint steps: ${CHECKPOINT_STEPS[*]}"
    echo "============================================================"

    # Base model evaluation
    run_eval "$language" "$base_output"

    # Checkpoint evaluations
    if [[ ! -d "$run_dir" ]]; then
        echo "[WARN] Missing run dir, skipping checkpoints for ${language}: $run_dir"
        echo
        return
    fi

    for step in "${CHECKPOINT_STEPS[@]}"; do
        local checkpoint_dir
        local ckpt_output

        checkpoint_dir="${run_dir}/checkpoint-${step}"
        ckpt_output="${eval_root}/qwen3_4b_opsd_${lang_lower}_step${step}_max${MAX_NEW_TOKENS}_valn${VAL_N}.json"

        if [[ ! -d "$checkpoint_dir" ]]; then
            echo "[SKIP] Checkpoint does not exist: $checkpoint_dir"
            continue
        fi

        run_eval "$language" "$ckpt_output" "$checkpoint_dir"
    done

    echo "[OK] Finished ${language}"
    echo
}

# =========================
# Run all languages
# =========================
for language in "${LANGUAGES[@]}"; do
    eval_one_language "$language"
done

echo "All evaluations finished."


# qwen3_4b_base_bn_max8192_valn12.json

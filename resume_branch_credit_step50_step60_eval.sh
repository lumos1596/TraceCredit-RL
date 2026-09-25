#!/usr/bin/env bash
# Resume the formal balanced-480 EM evaluations after training has completed.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUN_NAME=multihop-tree-dapo-branch-credit-correctness-step40to60-20260910
RUN_DIR="$PROJECT_DIR/verl_checkpoints/$RUN_NAME"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
REPORT_BUILDER="$PROJECT_DIR/evaluation/formal_em/build_report.py"
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"

validate_checkpoint() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for rank in 0 1 2; do
            [[ -s "$checkpoint/${kind}_world_size_3_rank_${rank}.pt" ]] || {
                echo "missing or empty checkpoint shard: $checkpoint/${kind}_world_size_3_rank_${rank}.pt" >&2
                return 1
            }
        done
    done
}

cd "$PROJECT_DIR"

for step in 50 60; do
    checkpoint="$RUN_DIR/actor/global_step_$step"
    validate_checkpoint "$checkpoint"
    LABEL="branch_credit_correctness_step${step}" \
    MODEL_PATH="$SFT_MODEL" \
    CHECKPOINT="$checkpoint" \
    CHECKPOINT_STEP="$step" \
    EVAL_SPLIT=balanced \
    SAMPLE_COUNT=480 \
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS:-0,1,2}" \
    "$EVAL_SCRIPT"
done

"$PYTHON_BIN" "$REPORT_BUILDER"
echo "RESUMED_EVALUATION_COMPLETE=$(date -Is)"

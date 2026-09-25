#!/usr/bin/env bash
# Restartable 3B TraceCredit + Self-OPD run followed by matched balanced-480 EM.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUN_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-tracecredit-self-opd-step40to50-20260918}
TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_tracecredit_self_opd_step40_to50.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$RUN_NAME/actor/global_step_50"
RESULT_ROOT="$PROJECT_DIR/evaluation/formal_em/tracecredit_self_opd_step50"

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
EXPERIMENT_NAME="$RUN_NAME" "$TRAIN_SCRIPT"
validate_checkpoint "$CHECKPOINT"

LABEL=tracecredit_self_opd_step50 \
MODEL_PATH="$SFT_MODEL" \
CHECKPOINT="$CHECKPOINT" \
CHECKPOINT_STEP=50 \
EVAL_SPLIT=balanced \
SAMPLE_COUNT=480 \
RESULT_ROOT="$RESULT_ROOT" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}" \
    "$EVAL_SCRIPT"

echo "TRACECREDIT_SELF_OPD_PIPELINE_COMPLETE=$(date -Is)"

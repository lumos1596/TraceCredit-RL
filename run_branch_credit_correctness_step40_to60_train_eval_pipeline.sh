#!/usr/bin/env bash
# Train corrected Branch-Credit through step 60, evaluate step 50/60 on the
# balanced 480-question EM suite, then refresh the overall report.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-correctness-step40to60-20260910}
TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_branch_credit_correctness_step40_to60.sh"
EVAL_SCRIPT="$PROJECT_DIR/evaluation/formal_em/run_eval.sh"
REPORT_BUILDER="$PROJECT_DIR/evaluation/formal_em/build_report.py"
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
SFT_MODEL="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
RUN_DIR="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME"

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

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 2
fi

export EXPERIMENT_NAME
"$TRAIN_SCRIPT"

for step in 50 60; do
    checkpoint="$RUN_DIR/actor/global_step_$step"
    validate_checkpoint "$checkpoint"
    LABEL="branch_credit_correctness_step${step}" \
    MODEL_PATH="$SFT_MODEL" \
    CHECKPOINT="$checkpoint" \
    CHECKPOINT_STEP="$step" \
    EVAL_SPLIT=balanced \
    SAMPLE_COUNT=480 \
    "$EVAL_SCRIPT"
done

"$PYTHON_BIN" "$REPORT_BUILDER"
echo "PIPELINE_COMPLETE=$(date -Is)"

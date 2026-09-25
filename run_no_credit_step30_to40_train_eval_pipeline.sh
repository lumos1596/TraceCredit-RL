#!/usr/bin/env bash
# Run the fair no-Credit continuation and evaluate step 35 and step 40.
# Intended to be launched by monitor_and_launch_branch_credit.sh inside a
# detached systemd user service after three GPUs are stably idle.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-no-credit-step30to40-20260909}
TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_tree_dapo_step30_to40.sh"
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME/actor"
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/$EXPERIMENT_NAME"}

checkpoint_complete() {
    local checkpoint_dir=$1
    local rank prefix
    for rank in 0 1 2; do
        for prefix in model optim extra_state; do
            [[ -s "$checkpoint_dir/${prefix}_world_size_3_rank_${rank}.pt" ]] || return 1
        done
    done
}

[[ -x "$TRAIN_SCRIPT" ]] || { echo "training script is not executable: $TRAIN_SCRIPT" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" ]] || { echo "refusing to overwrite evaluation results: $RESULT_ROOT" >&2; exit 3; }

echo "PIPELINE_START=$(date -Is)"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$TRAIN_SCRIPT"

for step in 35 40; do
    checkpoint_dir="$CHECKPOINT_ROOT/global_step_$step"
    if ! checkpoint_complete "$checkpoint_dir"; then
        echo "incomplete checkpoint: $checkpoint_dir" >&2
        exit 4
    fi
done

for step in 35 40; do
    checkpoint_dir="$CHECKPOINT_ROOT/global_step_$step"
    echo "EVALUATING_STEP=$step START=$(date -Is)"
    CHECKPOINT="$checkpoint_dir" \
        CHECKPOINT_STEP="$step" \
        RESULT_ROOT="$RESULT_ROOT/step$step" \
        "$PROJECT_DIR/run_branch_credit_step30_eval.sh"
    echo "EVALUATING_STEP=$step END=$(date -Is)"
done

echo "PIPELINE_END=$(date -Is)"

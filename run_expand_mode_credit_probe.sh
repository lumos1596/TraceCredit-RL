#!/usr/bin/env bash
# One fixed-budget, no-update rollout probe for expansion-node selection.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
MODE=${1:-}
case "$MODE" in
    random|uncertainty_balanced|outcome_prior) ;;
    *) echo "usage: $0 {random|uncertainty_balanced|outcome_prior}" >&2; exit 2 ;;
esac

STEP60="$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-branch-credit-correctness-step40to60-20260910/actor/global_step_60"
[[ -s "$STEP60/model_world_size_3_rank_0.pt" ]] || {
    echo "missing step60 checkpoint: $STEP60" >&2
    exit 2
}

export EXPERIMENT_NAME="branch-credit-expand-probe-${MODE}-step60"
export INIT_CHECKPOINT="$STEP60"
export RESUME_GLOBAL_STEP=60
export TOTAL_TRAINING_STEPS=62
export SAVE_FREQ=-1
export SAVE_AT_END=false
export LOGGER="['console']"
export BRANCH_CREDIT_VALUE_MODE=correctness
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8
export EXPAND_MODE="$MODE"
export ACTOR_LR=0
export CRITIC_WARMUP=9999
export SHUFFLE_TRAIN_DATALOADER=false
export ROLLOUT_ACCUMULATION_STEPS=${ROLLOUT_ACCUMULATION_STEPS:-8}
export DAPO_DYNAMIC_SAMPLING=false

exec "$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh"

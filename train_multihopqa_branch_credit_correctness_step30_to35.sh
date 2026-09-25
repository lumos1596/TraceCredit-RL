#!/usr/bin/env bash
# Diagnostic continuation of Branch-Credit from step 30 through step 35.
# Restores actor optimizer/scheduler/worker RNG state and computes local
# sibling credit from binary answer correctness instead of shaped reward.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_RUN=multihop-tree-dapo-branch-credit-step20to30-retry-20260906-215732

export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/$BASE_RUN/actor/global_step_30"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-correctness-step30to35}
export BRANCH_CREDIT_VALUE_MODE=${BRANCH_CREDIT_VALUE_MODE:-correctness}
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=${BRANCH_CREDIT_CORRECTNESS_THRESHOLD:-0.8}
export RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-30}
export RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-true}
# The accumulation loop increments before checking its exclusive upper bound;
# 36 therefore performs and logs optimizer steps 31, 32, 33, 34, and 35.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-36}
export SAVE_FREQ=${SAVE_FREQ:-5}

exec "$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

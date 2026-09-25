#!/usr/bin/env bash
# Continue the corrected Branch-Credit run from optimizer step 40 through 60.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_RUN=multihop-tree-dapo-branch-credit-correctness-step35to40-20260908

export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/$BASE_RUN/actor/global_step_40"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-correctness-step40to60-20260910}
export BRANCH_CREDIT_VALUE_MODE=${BRANCH_CREDIT_VALUE_MODE:-correctness}
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=${BRANCH_CREDIT_CORRECTNESS_THRESHOLD:-0.8}
export RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-40}
export RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-true}
# The loop uses an exclusive upper bound: 61 performs steps 41 through 60.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-61}
export SAVE_FREQ=${SAVE_FREQ:-10}

exec "$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

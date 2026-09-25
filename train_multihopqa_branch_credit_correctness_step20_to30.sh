#!/usr/bin/env bash
# Fair rerun from the shared Tree-GRPO step-20 actor using correctness-only
# local branch credit. All other formal Branch-Credit/DAPO settings are kept.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO

export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/multihopqa-tree-grpo-sft350-3gpu-budget48-persistent-20260826/actor/global_step_20"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-correctness-step20to30}
export BRANCH_CREDIT_VALUE_MODE=${BRANCH_CREDIT_VALUE_MODE:-correctness}
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=${BRANCH_CREDIT_CORRECTNESS_THRESHOLD:-0.8}
export RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-20}
# This is a fair new run from the common actor checkpoint, so Adam and RNG are
# initialized afresh just as in the original DAPO and Branch-Credit runs.
export RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-false}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-31}
export SAVE_FREQ=${SAVE_FREQ:-10}

exec "$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
: "${TRAIN_RUN_NAME:?TRAIN_RUN_NAME is required}"

export CHECKPOINT=${CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/$TRAIN_RUN_NAME/actor/global_step_30"}
export RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/$TRAIN_RUN_NAME"}

exec "$PROJECT_DIR/run_branch_credit_step30_eval.sh"

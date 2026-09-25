#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RERUN_NAME=multihop-tree-dapo-branch-credit-correctness-step20to30-20260907-195640
CONTINUATION_NAME=multihop-tree-dapo-branch-credit-correctness-step30to35-from-old-step30-20260907

CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$RERUN_NAME/actor/global_step_30" \
CHECKPOINT_STEP=30 \
RESULT_ROOT="$PROJECT_DIR/evaluation/$RERUN_NAME" \
    "$PROJECT_DIR/run_branch_credit_step30_eval.sh"

CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$CONTINUATION_NAME/actor/global_step_35" \
CHECKPOINT_STEP=35 \
RESULT_ROOT="$PROJECT_DIR/evaluation/$CONTINUATION_NAME" \
    "$PROJECT_DIR/run_branch_credit_step30_eval.sh"

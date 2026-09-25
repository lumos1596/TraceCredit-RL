#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUNNER="$PROJECT_DIR/evaluation/multihop_step30_dapo_20260830/run_eval.sh"
CHECKPOINT=${CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-branch-credit-step20to30-retry-20260906-215732/actor/global_step_30"}
RESULT_ROOT=${RESULT_ROOT:-"$PROJECT_DIR/evaluation/multihop_step30_branch_credit_20260907"}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-30}

/usr/bin/curl --fail --silent --show-error --max-time 15 \
    -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Tree-GRPO branch-credit evaluation health check"],"topk":1,"return_scores":true}' \
    >/dev/null

CHECKPOINT="$CHECKPOINT" CHECKPOINT_STEP="$CHECKPOINT_STEP" RESULT_ROOT="$RESULT_ROOT" \
    EVAL_SPLIT=natural "$RUNNER" 120

CHECKPOINT="$CHECKPOINT" CHECKPOINT_STEP="$CHECKPOINT_STEP" RESULT_ROOT="$RESULT_ROOT" \
    EVAL_SPLIT=balanced "$RUNNER" 480

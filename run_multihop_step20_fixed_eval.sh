#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
export CHECKPOINT="$PROJECT_DIR/verl_checkpoints/multihopqa-tree-grpo-sft350-3gpu-budget48-persistent-20260826/actor/global_step_20"
export CHECKPOINT_STEP=20
export RESULT_ROOT="$PROJECT_DIR/evaluation/multihop_step20_fixed_20260830"
export EVAL_SPLIT=natural

/usr/bin/curl --fail --silent --show-error --max-time 15 \
    -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Tree-GRPO step20 evaluation health check"],"topk":1,"return_scores":true}' \
    >/dev/null

exec "$PROJECT_DIR/evaluation/multihop_step30_dapo_20260830/run_eval.sh" 120

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO

/usr/bin/curl --fail --silent --show-error --max-time 15 \
    -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Tree-GRPO retrieval health check"],"topk":1,"return_scores":true}' \
    >/dev/null

"$PROJECT_DIR/train_multihopqa_tree_dapo_step20_to30.sh"
EVAL_SPLIT=natural "$PROJECT_DIR/evaluation/multihop_step30_dapo_20260830/run_eval.sh" 120
EVAL_SPLIT=balanced "$PROJECT_DIR/evaluation/multihop_step30_dapo_20260830/run_eval.sh" 480

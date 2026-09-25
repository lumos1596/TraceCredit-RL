#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RUNNER="$PROJECT_DIR/evaluation/multihop_step30_dapo_20260830/run_eval.sh"

/usr/bin/curl --fail --silent --show-error --max-time 15 \
    -X POST http://127.0.0.1:8000/retrieve \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Tree-GRPO evaluation health check"],"topk":1,"return_scores":true}' \
    >/dev/null

EVAL_SPLIT=natural "$RUNNER" 120
EVAL_SPLIT=balanced "$RUNNER" 480

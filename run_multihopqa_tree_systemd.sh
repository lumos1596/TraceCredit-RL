#!/usr/bin/env bash
# Durable systemd entrypoint for the local multi-hop Tree-GRPO run.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
RETRIEVER_URL=http://127.0.0.1:8000/retrieve

# Fail before allocating training GPUs if the retrieval dependency is down.
/usr/bin/curl --fail --silent --show-error --max-time 15 \
    -X POST "$RETRIEVER_URL" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["Tree-GRPO retrieval health check"],"topk":1,"return_scores":true}' \
    >/dev/null

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihopqa-tree-grpo-sft350-3gpu-budget48-entropy0-systemd-20260821}
export SAVE_FREQ=${SAVE_FREQ:-2}
export TEST_FREQ=${TEST_FREQ:-20}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}

exec "$PROJECT_DIR/train_multihopqa_tree_search_local.sh"

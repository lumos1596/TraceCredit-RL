#!/usr/bin/env bash
set -euo pipefail

ROOT=${PROJECT_DIR:-/home/luwa/Documents/Tree-GRPO}
PYTHON="$ROOT/.conda/envs/treegrpo/bin/python"
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
ANALYZER_URL=${SELF_OPD_ANALYZER_URL:-http://127.0.0.1:8127}

for path in \
    "$PYTHON" \
    "$ROOT/data/tree_seed_node_skill_generation_sft_3b_v4_20260923/merged/config.json" \
    "$ROOT/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350/config.json" \
    "$ROOT/data/multihopqa_search_mixed_402020_20260830/train.parquet" \
    "$ROOT/data/multihopqa_search_mixed_402020_20260830/test.parquet"; do
    [[ -s "$path" ]] || exit 1
done

analyzer_health=$(curl -fsS --max-time 5 "$ANALYZER_URL/health" 2>/dev/null) || exit 1
printf '%s' "$analyzer_health" \
    | "$PYTHON" -c 'import json,sys; assert json.load(sys.stdin).get("ok") is True' >/dev/null
retriever_health=$(curl -fsS --max-time 10 -H 'Content-Type: application/json' \
    -d '{"queries":["NodeSkill OPD readiness probe"],"topk":1,"return_scores":true}' \
    "$RETRIEVER_URL" 2>/dev/null) || exit 1
printf '%s' "$retriever_health" \
    | "$PYTHON" -c 'import json,sys; x=json.load(sys.stdin); assert x.get("result") and x["result"][0]' >/dev/null

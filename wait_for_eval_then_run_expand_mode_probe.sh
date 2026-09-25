#!/usr/bin/env bash
# Wait for successful step50/60 formal EM evaluation, then compare branch
# expansion selectors with fixed no-update rollout probes.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
EVAL_UNIT=${EVAL_UNIT:-tree-grpo-branch-credit-step50-step60-eval.service}
PROBE_UNIT_NAME=${PROBE_UNIT_NAME:-tree-grpo-expand-selector-probe}
EVAL_TIMEOUT_SECONDS=${EVAL_TIMEOUT_SECONDS:-43200}
POLL_SECONDS=${POLL_SECONDS:-30}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
GPU_UTIL_LIMIT=${GPU_UTIL_LIMIT:-5}
HOST_AVAILABLE_MIN_GIB=${HOST_AVAILABLE_MIN_GIB:-150}
GPU_STABLE_SAMPLES=${GPU_STABLE_SAMPLES:-6}
EVAL_GPUS=${EVAL_GPUS:-0,1,2}
STATE_DIR="$PROJECT_DIR/pipeline_state/$PROBE_UNIT_NAME"
STATE_FILE="$STATE_DIR/state"
LOCK_FILE="$STATE_DIR/pipeline.lock"
STEP50_RESULT="$PROJECT_DIR/evaluation/formal_em/branch_credit_correctness_step50/balanced_n480/result.json"
STEP60_RESULT="$PROJECT_DIR/evaluation/formal_em/branch_credit_correctness_step60/balanced_n480/result.json"
COMPARISON_RESULT="$PROJECT_DIR/branch_credit_expand_mode_comparison.json"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "another selector-probe pipeline already holds $LOCK_FILE" >&2
    exit 3
fi

record_state() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee "$STATE_FILE"
}

validate_eval_result() {
    local result=$1 expected_step=$2
    [[ -s "$result" ]] || return 1
    "$PYTHON_BIN" -c '
import json, sys
path, expected_step = sys.argv[1], int(sys.argv[2])
result = json.load(open(path, encoding="utf-8"))
assert result.get("checkpoint_step") == expected_step
assert result.get("sample_count") == 480
assert sum(result.get("dataset_counts", {}).values()) == 480
assert not any(result.get("error_checks", {}).values())
assert all(result.get("protocol_checks", {}).values())
' "$result" "$expected_step" >/dev/null
}

evaluation_complete() {
    validate_eval_result "$STEP50_RESULT" 50 && validate_eval_result "$STEP60_RESULT" 60
}

wait_for_evaluation() {
    local started now active_state result status
    started=$(date +%s)
    while ! evaluation_complete; do
        now=$(date +%s)
        if (( now - started >= EVAL_TIMEOUT_SECONDS )); then
            record_state "FAILED evaluation wait timed out"
            return 1
        fi

        active_state=$(systemctl --user show "$EVAL_UNIT" -p ActiveState --value 2>/dev/null || true)
        result=$(systemctl --user show "$EVAL_UNIT" -p Result --value 2>/dev/null || true)
        status=$(systemctl --user show "$EVAL_UNIT" -p ExecMainStatus --value 2>/dev/null || true)
        case "$active_state" in
            active|activating|reloading) ;;
            *)
                record_state "FAILED evaluation stopped before valid step50/60 results: active=$active_state result=$result status=$status"
                return 1
                ;;
        esac
        record_state "WAITING evaluation active=$active_state"
        sleep "$POLL_SECONDS"
    done

    while :; do
        active_state=$(systemctl --user show "$EVAL_UNIT" -p ActiveState --value 2>/dev/null || true)
        result=$(systemctl --user show "$EVAL_UNIT" -p Result --value 2>/dev/null || true)
        status=$(systemctl --user show "$EVAL_UNIT" -p ExecMainStatus --value 2>/dev/null || true)
        if [[ "$active_state" == inactive && "$result" == success && "$status" == 0 ]]; then
            record_state "EVALUATION_COMPLETE"
            return 0
        fi
        if [[ "$active_state" != active && "$active_state" != activating && "$active_state" != reloading ]]; then
            record_state "FAILED evaluation results exist but service result is not successful: active=$active_state result=$result status=$status"
            return 1
        fi
        sleep 5
    done
}

gpu_set_is_idle() {
    local query
    query=$(nvidia-smi --id="$EVAL_GPUS" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits) || return 1
    awk -F, -v mem="$GPU_MEMORY_LIMIT_MIB" -v util="$GPU_UTIL_LIMIT" '
        {gsub(/ /, "", $1); gsub(/ /, "", $2)}
        $1 > mem || $2 > util {exit 1}
    ' <<<"$query"
}

host_memory_is_safe() {
    local available_kib required_kib
    available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    required_kib=$((HOST_AVAILABLE_MIN_GIB * 1024 * 1024))
    (( available_kib >= required_kib ))
}

wait_for_resources() {
    local stable=0
    while (( stable < GPU_STABLE_SAMPLES )); do
        if gpu_set_is_idle && host_memory_is_safe; then
            stable=$((stable + 1))
            record_state "WAITING resources stable=$stable/$GPU_STABLE_SAMPLES GPUs=$EVAL_GPUS"
        else
            stable=0
            record_state "WAITING resources GPUs=$EVAL_GPUS or host memory not ready"
        fi
        sleep 5
    done
}

retriever_is_ready() {
    curl -fsS --max-time 30 \
        -H 'Content-Type: application/json' \
        -d '{"queries":["branch credit probe"],"topk":1,"return_scores":true}' \
        http://127.0.0.1:8000/retrieve \
        | "$PYTHON_BIN" -c '
import json, sys
payload = json.load(sys.stdin)
assert payload.get("result") and payload["result"][0]
' >/dev/null
}

validate_comparison() {
    [[ -s "$COMPARISON_RESULT" ]] || return 1
    "$PYTHON_BIN" -c '
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
for mode in ("random", "uncertainty_balanced"):
    stats = result[mode]
    assert stats["records"] > 0
    assert stats["parents_with_multiple_children"] > 0
    assert stats["selection_stats_records"] > 0
comparison = result["comparison"]
assert comparison["effective_parent_rate_absolute_change"] is not None
assert comparison["nonzero_sibling_edge_rate_absolute_change"] is not None
' "$COMPARISON_RESULT" >/dev/null
}

cd "$PROJECT_DIR"
record_state "WAITING evaluation unit=$EVAL_UNIT"
wait_for_evaluation
wait_for_resources
if ! retriever_is_ready; then
    record_state "FAILED retriever is unavailable"
    exit 1
fi

record_state "RUNNING fixed-budget selector probes"
CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
ROLLOUT_ACCUMULATION_STEPS=16 \
"$PROJECT_DIR/run_expand_mode_credit_comparison.sh"

validate_comparison
record_state "COMPLETE result=$COMPARISON_RESULT"

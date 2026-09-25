#!/usr/bin/env bash
# Wait for both corrected training runs, then acquire three idle GPUs and
# evaluate both checkpoints sequentially under the same protocol.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/luwa/Documents/Tree-GRPO}
RERUN_SERVICE=tree-grpo-branch-credit-correctness-monitor.service
CONTINUATION_SERVICE=tree-grpo-branch-credit-continuation-watcher.service
RERUN_NAME=multihop-tree-dapo-branch-credit-correctness-step20to30-20260907-195640
CONTINUATION_NAME=multihop-tree-dapo-branch-credit-correctness-step30to35-from-old-step30-20260907
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-60}

checkpoint_complete() {
    local path=$1 rank kind
    for rank in 0 1 2; do
        for kind in model optim extra_state; do
            [[ -s "$path/${kind}_world_size_3_rank_${rank}.pt" ]] || return 1
        done
    done
}

service_succeeded() {
    local service=$1 active_state result
    active_state=$(systemctl --user show "$service" -p ActiveState --value 2>/dev/null || true)
    result=$(systemctl --user show "$service" -p Result --value 2>/dev/null || true)
    [[ "$active_state" == inactive && "$result" == success ]]
}

while true; do
    if service_succeeded "$RERUN_SERVICE" \
        && service_succeeded "$CONTINUATION_SERVICE" \
        && checkpoint_complete "$PROJECT_DIR/verl_checkpoints/$RERUN_NAME/actor/global_step_30" \
        && checkpoint_complete "$PROJECT_DIR/verl_checkpoints/$CONTINUATION_NAME/actor/global_step_35"; then
        break
    fi
    sleep "$POLL_INTERVAL_SECONDS"
done

export TRAIN_SCRIPT="$PROJECT_DIR/run_branch_credit_correctness_all_eval.sh"
export EXPERIMENT_NAME=branch-credit-correctness-step30-and-step35-formal-eval
export MONITOR_LOG="$PROJECT_DIR/verl_log/branch_credit_correctness_all_eval_gpu_monitor.log"
export STATE_FILE="$PROJECT_DIR/verl_log/branch_credit_correctness_all_eval_gpu_monitor.state"
export LOCK_FILE="$PROJECT_DIR/verl_log/branch_credit_correctness_all_eval_gpu_monitor.lock"
export STABLE_SAMPLES=${STABLE_SAMPLES:-15}
export POLL_INTERVAL_SECONDS=2

exec "$PROJECT_DIR/scripts/monitor_and_launch_branch_credit.sh"

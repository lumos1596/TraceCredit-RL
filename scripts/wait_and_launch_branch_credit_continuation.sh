#!/usr/bin/env bash
# Queue the corrected step30->35 continuation until the corrected step20->30
# rerun has completed and released its host/GPU resources.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/luwa/Documents/Tree-GRPO}
PRIOR_SERVICE=${PRIOR_SERVICE:-tree-grpo-branch-credit-correctness-monitor.service}
PRIOR_RUN_NAME=${PRIOR_RUN_NAME:-multihop-tree-dapo-branch-credit-correctness-step20to30-20260907-195640}
CONTINUATION_RUN_NAME=${CONTINUATION_RUN_NAME:-multihop-tree-dapo-branch-credit-correctness-step30to35-from-old-step30-20260907}
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-60}

PRIOR_CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$PRIOR_RUN_NAME/actor/global_step_30"

checkpoint_complete() {
    local rank kind
    for rank in 0 1 2; do
        for kind in model optim extra_state; do
            [[ -s "$PRIOR_CHECKPOINT/${kind}_world_size_3_rank_${rank}.pt" ]] || return 1
        done
    done
}

while true; do
    active_state=$(systemctl --user show "$PRIOR_SERVICE" -p ActiveState --value 2>/dev/null || true)
    result=$(systemctl --user show "$PRIOR_SERVICE" -p Result --value 2>/dev/null || true)
    if [[ "$active_state" == inactive && "$result" == success ]] && checkpoint_complete; then
        break
    fi
    sleep "$POLL_INTERVAL_SECONDS"
done

export TRAIN_SCRIPT="$PROJECT_DIR/train_multihopqa_branch_credit_correctness_step30_to35.sh"
export EXPERIMENT_NAME="$CONTINUATION_RUN_NAME"
export MONITOR_LOG="$PROJECT_DIR/verl_log/${CONTINUATION_RUN_NAME}-gpu-monitor.log"
export STATE_FILE="$PROJECT_DIR/verl_log/${CONTINUATION_RUN_NAME}-gpu-monitor.state"
export LOCK_FILE="$PROJECT_DIR/verl_log/${CONTINUATION_RUN_NAME}-gpu-monitor.lock"
export STABLE_SAMPLES=${STABLE_SAMPLES:-15}
export POLL_INTERVAL_SECONDS=2

exec "$PROJECT_DIR/scripts/monitor_and_launch_branch_credit.sh"

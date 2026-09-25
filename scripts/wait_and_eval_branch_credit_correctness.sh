#!/usr/bin/env bash
# Wait for a successful detached training completion, then wait for three idle
# GPUs and run both formal evaluations. This process is itself run by systemd.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/luwa/Documents/Tree-GRPO}
TRAIN_SERVICE=${TRAIN_SERVICE:-tree-grpo-branch-credit-correctness-monitor.service}
: "${TRAIN_RUN_NAME:?TRAIN_RUN_NAME is required}"

CHECKPOINT="$PROJECT_DIR/verl_checkpoints/$TRAIN_RUN_NAME/actor/global_step_30"
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-60}

checkpoint_complete() {
    local rank kind
    for rank in 0 1 2; do
        for kind in model optim extra_state; do
            [[ -s "$CHECKPOINT/${kind}_world_size_3_rank_${rank}.pt" ]] || return 1
        done
    done
}

while true; do
    active_state=$(systemctl --user show "$TRAIN_SERVICE" -p ActiveState --value 2>/dev/null || true)
    result=$(systemctl --user show "$TRAIN_SERVICE" -p Result --value 2>/dev/null || true)
    if [[ "$active_state" == inactive && "$result" == success ]] && checkpoint_complete; then
        break
    fi
    sleep "$POLL_INTERVAL_SECONDS"
done

export TRAIN_SCRIPT="$PROJECT_DIR/run_branch_credit_correctness_step30_eval.sh"
export EXPERIMENT_NAME="${TRAIN_RUN_NAME}-formal-eval"
export MONITOR_LOG="$PROJECT_DIR/verl_log/${TRAIN_RUN_NAME}-eval-gpu-monitor.log"
export STATE_FILE="$PROJECT_DIR/verl_log/${TRAIN_RUN_NAME}-eval-gpu-monitor.state"
export LOCK_FILE="$PROJECT_DIR/verl_log/${TRAIN_RUN_NAME}-eval-gpu-monitor.lock"
export STABLE_SAMPLES=${STABLE_SAMPLES:-15}
export POLL_INTERVAL_SECONDS=2

exec "$PROJECT_DIR/scripts/monitor_and_launch_branch_credit.sh"

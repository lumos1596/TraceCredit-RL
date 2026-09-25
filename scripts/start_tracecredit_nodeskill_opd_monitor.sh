#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/luwa/Documents/Tree-GRPO
export PROJECT_DIR="$ROOT"
export TRAIN_SCRIPT="$ROOT/run_tracecredit_nodeskill_opd_smoke.sh"
export PREFLIGHT_SCRIPT="$ROOT/scripts/preflight_tracecredit_nodeskill_opd.sh"
export CANDIDATE_GPUS=0,1,2
export REQUIRED_GPUS=3
export IDLE_MEMORY_MIB=750
export IDLE_UTILIZATION_PERCENT=5
# Six one-second samples span at least five continuous seconds.
export STABLE_SAMPLES=6
export POLL_INTERVAL_SECONDS=1
export MIN_AVAILABLE_MEMORY_MIB=155000
export EXPERIMENT_PREFIX=tracecredit-nodeskill-sampled-nll-opd-smoke-sft350
export MONITOR_LOG="$ROOT/verl_log/tracecredit_nodeskill_opd_gpu_monitor.log"
export STATE_FILE="$ROOT/verl_log/tracecredit_nodeskill_opd_gpu_monitor.state"
export LOCK_FILE="$ROOT/verl_log/tracecredit_nodeskill_opd_gpu_monitor.lock"

exec "$ROOT/scripts/monitor_and_launch_branch_credit.sh" "$@"

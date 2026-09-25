#!/usr/bin/env bash
# Wait for three stably idle GPUs, then replace this monitor process with the
# formal Branch-Credit Tree-GRPO training process.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/luwa/Documents/Tree-GRPO}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh}
NVIDIA_SMI_BIN=${NVIDIA_SMI_BIN:-nvidia-smi}

# GPU 4 hosts the retrieval service and is deliberately excluded by default.
CANDIDATE_GPUS=${CANDIDATE_GPUS:-0,1,2,3,5,6,7}
REQUIRED_GPUS=${REQUIRED_GPUS:-3}
IDLE_MEMORY_MIB=${IDLE_MEMORY_MIB:-500}
IDLE_UTILIZATION_PERCENT=${IDLE_UTILIZATION_PERCENT:-5}
STABLE_SAMPLES=${STABLE_SAMPLES:-15}
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-2}
# Three FSDP workers plus the retrieval index need substantial host RAM.  A
# GPU-only gate can otherwise launch during another user's job-transition
# window and let Ray kill the run at its 95% node-memory threshold.
MIN_AVAILABLE_MEMORY_MIB=${MIN_AVAILABLE_MEMORY_MIB:-150000}
MEMINFO_FILE=${MEMINFO_FILE:-/proc/meminfo}

MONITOR_LOG=${MONITOR_LOG:-$PROJECT_DIR/verl_log/branch_credit_gpu_monitor.log}
STATE_FILE=${STATE_FILE:-$PROJECT_DIR/verl_log/branch_credit_gpu_monitor.state}
LOCK_FILE=${LOCK_FILE:-$PROJECT_DIR/verl_log/branch_credit_gpu_monitor.lock}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-multihop-tree-dapo-branch-credit-step20to30-retry}
DRY_RUN=${DRY_RUN:-false}
ONCE=${ONCE:-false}
PREFLIGHT_SCRIPT=${PREFLIGHT_SCRIPT:-}

usage() {
    cat <<'EOF'
Usage: monitor_and_launch_branch_credit.sh [--dry-run] [--once]

Environment overrides:
  CANDIDATE_GPUS=0,1,2,3,5,6,7  Physical GPUs eligible for training
  REQUIRED_GPUS=3                  Number of GPUs required
  IDLE_MEMORY_MIB=500              Maximum used memory for an idle GPU
  IDLE_UTILIZATION_PERCENT=5       Maximum utilization for an idle GPU
  STABLE_SAMPLES=15                Consecutive identical idle selections
  POLL_INTERVAL_SECONDS=2          Delay between samples (~28s to 15 samples)
  MIN_AVAILABLE_MEMORY_MIB=150000  Required host MemAvailable before launch
  MEMINFO_FILE=/proc/meminfo       Injectable memory source for tests
  EXPERIMENT_NAME=...              Optional exact output name
  NVIDIA_SMI_BIN=...               Injectable nvidia-smi path for tests
  TRAIN_SCRIPT=...                 Injectable training command for tests
EOF
}

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=true ;;
        --once) ONCE=true ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_integer "$REQUIRED_GPUS" || { echo "REQUIRED_GPUS must be positive" >&2; exit 2; }
is_nonnegative_integer "$IDLE_MEMORY_MIB" || { echo "IDLE_MEMORY_MIB must be non-negative" >&2; exit 2; }
is_nonnegative_integer "$IDLE_UTILIZATION_PERCENT" || { echo "IDLE_UTILIZATION_PERCENT must be non-negative" >&2; exit 2; }
is_positive_integer "$STABLE_SAMPLES" || { echo "STABLE_SAMPLES must be positive" >&2; exit 2; }
is_nonnegative_integer "$POLL_INTERVAL_SECONDS" || { echo "POLL_INTERVAL_SECONDS must be non-negative" >&2; exit 2; }
is_nonnegative_integer "$MIN_AVAILABLE_MEMORY_MIB" || { echo "MIN_AVAILABLE_MEMORY_MIB must be non-negative" >&2; exit 2; }
[[ "$DRY_RUN" == true || "$DRY_RUN" == false ]] || { echo "DRY_RUN must be true or false" >&2; exit 2; }
[[ "$ONCE" == true || "$ONCE" == false ]] || { echo "ONCE must be true or false" >&2; exit 2; }
[[ -x "$TRAIN_SCRIPT" ]] || { echo "training script is not executable: $TRAIN_SCRIPT" >&2; exit 2; }
[[ -z "$PREFLIGHT_SCRIPT" || -x "$PREFLIGHT_SCRIPT" ]] || {
    echo "preflight script is not executable: $PREFLIGHT_SCRIPT" >&2; exit 2;
}

IFS=',' read -r -a candidate_array <<< "$CANDIDATE_GPUS"
declare -A candidate_set=()
for gpu in "${candidate_array[@]}"; do
    gpu=${gpu//[[:space:]]/}
    is_nonnegative_integer "$gpu" || { echo "invalid GPU index in CANDIDATE_GPUS: $gpu" >&2; exit 2; }
    [[ -z "${candidate_set[$gpu]+x}" ]] || { echo "duplicate GPU index: $gpu" >&2; exit 2; }
    candidate_set[$gpu]=1
done
(( ${#candidate_set[@]} >= REQUIRED_GPUS )) || {
    echo "candidate GPU count is smaller than REQUIRED_GPUS" >&2
    exit 2
}

mkdir -p "$(dirname "$MONITOR_LOG")" "$(dirname "$STATE_FILE")" "$(dirname "$LOCK_FILE")"
touch "$MONITOR_LOG" "$STATE_FILE" "$LOCK_FILE"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "another branch-credit GPU monitor/training launch owns $LOCK_FILE" >&2
    exit 4
fi

log() {
    local line
    line="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$line" | tee -a "$MONITOR_LOG"
}

write_state() {
    local status=$1
    local selection=${2:-}
    local consecutive=${3:-0}
    local available_memory_mib=${4:-unknown}
    local tmp_file="${STATE_FILE}.tmp.$$"
    {
        echo "status=$status"
        echo "updated_at=$(date --iso-8601=seconds)"
        echo "candidate_gpus=$CANDIDATE_GPUS"
        echo "selected_gpus=$selection"
        echo "consecutive_samples=$consecutive"
        echo "required_samples=$STABLE_SAMPLES"
        echo "available_memory_mib=$available_memory_mib"
        echo "required_available_memory_mib=$MIN_AVAILABLE_MEMORY_MIB"
        echo "monitor_pid=$$"
    } > "$tmp_file"
    mv "$tmp_file" "$STATE_FILE"
}

query_available_memory_mib() {
    local key value unit
    while read -r key value unit; do
        if [[ "$key" == "MemAvailable:" && "$value" =~ ^[0-9]+$ && "$unit" == "kB" ]]; then
            printf '%s\n' "$((value / 1024))"
            return 0
        fi
    done < "$MEMINFO_FILE"
    return 1
}

query_idle_gpus() {
    local output index memory utilization
    local -a idle=()
    if ! output=$("$NVIDIA_SMI_BIN" \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits); then
        return 1
    fi
    while IFS=',' read -r index memory utilization; do
        index=${index//[[:space:]]/}
        memory=${memory//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        [[ -n "$index" ]] || continue
        [[ -n "${candidate_set[$index]+x}" ]] || continue
        is_nonnegative_integer "$memory" || continue
        is_nonnegative_integer "$utilization" || continue
        if (( memory <= IDLE_MEMORY_MIB && utilization <= IDLE_UTILIZATION_PERCENT )); then
            idle+=("$index")
        fi
    done <<< "$output"
    printf '%s\n' "${idle[@]}"
}

choose_gpus() {
    local -a idle=("$@")
    local -a selected=("${idle[@]:0:REQUIRED_GPUS}")
    local joined
    ((${#selected[@]} == REQUIRED_GPUS)) || return 1
    IFS=','; joined="${selected[*]}"; unset IFS
    printf '%s\n' "$joined"
}

make_experiment_name() {
    local proposed
    if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
        proposed=$EXPERIMENT_NAME
    else
        proposed="${EXPERIMENT_PREFIX}-$(date '+%Y%m%d-%H%M%S')"
    fi
    if [[ -e "$PROJECT_DIR/verl_checkpoints/$proposed" ||
          -e "$PROJECT_DIR/rollouts/$proposed" ||
          -e "$PROJECT_DIR/verl_log/$proposed.log" ]]; then
        echo "refusing to overwrite existing experiment outputs: $proposed" >&2
        return 1
    fi
    printf '%s\n' "$proposed"
}

consecutive=0
last_selection=
write_state waiting
log "monitor started: candidates=$CANDIDATE_GPUS required=$REQUIRED_GPUS gpu_memory<=${IDLE_MEMORY_MIB}MiB util<=${IDLE_UTILIZATION_PERCENT}% host_available>=${MIN_AVAILABLE_MEMORY_MIB}MiB stable_samples=$STABLE_SAMPLES"

while true; do
    available_memory_mib=$(query_available_memory_mib) || {
        consecutive=0
        last_selection=
        write_state query_failed "" 0 unknown
        log "host MemAvailable query failed; retrying"
        if [[ "$ONCE" == true ]]; then exit 3; fi
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    }
    idle_output=$(query_idle_gpus) || {
        consecutive=0
        last_selection=
        write_state query_failed "" 0 "$available_memory_mib"
        log "nvidia-smi query failed; retrying"
        if [[ "$ONCE" == true ]]; then exit 3; fi
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    }
    idle_gpus=()
    if [[ -n "$idle_output" ]]; then
        mapfile -t idle_gpus <<< "$idle_output"
    fi

    selection=$(choose_gpus "${idle_gpus[@]}") || selection=
    if (( available_memory_mib < MIN_AVAILABLE_MEMORY_MIB )); then
        selection=
    fi
    if [[ -n "$selection" && -n "$PREFLIGHT_SCRIPT" ]] && ! "$PREFLIGHT_SCRIPT"; then
        selection=
        log "preflight services or files are not ready; continuing to wait"
    fi
    if [[ -z "$selection" ]]; then
        consecutive=0
        last_selection=
        write_state waiting "" 0 "$available_memory_mib"
        log "waiting: idle_candidates=${idle_gpus[*]:-none} host_available=${available_memory_mib}MiB required=${MIN_AVAILABLE_MEMORY_MIB}MiB"
    elif [[ "$selection" == "$last_selection" ]]; then
        ((consecutive += 1))
        write_state stabilizing "$selection" "$consecutive" "$available_memory_mib"
        log "stable candidate $selection ($consecutive/$STABLE_SAMPLES)"
    else
        last_selection=$selection
        consecutive=1
        write_state stabilizing "$selection" "$consecutive" "$available_memory_mib"
        log "new candidate $selection (1/$STABLE_SAMPLES)"
    fi

    if [[ -n "$selection" && "$consecutive" -ge "$STABLE_SAMPLES" ]]; then
        # Close the check-to-launch window as much as possible with one final
        # query. A cluster-wide scheduler is still required for perfect GPU
        # reservation against unrelated users.
        final_output=$(query_idle_gpus) || final_output=
        final_available_memory_mib=$(query_available_memory_mib) || final_available_memory_mib=0
        final_idle=()
        if [[ -n "$final_output" ]]; then
            mapfile -t final_idle <<< "$final_output"
        fi
        final_selection=$(choose_gpus "${final_idle[@]}") || final_selection=
        if [[ -n "$final_selection" && -n "$PREFLIGHT_SCRIPT" ]] && ! "$PREFLIGHT_SCRIPT"; then
            final_selection=
        fi
        if [[ "$final_selection" != "$selection" || "$final_available_memory_mib" -lt "$MIN_AVAILABLE_MEMORY_MIB" ]]; then
            log "GPU or host-memory availability changed during final recheck; returning to wait"
            consecutive=0
            last_selection=
        else
            experiment_name=$(make_experiment_name) || exit 5
            write_state ready "$selection" "$consecutive" "$final_available_memory_mib"
            log "GPUs $selection are stably idle; experiment=$experiment_name"
            if [[ "$DRY_RUN" == true ]]; then
                log "dry-run: training was not launched"
                write_state dry_run "$selection" "$consecutive" "$final_available_memory_mib"
                exit 0
            fi
            write_state launching "$selection" "$consecutive" "$final_available_memory_mib"
            log "launching $TRAIN_SCRIPT with CUDA_VISIBLE_DEVICES=$selection"
            export CUDA_VISIBLE_DEVICES=$selection
            export EXPERIMENT_NAME=$experiment_name
            exec "$TRAIN_SCRIPT"
        fi
    fi

    if [[ "$ONCE" == true ]]; then
        write_state not_ready "$selection" "$consecutive" "$available_memory_mib"
        exit 3
    fi
    sleep "$POLL_INTERVAL_SECONDS"
done

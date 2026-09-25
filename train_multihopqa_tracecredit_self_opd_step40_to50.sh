#!/usr/bin/env bash
# Matched TraceCredit + Self-OPD continuation for the grouped-branch baseline.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-tracecredit-self-opd-step40to50-20260918}
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME/actor"
BASE_CHECKPOINT="$PROJECT_DIR/verl_checkpoints/multihop-tree-dapo-branch-credit-correctness-step35to40-20260908/actor/global_step_40"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}

checkpoint_complete() {
    local checkpoint=$1 rank kind
    for kind in model optim extra_state; do
        for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
            [[ -s "$checkpoint/${kind}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt" ]] || return 1
        done
    done
}

latest_checkpoint=
if [[ -d "$CHECKPOINT_ROOT" ]]; then
    mapfile -t checkpoint_candidates < <(
        find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'global_step_*' -printf '%f\n' | sort -Vr
    )
    for candidate in "${checkpoint_candidates[@]}"; do
        if checkpoint_complete "$CHECKPOINT_ROOT/$candidate"; then
            latest_checkpoint=$candidate
            break
        fi
        echo "ignoring incomplete checkpoint: $CHECKPOINT_ROOT/$candidate" >&2
    done
fi

if [[ -n "$latest_checkpoint" ]]; then
    checkpoint_step=${latest_checkpoint#global_step_}
    [[ "$checkpoint_step" =~ ^[0-9]+$ ]] || {
        echo "invalid checkpoint step: $latest_checkpoint" >&2
        exit 2
    }
    if (( checkpoint_step >= 50 )); then
        echo "TraceCredit + Self-OPD training already complete at step $checkpoint_step"
        exit 0
    fi
    export INIT_CHECKPOINT="$CHECKPOINT_ROOT/$latest_checkpoint"
    export RESUME_GLOBAL_STEP="$checkpoint_step"
else
    checkpoint_complete "$BASE_CHECKPOINT" || {
        echo "incomplete base checkpoint: $BASE_CHECKPOINT" >&2
        exit 2
    }
    export INIT_CHECKPOINT="$BASE_CHECKPOINT"
    export RESUME_GLOBAL_STEP=40
fi

export EXPERIMENT_NAME
export RESUME_ACTOR_STATE=true
export BRANCH_CREDIT_COEF=1.0
export BRANCH_CREDIT_VALUE_MODE=correctness
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8
export EXPAND_MODE=grouped_random
export SELF_OPD_ENABLED=true
export SELF_OPD_COEF=${SELF_OPD_COEF:-0.001}
export SELF_OPD_TEMPERATURE=${SELF_OPD_TEMPERATURE:-1.0}
export SELF_OPD_MAX_EVENTS=${SELF_OPD_MAX_EVENTS:-3}
export SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-512}
export SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-64}
export SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-96}
export SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25}
export SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0}
export SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0}
# The trainer uses an exclusive upper bound: 51 performs steps 41--50.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-51}
export SAVE_FREQ=${SAVE_FREQ:-1}
export SAVE_AT_END=true
export DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24}
export DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48}
export N_GPUS_PER_NODE

exec "$PROJECT_DIR/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

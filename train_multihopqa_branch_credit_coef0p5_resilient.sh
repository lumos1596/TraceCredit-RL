#!/usr/bin/env bash
# Resilient coefficient-0.5 ablation. The first launch starts actor-only from
# the shared Tree-GRPO step-20 checkpoint. A service restart resumes the latest
# checkpoint with optimizer, scheduler, and RNG state intact.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-correctness-coef0p5-step20to40-fixed}
CHECKPOINT_ROOT="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME/actor"
SHARED_STEP20="$PROJECT_DIR/verl_checkpoints/multihopqa-tree-grpo-sft350-3gpu-budget48-persistent-20260826/actor/global_step_20"
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}

latest_checkpoint=
if [[ -d "$CHECKPOINT_ROOT" ]]; then
    mapfile -t checkpoint_candidates < <(
        find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'global_step_*' -printf '%f\n' | sort -Vr
    )
    for candidate in "${checkpoint_candidates[@]}"; do
        complete=true
        for ((rank = 0; rank < N_GPUS_PER_NODE; rank++)); do
            for shard_type in model optim extra_state; do
                shard="$CHECKPOINT_ROOT/$candidate/${shard_type}_world_size_${N_GPUS_PER_NODE}_rank_${rank}.pt"
                if [[ ! -s "$shard" ]]; then
                    complete=false
                    break 2
                fi
            done
        done
        if [[ "$complete" == true ]]; then
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
    export INIT_CHECKPOINT="$CHECKPOINT_ROOT/$latest_checkpoint"
    export RESUME_GLOBAL_STEP="$checkpoint_step"
    export RESUME_ACTOR_STATE=true
    echo "resuming full training state from $INIT_CHECKPOINT"
else
    export INIT_CHECKPOINT="$SHARED_STEP20"
    export RESUME_GLOBAL_STEP=20
    export RESUME_ACTOR_STATE=false
    echo "starting coefficient-0.5 ablation from shared actor step20"
fi

export EXPERIMENT_NAME
export BRANCH_CREDIT_COEF=0.5
export BRANCH_CREDIT_VALUE_MODE=correctness
export BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-41}
export SAVE_FREQ=${SAVE_FREQ:-1}
export SAVE_AT_END=true
export DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24}
export DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48}
export N_GPUS_PER_NODE
export EXPAND_MODE=${EXPAND_MODE:-random}

exec "$PROJECT_DIR/train_multihopqa_branch_credit_correctness_step20_to30.sh"

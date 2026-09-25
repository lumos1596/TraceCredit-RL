#!/usr/bin/env bash
# Fair no-Credit Tree-GRPO continuation from the original step-30 checkpoint.
# Restores model, Adam optimizer, scheduler, and worker RNG state, then saves
# both step 35 and step 40 for matched evaluation.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
BASE_RUN=${BASE_RUN:-multihop-tree-dapo-mixed-step20to30-20260830}

export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/$BASE_RUN/actor/global_step_30"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-no-credit-step30to40-20260909}
export RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-30}
export RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-true}
# The accumulation loop treats this as an exclusive upper bound, so 41 runs
# optimizer steps 31 through 40.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-41}
export SAVE_FREQ=${SAVE_FREQ:-5}
# Match the memory-safe full-state continuation settings already validated by
# the corrected-Credit step30->40 runs.
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
# The host currently has a 56-GiB retrieval index resident. Keeping the FSDP
# parameter shards and gradients on GPU saves host RAM without changing the
# optimizer or training math; Adam state remains explicitly offloaded.
export PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
export GRAD_OFFLOAD=${GRAD_OFFLOAD:-false}

exec "$PROJECT_DIR/train_multihopqa_tree_dapo_step20_to30.sh" "$@"

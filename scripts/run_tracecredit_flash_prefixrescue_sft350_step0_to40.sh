#!/usr/bin/env bash
# From-scratch run from the SFT350 checkpoint (step 0 -> step 40) using the
# forkable teacher-prefix TREE rescue (trainer.teacher_rescue_style=prefix).
#
# This is the clean-sheet counterpart of the step28->40 continuation: it does
# NOT resume from any RL checkpoint. An empty INIT_CHECKPOINT makes the trainer
# start directly from MODEL_DIR, i.e.
#   verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350
#
# Services (see README "Current experiment"):
#   - One Flash service instance is enough on a fresh machine because the same
#     process serves /generate (Self-OPD analyzer), /semantic_judge (semantic
#     reward) and /prefix_rescue. Override FLASH_SERVICE_URL to split traffic
#     across two instances if desired.
#   - Retriever on RETRIEVER_URL.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.conda/envs/treegrpo/bin/python"}
export CC=${CC:-cc}
export EXPERIMENT_NAME=tracecredit-opid-flash-prefix-tree-rescue-sft350-step0to40-20261001
# Dataset and SFT350 base model; empty INIT_CHECKPOINT => start directly from MODEL_DIR.
export DATA_DIR=${DATA_DIR:-"$ROOT/data/multihopqa_search_mixed_402020_20260830"}
export MODEL_DIR=${MODEL_DIR:-"$ROOT/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"}
export INIT_CHECKPOINT=""
export RESUME_GLOBAL_STEP=0
export RESUME_ACTOR_STATE=false
export TOTAL_TRAINING_STEPS=41
export SAVE_FREQ=1
export TEST_FREQ=4
export VAL_BEFORE_TRAIN=false
export VAL_AT_END=true
export SAVE_AT_END=true
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")}
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/ray_prefixrescue
export LOGGER="['console']"
export LOG_POLICY_ENTROPY=true
export ENTROPY_TOKEN_CHUNK_SIZE=16
export SELF_OPD_ENABLED=true
export PPO_EPOCHS=2
export SELF_OPD_LOSS=opid_advantage
export SELF_OPD_COEF=1.0
export SELF_OPD_DELTA_MIN=0.0
export SELF_OPD_THINK_COEF=0.2
export SELF_OPD_QUERY_COEF=1.0
export SELF_OPD_MIN_VALUE_GAP=0.1
export SELF_OPD_MIN_RAW_ADVANTAGE=0.0
export SELF_OPD_MAX_EVENTS=3
export SELF_OPD_MAX_TEACHER_LENGTH=512
export SELF_OPD_MAX_ACTION_TOKENS=256
export SELF_OPD_MAX_QUERY_TOKENS=64
export SELF_OPD_MAX_EVIDENCE_TOKENS=128
export SELF_OPD_TEACHER_CONTEXT=opid_global_failure
export SELF_OPD_EVENT_SELECTION=all
# One Flash instance serves OPD analyzer, semantic judge and prefix rescue.
export FLASH_SERVICE_URL=${FLASH_SERVICE_URL:-http://127.0.0.1:8130}
export SELF_OPD_ANALYZER_URL="$FLASH_SERVICE_URL"
export SEMANTIC_REWARD_URL="$FLASH_SERVICE_URL"
export TEACHER_RESCUE_ENABLED=true
export TEACHER_RESCUE_STYLE=prefix
export TEACHER_RESCUE_PREFIX_HOPS=2
export TEACHER_RESCUE_URL="$FLASH_SERVICE_URL"
export TEACHER_RESCUE_KD_COEF=0.01
export ACTOR_PARAM_OFFLOAD=false
export ACTOR_GRAD_OFFLOAD=true
export OPTIMIZER_OFFLOAD=true
export REF_PARAM_OFFLOAD=false
export GPU_MEMORY_UTILIZATION=0.36
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=3072
export ROLLOUT_MAX_NUM_SEQS=32
export DAPO_TARGET_EFFECTIVE_PROMPTS=12
export DAPO_MAX_CHUNKS=48
export ROLLOUT_ACCUMULATION_STEPS=8
export MAX_TURNS=5
export MAX_START_LENGTH=3072
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8002/retrieve}

curl -fsS --max-time 5 "$FLASH_SERVICE_URL/health" >/dev/null
curl -fsS --max-time 5 "${RETRIEVER_URL%/retrieve}/health" >/dev/null
exec "$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

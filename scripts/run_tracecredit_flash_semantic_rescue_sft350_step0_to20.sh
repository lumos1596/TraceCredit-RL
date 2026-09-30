#!/usr/bin/env bash
# SFT350-start RL+OPD with Flash semantic reward and evidence-gated rescue.
set -euo pipefail
ROOT=/home/luwa/Documents/Tree-GRPO
export PYTHON_BIN=/home/luwa/Documents/miniforge3/envs/treegrpo/bin/python
export CC=/home/luwa/Documents/miniforge3/envs/searchr1/bin/x86_64-conda-linux-gnu-gcc
export EXPERIMENT_NAME=tracecredit-opid-flash-semantic-rescue-sft350-step0to20-20260929
export INIT_CHECKPOINT=""
export RESUME_GLOBAL_STEP=0
export RESUME_ACTOR_STATE=false
export TOTAL_TRAINING_STEPS=21
export SAVE_FREQ=1
export TEST_FREQ=4
export VAL_BEFORE_TRAIN=false
export VAL_AT_END=true
export SAVE_AT_END=true
export N_GPUS_PER_NODE=3
export CUDA_VISIBLE_DEVICES=0,4,5
export RAY_ADDRESS=local
export RAY_TMPDIR=/tmp/ray_flashrescue
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
export SELF_OPD_ANALYZER_URL=http://127.0.0.1:8130
export SEMANTIC_REWARD_URL=http://127.0.0.1:8130
export TEACHER_RESCUE_ENABLED=true
export TEACHER_RESCUE_URL=http://127.0.0.1:8130
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
export RETRIEVER_URL=http://127.0.0.1:8002/retrieve

curl -fsS --max-time 5 http://127.0.0.1:8130/health >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8002/health >/dev/null
exec "$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

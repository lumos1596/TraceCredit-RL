#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/luwa/Documents/Tree-GRPO

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-tracecredit-nodeskill-sampled-nll-opd-formal-entropy-chunked-20260924}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$ROOT/verl_checkpoints/tracecredit-nodeskill-sampled-nll-opd-smoke-sft350-retry-20260924-1942/actor/global_step_1"}
export RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-1}
export RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-true}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-21}
export SAVE_FREQ=${SAVE_FREQ:-1}
export SAVE_AT_END=${SAVE_AT_END:-true}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}

export LOGGER="['console','wandb']"
export WANDB_MODE=online
export LOG_POLICY_ENTROPY=true
export ENTROPY_TOKEN_CHUNK_SIZE=16

export SELF_OPD_ENABLED=true
export SELF_OPD_LOSS=sampled_nll
export SELF_OPD_COEF=0.001
export SELF_OPD_BETA=5.0
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
export SELF_OPD_TEACHER_CONTEXT=node_skill
export SELF_OPD_EVENT_SELECTION=contrast
# The local NodeSkill server exposes POST /generate; self_opd appends
# "/generate" to this base URL.
export SELF_OPD_ANALYZER_URL=http://127.0.0.1:8127

export ACTOR_PARAM_OFFLOAD=true
export ACTOR_GRAD_OFFLOAD=true
export OPTIMIZER_OFFLOAD=true
export REF_PARAM_OFFLOAD=true
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.36}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=3072
export ROLLOUT_MAX_NUM_SEQS=32
export DAPO_TARGET_EFFECTIVE_PROMPTS=12
export DAPO_MAX_CHUNKS=24
export ROLLOUT_ACCUMULATION_STEPS=8
# Services (retriever :8000, node-skill analyzer :8127) run on the service host.
# Defaults target the service host itself; on a training-only machine, export
# RETRIEVER_URL / SELF_OPD_ANALYZER_URL pointing at the service host's LAN IP
# (see README "Cross-machine handoff").
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}
export SELF_OPD_ANALYZER_URL=${SELF_OPD_ANALYZER_URL:-http://127.0.0.1:8127}

exec "$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh"

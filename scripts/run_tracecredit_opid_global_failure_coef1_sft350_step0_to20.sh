#!/usr/bin/env bash
# Start from the SFT step350 actor for the OPID coefficient 1.0 stress test.
set -euo pipefail

ROOT=/home/luwa/Documents/Tree-GRPO

export PYTHON_BIN=/home/luwa/Documents/miniforge3/envs/treegrpo/bin/python
export CC=/home/luwa/Documents/miniforge3/envs/searchr1/bin/x86_64-conda-linux-gnu-gcc
export EXPERIMENT_NAME=tracecredit-opid-global-failure-coef1-sft350-step0to20-20260928
export INIT_CHECKPOINT=""
export RESUME_GLOBAL_STEP=0
export RESUME_ACTOR_STATE=false
export TOTAL_TRAINING_STEPS=21
export SAVE_FREQ=1
export TEST_FREQ=4
export SAVE_AT_END=true
export N_GPUS_PER_NODE=3
export CUDA_VISIBLE_DEVICES=1,2,3

# Console logging provides a durable metric stream for the separate W&B sync
# process, whose dependencies are isolated from the vLLM training environment.
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
export SELF_OPD_ANALYZER_URL=http://127.0.0.1:8128
export ACTOR_PARAM_OFFLOAD=true
export ACTOR_GRAD_OFFLOAD=true
export OPTIMIZER_OFFLOAD=true
export REF_PARAM_OFFLOAD=true
export GPU_MEMORY_UTILIZATION=0.36
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=3072
export ROLLOUT_MAX_NUM_SEQS=32
export DAPO_TARGET_EFFECTIVE_PROMPTS=12
export DAPO_MAX_CHUNKS=48
export ROLLOUT_ACCUMULATION_STEPS=8
export RETRIEVER_URL=http://127.0.0.1:8000/retrieve

if ! tmux has-session -t tracecredit_opid_global_failure_coef1_sft350_step0to20_wandb 2>/dev/null; then
    tmux new-session -d -s tracecredit_opid_global_failure_coef1_sft350_step0to20_wandb \
        "cd '$ROOT' && exec '$ROOT/.venv-wandb/bin/python' '$ROOT/scripts/sync_training_metrics_to_wandb.py' --log '$ROOT/verl_log/$EXPERIMENT_NAME.log' --experiment '$EXPERIMENT_NAME' --run-id tracecredit-opid-global-failure-coef1-sft350-step0to20-20260928 --run-name '$EXPERIMENT_NAME' --watch-pid $$ >> '$ROOT/verl_log/tracecredit_opid_global_failure_coef1_sft350_step0to20_wandb_sync.log' 2>&1"
fi

exec "$ROOT/train_multihopqa_branch_credit_dapo_step20_to30.sh" "$@"

#!/usr/bin/env bash
# Fair step20 -> step30 DAPO run with signed sibling-counterfactual branch credit.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
DATA_DIR="$PROJECT_DIR/data/multihopqa_search_mixed_402020_20260830"
MODEL_DIR=${MODEL_DIR:-"$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"}
# An empty INIT_CHECKPOINT intentionally starts directly from MODEL_DIR.  This
# is useful after converting a differently-sharded FSDP checkpoint to a normal
# Hugging Face model directory (for example, a 3-rank checkpoint reused by a
# 5-rank job).
INIT_CHECKPOINT=${INIT_CHECKPOINT-"$PROJECT_DIR/verl_checkpoints/multihopqa-tree-grpo-sft350-3gpu-budget48-persistent-20260826/actor/global_step_20"}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-branch-credit-step20to30-20260831}
DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24}
DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48}
BRANCH_CREDIT_COEF=${BRANCH_CREDIT_COEF:-1.0}
BRANCH_CREDIT_NORMALIZATION=${BRANCH_CREDIT_NORMALIZATION:-sibling_std}
BRANCH_CREDIT_CLIP=${BRANCH_CREDIT_CLIP:-3.0}
BRANCH_CREDIT_VALUE_MODE=${BRANCH_CREDIT_VALUE_MODE:-reward}
BRANCH_CREDIT_CORRECTNESS_THRESHOLD=${BRANCH_CREDIT_CORRECTNESS_THRESHOLD:-0.8}
SELF_OPD_ENABLED=${SELF_OPD_ENABLED:-false}
SELF_OPD_COEF=${SELF_OPD_COEF:-0.001}
SELF_OPD_TEMPERATURE=${SELF_OPD_TEMPERATURE:-1.0}
SELF_OPD_MIN_DIRECTIONAL_LIFT=${SELF_OPD_MIN_DIRECTIONAL_LIFT:-0.0}
SELF_OPD_THINK_COEF=${SELF_OPD_THINK_COEF:-0.2}
SELF_OPD_QUERY_COEF=${SELF_OPD_QUERY_COEF:-1.0}
SELF_OPD_LOSS=${SELF_OPD_LOSS:-jsd}
SELF_OPD_BETA=${SELF_OPD_BETA:-5.0}
SELF_OPD_DELTA_MIN=${SELF_OPD_DELTA_MIN:-0.0}
SELF_OPD_MAX_EVENTS=${SELF_OPD_MAX_EVENTS:-3}
SELF_OPD_MAX_TEACHER_LENGTH=${SELF_OPD_MAX_TEACHER_LENGTH:-512}
SELF_OPD_MAX_QUERY_TOKENS=${SELF_OPD_MAX_QUERY_TOKENS:-64}
SELF_OPD_MAX_ACTION_TOKENS=${SELF_OPD_MAX_ACTION_TOKENS:-256}
SELF_OPD_MAX_EVIDENCE_TOKENS=${SELF_OPD_MAX_EVIDENCE_TOKENS:-128}
SELF_OPD_MIN_VALUE_GAP=${SELF_OPD_MIN_VALUE_GAP:-0.25}
SELF_OPD_MIN_RAW_ADVANTAGE=${SELF_OPD_MIN_RAW_ADVANTAGE:-0.0}
SELF_OPD_MAX_ADVANTAGE_WEIGHT=${SELF_OPD_MAX_ADVANTAGE_WEIGHT:-3.0}
# Privileged-teacher variant: 'hindsight' (legacy self-teacher context) or
# 'answer' (gold-answer cheat sheet, scored by SELF_OPD_TEACHER_URL when set).
SELF_OPD_TEACHER_CONTEXT=${SELF_OPD_TEACHER_CONTEXT:-hindsight}
SELF_OPD_ANALYZER_URL=${SELF_OPD_ANALYZER_URL:-}
SELF_OPD_EVENT_SELECTION=${SELF_OPD_EVENT_SELECTION:-contrast}
SELF_OPD_TEACHER_URL=${SELF_OPD_TEACHER_URL:-}
RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-20}
RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-false}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-31}
SAVE_FREQ=${SAVE_FREQ:-10}
# HFRollout disables the actor's layer-wise FSDP auto-wrap in this verl
# version.  For Qwen2.5-3B that transiently materializes almost 24 GiB per
# rank before sharding and deterministically OOMs during FSDP init.  vLLM with
# the conservative limits below has already completed multiple rollout chunks;
# the monitor's host-memory gate prevents the separate Ray node-memory failure.
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-vllm}
# Keep enough headroom for the co-located FSDP actor when vLLM wakes from
# sleep if the optional vLLM backend is selected.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
# Adam states must be offloaded before vLLM wakes for the next optimizer step.
# Keeping them on GPU lets step 21 finish but deterministically leaves too
# little room to remap the 5.79-GiB vLLM weights at step 22.  Host-memory safety
# is enforced by the monitor's higher launch threshold instead.
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-true}
# Keep these independently configurable: on 24-GiB 3090s, retaining the
# sharded actor parameters and gradients on GPU uses otherwise idle VRAM and
# avoids exhausting host RAM during updates/checkpoint saves.  Adam remains
# offloaded because its state is substantially larger.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-true}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}
LOGGER=${LOGGER:-"['console','wandb']"}
LOG_POLICY_ENTROPY=${LOG_POLICY_ENTROPY:-false}
ENTROPY_TOKEN_CHUNK_SIZE=${ENTROPY_TOKEN_CHUNK_SIZE:-16}
EXPAND_MODE=${EXPAND_MODE:-random}
EXPAND_OUTCOME_PRIOR_ROOT=${EXPAND_OUTCOME_PRIOR_ROOT:-0.23255813953488372}
EXPAND_OUTCOME_PRIOR_PRE_SEARCH=${EXPAND_OUTCOME_PRIOR_PRE_SEARCH:-0.17880794701986755}
EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH1=${EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH1:-0.20754716981132076}
EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH2=${EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH2:-0.11363636363636363}
EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH3PLUS=${EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH3PLUS:-0.09090909090909091}
EXPAND_OUTCOME_PRIOR_CONCENTRATION_POWER=${EXPAND_OUTCOME_PRIOR_CONCENTRATION_POWER:-2.0}
ACTOR_LR=${ACTOR_LR:-3e-7}
SHUFFLE_TRAIN_DATALOADER=${SHUFFLE_TRAIN_DATALOADER:-true}
ROLLOUT_ACCUMULATION_STEPS=${ROLLOUT_ACCUMULATION_STEPS:-16}
DAPO_DYNAMIC_SAMPLING=${DAPO_DYNAMIC_SAMPLING:-true}
CRITIC_WARMUP=${CRITIC_WARMUP:-0}
SAVE_AT_END=${SAVE_AT_END:-false}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-3}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}

# A GPU monitor may select any safe three-card set. Direct invocations keep
# the proven 1,2,3 default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export WG_BACKEND=ray
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export CC=/home/luwa/.conda/envs/dsclr/bin/gcc
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# vLLM sleep mode uses its CuMem memory pool, which explicitly rejects
# PyTorch expandable segments.  Clear an inherited setting as well so a
# detached service and a direct shell launch behave identically.
unset PYTORCH_CUDA_ALLOC_CONF

for required_path in \
    "$PYTHON_BIN" \
    "$DATA_DIR/train.parquet" \
    "$DATA_DIR/test.parquet" \
    "$MODEL_DIR/config.json"; do
    if [[ ! -e "$required_path" ]]; then
        echo "missing required path: $required_path" >&2
        exit 2
    fi
done

TRAINER_INIT_ARGS=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
    if [[ ! -e "$INIT_CHECKPOINT/model_world_size_${N_GPUS_PER_NODE}_rank_0.pt" ]]; then
        echo "checkpoint is not sharded for ${N_GPUS_PER_NODE} ranks: $INIT_CHECKPOINT" >&2
        exit 2
    fi
    TRAINER_INIT_ARGS+=(
        "+trainer.init_actor_checkpoint=$INIT_CHECKPOINT"
        "+trainer.resume_actor_state=$RESUME_ACTOR_STATE"
    )
fi

mkdir -p "$PROJECT_DIR/verl_log" "$PROJECT_DIR/verl_checkpoints" "$PROJECT_DIR/rollouts/$EXPERIMENT_NAME"
cd "$PROJECT_DIR"
ulimit -n 65535

"$PYTHON_BIN" -m verl.trainer.main_ppo_format_ts \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=120 \
    data.train_batch_size=3 \
    data.val_batch_size=3 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader="$SHUFFLE_TRAIN_DATALOADER" \
    algorithm.adv_estimator=branch_credit \
    algorithm.branch_credit_coef="$BRANCH_CREDIT_COEF" \
    algorithm.branch_credit_normalization="$BRANCH_CREDIT_NORMALIZATION" \
    algorithm.branch_credit_clip="$BRANCH_CREDIT_CLIP" \
    algorithm.branch_credit_no_sibling=zero \
    +algorithm.branch_credit_value_mode="$BRANCH_CREDIT_VALUE_MODE" \
    +algorithm.branch_credit_correctness_threshold="$BRANCH_CREDIT_CORRECTNESS_THRESHOLD" \
    algorithm.self_opd_max_events="$SELF_OPD_MAX_EVENTS" \
    algorithm.self_opd_max_teacher_length="$SELF_OPD_MAX_TEACHER_LENGTH" \
    algorithm.self_opd_max_query_tokens="$SELF_OPD_MAX_QUERY_TOKENS" \
    algorithm.self_opd_max_action_tokens="$SELF_OPD_MAX_ACTION_TOKENS" \
    algorithm.self_opd_max_evidence_tokens="$SELF_OPD_MAX_EVIDENCE_TOKENS" \
    algorithm.self_opd_min_value_gap="$SELF_OPD_MIN_VALUE_GAP" \
    algorithm.self_opd_min_raw_advantage="$SELF_OPD_MIN_RAW_ADVANTAGE" \
    algorithm.self_opd_max_advantage_weight="$SELF_OPD_MAX_ADVANTAGE_WEIGHT" \
    algorithm.self_opd_teacher_context="$SELF_OPD_TEACHER_CONTEXT" \
    algorithm.self_opd_analyzer_url="$SELF_OPD_ANALYZER_URL" \
    algorithm.self_opd_event_selection="$SELF_OPD_EVENT_SELECTION" \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.actor.log_policy_entropy="$LOG_POLICY_ENTROPY" \
    +actor_rollout_ref.actor.entropy_token_chunk_size="$ENTROPY_TOKEN_CHUNK_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.grad_offload="$ACTOR_GRAD_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=2 \
    actor_rollout_ref.rollout.ts_n=2 \
    actor_rollout_ref.rollout.ts_l=1 \
    actor_rollout_ref.rollout.ts_k=3 \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode="$EXPAND_MODE" \
    actor_rollout_ref.rollout.expand_outcome_prior_root="$EXPAND_OUTCOME_PRIOR_ROOT" \
    actor_rollout_ref.rollout.expand_outcome_prior_pre_search="$EXPAND_OUTCOME_PRIOR_PRE_SEARCH" \
    actor_rollout_ref.rollout.expand_outcome_prior_after_search_depth1="$EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH1" \
    actor_rollout_ref.rollout.expand_outcome_prior_after_search_depth2="$EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH2" \
    actor_rollout_ref.rollout.expand_outcome_prior_after_search_depth3plus="$EXPAND_OUTCOME_PRIOR_AFTER_SEARCH_DEPTH3PLUS" \
    actor_rollout_ref.rollout.expand_outcome_prior_concentration_power="$EXPAND_OUTCOME_PRIOR_CONCENTRATION_POWER" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=3 \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.self_opd_enabled="$SELF_OPD_ENABLED" \
    actor_rollout_ref.actor.self_opd_coef="$SELF_OPD_COEF" \
    actor_rollout_ref.actor.self_opd_temperature="$SELF_OPD_TEMPERATURE" \
    actor_rollout_ref.actor.self_opd_min_directional_lift="$SELF_OPD_MIN_DIRECTIONAL_LIFT" \
    actor_rollout_ref.actor.self_opd_think_coef="$SELF_OPD_THINK_COEF" \
    actor_rollout_ref.actor.self_opd_query_coef="$SELF_OPD_QUERY_COEF" \
    actor_rollout_ref.actor.self_opd_loss="$SELF_OPD_LOSS" \
    actor_rollout_ref.actor.self_opd_beta="$SELF_OPD_BETA" \
    actor_rollout_ref.actor.self_opd_delta_min="$SELF_OPD_DELTA_MIN" \
    actor_rollout_ref.actor.self_opd_teacher_url="$SELF_OPD_TEACHER_URL" \
    trainer.logger="$LOGGER" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    +trainer.val_at_end=false \
    +trainer.rollout_accumulation_steps="$ROLLOUT_ACCUMULATION_STEPS" \
    +trainer.dapo_dynamic_sampling="$DAPO_DYNAMIC_SAMPLING" \
    +trainer.dapo_target_effective_prompts="$DAPO_TARGET_EFFECTIVE_PROMPTS" \
    +trainer.dapo_max_chunks="$DAPO_MAX_CHUNKS" \
    +trainer.resume_global_step="$RESUME_GLOBAL_STEP" \
    "${TRAINER_INIT_ARGS[@]}" \
    +trainer.save_at_end="$SAVE_AT_END" \
    trainer.critic_warmup="$CRITIC_WARMUP" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
    trainer.project_name=Tree-GRPO \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME" \
    +trainer.rollout_dump_dir="$PROJECT_DIR/rollouts/$EXPERIMENT_NAME" \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0.1 \
    do_search=true \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    "$@" \
    2>&1 | tee "$PROJECT_DIR/verl_log/$EXPERIMENT_NAME.log"

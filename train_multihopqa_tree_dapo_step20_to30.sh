#!/usr/bin/env bash
# Continue the proven Tree-GRPO actor from step 20 with a fresh, conservative
# optimizer and DAPO dynamic sampling on the mixed multi-hop training set.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
DATA_DIR="$PROJECT_DIR/data/multihopqa_search_mixed_402020_20260830"
MODEL_DIR="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"
INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PROJECT_DIR/verl_checkpoints/multihopqa-tree-grpo-sft350-3gpu-budget48-persistent-20260826/actor/global_step_20"}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihop-tree-dapo-mixed-step20to30-20260830}
DAPO_TARGET_EFFECTIVE_PROMPTS=${DAPO_TARGET_EFFECTIVE_PROMPTS:-24}
DAPO_MAX_CHUNKS=${DAPO_MAX_CHUNKS:-48}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-true}
GRAD_OFFLOAD=${GRAD_OFFLOAD:-true}
LOGGER=${LOGGER:-"['console','wandb']"}
RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-20}
RESUME_ACTOR_STATE=${RESUME_ACTOR_STATE:-false}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-31}
SAVE_FREQ=${SAVE_FREQ:-10}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export WG_BACKEND=ray
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export CC=/home/luwa/.conda/envs/dsclr/bin/gcc
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for required_path in \
    "$PYTHON_BIN" \
    "$DATA_DIR/train.parquet" \
    "$DATA_DIR/test.parquet" \
    "$MODEL_DIR/config.json" \
    "$INIT_CHECKPOINT/model_world_size_3_rank_0.pt"; do
    if [[ ! -e "$required_path" ]]; then
        echo "missing required path: $required_path" >&2
        exit 2
    fi
done

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
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=tree \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.grad_offload="$GRAD_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=2 \
    actor_rollout_ref.rollout.ts_n=2 \
    actor_rollout_ref.rollout.ts_l=1 \
    actor_rollout_ref.rollout.ts_k=3 \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode=random \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=3 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="$LOGGER" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    +trainer.val_at_end=false \
    +trainer.rollout_accumulation_steps=16 \
    +trainer.dapo_dynamic_sampling=true \
    +trainer.dapo_target_effective_prompts="$DAPO_TARGET_EFFECTIVE_PROMPTS" \
    +trainer.dapo_max_chunks="$DAPO_MAX_CHUNKS" \
    +trainer.resume_global_step="$RESUME_GLOBAL_STEP" \
    +trainer.init_actor_checkpoint="$INIT_CHECKPOINT" \
    +trainer.resume_actor_state="$RESUME_ACTOR_STATE" \
    +trainer.save_at_end=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
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
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3 \
    "$@" \
    2>&1 | tee "$PROJECT_DIR/verl_log/$EXPERIMENT_NAME.log"

#!/usr/bin/env bash
# One-step end-to-end GRPO smoke test using the prepared Search-R1 assets.
# The existing retriever is expected at http://127.0.0.1:8000/retrieve.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN=$PROJECT_DIR/.conda/envs/treegrpo/bin/python
DATA_DIR=/home/luwa/Documents/Search-R1/data/nq_search
MODEL_DIR=/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
EXPERIMENT_NAME=singlehopqa-grpo-smoke-qwen2.5-3b

export CUDA_VISIBLE_DEVICES=0,1,2
export VLLM_ATTENTION_BACKEND=XFORMERS
export CC=/home/luwa/.conda/envs/dsclr/bin/gcc
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

mkdir -p "$PROJECT_DIR/verl_log" "$PROJECT_DIR/verl_checkpoints"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo_format \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=6 \
    data.val_data_num=6 \
    data.train_batch_size=6 \
    data.val_batch_size=6 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.max_start_length=1024 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    +trainer.save_at_end=false \
    trainer.test_freq=-1 \
    trainer.project_name=Tree-GRPO-smoke \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME" \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    do_search=true \
    max_turns=2 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3 \
    "$@" \
    2>&1 | tee "$PROJECT_DIR/verl_log/$EXPERIMENT_NAME.log"

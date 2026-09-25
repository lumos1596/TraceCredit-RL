#!/usr/bin/env bash
# Local multi-hop Tree-GRPO launcher derived from the proven three-GPU
# single-hop budget48 setup. GPU 4 is reserved for the E5 retriever.

set -euo pipefail

PROJECT_DIR=/home/luwa/Documents/Tree-GRPO
PYTHON_BIN="$PROJECT_DIR/.conda/envs/treegrpo/bin/python"
DATA_DIR="$PROJECT_DIR/data/multihopqa_search"
MODEL_DIR="$PROJECT_DIR/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-multihopqa-tree-grpo-sft350-3gpu-budget48-20260820}
TRAIN_STEPS=${TRAIN_STEPS:-180}
TRAIN_DATA_NUM=${TRAIN_DATA_NUM:-null}
VAL_DATA_NUM=${VAL_DATA_NUM:-120}
ROLLOUT_ACCUMULATION_STEPS=${ROLLOUT_ACCUMULATION_STEPS:-16}
SAVE_FREQ=${SAVE_FREQ:-2}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_AT_END=${SAVE_AT_END:-true}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
LOGGER=${LOGGER:-"['console','wandb']"}

export CUDA_VISIBLE_DEVICES=1,2,3
export WG_BACKEND=ray
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export CC=/home/luwa/.conda/envs/dsclr/bin/gcc
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

for required_path in "$PYTHON_BIN" "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet" "$MODEL_DIR/config.json"; do
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
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
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
    +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    +trainer.rollout_accumulation_steps="$ROLLOUT_ACCUMULATION_STEPS" \
    +trainer.save_at_end="$SAVE_AT_END" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.project_name=Tree-GRPO \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="$TRAIN_STEPS" \
    trainer.default_local_dir="$PROJECT_DIR/verl_checkpoints/$EXPERIMENT_NAME" \
    +trainer.rollout_dump_dir="$PROJECT_DIR/rollouts/$EXPERIMENT_NAME" \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    do_search=true \
    max_turns=3 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=3 \
    "$@" \
    2>&1 | tee "$PROJECT_DIR/verl_log/$EXPERIMENT_NAME.log"

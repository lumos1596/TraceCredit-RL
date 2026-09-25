#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/luwa/Documents/Tree-GRPO"
PROJECT_PYTHON="${PROJECT_PYTHON:-${PROJECT_ROOT}/.conda/envs/treegrpo/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#VISIBLE_DEVICES[@]}"

LOG_DIR="${PROJECT_ROOT}/verl_log"
CHECKPOINT_DIR="${PROJECT_ROOT}/verl_checkpoints"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

cd "${PROJECT_ROOT}"

"${PROJECT_PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path="${PROJECT_ROOT}/verl/trainer/config" \
  --config-name=sft_trainer \
  data.train_files=data/search_r1_sft/train.parquet \
  data.val_files=data/search_r1_sft/validation.parquet \
  data.prompt_key=messages \
  data.response_key=null \
  data.max_length=2048 \
  data.truncation=right \
  data.train_batch_size=24 \
  data.micro_batch_size=3 \
  data.balance_dp_token=false \
  data.num_workers=0 \
  model.partial_pretrain=models/Qwen2.5-3B-Instruct \
  model.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=true \
  model.lora_rank=0 \
  model.lora_alpha=0 \
  optim.lr=2.0e-6 \
  optim.warmup_steps_ratio=0.03 \
  optim.clip_grad=1.0 \
  trainer.total_epochs=1 \
  trainer.validate_before_training=true \
  trainer.validation_freq=50 \
  trainer.save_freq=50 \
  trainer.log_freq=1 \
  trainer.project_name=Tree-GRPO \
  trainer.experiment_name=singlehopqa-sft-search-r1-qwen2.5-3b-instruct \
  trainer.default_local_dir=verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct \
  trainer.logger='[console,wandb]' \
  trainer.resume_path=null \
  trainer.default_hdfs_dir=null \
  "$@" \
  2>&1 | tee "${LOG_DIR}/singlehopqa-sft-$(date +%Y%m%d-%H%M%S).log"

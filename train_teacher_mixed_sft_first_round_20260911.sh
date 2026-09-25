#!/usr/bin/env bash
set -euo pipefail

# Launch only after the production accepted JSONL files have been built and
# reviewed.  This script contains no generation or RL path; it runs the
# existing full-parameter FSDP SFT trainer.

PROJECT_ROOT="/home/luwa/Documents/Tree-GRPO"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.conda/envs/treegrpo/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-3B-Instruct}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/teacher_sft_mixed_1500_20260911}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/verl_checkpoints/teacher-mixed-sft-qwen2.5-3b-first-round-20260911}"
LOG_DIR="${PROJECT_ROOT}/verl_log"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${RUN_NAME:-teacher-mixed-sft-qwen2.5-3b-first-round-20260911-${RUN_STAMP}}"
RESUME_PATH="${RESUME_PATH:-null}"

VISIBLE_CUDA="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
if [[ "${VISIBLE_CUDA}" != "0,1,2,3,4" ]]; then
  echo "This SFT experiment is pinned to CUDA_VISIBLE_DEVICES=0,1,2,3,4 (five cards; GPU 5 is reserved for the retriever); got ${VISIBLE_CUDA}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${VISIBLE_CUDA}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

[[ -x "${PYTHON_BIN}" ]] || { echo "missing executable Python: ${PYTHON_BIN}" >&2; exit 2; }
# Perform the safety gate through PyTorch and require ample free memory on all
# five visible GPUs. GPU 5 is reserved for the retriever and is not exposed here.
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 5:
    raise SystemExit(f"expected exactly 5 visible CUDA devices, got {torch.cuda.device_count()}")
minimum = 18 * 1024**3
for logical_id in range(5):
    free_bytes, total_bytes = torch.cuda.mem_get_info(logical_id)
    print(f"GPU_CHECK logical={logical_id} free_gib={free_bytes/1024**3:.2f} total_gib={total_bytes/1024**3:.2f}")
    if free_bytes < minimum:
        raise SystemExit(f"GPU {logical_id} has less than 18 GiB free")
PY

for required_path in \
  "${MODEL_PATH}/config.json" \
  "${DATA_DIR}/train.parquet" \
  "${DATA_DIR}/validation.parquet" \
  "${DATA_DIR}/stats.json" \
  "${PROJECT_ROOT}/verl/trainer/config/sft_trainer.yaml" \
  "${PROJECT_ROOT}/verl/trainer/fsdp_sft_trainer.py"; do
  [[ -e "${required_path}" ]] || { echo "missing required path: ${required_path}" >&2; exit 2; }
done

# The builder already performs the expensive tokenizer/SFTDataset checks.  A
# small row-count gate prevents accidentally launching on a partial dataset.
"${PYTHON_BIN}" - "${DATA_DIR}/train.parquet" "${DATA_DIR}/validation.parquet" "${DATA_DIR}/stats.json" <<'PY'
import json
import sys

train_path, validation_path, stats_path = sys.argv[1:]
try:
    import pyarrow.parquet as parquet
except ImportError:
    import pandas as pd
    train_count = len(pd.read_parquet(train_path))
    validation_count = len(pd.read_parquet(validation_path))
else:
    train_count = parquet.read_metadata(train_path).num_rows
    validation_count = parquet.read_metadata(validation_path).num_rows

stats = json.load(open(stats_path, encoding="utf-8"))
assert train_count == 1350, train_count
assert validation_count == 150, validation_count
assert stats["counts"] == {"total": 1500, "train": 1350, "validation": 150}, stats["counts"]
assert stats["tokenization"]["max_length"] == 2560, stats["tokenization"]
assert stats["tokenization"]["max_token_length"] <= 2560, stats["tokenization"]
assert stats["masking"]["truncation"] == "error"
print(
    f"SFT_DATA_CHECK train={train_count} validation={validation_count} "
    f"actual_max={stats['tokenization']['max_token_length']} train_max_length=2560"
)
PY

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=online
export WANDB_PROJECT=Tree-GRPO

LOG_FILE="${LOG_DIR}/teacher-mixed-sft-qwen2.5-3b-first-round-20260911-${RUN_STAMP}.log"
echo "SFT_START=$(date -Is)"
echo "SFT_RUN=${RUN_NAME}"
echo "SFT_LOG=${LOG_FILE}"
echo "SFT_CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "SFT_PROTOCOL=five-card full-parameter-FSDP; visible_cards=0,1,2,3,4; GPU5=reserved_for_retriever; global_batch=25; configured_micro_batch=5; per-rank_micro_batch=1_after_dp5_division; max_length=2560; truncation=error; epochs=3; validation/save_every=50; steps_per_epoch=54; expected_steps=162"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=5 \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path="${PROJECT_ROOT}/verl/trainer/config" \
  --config-name=sft_trainer \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/validation.parquet" \
  data.prompt_key=messages \
  data.response_key=null \
  data.max_length=2560 \
  data.truncation=error \
  data.train_batch_size=25 \
  data.micro_batch_size=5 \
  data.balance_dp_token=false \
  data.num_workers=0 \
  model.partial_pretrain="${MODEL_PATH}" \
  model.attn_implementation=sdpa \
  model.enable_gradient_checkpointing=true \
  model.lora_rank=0 \
  model.lora_alpha=0 \
  optim.lr=2.0e-6 \
  optim.warmup_steps_ratio=0.03 \
  optim.clip_grad=1.0 \
  trainer.total_epochs=3 \
  trainer.total_training_steps=null \
  trainer.validate_before_training=true \
  trainer.validation_freq=50 \
  trainer.save_freq=50 \
  trainer.log_freq=1 \
  trainer.project_name=Tree-GRPO \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.logger='[console,wandb]' \
  trainer.resume_path="${RESUME_PATH}" \
  2>&1 | tee "${LOG_FILE}"

echo "SFT_END=$(date -Is)"

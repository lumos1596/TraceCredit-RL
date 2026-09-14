python ./model_merger.py merge \
    --backend fsdp \
    --hf_model_path /mnt/workspace/common/models/Qwen2.5-3B \
    --local_dir /path/to/TraceCredit-RL/verl_checkpoints/example/actor/global_step_160 \
    --target_dir /path/to/TraceCredit-RL/scripts/merge_ckpt/exported-model

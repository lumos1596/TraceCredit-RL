# TraceCredit-RL

TraceCredit-RL is a reinforcement-learning framework for assigning credit to individual retrieval and reasoning decisions in tree-structured agent trajectories.

Sparse outcome rewards normally indicate whether an entire trajectory succeeded, but they do not identify which intermediate decision helped or hurt. TraceCredit-RL compares sibling branches that share the same prefix and derives signed, branch-level advantages from their outcome differences. These advantages are applied only to the tokens associated with the corresponding decision.

## Highlights

- Tree-structured rollouts for retrieval-augmented language-model agents.
- Signed counterfactual credit assignment between sibling branches.
- Configurable credit normalization, clipping, and fallback behavior.
- Support for reward-based and correctness-based branch values.
- DAPO-style dynamic sampling and rollout accumulation.
- Metrics for branch coverage, positive/negative credit, clipping, and sibling availability.

## Repository layout

```text
search_r1/
  llm_agent/        Tree rollout generation and branch-credit construction
  search/           Local and web retrieval services
verl/
  trainer/          PPO/GRPO training entry points and configuration
  workers/          Distributed rollout, actor, critic, and reward workers
scripts/
  data_process/     Dataset preparation utilities
  merge_ckpt/       Checkpoint conversion utilities
```

Generated datasets, model weights, checkpoints, logs, local environments, historical experiments, and paper files are intentionally excluded from this repository.

## Installation

Python 3.12 and a CUDA-capable PyTorch environment are recommended.

```bash
conda create -n tracecredit-rl python=3.12
conda activate tracecredit-rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install vllm==0.8.5.post1
pip install -e .
pip install flash-attn --no-build-isolation
```

Install the retrieval dependencies in a separate environment if desired:

```bash
conda create -n tracecredit-retriever python=3.10
conda activate tracecredit-retriever
pip install torch transformers datasets pyserini faiss-gpu uvicorn fastapi
```

## Data preparation

Prepare single-hop or multi-hop retrieval datasets with the utilities under `scripts/data_process/`. Paths in the shell templates are placeholders and must be changed for your environment.

```bash
bash scripts/data_process/data_process_singlehop.sh
bash scripts/data_process/data_process_multihop.sh
```

## Start a retrieval service

Set the index, corpus, and retriever paths in `local_retrieval_launch.sh`, then run:

```bash
bash local_retrieval_launch.sh
```

For web retrieval, set the API credential locally in `bing_search_launch.sh` or pass it through your own secret-management mechanism. Never commit credentials.

## Training

The branch-credit estimator is enabled with `algorithm.adv_estimator=branch_credit`:

```bash
python -m verl.trainer.main_ppo_ts \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  actor_rollout_ref.model.path=/path/to/model \
  algorithm.adv_estimator=branch_credit \
  algorithm.branch_credit_coef=1.0 \
  algorithm.branch_credit_normalization=sibling_std \
  algorithm.branch_credit_clip=3.0 \
  algorithm.branch_credit_no_sibling=zero \
  actor_rollout_ref.rollout.tree_search=true \
  retriever.url=http://127.0.0.1:8000/retrieve
```

Important options:

| Option | Purpose |
|---|---|
| `algorithm.branch_credit_coef` | Scales the local branch-credit term. |
| `algorithm.branch_credit_normalization` | Controls sibling-group normalization. |
| `algorithm.branch_credit_clip` | Bounds normalized branch advantages. |
| `algorithm.branch_credit_no_sibling` | Defines fallback behavior when no sibling comparison exists. |
| `algorithm.branch_credit_value_mode` | Selects reward-based or correctness-based branch values. |
| `algorithm.branch_credit_correctness_threshold` | Sets the correctness cutoff when correctness mode is used. |

Batch sizes, rollout budgets, GPU counts, checkpoint paths, and logging backends should be supplied as Hydra overrides for the target environment.

## License

See [LICENSE](LICENSE).

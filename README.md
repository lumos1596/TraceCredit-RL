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

## Current experiment: forkable teacher-prefix tree rescue (2026-10-01)

The active training scheme is **Tree-GRPO (`branch_credit`) + DAPO dynamic
sampling + Self-OPD + gold-free teacher-prefix rescue**. When DAPO produces an
all-wrong group (all `m*k` trajectories wrong, which it normally discards and
gets no gradient from), a teacher supplies the first two
`think + search` actions; the standard tree rollout then runs on top of that
prefix and, if any of the resulting leaves answers correctly, the group
becomes a normal mixed group and re-enters advantage/gradient computation.
All-wrong groups with no rescued leaf stay invalid and are discarded (they are
never force-included).

The seeded tree is **forkable**: the tree root holds teacher hop 1 (shared by
all leaves); the complete 2-hop teacher prefix is one child of the root, while
the ordinary expansion round (`ts_n=2, ts_l=1`) can also expand directly from
the root — i.e. fork *before teacher action 2* — so the student may replace
hop 2 with its own search. Tree parameters, expansion selection,
`sample_leaf(k=3)`, and branch credit are otherwise identical to a normal
rollout (`ts_m=2, ts_n=2, ts_l=1, ts_k=3`); the only added variable is the
teacher prefix. Teacher `think/search` tokens are KD-only (zero on
observations, excluded from PPO via `info_mask`/`loss_mask`), and the teacher
root receives no branch advantage by `TreeNode`'s root-skipping path rule.

### Code map

| Component | File / symbol |
|---|---|
| Rescue injection (forkable tree build, tree rollout, per-leaf KD masks, reward/branch-credit recompute, all-wrong discard) | `search_r1/llm_agent/teacher_rescue.py`: `rescue_all_wrong_prefix`, `_build_seeded_prefix_tree`, `_prefix_tree_rollout`, `_leaf_kd_mask`, `request_prefix_rescue` |
| Legacy closed-loop rescue (unchanged, still the default style) | `search_r1/llm_agent/teacher_rescue.py`: `rescue_all_wrong` |
| Trainer dispatch by style | `verl/trainer/ppo/ray_trainer_ts.py` (`trainer.teacher_rescue_style` = `legacy` / `prefix`) |
| Teacher HTTP service (`/prefix_rescue` endpoint, independent `prefix_cache`) | `scripts/run_deepseek_flash_training_service.py` |
| Launch wiring (env → Hydra overrides) | `train_multihopqa_branch_credit_dapo_step20_to30.sh`: `ROOT`, `DATA_DIR`, `TEACHER_RESCUE_STYLE`, `TEACHER_RESCUE_PREFIX_HOPS`, `TEACHER_RESCUE_URL`, `TEACHER_RESCUE_KD_COEF` |
| Concrete fresh-machine launch (SFT350, step 0 → step 40, no RL resume) | `scripts/run_tracecredit_flash_prefixrescue_sft350_step0_to40.sh` |
| Local continuation launch (step28 → step40, weights-only resume from `global_step_28`) | `scripts/run_tracecredit_flash_prefixrescue_step28_to40.sh` |
| Earlier continuation/ablation launchers | `scripts/run_tracecredit_flash_prefixrescue_sft350_step12_to20.sh`, `scripts/run_tracecredit_flash_semantic_rescue_sft350_step12_to20_resume.sh` |
| Offline RescueRate evaluation (prefix vs fresh-resample arms) | `scripts/evaluation/evaluate_direct_prefix_teacher.py` |

### Fresh-machine launch from SFT350

Use `scripts/run_tracecredit_flash_prefixrescue_sft350_step0_to40.sh` for a new
run. It starts at RL step 0 from the SFT checkpoint and does **not** load any
intermediate RL checkpoint:

- `MODEL_DIR=$ROOT/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350`
- `INIT_CHECKPOINT=""` (empty means start directly from `MODEL_DIR`)
- `RESUME_GLOBAL_STEP=0`, `RESUME_ACTOR_STATE=false`
- `TOTAL_TRAINING_STEPS=41` (run step 0 through step 40), `SAVE_FREQ=1`,
  `TEST_FREQ=4`, `VAL_AT_END=true`
- `TEACHER_RESCUE_ENABLED=true`, `TEACHER_RESCUE_STYLE=prefix`,
  `TEACHER_RESCUE_PREFIX_HOPS=2`, `SELF_OPD_COEF=1.0`

Required files on the new machine:

```text
$ROOT/data/multihopqa_search_mixed_402020_20260830/train.parquet
$ROOT/data/multihopqa_search_mixed_402020_20260830/balanced_test.parquet
$ROOT/verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350/
```

The launcher derives `ROOT` from its own path, so it normally finds the data and
SFT350 checkpoint after cloning/copying the repository into the same root. Set
`ROOT`, `DATA_DIR`, or `MODEL_DIR` explicitly if the layout differs. The
default Python is `$ROOT/.conda/envs/treegrpo/bin/python`; override `PYTHON_BIN`
if the environment is elsewhere.

Start the two local services first. Configure and start the retriever as
described in [Start a retrieval service](#start-a-retrieval-service), using
port `8002`. Then, from the repository root with the training environment
activated, start one Flash service on `8130`; this single process serves
`/generate` (Self-OPD analyzer), `/semantic_judge`, and `/prefix_rescue`:

```bash
cd /path/to/Tree-GRPO
export DEEPSEEK_API_KEY=...
python scripts/run_deepseek_flash_training_service.py \
  --host 127.0.0.1 \
  --port 8130 \
  --retriever-url http://127.0.0.1:8002/retrieve
```

Launch training in another shell:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_tracecredit_flash_prefixrescue_sft350_step0_to40.sh
```

If services run on another host, override both URLs:

```bash
FLASH_SERVICE_URL=http://<service-host>:8130 \
RETRIEVER_URL=http://<service-host>:8002/retrieve \
CUDA_VISIBLE_DEVICES=0,1,2 \
bash scripts/run_tracecredit_flash_prefixrescue_sft350_step0_to40.sh
```

`scripts/run_tracecredit_flash_prefixrescue_step28_to40.sh` remains only the
machine-specific continuation script for the existing step28 run. The prefix
style is opt-in; omitting `TEACHER_RESCUE_STYLE` keeps the legacy closed-loop
rescue path unchanged.

### Known edge case

If teacher observations + student continuation exceed `max_start_length`, the
existing keep-tail trajectory truncation can drop the hop-1 head; such groups
fail the hop-1 prefix check and are safely skipped (treated like any all-wrong
group). Observed rate ~7% of eligible groups.

## 7B experiment handoff: 4 x A40 40GB

The recommended next experiment scales the current multi-hop Tree-GRPO setup
from Qwen2.5-3B to a Qwen2.5-7B model on four A40 40GB GPUs. The objective is
to change model scale while retaining the tree rollout, DAPO sampling, reward,
and branch-credit definitions used by the current experiment.

### Experiment definition

- Model: a **7B SFT checkpoint** trained for the same
  `think -> search -> answer` protocol. A 3B checkpoint cannot be loaded into a
  7B model.
- Algorithm: Tree-GRPO with DAPO-style dynamic sampling and signed
  sibling-counterfactual branch credit.
- Branch value: correctness-only, using a correctness threshold of `0.8`.
- Main comparison: `branch_credit_coef=1.0`; the planned ablation uses `0.5`.
- Tree parameters: `M=2`, `N=2`, `L=1`, `K=3`.
- Context limits: prompt `4096`, generated response `500`, initial context
  `2048`, retrieved observation `500`, and at most three search turns.
- Training precision/placement: four-way FSDP, gradient checkpointing, and
  parameter/gradient/optimizer/reference-model CPU offload.
- Rollout backend: vLLM with tensor parallelism `1` initially.

The implementation entry point is
`train_multihopqa_branch_credit_dapo_step20_to30.sh`. Its defaults describe the
existing 3B run, so the 7B launch must explicitly override the model,
checkpoint, GPU count, and memory-sensitive rollout settings.

### Server prerequisites

Before launching, verify all of the following:

1. Four A40 GPUs are visible and idle, with approximately 40GB free on each.
2. The server has at least 128GB system RAM; 192GB or more is preferred because
   optimizer and model states are offloaded to the host.
3. A compatible CUDA/PyTorch/vLLM environment is installed and the retrieval
   service answers `http://127.0.0.1:8000/retrieve`.
4. The multi-hop train/test parquet files are available.
5. `MODEL_DIR` points to a normal Hugging Face-format **7B SFT model** containing
   at least `config.json`, tokenizer files, and model weights.
6. W&B credentials are configured if `LOGGER="['console','wandb']"` is used.

Do not set `INIT_CHECKPOINT` to any existing 3B checkpoint. Existing FSDP
checkpoints are also tied to their saved world size. A three-rank checkpoint
must be merged into a Hugging Face model and reshared before a four-rank job can
resume from it.

### Preflight

Run these checks and record their output in the experiment log:

```bash
nvidia-smi
curl --fail http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Tree-GRPO"],"topk":1,"return_scores":true}'
test -f /path/to/qwen2.5-7b-sft/config.json
test -f data/multihopqa_search_mixed_402020_20260830/train.parquet
```

Also update the machine-specific `PROJECT_DIR`, `PYTHON_BIN`, `DATA_DIR`, and
compiler path near the top of the launch script when the A40 server uses a
different directory layout.

### One-step smoke test

Start conservatively. The empty `INIT_CHECKPOINT` is intentional: it starts
from `MODEL_DIR` instead of trying to read the incompatible 3B RL checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DIR=/path/to/qwen2.5-7b-sft \
INIT_CHECKPOINT= \
EXPERIMENT_NAME=multihop-tree-dapo-credit-7b-a40-smoke \
N_GPUS_PER_NODE=4 \
RESUME_GLOBAL_STEP=0 \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=1 \
SAVE_AT_END=true \
BRANCH_CREDIT_VALUE_MODE=correctness \
BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8 \
BRANCH_CREDIT_COEF=1.0 \
GPU_MEMORY_UTILIZATION=0.40 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
ROLLOUT_MAX_NUM_SEQS=32 \
bash train_multihopqa_branch_credit_dapo_step20_to30.sh \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1
```

The smoke test passes only if it completes one optimizer update and checkpoint
save with no OOM, NaN/Inf, Ray worker death, shape mismatch, or retrieval
failure. Confirm that all four ranks are represented in the saved checkpoint.
Inspect the log for nonzero effective DAPO groups and branch-credit coverage;
finishing a rollout without an optimizer update is not sufficient.

### Formal run

After the smoke test passes, use a new experiment name and the intended step
budget. Keep the sampling and optimization settings unchanged so that the 7B
result remains comparable with the 3B experiment.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DIR=/path/to/qwen2.5-7b-sft \
INIT_CHECKPOINT= \
EXPERIMENT_NAME=multihop-tree-dapo-credit-7b-a40-coef1 \
N_GPUS_PER_NODE=4 \
RESUME_GLOBAL_STEP=0 \
TOTAL_TRAINING_STEPS=40 \
SAVE_FREQ=5 \
SAVE_AT_END=true \
BRANCH_CREDIT_VALUE_MODE=correctness \
BRANCH_CREDIT_CORRECTNESS_THRESHOLD=0.8 \
BRANCH_CREDIT_COEF=1.0 \
GPU_MEMORY_UTILIZATION=0.40 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 \
ROLLOUT_MAX_NUM_SEQS=32 \
bash train_multihopqa_branch_credit_dapo_step20_to30.sh \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1
```

Launch long jobs through the server's persistent job mechanism (`systemd`, a
scheduler, or an existing detached monitor), not an interactive SSH process.
The experiment must continue after the terminal and Codex session disconnect.

If the run is stable, increase vLLM memory utilization gradually from `0.40`
to `0.45` and then `0.50`; change only one memory setting at a time. If it
OOMs, first reduce `ROLLOUT_MAX_NUM_SEQS` to `16`, followed by
`ROLLOUT_MAX_NUM_BATCHED_TOKENS` to `2048`. Do not shorten the 4096/500
prompt-response limits merely to make the model fit, because that changes the
task and may truncate successful trajectories.

### Required monitoring and evaluation

For each optimizer step, retain at least:

- total reward and correctness reward;
- intra-tree, inter-tree, and final combined advantages;
- branch-credit mean/std, positive/negative proportions, coverage, and the
  fraction of comparable branches with different correctness outcomes;
- DAPO effective prompt/group ratio and sampled chunk count;
- KL divergence, policy entropy, gradient norm, token lengths, truncated
  trajectory ratio, wall-clock step time, and peak GPU/host memory;
- rollout dumps and retrieval-call counts.

Evaluate the untrained 7B base model, the 7B SFT starting point, and the RL
checkpoints with the same full test set and the same exact-match definition.
The minimum comparison table should contain the 3B SFT/Tree-GRPO/credit runs
and the corresponding 7B SFT and 7B credit run. Do not substitute pass@k for
the paper-style EM metric.

### Optional coefficient ablation

Only after the coefficient-1.0 run is healthy, repeat the matched experiment
with `BRANCH_CREDIT_COEF=0.5`. Keep the initial 7B SFT checkpoint, data order,
training-step budget, DAPO target, tree parameters, temperature, context limits,
and evaluation set identical. This is the valid test of whether local credit is
overweighted; changing model scale or rollout budget at the same time would
confound the result.

## Cross-machine handoff (RL+OPD checkpoints)

Large artifacts are hosted on Hugging Face:
[`lumos2548/treegrpo-rl-opd-handoff`](https://huggingface.co/datasets/lumos2548/treegrpo-rl-opd-handoff)

| Artifact | Path in the HF repo | Size |
|---|---|---|
| Student SFT init | `singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350/` | ~36 GB |
| RL+OPD latest checkpoint | `tracecredit-nodeskill-sampled-nll-opd-formal-entropy-chunked-retry-20260925/actor/global_step_9/` | ~36 GB |
| Training/eval data | `data/multihopqa_search_mixed_402020_20260830/` | ~1 GB |

To resume on a new machine (scripts hard-code `/home/luwa/Documents/Tree-GRPO`,
so use the same path or adjust `PROJECT_DIR`):

```bash
# 1. Code
git clone https://github.com/lumos1596/TraceCredit-RL.git /home/luwa/Documents/Tree-GRPO
cd /home/luwa/Documents/Tree-GRPO

# 2. Checkpoints + data
hf download lumos2548/treegrpo-rl-opd-handoff --repo-type dataset --local-dir .

# 3. Restore checkpoint layout (HF paths are flat, local layout is nested)
mkdir -p verl_checkpoints
mv singlehopqa-sft-search-r1-qwen2.5-3b-instruct verl_checkpoints/
mv tracecredit-nodeskill-sampled-nll-opd-formal-entropy-chunked-retry-20260925 verl_checkpoints/

# 4. Services run on the service host (this machine, LAN IP 172.18.167.248)
#    - Retriever (e5 + wiki-18 FAISS index):  http://172.18.167.248:8000/retrieve
#    - Node-skill analyzer (3B, GPU 0):       http://172.18.167.248:8127
#    On a training-only machine, export before launching:
#      export RETRIEVER_URL=http://172.18.167.248:8000/retrieve
#      export SELF_OPD_ANALYZER_URL=http://172.18.167.248:8127
#    Off-LAN, tunnel instead:
#      ssh -L 8000:127.0.0.1:8000 -L 8127:127.0.0.1:8127 <service-host>
#    Both services auto-start with the run scripts; defaults point to 127.0.0.1,
#    which is correct on the service host itself.
#    Teacher (dense answer-OPD runs only): Qwen/Qwen2.5-7B-Instruct from official HF
#    Env: rebuild .conda/envs/treegrpo from pyproject.toml / requirements.txt

# 5. Resume (auto-detects the latest complete checkpoint)
bash scripts/run_tracecredit_nodeskill_opd_formal_entropy.sh
```

## License

See [LICENSE](LICENSE).

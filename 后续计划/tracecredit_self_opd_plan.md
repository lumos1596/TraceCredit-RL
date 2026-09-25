# Tree-GRPO NodeSkill Self-OPD 执行计划

更新时间：2026-09-23

## 当前可用模型

NodeSkill 教师已经完成训练和 held-out 验证，可作为下一阶段 OPD 的 frozen analyzer：

```text
data/tree_seed_node_skill_generation_sft_3b_v4_20260923/merged/
```

RL actor 起点：

```text
verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350
```

RL+OPD 从该 SFT checkpoint 直接开始，训练 step 从 0 重新计数，并重新初始化 optimizer 和
scheduler。NodeSkill 教师使用上面的 merged checkpoint 并在训练中冻结。OPD 从第一个 RL
update 启用；若当前 tree 没有合格 event，则该 update 的 `L_OPD=0`。

该模型输入当前节点的 question、历史、observation、branch 对比信息和局部 evidence，输出四字段 JSON：

```json
{
  "failure_type": "...",
  "missing_relation": "...",
  "next_operation": "...",
  "stop_condition": "..."
}
```

skill 表达抽象的实体/关系/证据缺口和下一步验证策略，作为学生已采样动作的辅助上下文。

## 本阶段实验范围

本阶段只运行一组主实验：`global_step_350 SFT actor → RL + NodeSkill sampled-NLL OPD`。
`plain` 与 `skill` 只是在同一个 update 内计算 OPD 权重所需的两个上下文，不构成独立实验。
RL-only、shuffled-skill、JSD 和 step-matched 对照统一放到后续阶段。

## 已完成验证

训练集：

```text
data/tree_seed_node_skill_generation_sft_v4_20260923/train.jsonl
data/tree_seed_node_skill_generation_sft_v4_20260923/validation.jsonl
```

- 603 条 train、123 条 validation；按 question UID 隔离。
- validation loss：`2.325 -> 0.921`。
- held-out skill JSON 通过率：`120/123 = 97.6%`。
- 严格答案无关过滤后：`118/123 = 95.9%`。
- 过滤规则：四字段 schema、非空、长度、answer alias、数字事实、原始 query、证据标题和特权元信息。

生成结果：

```text
verl_log/generated_skills_node_sft_v4_heldout_v1.jsonl
verl_log/generated_skills_node_sft_v4_consumer_heldout_strict.jsonl
```

固定学生动作的 real/plain/shuffled 审计：

```text
verl_log/node_skill_likelihood_local_generated_v4_actor_base_heldout.json
```

search high-minus-low margin：

- real 相对 plain：`+0.073`，95% CI 约为 `[+0.032, +0.115]`；
- real 相对 shuffled：`+0.081`，95% CI 约为 `[+0.041, +0.121]`。

这说明生成的 skill 能改变学生对正确 search 动作的偏好，具备进入 OPD smoke 的条件。

## 训练数据和脚本

- 构造数据：`scripts/data_process/export_node_skill_generation_sft.py`
- SFT：`scripts/train_analyzer_sft_lora.py`
- 在线生成和校验：`scripts/evaluate_node_skill_generation.py`
- 生成结果转审计格式：`scripts/prepare_generated_skill_audit.py`
- 固定动作 likelihood：`scripts/audit_node_skill_likelihood.py`
- sampled-NLL 定义和实现位置：`search_r1/llm_agent/self_opd.py`、`verl/workers/actor/dp_actor.py`

## 下一步：接入在线 OPD

### 1. 在线 event 构造

每轮使用当前策略产生 on-policy tree，并选取：

- positive branch advantage；
- `value_gap >= 0.1`；
- 完整的学生 `<think>...</think><search>...</search>` 动作。

使用 frozen NodeSkill 教师为节点生成 skill，并经过统一 validator。通过的 skill 用于对学生
已采样动作进行辅助重打分。

### 2. 同 token 重打分

对同一条学生已采样 token 计算：

```text
logp_plain = log πθ(action | ordinary_context)
logp_skill = log πθ(action | ordinary_context + skill)
delta      = stop_gradient(logp_skill - logp_plain)
```

skill 只用于训练时的辅助重打分，部署时学生仍使用 ordinary context。

### 3. sampled-NLL OPD

```text
w_i = I[delta_i > delta_min]
      * max(branch_advantage, 0)
      * sigmoid(beta_opd * delta_i)

L = L_TraceCredit + lambda_opd * L_OPD
```

OPD 只作用于学生实际采样的 think/search token；think/search 分开归一化。建议初始配置：

```text
self_opd_loss = sampled_nll
lambda_opd    = 0.001
beta_opd      = 5.0
delta_min     = 0.0
think_coef    = 0.2
query_coef    = 1.0
value_gap_min = 0.1
```

无有效 event 时，`L_OPD = 0`。

### 4. Smoke 和连续训练

先运行 step1–4 smoke，记录：

- plain/skill 的 think/search log-prob 和 margin；
- skill valid rate、answer-leak rate、有效 token 数；
- OPD loss、梯度范数和 TraceCredit loss；
- natural-120 EM、supporting-title hit/recall、重复/无效 query 率。

smoke 正常后，沿同一次运行继续训练至 step20：

```text
global_step_350 SFT actor
→ TraceCredit-RL + NodeSkill sampled-NLL OPD
→ RL step1–20
```

保存每个评测点的 checkpoint、event JSONL、skill 生成结果和逐 step 指标；本阶段只形成这一组主实验结果。

## 验收标准

从 smoke 继续训练前，满足：

- skill valid rate ≥ 0.90；
- answer-leak rate = 0；
- skill context 下的 high-value search margin > plain；
- high-value search 的 `delta` 为正；
- smoke 中 OPD loss 非零且 backward 稳定。

本轮先报告 RL+OPD 的 natural-120 EM、supporting-title recall、有效 skill/event 数、OPD gate
通过率和训练稳定性；消融与对照实验后续补充。

## 执行记录（2026-09-23，本轮）

- 已核验 actor 起点与 frozen NodeSkill checkpoint 文件完整存在。
- 已在 `dp_actor.py` 新增 `self_opd_loss=sampled_nll`：逐 token 计算 detached
  `delta`，执行 `delta > delta_min` 硬门控、正 branch advantage、
  `sigmoid(beta_opd * delta)` 和 think/search 独立归一化；保留原 JSD 路径。
- 已增加 `self_opd_beta`、`self_opd_delta_min` 配置和训练脚本透传。
- `tests/test_self_opd.py` 共 11 项通过，覆盖负 delta、非正 advantage、空 event、
  skill/delta/advantage stop-gradient 以及原 JSD 回归。
- 已新增 `scripts/run_node_skill_server.py`，并将 `teacher_context=node_skill` 接入在线
  rollout builder。builder 向 frozen analyzer 发送当前可见状态、正负 sibling 动作和已脱敏
  evidence；返回 JSON 会在 rollout 端再次经过 schema、长度、answer、query、数字事实和
  privileged-meta 校验，通过后才加入 `<analyzer_skill>` context。
- 已在 GPU 3 实际加载 frozen checkpoint，`/health` 返回正常，并使用 held-out validation
  prompt 完成一次真实四字段 JSON 生成。服务地址为 `http://127.0.0.1:8127`。
- NodeSkill 接入后的 Self-OPD 测试共 13 项通过；包括 skill context 中目标 token 对齐。
- 尚未启动 RL smoke：GPU 0/1/2 分别已有约 17.7/17.7/17.3 GiB 占用，GPU 4--7
  约 22--23 GiB 占用；GPU 3 正运行 analyzer（约 6.6 GiB）。没有三张满足既有训练配置
  显存余量的卡，不能在不干扰其他任务的情况下安全启动。
- 资源检查时仅 GPU 3 基本空闲，其余 7 张卡占用约 17--23 GiB；现有训练入口默认需要
  3 张 24-GiB GPU，因此当前也不满足已验证的训练资源条件。

资源释放后的 smoke 配置应使用：

```bash
SELF_OPD_ENABLED=true \
SELF_OPD_LOSS=sampled_nll \
SELF_OPD_COEF=0.001 \
SELF_OPD_BETA=5.0 \
SELF_OPD_DELTA_MIN=0.0 \
SELF_OPD_MIN_VALUE_GAP=0.1 \
SELF_OPD_TEACHER_CONTEXT=node_skill \
SELF_OPD_ANALYZER_URL=http://127.0.0.1:8127 \
INIT_CHECKPOINT= RESUME_GLOBAL_STEP=0 RESUME_ACTOR_STATE=false \
TOTAL_TRAINING_STEPS=5 SAVE_FREQ=1 \
bash train_multihopqa_branch_credit_dapo_step20_to30.sh
```

### 自动监控状态

监控已于 `2026-09-24 19:32 +08:00` 按当前资源重新启动：

- 固定训练卡：GPU `0,1,2`；GPU 3 专用于 frozen analyzer；GPU 5 提供检索服务。
- 启动条件：三张训练卡 `memory.used <= 750 MiB`、利用率 `<= 5%`，连续六次
  每秒采样通过（首尾至少五秒），同时主机 `MemAvailable >= 155000 MiB`。
- 最终启动前再次原子式复查 GPU、主存、analyzer `/health`、retriever 实际查询和所有
  checkpoint/data 文件；使用文件锁避免重复启动。
- smoke 使用 vLLM 显存比例 `0.30`、`max_num_batched_tokens=3072`、`max_num_seqs=32`、
  PPO micro-batch 1、12 个有效 prompt、最多 24 rollout chunks。actor 参数和梯度保留在
  GPU，optimizer 与 reference 参数 offload，减少主存与显存峰值冲突。
- 状态：`verl_log/tracecredit_nodeskill_opd_gpu_monitor.state`
- 日志：`verl_log/tracecredit_nodeskill_opd_gpu_monitor.log`
- 启动器：`scripts/start_tracecredit_nodeskill_opd_monitor.sh`
- smoke 启动器已改为自动两阶段守护：5-step smoke 结束后必须同时满足无 OOM/traceback/NaN、
  `self_opd/loss > 0`、有效 weighted OPD token > 0、至少一个 gated/search token，且
  `global_step_5` 的 model/optimizer/extra-state 三卡 checkpoint 完整；通过后自动以
  `RESUME_GLOBAL_STEP=5`、`RESUME_ACTOR_STATE=true` 继续 formal run（目标 step 20），并使用
  独立的 `-formal` 实验目录保留两阶段日志和 checkpoint。

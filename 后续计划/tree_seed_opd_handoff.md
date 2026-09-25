# Tree-SEED OPD 实施指示文档

> 用途：将本文件直接交给新的 Codex 对话，作为下一阶段实现、审计和实验启动的唯一任务说明。
>
> 日期：2026-09-23

## 1. 任务目标

在现有 Tree-GRPO / TraceCredit 框架中实现一种受 SEED 启发的自演化 OPD：当前策略生成
on-policy 搜索轨迹，完成轨迹后由同步 analyzer 提取不包含答案的抽象 hindsight skill；
随后在普通上下文和 skill 增强上下文中，对学生已经采样的同一组 `<think>` / `<search>`
token 重新打分，并将 skill 引起的正向概率变化作为稠密辅助监督，与 TraceCredit-RL
联合优化。

本方案不要求教师生成新的 correction、rationale 或 query。已有实验已经证明，7B 教师
生成的新 query 并不稳定优于学生，因此不要恢复该路线。

## 2. 必须保持的 OPD 定义

训练数据必须来自当前或最近同步策略的 on-policy rollout。对于学生实际采样的 token
`a_i`，计算：

```text
l_plain(i) = log pi_theta(a_i | h_i)
l_skill(i) = log pi_theta(a_i | h_i, skill_tau)
delta(i)   = stop_gradient(l_skill(i) - l_plain(i))
```

Teacher 与 Student 使用同一策略 checkpoint，但上下文不同；skill branch 必须
stop-gradient。部署和评测时只使用普通策略，不提供 skill、analyzer 或额外检索模块。

离线阶段只允许用于初始化 trajectory/tree → skill 的分析能力。真正的行为监督必须来自
RL 阶段当前策略生成的轨迹和同步更新的 analyzer，因此该方法仍属于 on-policy
distillation，而不是离线 response distillation。

## 3. 与原版 SEED 的关键差异

不要原样复现 SEED 的纯 sigmoid gate。Tree-GRPO 已经提供 sibling branch value 和
TraceCredit，应利用它避免强化失败动作。

推荐权重：

```text
w_i = I[delta_i > delta_min]
      * max(A_branch, 0)
      * sigmoid(beta_opd * delta_i)
      * span_weight_i

L_OPD = -sum_i w_i * log pi_theta(a_i | h_i) / sum_i w_i
L     = L_TraceCredit + lambda_opd * L_OPD
```

其中：

- 只监督同父 sibling 中 positive-credit 的高价值 branch；
- `delta_i <= delta_min` 的 token 权重必须为零，不能仅用一个始终大于零的 sigmoid；
- `<think>` 与 `<search>` 分别按 span 长度归一化；
- 初始 `think_coef=0.2`、`query_coef=1.0`；
- final answer、environment observation、控制标签和 padding token 不参与 OPD；
- 无合格 token/event 时，OPD loss 必须安全为零，训练退化为纯 TraceCredit-RL；
- 保留现有 JSD 实现作为消融，不要删除。新增 `sampled_nll` loss 模式。

## 4. Analyzer 初始化

在 RL 前增加小规模 Analyzer SFT，使 SFT checkpoint 学会从完整 trajectory 或 tree
生成简短的、可执行的 hindsight skill。训练目标不是标准答案或目标 query，而是错误诊断
与可迁移搜索策略。

建议输出格式：

```xml
<diagnosis>当前搜索在实体、关系、证据链或验证步骤上的问题</diagnosis>
<strategy>下一步应验证哪类关系、消除哪个歧义或补齐哪一跳</strategy>
```

成功轨迹应总结有效 workflow；失败轨迹应总结 failure-avoidance rule。优先使用完整轨迹
以及匿名化的 sibling outcome/value 来生成 skill，但不得显示目标 query 或答案。

初始 skill 标注可以由较强外部模型离线生成，但必须经过格式、防泄漏和行为有效性过滤。
Analyzer SFT 后，同一个 checkpoint 同时初始化 actor 和 analyzer。进入 RL 后，每轮或每个
固定同步周期都使用最新冻结 policy snapshot 生成新 skill，不再依赖静态 skill bank。

## 5. 严格防答案泄漏约束

Analyzer 在生成 skill 时可以读取完成后的轨迹和 outcome，但输出必须满足：

- 不包含 gold answer、answer alias 或“答案是 X”一类断言；
- 不包含 gold supporting title，除非该 title 在当前决策状态之前已经由学生检索可见；
- 不包含最优 sibling query、目标 query 或其近似复述；
- 不包含能够直接反推出答案的完整事实句；
- 不指示模型根据已知答案反向构造 query；
- 只允许输出实体类型、关系类型、歧义类别、证据缺口和搜索/验证策略。

继续复用并加强已有 answer redaction。gold answer 只能供 validator 做脱敏检测，绝不能拼入
teacher/analyzer 输入中的指令或输出目标。任何格式非法或疑似泄漏的 skill 整条作废，
对应 OPD 权重为零；统计时仍计入分母，避免 survivor bias。

## 6. 训练前必须完成的 preflight

不要以“教师能否生成更好的 query”作为资格门。新的固定-state 审计应直接比较同一批
on-policy token 在以下三种上下文中的 log-prob：

1. `plain`：无 skill；
2. `real`：由本轨迹/本 tree 生成的真实 skill；
3. `shuffled`：来自其他问题、长度和格式尽量匹配的 skill。

至少报告：

- 高价值 branch 的 mean `delta(real)`；
- 低价值 sibling 的 mean `delta(real)`；
- `real - shuffled` 的 token lift 和 event lift；
- real skill 对“高价值 query − 低价值 query”likelihood margin 的提升；
- think/search 分项统计；
- supporting-title 命中与未命中 branch 的分组统计；
- skill 格式有效率、答案泄漏率、gate 通过率和有效 token 数；
- event-level bootstrap 95% CI。

建议训练准入门槛：

```text
answer_leak_rate == 0
skill_valid_rate >= 0.90
real high-value delta > 0
real high-value delta - shuffled high-value delta > 0
real 对 high-vs-low margin 的提升 > 0
至少后两项的 bootstrap CI 下界不低于 0，且点估计有实际幅度
```

如果 real skill 只对所有 query 都产生相似的通用增益，或不优于 shuffled skill，不得启动
OPD 训练。应先修复 Analyzer SFT 数据/提示，而不是增大 `lambda_opd`。

## 7. 实现范围

优先检查和扩展以下现有代码：

- `search_r1/llm_agent/self_opd.py`
  - 增加 trajectory/tree → abstract skill 的上下文构造；
  - 保持目标 action/query 文本从 privileged context 中隐藏；
  - 保留 think/search token 对齐和 mask。
- `verl/workers/actor/dp_actor.py`
  - 新增 `self_opd_loss=sampled_nll`；
  - 实现 token-level `delta`、硬正向门控、branch advantage 权重和 stop-gradient；
  - 保留当前 `jsd` 路径作为消融；
  - 保证 FSDP 各 rank forward 次数一致，全局 denominator 正确 all-reduce。
- `verl/trainer/config/ppo_trainer.yaml`
  - 增加 loss mode、`beta_opd`、`delta_min`、skill 同步周期等配置及校验。
- 新增 Analyzer SFT 数据构造、泄漏 validator、preflight audit、测试和运行脚本。

必须先阅读现有实现和测试，保护工作区中的用户修改，不要重写无关训练框架。

## 8. 验证要求

实现后至少验证：

- token 对齐：plain/skill 分支评分的是完全相同的 on-policy token；
- stop-gradient：skill branch、delta、gate 和 advantage 不接收梯度；
- 负 delta、非正 branch advantage、泄漏 skill 的 OPD 权重严格为零；
- think/search 独立归一化正确；
- 空 event、全无效 event、多 rank 局部事件数不一致时不死锁、不产生 NaN；
- JSD 旧路径仍可运行；
- sampled-NLL loss 的手算单元测试一致；
- preflight 的 shuffled control、bootstrap CI 和 survivor-bias 处理正确；
- smoke test 中 OPD loss 非零且 backward 成功，但未通过资格门前不得开启正式训练。

## 9. 首轮实验矩阵

只有 preflight 通过后才运行：

```text
A. TraceCredit-RL only
B. TraceCredit + Tree-SEED sampled-NLL
C. TraceCredit + shuffled-skill sampled-NLL
D. TraceCredit + 原 JSD OPD（消融）
```

所有实验必须从同一个 Analyzer-SFT checkpoint、相同数据顺序、rollout budget 和评测点
开始。建议先做短程 smoke/step1–4，再做 step1–20。

`lambda_opd` 建议测试 `0.001 / 0.003 / 0.01`，但第一轮正式对照只固定一个通过 smoke
的保守值，不能根据验证集结果事后挑选。`beta_opd` 可先采用 SEED 的 5.0，另记录硬门控
后的实际权重分布。

评测除 natural-120 EM 外，还要记录：

- supporting-title hit/recall；
- 每步 search 数和无效/重复 query 率；
- real/shuffled gate 通过率；
- high/low branch delta 分布；
- OPD 有效 token 占比与梯度范数；
- RL-only 与 OPD 的 step-matched 曲线，而不只比较最终 checkpoint。

## 10. 已知失败结果，不要重复

现有 7B generated-correction preflight（25 events）结果：

```text
hit@3: student 44%, real 36%, no-cheat 24%, shuffled 40%
real - student = -8pp
real - shuffled = -4pp
```

因此：

- 不要让 7B/3B teacher 生成新 think/search 作为蒸馏 target；
- 不要把 gold answer 放入 prompt 后做 backward query generation；
- 不要因 real 优于 no-cheat 就忽略它仍差于 student/shuffled；
- 不要在 preflight 失败时自动启动训练；
- 不要删除或覆盖已有审计结果。

相关现有文件：

- `后续计划/tracecredit_self_opd_plan.md`
- `scripts/audit_teacher_correction_retrieval.py`
- `scripts/run_teacher_correction_preflight.sh`
- `verl_log/teacher_correction_retrieval_7b_v2.json`

## 11. 新对话的执行顺序

1. 阅读本文件、原计划、现有 Self-OPD 代码、配置和测试。
2. 给出简短实施计划并确认当前 GPU/进程状态；不要立即启动训练。
3. 先实现 Analyzer skill schema、泄漏 validator 和固定-state preflight。
4. 利用 deepseek 构造小规模 Analyzer SFT 数据并做人工抽查。
5. 完成 Analyzer SFT 后运行 real/plain/shuffled likelihood audit。
6. 只有资格门通过，才实现或启用 sampled-NLL 正式训练路径。
7. 单元测试与短程 smoke 通过后，再启动 A/B/C/D 对照。
8. 若 GPU 暂时不可用，只排队已经通过全部门控的实验；不得绕过门控。
9. 将命令、日志路径、checkpoint、指标和结论持续回写到计划文档。

## 12. 完成标准

只有同时满足以下条件，才算完成本阶段：

- Analyzer 能稳定生成无答案泄漏、格式正确的抽象 skill；
- real skill 在固定-state 审计中显著优于 plain/shuffled，并更偏向高价值 branch；
- sampled-NLL OPD 实现通过测试和 smoke；
- RL-only、real-skill、shuffled-skill 至少形成可比较的 step-matched 曲线；
- 结论能够区分“skill 无信息”“loss/gate 不合理”和“OPD 确有增益”三种情况。

若资格门失败，应明确记录失败原因并停止，不得以扩大训练规模代替方法验证。

## 13. Analyzer SFT 数据构造状态（2026-09-23）

已从 step-21 的 396 条已脱敏 TraceCredit rollout prompt 构造并审计首版数据。DeepSeek
只见到去掉 `<answer>` 内容后的轨迹与问题；答案别名仅在本地用于 redaction/validator，不能
进入训练 prompt 或 response。

- 候选：396 条（66 个问题）；通过：324 条（64 个问题）；拒绝并隔离：72 条。
- 72 条中包含 6 条实际 `answer_leak`；其余主要为 answer-claim、标题/query 复述或
  redaction artifact。**训练只能使用 accepted/export，不得使用 candidates、rejected 或 raw
  API response。**
- 对 324 条 accepted 逐条独立复验：0 个 schema/泄漏违规。
- 已按 `uid`（问题）而非 rollout 切分，保证同题轨迹不跨集合：train 293 条/58 问题，validation
  31 条/6 问题；split 中 uid overlap=0。

可直接供 Analyzer SFT 使用的文件：

- `data/tree_seed_analyzer_sft_deepseek_full/export/train.jsonl`
- `data/tree_seed_analyzer_sft_deepseek_full/export/validation.jsonl`
- 对应 `train.parquet`、`validation.parquet` 及 `manifest.json`

生成与导出脚本：

- `scripts/data_process/build_tree_seed_analyzer_sft.py`
- `scripts/data_process/export_tree_seed_analyzer_sft.py`

## 14. 3B 使用静态 Analyzer skill 的生成式预检（2026-09-23）

已在 `models/Qwen2.5-3B-Instruct` 上完成 fixed-state、生成式的 real/plain/shuffled
对照（24 个可审计 sibling event）。这是对“3B 能否利用特权 skill 纠正下一步
think/search”的必要但不充分检查，**不是** sampled-NLL 的最终 likelihood 资格门。

- real skill：45.8% hit@3；student 原动作：41.7%（`+4.2pp`，95% CI `[-12.5, +20.8]`）；
  shuffled skill：37.5%（`+8.3pp`，CI `[-8.3, +25.0]`）；plain：20.8%。
- real skill 的格式有效率 95.8%，答案泄漏率 0%；plain 为 50.0%，shuffled 为 100%。
- 24/24 real skill 来自同一问题；24/24 shuffled skill 来自不同问题。对照没有同题泄漏。
- 结论：有“真实 skill 优于 plain/shuffled”的点估计迹象，但 real 相对 student 未达到
  预设 `+5pp`，且 real-vs-shuffled 的 bootstrap CI 跨 0。因此 **gate=false，不得启动
  OPD 训练**。当前证据不能证明方案可行，只能证明 3B 能够在部分状态下格式化地利用 skill。

可复现产物与脚本：

- `verl_log/analyzer_skill_correction_3b_v1.json`
- `scripts/audit_analyzer_skill_correction_retrieval.py`

下一步应扩大独立 question/event 覆盖（不能复用 rejected skill），然后运行规划中的
same-token real/plain/shuffled likelihood audit；只有该审计通过才可进入 sampled-NLL OPD。

## 15. 固定学生动作的 3B likelihood 预检（2026-09-23）

按简化要求，已不让 3B 重新生成或执行搜索，而是固定同一批树中已有的 high-value/low-value
学生动作，直接比较其 token 平均 log-prob。24/24 real skill 与本题 UID 对齐，24/24
shuffle skill 来自其他问题。

- real 相对 plain 的 high-value 动作 log-prob lift：think `-0.184`，search `-0.083`，
  all `-0.155`；因此真实 skill 没有提高正确动作的绝对概率，反而降低了概率。
- real 对 high-vs-low margin 的 lift：think `+0.026`，search `-0.033`，all `+0.015`。
  其中真正决定下一跳的 search margin 是负的。
- 结论：这批 Analyzer skill 不能作为当前 3B sampled-NLL OPD 的直接 teacher signal；不要
  启动训练。结果文件为 `verl_log/analyzer_skill_likelihood_3b_v1.json`，脚本为
  `scripts/audit_analyzer_skill_likelihood.py`。

## 16. 3B Analyzer-SFT 后的固定动作复测（2026-09-23）

使用 293 条 train / 31 条 validation 做了两轮 LoRA Analyzer-SFT。训练的 validation loss
从 `2.671` 降到 `1.607`，产物为：

- `data/tree_seed_analyzer_sft_3b_20260923/adapter`
- `data/tree_seed_analyzer_sft_3b_20260923/merged`
- `data/tree_seed_analyzer_sft_3b_20260923/manifest.json`

随后在完全相同的 24 个固定 high-value/low-value 动作上复测：

- real skill 对 high-value search 的绝对 log-prob lift 从 base 的 `-0.083` 改善到
  `-0.005`，说明 SFT 确实让 3B 更适应 skill 输入格式；但仍没有正向提高正确 search
  token 的概率。
- real skill 的 search good-minus-bad margin lift 为 `+0.011`，shuffled skill 为 `+0.017`；
  real 并不优于 shuffled，且 CI 均跨 0。
- real skill 的 all-span margin lift 为 `+0.033`，但 shuffled 也为 `+0.023`，不能归因于
  本题 skill 的信息。

因此：Analyzer-SFT 解决了“3B 不认识 skill 格式”的一部分问题，但没有证明 skill 能把
3B 推向本题的正确下一跳。sampled-NLL OPD 仍保持关闭；下一步应改进节点级 skill 对齐和
可执行性，而不是直接增大 OPD 权重。

## 17. 节点级 skill-consumption pilot 与混合 SFT（2026-09-23）

经用户明确授权，将 31 条答案脱敏的节点级 prompt 发送给 DeepSeek。自动 validator 接受
30 条、拒绝 1 条标题复制；人工抽检发现低 value-gap 节点存在策略错配，且部分 target
动作含 `[REDACTED]`，因此进一步收紧为：

- `value_gap >= 0.25`；
- target 必须是完整 `<think>...</think><search>...</search>`；
- target 不得含 `[REDACTED]`；
- skill 不得复制答案、标题或完整 target query。

最终严格保留 11 条节点样本（8 个问题）：9 条训练、2 条验证。训练中仅对 9 条
skill-consumer 样本有限重复 16 次，使混合训练集包含 293 条 Analyzer-generation 和
144 条 skill-consumer；验证集为 31 + 2 条，问题 UID 不跨集合。由于独立节点样本很少，
该实验只用于验证接口能否被 bootstrap，不能作为泛化能力结论。

从原始 agent SFT checkpoint
`verl_checkpoints/singlehopqa-sft-search-r1-qwen2.5-3b-instruct/global_step_350`
在 GPU 3 完成两轮 LoRA 混合 SFT：110 updates，validation loss 从 `2.538` 降到 `1.482`。

产物：

- 数据：`data/tree_seed_mixed_node_skill_sft_v2/`
- adapter：`data/tree_seed_mixed_node_skill_sft_3b_v2_20260923/adapter/`
- merged checkpoint：`data/tree_seed_mixed_node_skill_sft_3b_v2_20260923/merged/`
- 训练日志：`data/tree_seed_mixed_node_skill_sft_3b_v2_20260923/training_log.jsonl`
- 构造脚本：`scripts/data_process/build_node_skill_consumption_sft.py`
- 混合导出：`scripts/data_process/export_mixed_node_skill_sft.py`

在扩大节点数据或启动 OPD 前，必须先在 held-out 节点上检查 real skill 是否提高 high-value
action likelihood 且优于 shuffled skill；当前验证节点只有 2 个，统计能力不足。

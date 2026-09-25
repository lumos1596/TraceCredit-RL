# TraceCredit + Self-OPD step41 smoke 实验报告

日期：2026-09-19

## 结论

原计划在方法方向上合理，且与 SD-Search 的核心设置一致：同模型 teacher、hindsight sibling context、teacher stop-gradient、仅在 query token 上计算 JSD，并与 TraceCredit 联合训练。

但原计划缺少可直接执行所需的若干约束：同前缀 sibling 的定义、未来信息隔离、teacher/student token 对齐、定长张量组织、无正负对照组时的退化行为，以及显存预算。此次实现补齐了这些约束，并完成共享 step40 到 step41 的 3B 配对 smoke。

工程可行性已经验证：Self-OPD 在真实 3B FSDP 训练中产生了非零信号并成功反向、保存 checkpoint。当前一步 smoke 的效果结果为负，不能据此声称 Self-OPD 带来收益，也不宜立即扩展到 7B 或长程正式训练。

## 实验设置

- 公共起点：TraceCredit step40 checkpoint。
- 对照组：已有 grouped TraceCredit step41 checkpoint。
- 实验组：TraceCredit + Self-OPD，从相同 step40 仅更新一次到 step41。
- 模型：Qwen2.5-3B。
- OPD 系数：`1e-3`。
- teacher 最大长度：512 tokens；query 最大长度：32 tokens。
- rollout：16 chunks，24 个有效 prompts，144 条被接受轨迹；动态采样共尝试 288 条轨迹。
- 评测：natural-120，greedy，`tree_search=false`、`n=1`、`n_agent=1`、`temperature=0`、`do_sample=false`。

## 训练验收

step41 的实际训练指标：

| 指标 | 数值 |
|---|---:|
| `self_opd/loss` | 0.0132808 |
| `self_opd/coef` | 0.001 |
| `self_opd/token_weight` | 1.1667 |
| `self_opd/events` | 0.1667 |
| `actor/pg_loss` | 0.103259 |
| `actor/grad_norm` | 2.43471 |
| `branch_credit/nonzero_fraction` | 0.214835 |

`self_opd/events` 与 `self_opd/token_weight` 是跨 mini-batch 聚合后的均值，不是原始事件总数。二者及 JSD loss 均非零，足以证明构造、teacher/student 对齐、stop-gradient 前向和反向链路实际生效。

step41 checkpoint 的 3 个 model、3 个 optimizer 和 3 个 extra-state rank 分片均已生成。

## 配对评测结果

| 方法 | natural-120 micro EM | macro EM | 相对 TraceCredit |
|---|---:|---:|---:|
| TraceCredit | 0.3667 | 0.2201 | — |
| TraceCredit + Self-OPD | 0.3417 | 0.2063 | -0.0250 |

micro EM 相差 0.025，即 Self-OPD 组在 120 条样本上少答对 3 题。两组的协议检查全部通过，且均无 OOM、traceback、Ray task error 或 trajectory-count error。

## 解释与下一步门槛

这次实验支持“实现可运行且确有 OPD 梯度信号”，不支持“Self-OPD 已提升效果”。一步更新和 120 条自然分布样本的方差较大，因此也不足以判定方法无效。

进一步投入前建议先做低成本诊断，而不是直接上 7B：

1. 将 OPD 原始事件数、query token 数、成功/失败 sibling 数改为总量指标，避免均值难以解释。
2. 在 3B 上做至少 3 个固定 seed 的 step40→step41 配对，或跑 3–5 个更新；报告均值、方差和 paired bootstrap 区间。
3. 对比 `alpha ∈ {1e-4, 3e-4, 1e-3}`。当前一步结果提示 `1e-3` 可能偏强，但不能单点定论。
4. 记录每种父节点类型的 OPD 覆盖率。离线审计中，102 棵已有树里有 63 个严格同父 query sibling 组，只有 11 组同时包含成功与失败，说明信号天然稀疏。
5. 只有当多 seed 3B 结果至少不劣于 TraceCredit，且 OPD 覆盖率与稳定性可接受时，再扩展到 7B。

## 产物

- 训练日志：`verl_log/multihop-tree-dapo-tracecredit-self-opd-smoke-step41-20260918-r2.log`
- 实验 checkpoint：`verl_checkpoints/multihop-tree-dapo-tracecredit-self-opd-smoke-step41-20260918-r2/actor/global_step_41`
- 配对结果：`evaluation/formal_em/tracecredit_self_opd_step41_smoke/comparison.json`
- TraceCredit 结果：`evaluation/formal_em/tracecredit_self_opd_step41_smoke/tracecredit/natural_n120/result.json`
- Self-OPD 结果：`evaluation/formal_em/tracecredit_self_opd_step41_smoke/tracecredit_self_opd/natural_n120/result.json`


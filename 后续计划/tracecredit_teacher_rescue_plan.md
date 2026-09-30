# TraceCredit + Teacher-Rescue 主实验执行计划

## 1. 实验目标

解决 DAPO 动态采样中大量 **all-wrong prompt 被直接丢弃**、导致模型最薄弱问题长期得不到训练的问题。

核心思路：

- 对正常“有对有错”的 group，继续使用现有 TraceCredit + DAPO 训练。
- 对 all-wrong group，不再直接丢弃，而是让 Teacher 在关键位置短暂接管一次，给出一个更合理的搜索动作，然后再交还给 Student 继续 rollout。
- Teacher 接管的 token 用 KD/OPD 学习；Teacher 之后由 Student 生成的 suffix 继续参与 RL。

## 2. 主训练流程

### A. Student 正常采样

每个 prompt 使用 Student 采样 6 条轨迹。

根据结果分两类：

1. **Mixed group：6 条中有对有错**
   - 按现有 TraceCredit / DAPO 流程训练。
   - 不增加 Teacher intervention。

2. **All-wrong group：6 条全部错误**
   - 不再丢弃。
   - 进入 Teacher-Rescue 流程。

## 3. Teacher-Rescue 流程

对 all-wrong prompt：

1. 从失败轨迹中选择一个较早的关键 search 决策位置。
2. 保留该位置之前的 Student prefix。
3. 将当前 prefix、问题、已有检索信息，以及可用的 privileged information（如 golden answer / skill）提供给 Teacher。
4. Teacher 只生成一个很短的 corrective action：
   - 优先只生成 `<search>...</search>`
   - 必要时可以生成短 `<think> + <search>`
   - 不允许 Teacher 直接完成整条轨迹或直接给最终答案。
5. 将 Teacher corrective action 写入 trajectory。
6. 从 Teacher action 之后重新交还给 Student。
7. Student 自己继续 search / think / answer，直到轨迹结束。

流程：

```text
Student prefix
    ↓
Teacher corrective search
    ↓
Student suffix rollout
    ↓
Final answer / reward
```

## 4. Loss 处理

### 4.1 Mixed group

保持现有训练方式：

```text
TraceCredit / DAPO RL loss
```

### 4.2 All-wrong rescued group

将轨迹分成两部分：

#### Teacher 接管 token

Teacher 生成的 corrective query 需要让 Student 学习，因此：

```text
Teacher tokens → KD / OPD loss
```

建议保持较小权重，例如：

```text
L_teacher = λ * L_KD
```

初始可以先使用：

```text
λ = 0.01
```

#### Student suffix token

Teacher 之后由 Student 自己生成的部分：

```text
Student suffix → 正常 RL / TraceCredit loss
```

Teacher token 不作为 Student 自己采样的 RL action 参与普通 GRPO ratio 计算。

最终：

```text
L_total
=
L_RL(mixed groups + rescued student suffix)
+
λ * L_KD(teacher corrective tokens)
```

## 5. 训练时必须记录的指标

不要再只看筛选后的 `critic/score/mean`。

至少记录：

- 所有原始 rollout 的纯 EM
- all-wrong prompt 比例
- mixed prompt 比例
- Teacher-Rescue 后成功的 prompt 比例
- rescued trajectory 的最终 EM
- 固定验证集纯 EM
- Teacher intervention 次数 / 总 prompt 数

重点关注：

```text
all-wrong prompt
→ Teacher rescue
→ Student 是否能够继续完成任务
```

## 6. 主实验对比

只保留最关键的两组：

### Baseline

```text
TraceCredit + DAPO
```

all-wrong group 直接丢弃。

### Proposed

```text
TraceCredit + DAPO
+
Teacher-Rescue for all-wrong groups
```

其中：

```text
Teacher corrective token → KD
Student suffix → RL
```

最终主要比较：

- 固定验证集 EM
- all-wrong prompt 比例是否下降
- 原始 rollout EM 是否提高
- Teacher-Rescue 后 Student 独立完成任务的能力是否提高

## 7. 最核心的实现原则

1. Teacher 只在 **all-wrong prompt** 上介入。
2. Teacher 只接管 **一个短的关键 search action**，不要代替 Student 完成整条轨迹。
3. Teacher token 用 KD/OPD 学习。
4. Teacher 之后必须重新交还给 Student 自己 rollout。
5. Student suffix 才参与正常 RL / TraceCredit 更新。
6. 训练效果以 **完整验证集 EM 和未筛选 rollout EM** 为主，不再用 DAPO 筛选后的 shaped reward 判断整体能力。

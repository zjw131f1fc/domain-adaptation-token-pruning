# Paper 用方法与 Loss 描述（Domain Adaptation Token Pruning）

本文档面向论文写作，描述我们方法的**结构**与**训练目标（loss）**。只包含当前实验设置中实际启用的模块与损失项。

---

## 1. 问题设定与总体思路

我们研究多模态大模型（MLLM）中的**视觉 token 剪枝**：在不显著损害回答准确率的前提下，减少参与后续语言层计算的视觉 token 数量，从而降低推理算力开销。

方法由两部分组成：

1. **Pruner（不变）**：在若干指定的 Transformer 层上，预测视觉 token 的保留/剪除掩码，并以“累积式”方式传播到后续层。
2. **Delayed Repair Adapter（部署版）**：在剪枝造成的信息缺失主要影响回答生成时，通过“延迟修复”的方式，仅对答案生成相关 token 的表示做轻量补偿。

---

## 2. 结构：累积式视觉 token 剪枝

### 2.1 累积掩码（cumulative mask）

设图像在序列中对应的视觉 token 数为 \(N\)。我们在若干剪枝层 \(\mathcal{L}_p=\{\ell_1,\ell_2,\dots\}\) 上预测当前层掩码：

\[
 m^{(\ell)} \in \{0,1\}^{N}, \quad 1=\text{keep},\ 0=\text{prune}.
\]

剪枝是**累积式**的。令累积掩码为 \(c^{(\ell)}\)，则：

\[
 c^{(\ell)} = c^{(\ell-1)} \odot m^{(\ell)}, \qquad c^{(0)}=\mathbf{1}.
\]

在任意层的 self-attention 中，我们使用 \(c^{(\ell)}\) 对视觉 token 的注意力权重做 post-softmax masking，并重新归一化，使得被剪除的视觉 token 不再参与聚合，同时序列位置保持不变（不做物理删除），从而保持训练稳定性。

### 2.2 Pruner 预测（Cross-attention scoring）

Pruner 使用可学习的 pruning queries 与视觉 token 表示做 cross-attention，输出每个视觉 token 的保留 logits（并可融合 question-conditioning 信息），再通过 Gumbel-Sigmoid / STE 方式得到二值掩码 \(m^{(\ell)}\)。我们在训练中使用分阶段策略（hybrid）逐步降低温度并在后期关闭噪声，以减小训练—推理差异。

---

## 3. Delayed Repair Adapter：语言侧、仅修复答案生成 token

视觉 token 剪枝会降低跨模态信息注入，从而影响生成答案。我们采用“Delayed Repair”策略：

1. **在剪枝层缓存修复上下文**：对剪枝后的视觉保留模式进行编码，形成低维上下文向量（例如 mask 表征与被剪除信息的聚合表征）。
2. **在若干 repair 层注入修复**：在语言侧的指定层 \(\mathcal{L}_r\) 上，使用上述上下文对隐藏状态做轻量调制，但仅作用于答案生成相关 token（gen_answer 区域），避免对无关 token 造成扰动。

### 3.1 gen_answer 区域

为了与训练时的 teacher-forcing 任务损失对齐，我们定义 gen_answer 为答案 token 的预测位置区间。直观上，它对应“开始生成答案之前一个位置起”到“答案结束”这一段的隐藏状态子序列。

### 3.2 上下文来源与映射

每个 repair 层从某个剪枝层获取上下文（可一一指定，也可按“最近的剪枝层”自动选择）。repair 模块输出增量 \(\Delta h\)，并通过门控只对 gen_answer 区域生效：

\[
 h \leftarrow h + \mathbf{g} \odot \Delta h,
\]

其中 \(\mathbf{g}\) 为 gen_answer 的二值掩码（非 gen_answer 位置为 0）。

### 3.3 梯度解耦（训练稳定性）

我们提供可选的 `repair_detach_input` 控制项，用于决定 repair 分支的输入是否 stop-gradient。启用时可避免 repair 目标通过主干路径对 pruner 形成不稳定的强监督；关闭时则允许更强的端到端耦合监督。

---

## 4. 训练目标（Loss）

当前实验设置中启用的 loss 由三部分组成：

\[
\mathcal{L} = \lambda_{\text{task}}\mathcal{L}_{\text{task}}
           + \lambda_{\text{repair}}\mathcal{L}_{\text{repair}}
           + \lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}}.
\]

### 4.1 任务损失 \(\mathcal{L}_{\text{task}}\)

对答案 token 进行交叉熵训练（teacher-forcing）。该项衡量模型在给定图像与问题条件下生成正确答案的能力。

### 4.2 修复对齐损失 \(\mathcal{L}_{\text{repair}}\)（Teacher–Student 分布对齐）

我们引入 teacher forward 作为对齐目标：

- **Teacher**：在 keep-all（不剪枝）模式下前向，得到 gen_answer 区域的中间层表示作为参考分布。
- **Student**：使用 pruning + delayed repair 的前向，在相同层捕获 gen_answer 区域表示。

在每个 repair 层上，我们在 gen_answer 区域计算表示分布的对齐损失，并对所有 repair 层取平均。这里采用**软性分布对齐（moment matching）**：不做逐 token 的点到点回归，而是对齐表示分布的一阶/二阶统计量（均值与方差）：

\[
\mathcal{L}_{\text{repair}} =
\lVert \mu_s-\mu_t \rVert_2^2 \;+\; \alpha \lVert \sigma_s^2-\sigma_t^2 \rVert_2^2,
\]

其中 \(\mu\) 与 \(\sigma^2\) 分别为 token 维度上的均值与方差，\(\alpha\) 为方差项权重。

### 4.3 稀疏度损失 \(\mathcal{L}_{\text{sparse}}\)

设目标保留 token 数为 \(K\)，则目标保留率为 \(r^* = K/N\)。我们用累积保留率估计“全层平均保留率/算力占比”，并最小化其与目标的差异：

\[
\mathcal{L}_{\text{sparse}} = \left| \bar{r} - r^* \right|.
\]

训练早期可对 \(r^*\) 进行退火：从 \(1.0\) 平滑过渡到最终目标 \(K/N\)，以避免过早强剪枝导致训练崩溃。

---

## 5. 论文中的超参数消融建议（只围绕启用模块）

为了回答“哪些设计带来收益”，建议围绕三块做最小且信息量大的消融：

### 5.1 剪枝强度与位置

1. **目标保留数 \(K\)**：例如 \(\{32, 64, 96, 128\}\)，绘制 Accuracy–FLOPs/Speed 曲线。
2. **剪枝层集合 \(\mathcal{L}_p\)**：固定 \(K\)，对比不同层位（浅层/中层/深层、两层 vs 三层）。
3. **Gumbel/温度策略**：hybrid vs never（纯 STE），对比稳定性与最终性能（尤其是推理一致性）。

### 5.2 Delayed Repair 的结构设计

4. **是否启用 delayed repair**：on/off（这是最关键的 “module ablation”）。
5. **repair 层集合 \(\mathcal{L}_r\)**：固定数量但移动位置（靠前/靠后），以及 1 段 vs 2 段 vs 3 段修复。
6. **repair 上下文来源映射**：显式一一对应 vs “最近剪枝层自动选择”。
7. **repair bottleneck / 强度**：bottleneck 维度与 adapter 缩放（或等效强度参数）的敏感性。

### 5.3 Loss 权重与耦合方式

8. **\(\lambda_{\text{repair}}\)**：例如 \(\{0, 1, 3, 5\}\)，观察修复对齐与任务准确率的权衡。
9. **\(\lambda_{\text{sparse}}\)** 与退火：不同稀疏权重、是否退火/退火时长，评估训练稳定性与最终稀疏度达成情况。
10. **`repair_detach_input`**：detach vs no-detach，对比端到端耦合监督是否带来收益或不稳定。

> 建议优先级：4（repair on/off）→ 1（K 扫描）→ 8（repair weight）→ 5（repair 层位）→ 10（detach）→ 2/3（pruner层位与温度策略）。

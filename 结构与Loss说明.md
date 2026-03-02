# 方法与 Loss 描述

## 1. 问题设定与总体思路

我们研究多模态大模型（MLLM）中的**视觉 token 剪枝**：在不显著损害回答准确率的前提下，减少参与后续语言层计算的视觉 token 数量，从而降低推理算力开销。

方法由两部分组成：

1. **Pruner**：在若干指定的 Transformer 层上，预测视觉 token 的保留/剪除掩码，并以"累积式"方式传播到后续层。
2. **Delayed Repair Adapter**：在剪枝造成的信息缺失主要影响回答生成时，通过"延迟修复"的方式，仅对答案生成相关 token 的表示做轻量补偿。

---

## 2. 结构：累积式视觉 token 剪枝

### 2.1 累积掩码（Cumulative Mask）

设图像在序列中对应的视觉 token 数为 $N$。我们在若干剪枝层 $\mathcal{L}_p=\{\ell_1,\ell_2,\dots\}$ 上预测当前层掩码：

$$
m^{(\ell)} \in \{0,1\}^{N}, \quad 1=\text{keep},\ 0=\text{prune}
$$

剪枝是**累积式**的。令累积掩码为 $c^{(\ell)}$，则：

$$
c^{(\ell)} = c^{(\ell-1)} \odot m^{(\ell)}, \qquad c^{(0)}=\mathbf{1}
$$

在任意层的 self-attention 中，我们使用 $c^{(\ell)}$ 对视觉 token 的注意力权重做 post-softmax masking，并重新归一化，使得被剪除的视觉 token 不再参与聚合。

### 2.2 CrossAttentionPruner 结构

Pruner 采用**残差设计**，以 LLM 的 question→vision attention 作为 baseline，学习修正量 delta：

$$
\text{keep\_logits} = \text{baseline} + \text{delta} + \text{bias}
$$

各组件说明：

**Baseline（来自 LLM attention）**：
- 提取 LLM 层中 question tokens 对 vision tokens 的 attention 权重
- 转换到 logit 空间：$\text{baseline} = \log(\text{q2v\_attn}) - \text{mean}$
- 计算 mean 时只考虑累积 mask 中保留的位置

**Delta（Pruner 学习的修正）**：
- Cross-Attention 分数：可学习的 pruning queries $Q_p \in \mathbb{R}^{n_q \times d}$ 与 vision tokens 做 cross-attention
  $$
  \text{attn\_score} = \text{Aggregate}(\text{softmax}(Q_p K_v^\top / \sqrt{d}) )
  $$
- Question Conditioning：将 question tokens 的 hidden states 做 masked mean 后投影，加到 pruning queries 上作为条件
  $$
  Q_p' = Q_p + \text{proj}\left( \frac{\sum_i h_q^{(i)}}{L_q} \right)
  $$
  其中 $L_q$ 为 question 的实际长度（排除 padding）
- Per-token 分数：MLP 对每个 vision token 独立评分
  $$
  \text{token\_score} = \text{MLP}(\text{proj}(h_v))
  $$
- 两者相加：$\text{delta} = \text{attn\_score} + \text{token\_score}$

**Bias**：
- 可学习标量，初始化为 2.0，鼓励初期保留更多 token

**Mask 生成（Gumbel-Sigmoid）**：

训练时使用 Gumbel-Sigmoid + STE：
$$
y_{\text{soft}} = \sigma\left(\frac{\text{logits} + \text{noise}}{\tau}\right), \quad
m = \mathbb{1}[y_{\text{soft}} > 0.5] - \text{sg}(y_{\text{soft}}) + y_{\text{soft}}
$$

其中 noise 为 Logistic 噪声，$\tau$ 为温度，sg 表示 stop-gradient。

推理时使用确定性阈值：$m = \mathbb{1}[\sigma(\text{logits}/\tau) > 0.5]$

**Key Padding Mask**：Cross-attention 中使用累积 mask 屏蔽已被剪掉的 tokens，避免对已剪位置重复决策。

---

## 3. Delayed Repair Adapter

视觉 token 剪枝会降低跨模态信息注入，从而影响生成答案。我们采用"Delayed Repair"策略：在剪枝层缓存修复上下文，在后续 repair 层对答案生成相关 token 做轻量补偿。

### 3.1 RepairContextEncoder：缓存修复上下文

在每个剪枝层，将 (vision_hidden, mask) 编码为低维向量，供后续 repair 层使用：

**Mask Embedding（MaskAttentionEncoder）**：
- 可学习位置编码 $P \in \mathbb{R}^{N \times d_{\text{pos}}}$
- 可学习 attention query $q \in \mathbb{R}^{d_{\text{pos}}}$
- 编码过程：
  $$
  \text{mask\_emb} = \text{proj}\left( \text{softmax}(q \cdot (m \odot P)^\top) \cdot (m \odot P) \right)
  $$
- 被剪掉的位置（mask=0）对应的 embedding 为 0，attention 自然降低

**Pruned Token Embedding（PrunedTokenAggregator）**：
- 聚合被剪掉 tokens 的 hidden states，提供"丢失了什么信息"的上下文
  $$
  \text{pruned\_emb} = \text{proj}\left( \frac{\sum_i (1-m_i) h_i}{\sum_i (1-m_i)} \right)
  $$

### 3.2 LightweightAdapter：FiLM 调制

Adapter 采用 FiLM (Feature-wise Linear Modulation) 机制，根据修复上下文动态调制隐藏状态：

**输入**：
- $x$：当前层的 attention output
- $\text{mask\_emb}$：来自上一剪枝层的 mask embedding
- $\text{pruned\_emb}$：来自上一剪枝层的被剪枝信息 embedding
- $\text{query}$：当前 token 的 attention query

**结构**：
$$
\begin{aligned}
h &= \text{GELU}(\text{Down}(x)) \\
\text{cond} &= \text{mask\_emb} + \text{pruned\_emb} + \text{QueryProj}(\text{query}) \\
\gamma &= 1 + W_\gamma \cdot \text{cond}, \quad \beta = W_\beta \cdot \text{cond} \\
h' &= \gamma \odot h + \beta \\
\Delta &= \alpha \cdot \text{Up}(h') \\
\text{output} &= x + \Delta
\end{aligned}
$$

**关键设计**：
- $\alpha$：可学习缩放因子，初始化为 0.1，控制修复强度
- FiLM 参数 $W_\gamma, W_\beta$ 零初始化，初始时 $\gamma=1, \beta=0$，adapter 输出接近 0
- Bottleneck 结构：Down 投影到低维，Up 投影回原维度

### 3.3 gen_answer 区域

为了与训练时的 teacher-forcing 任务损失对齐，我们定义 gen_answer 为答案 token 的预测位置区间。直观上，它对应"开始生成答案之前一个位置起"到"答案结束"这一段的隐藏状态子序列。

### 3.4 上下文来源与映射

每个 repair 层从最近的某个剪枝层获取上下文。repair 模块输出增量 $\Delta h$，并通过门控只对 gen_answer 区域生效：

$$
h \leftarrow h + \mathbf{g} \odot \Delta h
$$

其中 $\mathbf{g}$ 为 gen_answer 的二值掩码（非 gen_answer 位置为 0）。

---

## 4. 训练目标（Loss）

当前实验设置中启用的 loss 由三部分组成：

$$
\mathcal{L} = \lambda_{\text{task}}\mathcal{L}_{\text{task}} + \lambda_{\text{repair}}\mathcal{L}_{\text{repair}} + \lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}}
$$

### 4.1 任务损失 $\mathcal{L}_{\text{task}}$

对答案 token 进行交叉熵训练（teacher-forcing）。该项衡量模型在给定图像与问题条件下生成正确答案的能力。

### 4.2 修复对齐损失 $\mathcal{L}_{\text{repair}}$（Teacher–Student 分布对齐）

我们引入 teacher forward 作为对齐目标：

- **Teacher**：预计算，在 keep-all（不剪枝）模式下前向，得到 gen_answer 区域的中间层表示作为参考分布。
- **Student**：使用 pruning + delayed repair 的前向，在相同层捕获 gen_answer 区域表示。

在每个 repair 层上，我们在 gen_answer 区域计算表示分布的对齐损失，并对所有 repair 层取平均。因为信息的损失本身是不可逆的，这里采用**软性分布对齐（moment matching）**：不做逐 token 的点到点回归，而是对齐表示分布的一阶/二阶统计量（均值与方差）：

$$
\mathcal{L}_{\text{repair}} = \lVert \mu_s-\mu_t \rVert_2^2 + \alpha \lVert \sigma_s^2-\sigma_t^2 \rVert_2^2
$$

其中 $\mu$ 与 $\sigma^2$ 分别为 token 维度上的均值与方差，$\alpha$ 为方差项权重。

### 4.3 稀疏度损失 $\mathcal{L}_{\text{sparse}}$

设目标保留 token 数为 $K$，则目标保留率为 $r^* = K/N$。我们用累积保留率计算全层平均保留率，并最小化其与目标的差异：

$$
\mathcal{L}_{\text{sparse}} = \left| \bar{r} - r^* \right|
$$

训练早期对 $r^*$ 进行退火：从 $1.0$ 平滑过渡到最终目标 $K/N$，以避免过早强剪枝导致训练崩溃。

---

## 5. 论文中的超参数消融建议（只围绕启用模块）

为了回答"哪些设计带来收益"，建议围绕三块做最小且信息量大的消融：

### 5.1 剪枝强度与位置

1. **目标保留数 $K$**：例如 $\{32, 64, 96, 128\}$，绘制 Accuracy–FLOPs/Speed 曲线。
2. **剪枝层集合 $\mathcal{L}_p$**：固定 $K$，对比不同层位（浅层/中层/深层、两层 vs 三层）。
3. **Gumbel/温度策略**：hybrid vs never（纯 STE），对比稳定性与最终性能（尤其是推理一致性）。

### 5.2 Delayed Repair 的结构设计

4. **是否启用 delayed repair**：on/off（这是最关键的 "module ablation"）。
5. **repair 层集合 $\mathcal{L}_r$**：固定数量但移动位置（靠前/靠后），以及 1 段 vs 2 段 vs 3 段修复。
6. **repair 上下文来源映射**：显式一一对应 vs "最近剪枝层自动选择"。
7. **repair bottleneck / 强度**：bottleneck 维度与 adapter 缩放（或等效强度参数）的敏感性。

### 5.3 Loss 权重与耦合方式

8. **$\lambda_{\text{repair}}$**：例如 $\{0, 1, 3, 5\}$，观察修复对齐与任务准确率的权衡。
9. **$\lambda_{\text{sparse}}$** 与退火：不同稀疏权重、是否退火/退火时长，评估训练稳定性与最终稀疏度达成情况。
10. **`repair_detach_input`**：detach vs no-detach，对比端到端耦合监督是否带来收益或不稳定。

> **建议优先级**：4（repair on/off）→ 1（K 扫描）→ 8（repair weight）→ 5（repair 层位）→ 10（detach）→ 2/3（pruner 层位与温度策略）。

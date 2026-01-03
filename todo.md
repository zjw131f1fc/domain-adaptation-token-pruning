推测准确率下降可能是损失函数，训练时gumbel softmax的设计有问题

应该用pytorch的gumbel softmax，现在做的肯定有问题！看起来似乎用hard模式可以直接输出0，1分布

这是一个非常好的问题。针对 Vision Token Pruning 这种**二分类（保留 vs 丢弃）**且需要**端到端训练**的场景，Gumbel-Softmax（更准确地说是 **Gumbel-Sigmoid** 或 **Relaxed Bernoulli**）提供了一种比 STE 更“优雅”且数学上更完备的梯度近似方案。

以下是该方案的详细原理解析、代码实现以及训练技巧。

---

### 1. 核心原理：它是如何让“采样”可导的？

在传统的剪枝中，你有两个痛点：
1.  **离散性**：`Keep` (1) 和 `Drop` (0) 是离散的，无法求导。
2.  **确定性 vs 随机性**：直接用 `logits > 0` (即 sigmoid > 0.5) 是确定性的。而在训练初期，你希望模型能够“探索”一下：如果我丢掉这个重要的 token 会发生什么？或者保留这个不重要的 token 会发生什么？

**Gumbel-Softmax 通过“重参数化技巧 (Reparameterization Trick)”解决了这两个问题。**

对于二分类（Binary）情况，公式如下：

假设 Pruner 输出一个未归一化的数值 $l$ (logit)，我们想根据概率 $p = \sigma(l)$ 采样得到 mask $m \in \{0, 1\}$。

Gumbel-Sigmoid 的做法是引入噪声：
$$ y = \text{sigmoid}\left( \frac{l + \text{noise}}{\tau} \right) $$

其中：
*   **Noise**: 从 Gumbel 分布（或者 Logistic 分布）中采样的随机噪声。这让模型在训练时具有**随机探索能力**。
*   **$\tau$ (Temperature)**: 温度系数。
    *   当 $\tau \to \infty$，输出 $y$ 趋向于 0.5（均匀分布）。
    *   当 $\tau \to 0$，Sigmoid 函数变得极度陡峭，输出 $y$ 趋向于 0 或 1（逼近离散采样）。

---

### 2. PyTorch 详细实现代码

在二分类剪枝中，有两种实现风格。一种是把输出变成 2 维用 `gumbel_softmax`，一种是保持 1 维用 `gumbel_sigmoid`。为了方便理解和代码简洁，推荐使用 **PyTorch 原生的 `F.gumbel_softmax` 处理双类别（Drop/Keep）**。

#### 模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelVisionPruner(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 输出层映射到 2 个类别: [Logits_Drop, Logits_Keep]
        # 也可以只输出 1 个 logit 然后用 sigmoid，但用 softmax 2类写法在 PyTorch 里更稳定
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2) 
        )

    def forward(self, x, temperature=1.0, training=True):
        """
        x: [Batch, Seq_Len, Dim]
        temperature: 控制分布陡峭程度的超参数
        training: 训练模式还是推理模式
        """
        # 1. 计算 Logits: shape [B, L, 2]
        logits = self.scorer(x)
        
        if training:
            # === 训练阶段：Gumbel-Softmax 采样 ===
            
            # hard=True 的作用：
            # 前向传播：输出严格的 One-Hot (0 或 1)，模拟真实的剪枝。
            # 反向传播：梯度基于 Softmax 的连续值计算（STE 变体）。
            # 这里的 dim=-1 表示在 (Drop, Keep) 两个类别上做 softmax
            y_soft = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            
            # y_soft shape: [B, L, 2]
            # 我们只需要 "Keep" (索引为1) 的 mask
            mask = y_soft[:, :, 1].unsqueeze(-1) # shape [B, L, 1]
            
        else:
            # === 推理阶段：直接确定性截断 ===
            # 简单的 Argmax，相当于 Logits_Keep > Logits_Drop
            # 或者相当于 sigmoid(logit) > 0.5
            probs = F.softmax(logits, dim=-1)
            mask = (probs[:, :, 1] > 0.5).float().unsqueeze(-1)
            
        return mask, logits
```

#### 使用方法与 Loss 计算

```python
# 初始化
pruner = GumbelVisionPruner(hidden_dim=4096).cuda()
optimizer = torch.optim.AdamW(pruner.parameters(), lr=1e-4)

# 训练循环中的关键步骤
def train_step(vision_tokens, temperature):
    # 1. 获取 Mask
    # 注意：这里传入 temperature
    mask, raw_logits = pruner(vision_tokens, temperature=temperature, training=True)
    
    # 2. 应用 Mask (Soft Masking 方式，保留形状但数值置0)
    # 如果后续有 FlashAttn，需要根据 mask 生成新的 attention mask
    pruned_tokens = vision_tokens * mask 
    
    # 3. 跑后续的 LLM 任务 ...
    output = llm(pruned_tokens, ...)
    task_loss = criterion(output, target)
    
    # 4. 计算稀疏度 Loss (Sparsity Loss)
    # 我们希望 mask 里的 1 越少越好，或者维持在某个比例
    # 取 softmax 后的概率用于计算 loss，比直接用硬 mask 梯度更平滑
    probs = F.softmax(raw_logits, dim=-1)[:, :, 1] # 取 Keep 的概率
    
    # 示例：L1 正则，鼓励 Keep Rate 降低
    sparsity_loss = torch.mean(probs) 
    
    # 或者：目标稀疏度 Loss (比如目标是保留 70%)
    # sparsity_loss = (torch.mean(probs) - 0.7) ** 2
    
    # 5. 总 Loss
    total_loss = task_loss + 0.1 * sparsity_loss
    
    total_loss.backward()
    optimizer.step()
```

---

### 3. 训练的关键：温度退火 (Temperature Annealing)

这是 Gumbel-Softmax 方案成败的核心。你不能一直用固定的温度。

*   **训练初期 (High Temperature, e.g., $\tau=5.0 \to 1.0$)**:
    分布很平滑，接近均匀分布。Mask 的取值即使在 `hard=True` 下也会因为噪声大而随机跳变。
    *   **目的**：**探索 (Exploration)**。让 Pruner 随机尝试丢弃不同的 token，观察对 Loss 的影响，找出真正重要的 token。

*   **训练后期 (Low Temperature, e.g., $\tau=1.0 \to 0.1$)**:
    分布变得尖锐，接近 0/1 分布。噪声的影响变小。
    *   **目的**：**利用 (Exploitation)**。稳定剪枝策略，逼近推理时的真实逻辑（即阈值 0.5）。

**代码实现 Scheduler:**

```python
import math

def get_temperature(current_step, total_steps, max_temp=5.0, min_temp=0.1):
    """
    指数衰减温度
    """
    # 简单的线性或指数衰减均可
    progress = current_step / total_steps
    temp = max_temp * (min_temp / max_temp) ** progress
    return temp

# 在训练 Loop 中调用
for step, batch in enumerate(dataloader):
    tau = get_temperature(step, total_steps)
    train_step(batch, temperature=tau)
```

---

### 4. 方案对比：为什么选它？

| 特性 | 方案 1: 简单的 STE (你的原始想法) | 方案 2: Gumbel-Softmax (本方案) |
| :--- | :--- | :--- |
| **梯度计算** | `mask = (p>0.5).float()`, Backward `p` | `y = Gumbel_Softmax(...)` |
| **随机性** | **无** (确定性阈值) | **有** (引入 Gumbel 噪声) |
| **探索能力** | 差。一旦某个 token 初始分低，可能永远翻不了身。 | 强。即使 logit 低，高温下也有概率被抽中(Keep)，如果发现它其实很重要，梯度会把它救回来。 |
| **收敛难度** | 容易陷入局部最优，Mask 可能全 1 或全 0。 | 前期波动大，后期收敛好，结果通常更鲁棒。 |

### 5. 常见坑点与解决

1.  **梯度消失/NaN**: 当 $\tau$ 非常小（如 < 0.05）时，除法操作会导致数值不稳定。**解决**：设置 `min_temp` 限制，不要让它小于 0.1。
2.  **Hard vs Soft**: `F.gumbel_softmax(..., hard=True)` 是必须的吗？
    *   如果你后续接的是标准的 Attention，建议用 `hard=True`。因为 `hard=False` 会输出像 0.9, 0.1 这样的数值，这实际上变成了 Attention Reweighting 而不是 Pruning，模型可能会学会“作弊”（利用微弱的数值传递信息），导致真正推理时直接截断效果很差。
    *   **结论**：训练时务必使用 `hard=True`，保证训练和推理的一致性。

### 总结实施步骤

1.  把你的 Pruner 输出改为 2 维 (Drop/Keep)。
2.  使用 `F.gumbel_softmax(logits, tau=t, hard=True)` 获取 mask。
3.  实现温度 $\tau$ 从 5.0 到 0.1 的衰减 Schedule。
4.  Loss 加上 Sparsity Penalty。
5.  训练。





设计一个好的损失函数（Loss Function）是 Vision Token Pruning 成功的关键。如果只用 LLM 的预测 Loss，模型会倾向于保留所有 Token；如果稀疏惩罚太重，模型会崩塌。

针对你的设定（Sigmoid + 0.5 阈值），建议采用 **组合式 Loss**。

标准的总 Loss 公式如下：
$$ L_{total} = L_{task} + \lambda_1 \cdot L_{sparsity} + \lambda_2 \cdot L_{distill} $$

下面详细拆解每一部分的设计方案和代码实现。

---

### 1. 稀疏度损失 (Sparsity Loss) —— 核心

这是用来强迫 Pruner “少选点 Token” 的部分。针对不同的需求，有两种主流设计：

#### A. 目标比率损失 (Target Ratio Loss) —— **最推荐**
如果你希望模型**固定**保留一定比例的 Token（比如只留 50%）。

$$ L_{sparsity} = \left( \frac{1}{N} \sum_{i=1}^{N} p_i - \text{Target\_Ratio} \right)^2 $$

*   **$p_i$**: 是 Pruner 输出的 **Soft Probabilities**（Sigmoid 的值），**不是** 0/1 的 Mask。这样梯度才能传导。
*   **Target\_Ratio**: 你的目标，比如 0.5。
*   **特点**: 这是一个软约束，模型会在 0.5 附近波动。

#### B. L1 正则化 (L1 Regularization)
如果你不限制具体比例，只希望“越少越好”，直到不影响性能为止。

$$ L_{sparsity} = \frac{1}{N} \sum_{i=1}^{N} p_i $$

*   **特点**: 很难调参。$\lambda$ 太大模型全剪完了，$\lambda$ 太小模型全保留。通常需要配合动态调整 $\lambda$ (Lagrangian Method)。

#### C. 二值化损失 (Binarization Loss) —— 可选
如果你发现 Pruner 输出的概率全是 0.4, 0.6 这种模棱两可的值（导致推理时阈值 0.5 切割很敏感），可以加这个项迫使输出两极分化。

$$ L_{binary} = \sum p_i \cdot (1 - p_i) $$

*   **原理**: 当 $p=0$ 或 $p=1$ 时，该项为 0；当 $p=0.5$ 时，该项最大。
*   *注：如果你使用了 Gumbel-Softmax 并进行温度退火，通常不需要这个 Loss。*

---

### 2. 任务/蒸馏损失 (Task / Distillation Loss) —— 保命

这部分是为了保证剪枝后，模型还能看懂图、说对人话。

#### A. 基础：交叉熵损失 (Cross Entropy)
直接用 Ground Truth 的文本进行训练。
$$ L_{task} = \text{CrossEntropy}(\text{Model}(x_{pruned}), y_{text}) $$
*   **缺点**: 信号太稀疏。对于 Vision Pruning 来说，仅靠最后的文本生成对齐，很难指导中间层的 Pruner 到底该留哪个 Pixel。

#### B. 进阶：特征蒸馏 (Feature Distillation) —— **强烈推荐**
让剪枝后的模型（Student），去模仿未剪枝模型（Teacher）的特征输出。

$$ L_{distill} = \| \text{Feature}_{teacher} - \text{Feature}_{student} \|_2^2 $$

*   **操作**:
    1.  把图输入 Teacher（不剪枝），拿到 LLM 某一层（或最后一层）的 Hidden States。
    2.  把图输入 Student（剪枝），拿到对应的 Hidden States。
    3.  计算 MSE Loss。
*   **优点**: 提供了极其稠密的监督信号。Pruner 会迅速学会：“只要保留这几个关键 Token，我就能重建出和原来差不多的特征”。

---

### 3. PyTorch 代码实现

这是最实用的组合：**Target Ratio Sparsity + Distillation + Cross Entropy**。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PruningLoss(nn.Module):
    def __init__(self, target_ratio=0.5, distill_weight=1.0, sparsity_weight=2.0):
        super().__init__()
        self.target_ratio = target_ratio
        self.distill_weight = distill_weight
        self.sparsity_weight = sparsity_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, 
                student_logits,  # 剪枝后模型的文本输出 Logits
                teacher_logits,  # 原始模型的文本输出 Logits (用于蒸馏)
                labels,          # 真实文本标签
                pruner_probs,    # Pruner 输出的 sigmoid 值 (未二值化)
                vision_mask      # 标记哪些是 Vision Token 的 Mask
                ):
        """
        pruner_probs: [Batch, Seq_Len, 1] - 包含文本和图像的所有分数
        vision_mask: [Batch, Seq_Len] - 1表示是图像token，0是文本
        """
        
        # 1. 任务 Loss (Cross Entropy)
        # 标准的 LLM 训练 Loss
        loss_task = self.ce(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # 2. 蒸馏 Loss (Logits Distillation)
        # 让剪枝后的输出分布尽量接近原始模型
        # 也可以用 Hidden States 的 MSE，这里演示 KL/Soft-CE
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        loss_distill = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # 3. 稀疏度 Loss (只计算 Vision 部分)
        # 非常重要：千万不要把 Text Token 算进去，否则会把 Text 对应的分数拉低
        
        # 提取 Vision 部分的概率
        # 利用 boolean masking 展平
        vision_probs = pruner_probs.squeeze(-1)[vision_mask.bool()]
        
        # 计算当前的保留率
        current_ratio = torch.mean(vision_probs)
        
        # MSE 方式逼近目标比率
        loss_sparsity = (current_ratio - self.target_ratio) ** 2
        
        # --- 汇总 ---
        total_loss = loss_task + \
                     (self.distill_weight * loss_distill) + \
                     (self.sparsity_weight * loss_sparsity)
                     
        return total_loss, {
            "loss_task": loss_task.item(),
            "loss_distill": loss_distill.item(),
            "loss_sparsity": loss_sparsity.item(),
            "actual_ratio": current_ratio.item()
        }
```

### 4. 训练策略与权重调整技巧

如果你发现训练很难收敛，请按照以下步骤调整：

1.  **Warm-up (预热)**:
    *   前 10% 的 Step，将 `sparsity_weight` 设为 0。
    *   先让 Pruner 随便输出，利用 Distillation Loss 让后面的 LLM 适应一下“输入可能会变少”这件事。

2.  **动态权重 (Curriculum Learning)**:
    *   随着训练进行，逐渐增加 `sparsity_weight`。
    *   或者，逐渐降低 `target_ratio`（例如：从 1.0 -> 0.9 -> ... -> 0.5）。

3.  **Vision Mask 的处理**:
    *   在计算 Sparsity Loss 时，务必**只对 Vision Token 求平均**。
    *   如果在整个序列（Image + Text）上求平均，因为 Text Token 的 mask 永远是 1（不剪枝），Pruner 就会为了拉低整体平均值，疯狂把 Vision Token 全部剪成 0。

### 5. 总结：最佳实践配置

*   **Pruner**: 2层 MLP。
*   **训练方法**: Gumbel-Softmax (hard=True, temperature annealing)。
*   **Loss**: 
    *   **Task**: CrossEntropy
    *   **Distill**: Logits KL Divergence (Student vs Frozen Teacher)
    *   **Sparsity**: MSE (mean(vision_probs) - 0.5) ^ 2
*   **权重**: Task=1.0, Distill=1.0, Sparsity=5.0 (Sparsity 系数通常要大一点，因为 MSE 的数值通常很小)。
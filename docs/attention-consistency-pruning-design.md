# Attention Consistency Pruning 设计文档

## 概述

本方案通过在每个剪枝层保证 **attention 聚合结果的一致性** 来实现 vision token 剪枝。核心思想是：如果剪枝前后，answer tokens 从前面 tokens 聚合得到的结果一致，则说明被剪掉的 tokens 是冗余的。

## 核心思想

### 1. 信息聚合一致性

在 Transformer 的 self-attention 中，每个 token 的输出是通过 attention 从前面 tokens 聚合 V 得到的：

```
h[p] = Σ attention[p, i] × V[i]    (i 遍历所有前面的 positions)
```

对于 answer token，它从 vision tokens 和 question tokens 聚合信息。如果剪掉某些 vision tokens 后，聚合结果不变，说明这些 tokens 对回答问题没有贡献。

### 2. 网络输出残差

网络（pruner）的输入是 LLM 该层的 **question→vision attention**（LLM 自己认为的重要性），输出是一个**残差调整**：

```
importance = llm_q2v_attention + residual
mask = gumbel_softmax(importance, hard=True)
```

这样网络学习的是：LLM 的原生 attention 哪里需要修正。

### 3. GAN 判别

使用 GAN 框架判别聚合结果的一致性：
- **h_real**：用完整 attention 从所有 tokens 聚合
- **h_fake**：用调整后的 attention 从剩余 tokens 聚合
- **Discriminator**：判断 h 是 real 还是 fake
- **目标**：让 h_fake 的分布接近 h_real

### 4. 局部一致性

每层独立保证一致性：
- 不需要完整的 real LLM forward
- 只在当前层比较剪枝前后的聚合结果
- real 和 fake 都接收上一层剪枝后的结果

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      LLaVA Model (继承)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 0-3: 正常 forward                                    │
│                                                             │
│  Layer 4 (剪枝层):                                          │
│    1. Self-Attention → 获取 attn_weights, V                 │
│    2. 计算 h_real = attn @ V (answer tokens)                │
│    3. Pruner 输出 residual → mask                           │
│    4. 计算 h_fake = adjusted_attn @ V                       │
│    5. Discriminator 判别 h_real vs h_fake                   │
│    6. 应用 mask 到 hidden_states                            │
│                                                             │
│  Layer 5-13: 正常 forward                                   │
│                                                             │
│  Layer 14 (剪枝层): 同上                                    │
│                                                             │
│  Layer 15-23: 正常 forward                                  │
│                                                             │
│  Layer 24 (剪枝层): 同上                                    │
│                                                             │
│  Layer 25-31: 正常 forward                                  │
│                                                             │
│  Output: logits → Task Loss                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 剪枝层详细流程

```python
# 在剪枝层 L 的 forward 中：

# === 1. 正常的 Self-Attention 计算 ===
Q = self.q_proj(layer_norm(hidden_states))
K = self.k_proj(layer_norm(hidden_states))
V = self.v_proj(layer_norm(hidden_states))

attn_weights = softmax(Q @ K.T / sqrt(d))  # (batch, heads, seq, seq)
attn_output = attn_weights @ V              # (batch, heads, seq, d)

# === 2. 计算 h_real (answer tokens 的聚合结果，不剪枝) ===
h_real = attn_weights[:, :, ans_range, :] @ V  # (batch, heads, n_ans, head_dim)
# 每个 answer token 独立：h_real[i] for i in range(n_ans)

# === 3. Pruner 生成 mask ===
q2v_attn = attn_weights[:, :, q_range, v_range].mean(dim=(1, 2))  # (batch, n_vision)
residual = pruner(q2v_attn)                                        # (batch, n_vision)
importance = q2v_attn + residual                                   # (batch, n_vision)
hard_mask = gumbel_softmax(importance, hard=True)                  # (batch, n_vision)

# === 4. 计算 h_fake (answer tokens 的聚合结果，剪枝后) ===
adjusted_attn = attn_weights.clone()
adjusted_attn[:, :, :, v_range] = adjusted_attn[:, :, :, v_range] * hard_mask
adjusted_attn = adjusted_attn / adjusted_attn.sum(dim=-1, keepdim=True)  # 重新归一化
h_fake = adjusted_attn[:, :, ans_range, :] @ V  # (batch, heads, n_ans, head_dim)

# === 5. GAN 判别 (每个 answer token 独立) ===
for i in range(n_ans):
    real_pred = discriminator(h_real[:, :, i, :])  # (batch, heads, head_dim) → scalar
    fake_pred = discriminator(h_fake[:, :, i, :])
    # 收集 loss

# === 6. 应用 mask 到 hidden_states ===
# 被剪掉的 vision tokens 缩放为 0
hidden_states[:, v_range, :] = hidden_states[:, v_range, :] * hard_mask.unsqueeze(-1)

# === 7. 继续正常的 forward (残差连接 + MLP) ===
hidden_states = hidden_states + attn_output
hidden_states = hidden_states + mlp(layer_norm(hidden_states))
```

### 组件设计

#### Pruner (每层独立)

```python
class LayerPruner(nn.Module):
    """轻量级剪枝网络，输出对 LLM attention 的残差调整"""

    def __init__(self, d_internal=128):
        self.mlp = nn.Sequential(
            nn.Linear(1, d_internal),
            nn.LayerNorm(d_internal),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal, 1)
        )
        # 初始化为零，初始时完全依赖 LLM attention
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.temperature = 1.0

    def forward(self, q2v_attn):
        """
        输入: q2v_attn (batch, n_vision) - LLM 的 question→vision attention
        输出: residual (batch, n_vision) - 残差调整
        """
        residual = self.mlp(q2v_attn.unsqueeze(-1)).squeeze(-1)
        return residual
```

#### Discriminator (每层独立，与 Pruner 配套)

```python
class LayerDiscriminator(nn.Module):
    """判别单个 answer token 的聚合结果是 real 还是 fake"""

    def __init__(self, num_heads, head_dim, d_hidden=256):
        input_dim = num_heads * head_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, h):
        """
        输入: h (batch, heads, head_dim) - 单个 answer token 的聚合结果
        输出: logit (batch,) - real/fake 判断
        """
        h_flat = h.view(h.shape[0], -1)  # (batch, heads * head_dim)
        return self.net(h_flat).squeeze(-1)
```

### Loss 设计

```python
# === 1. Task Loss ===
# 最终输出的交叉熵损失
task_loss = cross_entropy(logits, answer_tokens)

# === 2. Adversarial Loss (每层) ===
# Pruner 的目标：让 h_fake 骗过 Discriminator
adv_loss = 0
for layer_idx in pruning_layers:
    for ans_idx in range(n_ans):
        fake_pred = discriminators[layer_idx](h_fake[layer_idx][:, :, ans_idx, :])
        adv_loss += BCE(fake_pred, ones)

# === 3. Discriminator Loss (每层) ===
disc_loss = 0
for layer_idx in pruning_layers:
    for ans_idx in range(n_ans):
        real_pred = discriminators[layer_idx](h_real[layer_idx][:, :, ans_idx, :])
        fake_pred = discriminators[layer_idx](h_fake[layer_idx][:, :, ans_idx, :].detach())
        disc_loss += BCE(real_pred, ones) + BCE(fake_pred, zeros)

# === 4. Sparsity Loss ===
sparsity_loss = 0
for layer_idx in pruning_layers:
    kept_ratio = masks[layer_idx].mean()
    sparsity_loss += |kept_ratio - target_ratio|

# === 总 Loss ===
pruner_loss = task_loss * w_task + adv_loss * w_adv + sparsity_loss * w_sparsity
disc_loss = disc_loss * w_disc
```

## 实现方案

### 继承 LLaVA 类

不使用 hook 机制，直接继承 LLaVA 的模型类，重写 forward 方法：

```python
class PrunableLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    """可剪枝的 LLaVA 模型"""

    def __init__(self, config, pruning_layers=[4, 14, 24]):
        super().__init__(config)
        self.pruning_layers = pruning_layers

        # 每层独立的 pruner 和 discriminator
        self.pruners = nn.ModuleDict({
            str(idx): LayerPruner() for idx in pruning_layers
        })
        self.discriminators = nn.ModuleDict({
            str(idx): LayerDiscriminator(
                num_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads
            ) for idx in pruning_layers
        })

    def forward(self, ...):
        # 重写 forward，在剪枝层添加剪枝逻辑
        ...
```

### 重写 Decoder Layer

```python
class PrunableLlamaDecoderLayer(LlamaDecoderLayer):
    """可剪枝的 Decoder Layer"""

    def __init__(self, config, layer_idx, pruner=None, discriminator=None):
        super().__init__(config, layer_idx)
        self.pruner = pruner
        self.discriminator = discriminator
        self.is_pruning_layer = pruner is not None

    def forward(self, hidden_states, attention_mask, position_ids,
                vision_range, question_range, answer_range, **kwargs):

        if self.is_pruning_layer:
            return self.forward_with_pruning(
                hidden_states, attention_mask, position_ids,
                vision_range, question_range, answer_range, **kwargs
            )
        else:
            return super().forward(hidden_states, attention_mask, position_ids, **kwargs)

    def forward_with_pruning(self, ...):
        # 实现带剪枝的 forward
        ...
```

## 关键设计决策

### 1. 为什么用 GAN 而不是 MSE

- 与 domain adaptation 思路一致
- 不强制精确匹配，只要分布接近
- 更灵活，允许一定程度的变化

### 2. 为什么每个 answer token 独立判别

- 粒度更细，信号更强
- 每个 answer token 对 vision 的依赖可能不同
- 避免 pooling 丢失信息

### 3. 为什么网络输出残差

- LLM 的 attention 已经是一个不错的 baseline
- 网络只需要学习"修正"
- 训练更稳定，更容易收敛

### 4. 为什么继承类而不用 hook

- Hook 机制容易出 bug，难以调试
- 继承类可以直接控制 forward 流程
- 更容易获取中间变量（V, attention weights）

## 训练流程

```
1. 冻结 LLM 参数
2. 只训练 pruners 和 discriminators
3. 交替训练：
   - 更新 discriminators（判别 h_real vs h_fake）
   - 更新 pruners（让 h_fake 骗过 discriminator，同时满足 sparsity 约束）
4. Task loss 同时作用于 pruners
```

## 与原方案对比

| 方面 | 原方案 | 新方案 |
|------|--------|--------|
| 比较对象 | 最终 LLM 输出的 hidden states | 每层的 attention 聚合结果 h |
| 需要 real forward | 是（完整 forward 两次） | 否（只在剪枝层计算 h_real/h_fake） |
| 监督粒度 | 全局（最终输出） | 每层局部 |
| 判别器输入 | hidden states | 聚合后的 h（每个 answer token） |
| 网络输入 | 自己训练的 cross-attention | LLM 原生的 q→v attention |
| 实现方式 | Hook 机制 | 继承类，重写 forward |

## 文件结构（建议）

```
method/
├── models/
│   ├── prunable_llava.py      # 可剪枝的 LLaVA 模型
│   ├── prunable_llama.py      # 可剪枝的 LLaMA decoder
│   ├── layer_pruner.py        # 每层的剪枝网络
│   └── layer_discriminator.py # 每层的判别器
├── training.py                 # 训练逻辑
├── losses.py                   # Loss 计算
└── utils.py                    # 工具函数
```

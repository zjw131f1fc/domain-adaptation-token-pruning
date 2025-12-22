# Vision Token Pruning 模型与训练总结

## 1. 核心思想

使用**GAN对抗训练**学习剪枝多模态大语言模型(MLLM)中的视觉tokens，降低计算成本同时保持性能。

## 2. 模型架构

### 2.1 整体流程

```
Image → Vision Encoder (CLIP) → Token Merger → LLM with Layer Pruners → Output
                                     ↑                    ↑
                                  可选阶段              多层剪枝
```

### 2.2 Token Merger (第一阶段剪枝)

现在并没有是用merger
**V3版本 - Fixed Pooling**:
- 使用可学习的query向量池化vision tokens
- Question调制: 用问题嵌入调制query向量
- Cross-Attention: queries关注vision tokens
- 输出固定数量的merged tokens

```
Input: (B, 576, d_vision) → Output: (B, M, d_vision)
```

### 2.3 Layer-Specific Pruner (第二阶段剪枝)

在LLM的多个层(如L6, L12, L18)独立剪枝:

```
Vision Hidden + Question → Cross-Attention → MLP → Gumbel-Softmax → Soft Mask
```

- 支持Attention Residual: 利用text→vision attention增强问题依赖
- 每层独立学习剪枝策略

### 2.4 Discriminator (判别器)

判别real(无剪枝)与fake(有剪枝)的text hidden states:

```
Multi-layer Hidden States → Per-layer MLP → Concat → MLP → Sigmoid
```

**输入预处理 - 加权池化与位置感知噪声**:

Discriminator的输入不是原始的text hidden states序列，而是经过加权池化后的向量。池化过程中加入位置感知噪声:

1. **位置感知噪声**: 对序列中每个token添加不同强度的高斯噪声
   - 前面的token噪声大 (noise_scale_start=0.05)
   - 后面的token噪声小 (noise_scale_end=0.01)
   - 原因: 序列前面的token还没"看到"完整信息，表示不稳定；后面的token已聚合上下文，表示更可靠

2. **加权池化**: 对序列进行加权平均
   - 前面token权重小 (start_weight=0.3)
   - 后面token权重大 (end_weight=1.0)
   - 原因: 后面token的hidden state包含更完整的上下文信息

**设计意图**: 让Discriminator更关注序列后部(信息完整)的表示质量，而非前部(信息不完整)的噪声差异。

**可选谱归一化**: 对Linear层应用spectral normalization，约束Lipschitz常数，增强GAN训练稳定性

## 3. 训练方法

### 3.1 损失函数

| 损失 | 作用 |
|-----|------|
| Task Loss | 预测answer的交叉熵 |
| Adversarial Loss | 欺骗discriminator |
| Sparsity Loss | 约束token保留率 |
| Token Count Loss | 鼓励减少tokens |
| Binarization Loss | 鼓励mask趋近0/1 |

### 3.2 关键训练技巧

1. **Temperature Annealing**: 从soft到hard assignment的平滑转变
2. **余弦调度损失权重**: 初期重任务性能，后期强化对抗
3. **Gumbel-Softmax**: 可微分的离散采样

### 3.3 训练流程

```
1. Batch预处理 → 获取embeddings和vision positions
2. Token Merge (可选) → 合并vision tokens
3. LLM Forward + Layer Hooks → 多层应用pruning masks
4. Discriminator判别 → real vs fake
5. 计算总损失 → 反向传播
```

## 4. 配置要点

```yaml
# 剪枝层
pruning_layers: [6, 12, 18]

# 目标保留
target_token_num: 200  # 或 target_sparsity: 0.3

# Temperature
temperature: 0.99 → 0.4  # 逐渐降低

# 损失权重调度
task_loss_weight: 10.0 → 10.0
adv_loss_weight: 1.0 → 10.0
```

## 5. 评估模式

- **origin**: 无剪枝baseline
- **hard**: 二值化mask (mask > 0.5)

## 6. 创新点

1. 两阶段剪枝: Token Merge + Layer-wise Pruning
2. 问题驱动的Cross-Attention剪枝
3. 对抗训练保证剪枝后表示质量
4. 动态损失权重调度
5. 批量化训练优化

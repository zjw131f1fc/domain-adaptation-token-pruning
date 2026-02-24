# Vision Token Pruning with GAN

基于 GAN 对抗训练的多模态大语言模型（MLLM）视觉 token 剪枝项目。

## 重要规则

**禁止私自简化代码逻辑！** 如果需要简化，必须先询问用户确认。

## 概述

本项目实现了针对 MLLM（如 LLaVA）的视觉 token 剪枝方法，通过在多个 LLM 层进行渐进式剪枝，在保持模型性能的同时大幅减少视觉 token 数量，加速推理。

## 核心特性

- **Gumbel-Sigmoid 三阶段混合训练**：探索期温度退火 + 稳定期低温 + 对齐期关闭noise
- **基于 GAN 的对抗训练**：Discriminator 确保剪枝后的表示保持质量
- **Cross-Attention Pruner**：使用可学习 queries 评估 vision tokens 重要性
- **Layer-wise Pruning**：在 LLM 内部多层进行渐进式剪枝
- **Lightweight Adapter**：Mask-Aware FiLM adapter 补偿剪枝信息损失

## Gumbel-Sigmoid 三阶段混合训练策略

### 配置

```yaml
gumbel_mode: "hybrid"  # "always" | "never" | "hybrid"

# 三阶段配置
hybrid_phase1_end: 0.6      # 探索期结束点
hybrid_phase2_end: 0.9      # 稳定期结束点
hybrid_phase1_temp_start: 1.5
hybrid_phase1_temp_end: 0.3
```

### 三阶段说明

1. **阶段1 探索期** (0% - 60%): 温度从 1.5 退火到 0.3，使用 Gumbel noise，探索哪些 token 重要
2. **阶段2 稳定期** (60% - 90%): 温度保持 0.3，继续使用 Gumbel noise，精细化决策边界
3. **阶段3 对齐期** (90% - 100%): 温度保持 0.3，关闭 Gumbel noise，让训练和推理行为一致

### 推理逻辑

推理时使用确定性阈值：`mask = (sigmoid(logits / temp) > 0.5).float()`，其中 temp 默认为 0.3（与阶段 3 一致）。这确保了训练和推理的完全对齐。

## 架构

### 训练流程

每个剪枝层的处理流程：

1. **计算 Attention**：LayerNorm + Q/K/V 投影 + RoPE + Attention Weights（causal mask）
2. **提取 Baseline**：question→vision attention 作为重要性的初始估计
3. **Pruner 生成 Mask**：
   - 输入：vision hidden states + q2v attention + 累积 mask
   - 输出：当前层的剪枝决策（0/1 mask）
4. **计算 h_real 和 h_fake**：
   - `h_real`：完整 attention 聚合（无剪枝）
   - `h_fake`：应用 current_mask 的 post-softmax masking + renormalize
5. **Adapter 修正**：对 h_fake 应用轻量级 adapter 补偿信息损失
6. **Discriminator 判别**：判断 h_real vs h_fake（answer 位置）
7. **更新累积 Mask**：`new_cumulative_mask = old_cumulative_mask * current_mask`

### 核心组件

#### CrossAttentionPruner
- **残差设计**：`keep_logits = baseline + delta + bias`
  - `baseline`：question→vision attention（log 变换 + 中心化）
  - `delta`：Pruner 学习的修正量
    - Cross-attention：可学习 queries 评估 vision tokens
    - Per-token score：MLP 评估单个 token 重要性
  - `bias`：可学习偏置（初始化为 2.0，鼓励保留更多 token）
- **Key Padding Mask**：使用累积 mask 屏蔽已被剪掉的 tokens

#### LayerDiscriminator
- 每个 answer token 独立判别（per-token discrimination）
- 支持谱归一化和随机重初始化（防止判别器过强）
- 输入：(batch, num_heads, seq_len, head_dim) 的 attention output

#### Lightweight Adapter（可选）
- **Mask-Aware 设计**：根据累积 mask 动态调整
- **分离式 Adapter**（可选）：
  - Vision Adapter：处理 vision tokens
  - Text Adapter：处理 question 和 answer tokens
- 使用 FiLM (Feature-wise Linear Modulation) 机制
- **可完全关闭**：设置 `use_adapter: false` 可关闭 Adapter，只使用 Pruner

### 训练 vs 推理

**训练时**：
- 使用 **post-softmax masking**（不做物理删除）
- 序列长度保持不变（576 个 vision tokens）
- 阶段 3 关闭 Gumbel noise，与推理对齐
- 计算 h_real 和 h_fake 用于对抗训练

**推理时**：
- **Prefill 阶段**：
  - 遍历所有层，在剪枝层计算 mask
  - **物理删除**被剪掉的 vision tokens
  - 更新 position_ids（保持原位置，不重新编号）
  - 更新 KV cache（删除对应位置）
- **Decode 阶段**：
  - 使用 Prefill 阶段保存的 mask
  - 在剪枝层应用 Adapter
  - 逐 token 生成

### 累积剪枝机制

剪枝是**累积的**，后续层只能在前面层保留的 tokens 上继续剪枝：

- **第一个剪枝层**（如 L4）：在所有 576 个 vision tokens 上决策
- **后续剪枝层**（如 L14, L24）：只能在前面层保留的 tokens 上继续剪枝
- **累积公式**：`new_cumulative_mask = old_cumulative_mask * current_mask`
- **非剪枝层**：应用累积 mask 的 post-softmax masking，不做新的剪枝决策

**示例**：
- L4 保留 30% → 173 tokens
- L14 在 173 tokens 上保留 70% → 121 tokens
- L24 在 121 tokens 上保留 80% → 97 tokens
- 最终保留率：97/576 ≈ 16.8%

## 配置说明

### 对抗训练模式

支持两种对抗训练模式：

1. **Discriminator 模式**（默认）：
   - 使用判别器进行对抗训练
   - 判别器学习区分 h_real 和 h_fake
   - Pruner 和 Adapter 学习欺骗判别器
   - 配置：`adversarial_mode: "discriminator"`

2. **MSE 模式**：
   - 直接使用 MSE 损失约束 h_real 和 h_fake 的一致性
   - 不需要判别器，训练更简单
   - 适合快速实验和消融研究
   - 配置：`adversarial_mode: "mse"`

### 关键超参数

```yaml
# 剪枝层
pruning_layers: [4, 14, 24]

# 剪枝目标
target_token_num: 64
sparsity_anneal_ratio: 0.3  # 前 30% 步数退火到目标

# 损失权重
task_loss_weight: 1.5
adv_loss_weight: 0.4
sparsity_weight: 0.5
```

## 使用方法

### 训练（DDP 分布式）

```bash
torchrun --nproc_per_node=4 main_acp_ddp.py --config configs/vision_token_pruning.yaml
```

### 评估

```bash
torchrun --nproc_per_node=4 eval_acp_ddp.py --config configs/vision_token_pruning.yaml --checkpoint outputs/checkpoints/checkpoint_final.pt
```

## 文件结构

```
├── main_acp_ddp.py              # DDP 训练脚本
├── eval_acp_ddp.py              # DDP 评估脚本
├── configs/
│   └── vision_token_pruning.yaml
├── method/models/
│   ├── layer_pruner_acp.py      # CrossAttentionPruner
│   ├── layer_discriminator.py   # LayerDiscriminator
│   ├── prunable_llava.py        # 可剪枝 LLaVA 模型
│   ├── prunable_llama_layer.py  # 可剪枝 LLaMA 层
│   └── adapter.py               # Adapter 模块
└── engine/
    ├── datas/                   # 数据集加载器
    └── configs/                 # 配置加载器
```

## 支持的数据集

- VQA v2（默认）
- GQA
- ScienceQA
- MME
- POPE

## 训练监控

日志输出示例：
```
Step 400 [Phase 2]: task_loss=1.23, adv_loss=0.45, sparsity_loss=0.02 (temp=0.30, noise=ON)
  Kept ratio: 22.15% (target: 22.22%) [L4=18.30%, L10=19.62%, L16=8.88%]
  Infer ratio (x>0): [L4=18.30%, L10=19.62%, L16=8.88%]
  Disc acc: 73.81% [L4=86%, L10=93%, L16=43%]
```

- `[Phase N]`: 当前训练阶段（hybrid 模式，1=探索期，2=稳定期，3=对齐期）
- `noise=ON/OFF`: 是否使用 Gumbel noise
- `Kept ratio`: 训练时累积保留率
- `Infer ratio`: 推理时累积保留率

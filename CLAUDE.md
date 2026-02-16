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

推理时始终使用 `mask = (logits > 0).float()`，这是 Gumbel-Sigmoid 的自然阈值（`sigmoid(0) = 0.5`）。

## 架构

### 核心组件

```
图像 → 视觉编码器 → 带剪枝层的 LLM → 输出（Fake）
                         ↓
          不带剪枝的 LLM → 输出（Real）
                         ↓
          Discriminator 判断 Real vs Fake
```

#### CrossAttentionPruner
- 可学习的 pruning queries 通过 cross-attention 评估 vision tokens
- 残差设计：`keep_logits = baseline + delta + bias`
- `keep_bias` 初始化为 2.0，初始保留更多 token

#### LayerDiscriminator
- 每个 answer token 独立判别
- 支持谱归一化和随机重初始化

#### Lightweight Adapter
- Mask-Aware FiLM adapter
- 根据剪枝 pattern 动态调整

## 配置说明

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

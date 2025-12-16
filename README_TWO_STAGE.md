# 两阶段Vision Token剪枝系统

基于GAN对抗训练的多模态大模型视觉token剪枝方法，实现LLaVA-1.5-7B的高效推理。

## 🎯 核心特性

- ✅ **两阶段剪枝架构**
  - Token Merge（LLM输入前）: 576 → 288 tokens
  - Layer-wise Pruning（LLM内部3层）: 渐进式剪枝

- ✅ **端到端可训练**
  - Gumbel-Top-K可微分采样
  - Temperature Annealing动态调节
  - GAN对抗训练保持性能

- ✅ **问题感知剪枝**
  - Cross-Attention机制
  - 基于问题内容动态决策

## 🚀 快速启动

### 1. 环境准备

```bash
cd /data/users/zjw/workspace/domain-adaptation-token-pruning
```

### 2. 直接运行

```bash
python main.py
```

### 3. 指定GPU

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python main.py
```

## 📊 系统架构

```
Input (Image + Question)
    ↓
CLIP Vision Encoder (576 tokens)
    ↓
Token Merger (Learnable) → 288 tokens
    ↓
Multi-Modal Projector
    ↓
LLM Layer 10: Pruner Head 1
LLM Layer 20: Pruner Head 2
LLM Layer 31: Pruner Head 3
    ↓
Discriminator (Real vs Fake)
    ↓
Multi-Objective Loss
```

## ⚙️ 配置文件

主配置文件: `configs/vision_token_pruning.yaml`

### 关键参数

```yaml
# Token Merger
merger_type: "simple"           # 或 "question_aware"
merge_ratio: 0.5                # 保留比例
target_merge_tokens: 288        # 目标token数

# Layer Pruners
pruning_layers: [10, 20, 31]    # 剪枝层位置
pruner_d_internal: 512          # 内部维度
pruner_num_heads: 4             # Attention头数

# Temperature Annealing
temperature: 1.0                # 初始温度
temperature_min: 0.1            # 最终温度
temperature_anneal_rate: 0.5    # Annealing比例

# Loss Weights
adv_loss_weight: 1.0            # 对抗损失
task_loss_weight: 25.0          # 任务损失
merge_sparsity_weight: 1e-4     # 稀疏性损失
```

## 📈 预期效果

| 指标 | 数值 |
|------|------|
| Token减少 | 60-70% |
| FLOPs减少 | 60-70% |
| 准确率下降 | < 3% |
| BLEU下降 | < 5% |

### Token数量变化

```
676 tokens (原始)
  ↓ Token Merge
388 tokens (~57%)
  ↓ Layer 10
~300 tokens
  ↓ Layer 20
~250 tokens
  ↓ Layer 31
~200 tokens (~30%)
```

## 📁 项目结构

```
.
├── configs/
│   └── vision_token_pruning.yaml    # 配置文件
├── method/
│   ├── models/
│   │   ├── token_merger.py          # Token合并模块
│   │   ├── layer_pruner.py          # 多层剪枝模块
│   │   └── discriminator.py         # 判别器
│   ├── training.py                  # 训练逻辑
│   ├── evaluation.py                # 评估逻辑
│   └── utils.py                     # 工具函数
├── main.py                          # 主入口
├── QUICKSTART.md                    # 快速启动指南
├── IMPLEMENTATION_SUMMARY.md        # 详细实现文档
└── CLAUDE.md                        # 完整技术规范
```

## 🔧 常见调整

### 调整剪枝强度

```yaml
# 更激进剪枝（更少token）
merge_ratio: 0.4
target_merge_tokens: 230

# 更保守剪枝（更多token）
merge_ratio: 0.6
target_merge_tokens: 346
```

### 切换Merger类型

```yaml
# 简单版（不依赖问题）
merger_type: "simple"

# 问题感知版（VQA推荐）
merger_type: "question_aware"
```

### 调整学习率

```yaml
optimizers:
  token_merger:
    lr: "2e-05"      # Token Merger
  layer_pruners:
    lr: "2e-05"      # Layer Pruners
  discriminator:
    lr: "5e-04"      # Discriminator
```

## 📚 文档索引

- **[QUICKSTART.md](QUICKSTART.md)** - 详细启动指南和故障排查
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - 完整实现细节
- **[CLAUDE.md](CLAUDE.md)** - 技术规范和设计文档

## 🔍 核心技术

### 1. Gumbel-Top-K Selection

可微分的离散token选择：
```python
gumbel_noise = -log(-log(uniform(0,1)))
perturbed_logits = importance_logits + gumbel_noise
top_k_indices = torch.topk(perturbed_logits, k=M)[1]
```

### 2. Soft Assignment Merge

温度控制的软合并：
```python
similarity = Q @ K.T / sqrt(d)
merge_weights = softmax(similarity / temperature)
merged = merge_weights.T @ vision_features
```

### 3. Cross-Attention Pruning

问题感知的重要性评估：
```python
attended_V = cross_attn(
    query=vision_tokens,
    key=question_tokens,
    value=question_tokens
)
keep_mask = mask_predictor(attended_V)
```

### 4. Temperature Annealing

训练过程动态调节：
```
Early: temp=1.0 → Soft (探索)
Late:  temp=0.1 → Hard (确定性)
```

## 🎓 引用

如果使用本项目，请引用：

```bibtex
@software{vision_token_pruning_2025,
  title={Two-Stage Vision Token Pruning with GAN},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/vision-token-pruning}
}
```

## 📝 许可证

MIT License

---

**状态**: ✅ 完全实现，可直接运行

**最后更新**: 2025-12-15

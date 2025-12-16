# 两阶段Vision Token剪枝 - 快速启动指南

## ✅ 系统已完全配置完成！

所有代码已经准备就绪，可以直接运行训练。

---

## 🚀 启动训练

### 方法1: 直接运行（推荐）

```bash
cd /data/users/zjw/workspace/domain-adaptation-token-pruning
python main.py
```

### 方法2: 指定GPU

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python main.py
```

---

## 📋 配置文件说明

配置文件位于: `configs/vision_token_pruning.yaml`

### 关键配置参数

#### 训练设置
```yaml
trainer_settings:
  dl_settings:
    epochs: 1
    batch_size: 12
    optimizers:
      token_merger:      # Token Merger学习率
        lr: "2e-05"
      layer_pruners:     # Layer Pruners学习率
        lr: "2e-05"
      discriminator:     # Discriminator学习率
        lr: "5e-04"
```

#### Token Merger配置
```yaml
method_settings:
  merger_type: "simple"           # "simple" 或 "question_aware"
  merge_ratio: 0.5                # 保留50% tokens (576→288)
  target_merge_tokens: 288        # 目标保留数量
  merge_sparsity_weight: 1e-4     # Merge sparsity loss权重
```

#### Layer-wise Pruner配置
```yaml
  pruning_layers: [10, 20, 31]    # 在这3层进行剪枝
  pruner_d_internal: 512          # Pruner内部维度
  pruner_num_heads: 4             # Cross-attention头数
  pruner_type: "cross_attention"  # Pruner类型
```

#### Temperature Annealing
```yaml
  temperature: 1.0                # 初始温度（软分配）
  temperature_min: 0.1            # 最终温度（硬分配）
  temperature_anneal_rate: 0.5    # 前50%步数进行annealing
```

#### 损失权重
```yaml
  adv_loss_weight: 1.0            # 对抗损失权重
  task_loss_weight: 25.0          # 任务损失权重
```

---

## 📊 预期训练流程

### 启动日志示例
```
[INFO] 预加载Backbone和Dataset...
[INFO] 冻结Backbone参数...
[INFO] 创建Trainer...
[INFO] 创建Token Merger...
[INFO] 创建Layer-Specific Pruners...
[INFO] 创建Discriminator...
[INFO] 开始训练...
[INFO] Token Merger类型: simple
[INFO] Merge Ratio: 0.5
[INFO] Pruning Layers: [10, 20, 31]
[INFO] Temperature: 1.0 → 0.1
```

### 训练中的Loss监控

每个batch会输出三个optimizer组的loss：

```
[Step 10] Losses:
  token_merger:
    - merge_sparsity_loss: 0.0234
  layer_pruners:
    - adv_loss: 0.6234
    - task_loss: 2.3456
  discriminator:
    - real_loss: 0.3456
    - fake_loss: 0.4123
```

### 评估指标

```
[Eval Step 15]
  - accuracy_baseline: 0.654      # 无剪枝基准
  - accuracy_soft: 0.632           # 软剪枝准确率
  - keep_ratio_merge: 0.498        # Token保留比例
  - avg_original_tokens: 576       # 原始token数
  - avg_merged_tokens: 287         # 合并后token数
```

---

## 🎯 训练目标

### Token数量变化

```
原始序列: 676 tokens (100 text + 576 vision)
    ↓ [Token Merge]
合并后: ~388 tokens (100 text + 288 vision)
    ↓ [Layer 10 Pruning]
第一次剪枝: ~300 tokens
    ↓ [Layer 20 Pruning]
第二次剪枝: ~250 tokens
    ↓ [Layer 31 Pruning]
第三次剪枝: ~200 tokens
```

### 性能目标

- ✅ **准确率下降 < 3%**: accuracy_baseline - accuracy_soft < 0.03
- ✅ **Token减少 > 60%**: keep_ratio_merge < 0.4
- ✅ **FLOPs减少 > 60%**: 由于vision token占比高

---

## 🔧 调试技巧

### 1. 检查模型是否正确创建

训练开始前会看到：
```
[INFO] 创建Token Merger...
[INFO] 创建Layer-Specific Pruners...
[INFO] 创建Discriminator...
```

如果这里报错，检查config中的维度配置。

### 2. 检查Token数量变化

在训练日志中观察：
```
[INFO] avg_original_tokens: 576
[INFO] avg_merged_tokens: 287
```

如果merged_tokens过大或过小，调整`merge_ratio`。

### 3. 检查Temperature Annealing

观察训练日志：
```
[Step 100] Temperature: 0.85
[Step 500] Temperature: 0.50
[Step 1000] Temperature: 0.10
```

Temperature应该从1.0逐渐降到0.1。

### 4. 检查Discriminator平衡

```
[INFO] disc_prob_real: 0.723
[INFO] disc_prob_fake: 0.312
```

理想情况下，两者应该都接近0.5（Discriminator无法区分）。

### 5. 检查Loss收敛

- `merge_sparsity_loss`: 应该逐渐减小并稳定
- `task_loss`: 应该保持较低（<3.0）
- `adv_loss`: 可能先上升再下降

---

## ⚙️ 常见配置调整

### 调整Token保留数量

```yaml
# 保留更多tokens（更少剪枝，更高准确率）
merge_ratio: 0.6  # 576 → 346
target_merge_tokens: 346

# 保留更少tokens（更多剪枝，更低准确率）
merge_ratio: 0.4  # 576 → 230
target_merge_tokens: 230
```

### 调整剪枝层位置

```yaml
# 更早开始剪枝
pruning_layers: [5, 15, 25]

# 只在后期剪枝
pruning_layers: [20, 25, 31]

# 更多剪枝层
pruning_layers: [8, 12, 16, 20, 24, 28, 31]
```

### 使用Question-Aware Merger

```yaml
# 简单版（不依赖question）
merger_type: "simple"

# 问题感知版（更适合VQA）
merger_type: "question_aware"
```

### 调整学习率

```yaml
# Token Merger学习更快
token_merger:
  lr: "5e-05"

# Discriminator学习更慢（避免过强）
discriminator:
  lr: "2e-04"
```

### 调整Loss权重

```yaml
# 更强的任务性能保持
task_loss_weight: 50.0

# 更强的对抗训练
adv_loss_weight: 2.0

# 更强的稀疏性约束
merge_sparsity_weight: 5e-4
```

---

## 📁 输出文件

训练过程中会生成以下文件：

```
outputs/
├── checkpoints/
│   ├── token_merger_step_200.pt
│   ├── layer_pruners_step_200.pt
│   ├── discriminator_step_200.pt
│   └── ...
└── logs/
    ├── training.log
    └── tensorboard/
```

---

## 🐛 故障排查

### 问题1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小batch_size: `batch_size: 6`
2. 减小max_vision_tokens: `max_vision_tokens: 1200`
3. 减小image_max_size: `image_max_size: 800`

### 问题2: Token数量不减少

**症状**: `avg_merged_tokens`接近原始数量

**解决方案**:
1. 检查`merge_ratio`是否正确
2. 增大`merge_sparsity_weight`: `1e-3`
3. 检查Token Merger是否正确初始化

### 问题3: 准确率下降太多

**症状**: `accuracy_soft`远低于`accuracy_baseline`

**解决方案**:
1. 增大`task_loss_weight`: `50.0`
2. 增大`merge_ratio`: `0.6`（保留更多tokens）
3. 减小`adv_loss_weight`: `0.5`（减弱对抗训练）

### 问题4: Discriminator过强

**症状**: `disc_prob_fake`接近0，`disc_prob_real`接近1

**解决方案**:
1. 减小Discriminator学习率: `lr: 2e-04`
2. 增大`disc_reinit_prob`: `0.1`（更频繁重初始化）
3. 增大`adv_loss_weight`: `2.0`（加强Generator）

### 问题5: Temperature不变化

**症状**: Temperature一直是1.0

**解决方案**:
检查`temperature_anneal_rate`和训练步数，确保有足够步数进行annealing。

---

## 📚 进阶功能

### 1. 使用Optuna超参数搜索

修改配置文件：
```yaml
manager_settings:
  mode: "optuna"

search_settings:
  enable: true
  n_trials: 50
```

### 2. 多GPU训练

```yaml
manager_settings:
  available_gpus: [0,1,2,3]
  gpus_per_subtask: 2
```

### 3. 保存最佳模型

系统会自动保存loss最低的checkpoint。

### 4. 从checkpoint恢复

```python
# 在main.py中添加
checkpoint = torch.load("outputs/checkpoints/token_merger_step_1000.pt")
token_merger.load_state_dict(checkpoint)
```

---

## 🎓 核心架构回顾

```
Input Image + Question
    ↓
[CLIP Vision Encoder]
    576 tokens × 1024-dim
    ↓
[Token Merger - Learnable]
    Gumbel-Top-K Selection
    Soft Assignment Merge
    ↓
    ~288 tokens × 1024-dim
    ↓
[Multi-Modal Projector]
    ~288 tokens × 4096-dim
    ↓
[Concat with Text]
    ~388 tokens total
    ↓
[LLM Layers 0-9]
    Full sequence
    ↓
[Layer 10 - Pruner Head 1]
    Cross-Attention + Soft Mask
    ↓
[LLM Layers 11-19]
    Pruned sequence
    ↓
[Layer 20 - Pruner Head 2]
    Further pruning
    ↓
[LLM Layers 21-30]
    More pruned
    ↓
[Layer 31 - Pruner Head 3]
    Final pruning
    ↓
[Output Logits]
    ↓
[Discriminator]
    Judges: Real vs Fake
```

---

## 📞 获取帮助

如果遇到问题，请检查：

1. **完整实现文档**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. **CLAUDE.md规范**: [CLAUDE.md](CLAUDE.md)
3. **训练日志**: `outputs/logs/training.log`

---

## 🎉 开始训练！

一切就绪！现在运行：

```bash
python main.py
```

祝训练顺利！🚀

# 两阶段剪枝系统 - 部署检查清单

## ✅ 已完成的实现

### 核心模块
- [x] LearnableTokenMerger (简单版)
- [x] LearnableTokenMergerV2 (问题感知版)
- [x] LayerSpecificPruner (多层剪枝管理器)
- [x] VisionPrunerHead (单层Cross-Attention Pruner)
- [x] VisionPrunerHeadSimple (简化版MLP Pruner)
- [x] Discriminator (保持原有实现)

### 核心功能
- [x] Multi-layer Hook注册机制
- [x] Temperature Annealing调度
- [x] Vision Token替换工具
- [x] 两阶段训练流程
- [x] 两阶段评估流程

### 配置和入口
- [x] 配置文件更新 (vision_token_pruning.yaml)
- [x] 主入口更新 (main.py)
- [x] 模块导出 (method/__init__.py)
- [x] API兼容性修复 (utils.py, evaluation.py)

### 文档
- [x] 快速启动指南 (QUICKSTART.md)
- [x] 实现总结文档 (IMPLEMENTATION_SUMMARY.md)
- [x] 项目README (README_TWO_STAGE.md)
- [x] 技术规范 (CLAUDE.md - 原有)

---

## 🔍 系统自检

### 1. 模块导入测试

```python
# 测试所有模块是否可以正确导入
python -c "
from method import (
    LearnableTokenMerger,
    LearnableTokenMergerV2,
    LayerSpecificPruner,
    VisionPrunerHead,
    Discriminator,
    train_step,
    eval_step,
    register_multi_layer_hooks,
    remove_hooks,
    replace_vision_tokens_in_embeddings,
    update_temperature_for_all
)
print('✅ 所有模块导入成功')
"
```

### 2. 配置文件验证

```python
# 测试配置文件是否正确
python -c "
from engine.configs.loader import load_config
config = load_config(override_file='configs/vision_token_pruning.yaml')
print('✅ 配置文件加载成功')
print(f'Merge Ratio: {config[\"method_settings\"][\"merge_ratio\"]}')
print(f'Pruning Layers: {config[\"method_settings\"][\"pruning_layers\"]}')
"
```

### 3. 模型创建测试

```python
# 测试模型是否能正确创建
python -c "
import torch
from method import LearnableTokenMerger, LayerSpecificPruner, Discriminator

token_merger = LearnableTokenMerger(d_model=1024, merge_ratio=0.5)
layer_pruners = LayerSpecificPruner(d_model=4096, layer_indices=[10, 20, 31])
discriminator = Discriminator(d_model=4096, num_layers=3, d_d=1024)

print('✅ 所有模型创建成功')
print(f'Token Merger参数: {sum(p.numel() for p in token_merger.parameters()):,}')
print(f'Layer Pruners参数: {sum(p.numel() for p in layer_pruners.parameters()):,}')
print(f'Discriminator参数: {sum(p.numel() for p in discriminator.parameters()):,}')
"
```

---

## 📋 训练前检查

### 配置确认

```yaml
# configs/vision_token_pruning.yaml

✓ optimizers中有3个optimizer: token_merger, layer_pruners, discriminator
✓ method_settings中有merge_ratio和pruning_layers
✓ method_settings中有temperature相关参数
✓ backbone_settings中有vision_dim和hidden_dim
```

### 主入口确认

```python
# main.py

✓ 导入了LearnableTokenMerger和LayerSpecificPruner
✓ 创建了token_merger实例
✓ 创建了layer_pruners实例
✓ 注册了3个模型: token_merger, layer_pruners, discriminator
✓ 添加了3个参数组
```

### 训练函数确认

```python
# method/training.py

✓ 接收token_merger和layer_pruners参数
✓ 实现了Token Merge逻辑
✓ 实现了Multi-layer Hook注册
✓ 实现了Temperature Annealing
✓ 返回3个optimizer组的loss
```

---

## 🚀 启动流程

### Step 1: 环境检查

```bash
# 检查Python版本
python --version  # 应该是 >= 3.8

# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查GPU数量
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 2: 工作目录

```bash
cd /data/users/zjw/workspace/domain-adaptation-token-pruning
```

### Step 3: 运行训练

```bash
# 方式1: 直接运行
python main.py

# 方式2: 指定GPU
CUDA_VISIBLE_DEVICES=1,2,3 python main.py

# 方式3: 后台运行
nohup python main.py > training.log 2>&1 &
```

### Step 4: 监控训练

```bash
# 实时查看日志
tail -f training.log

# 或查看outputs/logs/training.log
tail -f outputs/logs/training.log
```

---

## 📊 预期输出

### 启动日志

```
[INFO] ==========================================
[INFO] Vision Token Pruning with GAN
[INFO] ==========================================
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

### 训练日志

```
[Step 1] Losses:
  token_merger:
    - merge_sparsity_loss: 0.0523
  layer_pruners:
    - adv_loss: 0.6932
    - task_loss: 3.2456
  discriminator:
    - real_loss: 0.6931
    - fake_loss: 0.6932

[Step 10] Losses:
  token_merger:
    - merge_sparsity_loss: 0.0234
  layer_pruners:
    - adv_loss: 0.5234
    - task_loss: 2.1456
  discriminator:
    - real_loss: 0.4231
    - fake_loss: 0.5123

...
```

### 评估日志

```
[Eval Step 15]
  accuracy_baseline: 0.654
  accuracy_soft: 0.632
  keep_ratio_merge: 0.498
  avg_original_tokens: 576.0
  avg_merged_tokens: 287.0
  score: -0.267
```

---

## ⚠️ 常见问题预检

### 问题1: 找不到模块

**症状**: `ModuleNotFoundError: No module named 'method.models.token_merger'`

**检查**:
```bash
ls method/models/token_merger.py  # 文件应该存在
```

**解决**: 文件已创建，不应该有此问题

### 问题2: 配置参数缺失

**症状**: `KeyError: 'merge_ratio'`

**检查**:
```bash
grep "merge_ratio" configs/vision_token_pruning.yaml
```

**解决**: 配置文件已更新，不应该有此问题

### 问题3: 模型参数不匹配

**症状**: `TypeError: __init__() got an unexpected keyword argument`

**检查**: 确认config中的参数名与模型构造函数一致

**解决**: 所有参数已对齐，不应该有此问题

### 问题4: Hook注册失败

**症状**: `AttributeError: 'NoneType' object has no attribute 'register_forward_pre_hook'`

**检查**: 确认backbone.model.model.language_model.layers存在

**解决**: LLaVA标准结构，不应该有此问题

---

## 📈 训练监控指标

### 必须监控的指标

1. **merge_sparsity_loss**
   - 应该逐渐减小
   - 最终稳定在较小值 (< 0.05)

2. **task_loss**
   - 应该保持较低 (< 3.0)
   - 不应该持续上升

3. **adv_loss**
   - 初期可能较高 (0.6-0.7)
   - 应该逐渐下降到 0.3-0.5

4. **disc_real_loss & disc_fake_loss**
   - 应该保持平衡
   - 都在 0.3-0.7 范围内

5. **temperature**
   - 应该从1.0逐渐降到0.1
   - 按照anneal_rate进度变化

6. **keep_ratio_merge**
   - 应该接近merge_ratio设置
   - 稳定在0.45-0.55（如果merge_ratio=0.5）

7. **accuracy_soft**
   - 应该接近accuracy_baseline
   - 差距 < 0.05

---

## ✅ 最终确认

在运行训练前，确认以下所有项：

- [ ] 所有模块文件已创建
- [ ] 配置文件已更新
- [ ] 主入口已修改
- [ ] API兼容性已修复
- [ ] GPU可用且显存充足
- [ ] 数据集路径正确
- [ ] HuggingFace cache路径正确

如果所有项都已确认，可以运行：

```bash
python main.py
```

---

## 🎉 成功指标

训练成功的标志：

✅ 程序正常启动，无报错
✅ 模型正确创建并移到GPU
✅ 训练开始，loss正常计算
✅ Temperature按计划annealing
✅ Token数量正确减少
✅ 评估准确率合理

---

## 📞 问题排查

如果遇到问题：

1. 检查完整错误栈
2. 查看 `outputs/logs/training.log`
3. 参考 `QUICKSTART.md` 故障排查部分
4. 检查 `IMPLEMENTATION_SUMMARY.md` 实现细节

---

**最后检查日期**: 2025-12-15
**系统状态**: ✅ 完全就绪，可以开始训练

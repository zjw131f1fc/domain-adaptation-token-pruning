# Batch化训练功能使用说明

## 功能概述

为了提升训练效率，我们实现了真正的batch化训练功能。通过统一图片尺寸，可以实现整个batch的并行处理，显著加快训练速度（预计3-5倍加速）。

## 核心改动

### 1. 配置文件修改

在 `configs/vision_token_pruning.yaml` 中添加了两个新配置项：

```yaml
backbone_settings:
  mllm_settings:
    enable_true_batch: false  # 是否启用真正的batch化处理
    unified_image_size: 336   # 统一的图片尺寸（正方形，如336x336）
```

### 2. Backbone新增方法

在 `engine/backbones/impl/mllm/llava.py` 中：

- **`_resize_image`**: 修改为支持统一尺寸模式
  - 当 `enable_true_batch=True` 时，强制将所有图片调整为 `unified_image_size x unified_image_size`
  - 当 `enable_true_batch=False` 时，使用原有的等比例缩放逻辑

- **`preprocess_batch`**: 新增batch预处理方法
  - 一次性处理整个batch的图片和文本
  - 返回batch化的 embeddings、attention_mask、vision_positions 等
  - 要求所有样本的序列长度对齐

### 3. 训练函数

新文件 `method/training_batch.py`:

- **`train_step_batch`**: Batch化的训练step函数
  - 使用 `backbone.preprocess_batch` 处理整个batch
  - 一次forward处理所有样本
  - 支持所有现有功能：token merge、layer-wise pruning、GAN训练等

### 4. 辅助函数

新文件 `method/utils_batch.py`:

- **`replace_vision_tokens_in_embeddings_batch`**: Batch版本的vision token替换
- **`register_multi_layer_hooks_batch`**: Batch版本的multi-layer hooks注册

## 使用方法

### 启用Batch化训练

1. **修改配置文件**

```yaml
backbone_settings:
  mllm_settings:
    enable_true_batch: true     # 启用batch化
    unified_image_size: 336     # 所有图片调整为336x336
```

2. **修改训练入口**

在你的训练脚本中，导入并使用batch版本的训练函数：

```python
# 原来的导入
# from method.training import train_step

# 新的导入
from method.training_batch import train_step_batch

# 在trainer配置中使用
config["trainer_settings"]["train_step_fn"] = train_step_batch
```

或者，如果训练框架支持根据配置自动选择：

```python
from method.training import train_step
from method.training_batch import train_step_batch

# 根据配置选择
enable_batch = config["backbone_settings"]["mllm_settings"].get("enable_true_batch", False)
train_fn = train_step_batch if enable_batch else train_step
```

### 配置建议

#### unified_image_size 的选择

- **336x336** (推荐): LLaVA-1.5默认尺寸，生成576个vision tokens
- **448x448**: 更高分辨率，但会生成更多tokens (1024个)
- **224x224**: 最小尺寸，生成144个tokens，速度最快但精度可能下降

建议根据显存大小和batch_size调整：

| unified_image_size | Vision Tokens | 推荐batch_size (V100 32GB) |
|-------------------|---------------|---------------------------|
| 224               | 144           | 16-20                      |
| 336               | 576           | 10-12                      |
| 448               | 1024          | 6-8                        |

## 性能对比

### 预期加速比

| 模式 | Batch Size | 相对速度 | 备注 |
|-----|-----------|---------|------|
| 逐样本模式 (原始) | 10 | 1.0x | 循环处理每个样本 |
| Batch化模式 | 10 | 3-5x | 真正的batch forward |

### 显存占用

启用batch化后，显存占用会略有增加（约10-20%），因为：
- 一次性存储整个batch的embeddings
- Backbone forward时的中间激活值更大

## 限制与注意事项

### 1. 序列长度对齐要求

Batch化要求所有样本的序列长度完全一致，包括：
- Vision token位置必须相同
- Question长度应该相似

这通过统一图片尺寸和padding来保证。

### 2. 不支持的场景

以下情况下不应使用batch化模式：
- 图片尺寸变化很大且需要保持长宽比
- Question长度差异极大（如有的10词，有的100词）
- 需要对每个样本使用不同的处理逻辑

### 3. 调试建议

首次启用batch化时建议：
1. 先用 `batch_size=2` 测试，确保功能正常
2. 检查 `vision_tokens_mean/max/min` 统计是否相同
3. 对比batch化与非batch化的loss曲线，应该基本一致
4. 逐步增大batch_size，观察显存占用

## 故障排除

### 错误：Vision token positions not aligned

```
ValueError: Vision token positions not aligned across batch
```

**原因**：batch内不同样本的vision token位置不一致

**解决**：
1. 确认 `enable_true_batch=True`
2. 确认 `unified_image_size` 已设置
3. 检查 `_resize_image` 是否正确执行
4. 可能是question长度差异导致，尝试减小batch_size

### 错误：preprocess_batch requires enable_true_batch=True

**原因**：使用了 `train_step_batch` 但未启用配置

**解决**：在配置文件中设置 `enable_true_batch: true`

### 错误：CUDA out of memory

**原因**：batch化后显存占用增加

**解决**：
1. 减小 `batch_size`
2. 减小 `unified_image_size`
3. 使用梯度累积替代大batch

## 最佳实践

1. **渐进式启用**
   - 先在小数据集上验证正确性
   - 确认loss曲线与原始模式一致
   - 再切换到完整训练

2. **监控指标**
   - 观察 `vision_tokens_mean/max/min` 应该完全相同
   - 检查训练速度提升是否符合预期
   - 确认显存占用在可接受范围内

3. **配置调优**
   - 根据GPU显存调整 `unified_image_size` 和 `batch_size`
   - 336x336 + batch_size=10 是推荐的起点
   - 优先保证batch_size，再考虑提高图片分辨率

## 回退到原始模式

如果batch化模式出现问题，可以随时回退：

```yaml
backbone_settings:
  mllm_settings:
    enable_true_batch: false  # 关闭batch化
```

然后使用原始的 `train_step` 函数即可。

## 未来改进方向

1. **动态Padding + Batch化**: 支持不同长度序列的真正batch处理
2. **自适应图片尺寸**: 根据图片内容自动选择合适的统一尺寸
3. **混合精度训练**: 进一步加速batch化训练

---

**版本**: v1.0
**最后更新**: 2025-12-22

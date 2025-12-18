# Attention Residual Implementation

## 概述

实现了**可配置的attention residual功能**：token pruner的输出可以残差连接到text token对vision token的平均attention score上。

## 核心思想

传统pruner仅基于cross-attention预测keep/drop决策：
```
keep_logits = mask_predictor(cross_attn(vision, question))
```

现在可以添加**真实的LLM attention**作为额外信号：
```
keep_logits = mask_predictor(...) + weight * text_to_vision_attention
```

其中`text_to_vision_attention`是从LLM当前层的**真实attention weights**中提取的：
- 计算所有text tokens对每个vision token的attention
- 跨heads和text tokens平均，得到每个vision token被关注的程度
- 高attention → 更可能保留

## 实现细节

### 1. VisionPrunerHead修改

**文件**: [method/models/layer_pruner.py](method/models/layer_pruner.py)

**新增参数**:
```python
class VisionPrunerHead(nn.Module):
    def __init__(
        self,
        ...,
        use_attn_residual: bool = False,          # 是否启用
        attn_residual_weight: float = 0.5,         # residual权重
        learnable_attn_weight: bool = False        # 权重是否可学习
    ):
```

**Forward修改**:
```python
def forward(
    self,
    vision_hidden,
    question_embeddings,
    use_gumbel=True,
    text_to_vision_attn: Optional[torch.Tensor] = None  # 新增参数
):
    ...
    keep_logits = mask_predictor(...)

    # 残差连接
    if self.use_attn_residual and text_to_vision_attn is not None:
        keep_logits = keep_logits + self.attn_residual_weight * text_to_vision_attn

    soft_mask = gumbel_softmax(keep_logits)
    return soft_mask
```

### 2. Hook机制修改

**文件**: [method/utils.py](method/utils.py)

#### 2.1 Attention捕获

在每个剪枝层的`self_attn`上注册forward hook：

```python
def create_attn_capture_hook(storage):
    def attn_hook(module, args, output):
        # LlamaAttention.forward返回: (attn_output, attn_weights)
        attn_output, attn_weights = output
        storage['attn_weights'] = attn_weights  # 捕获
        return output
    return attn_hook

# 注册到 target_layer.self_attn
attn_handle = target_layer.self_attn.register_forward_hook(
    create_attn_capture_hook(attention_storage)
)
```

**关键发现**: LlamaAttention.forward **确实返回attn_weights**（即使LlamaDecoderLayer丢弃了它）！

#### 2.2 Text→Vision Attention计算

在modifier函数（forward pre-hook）中：

```python
def modifier(hidden_states, attention_mask):
    # Step 1: 获取捕获的attention weights
    attn_weights = attention_storage['attn_weights']  # (batch, heads, seq, seq)

    # Step 2: 提取text→vision attention
    text_indices = list(range(0, v_start)) + list(range(v_end+1, seq_len))
    vision_indices = list(range(v_start, v_end+1))

    text_to_vision = attn_weights[:, :, text_indices, :][:, :, :, vision_indices]
    # (batch, heads, n_text, n_vision)

    # Step 3: 平均（跨heads和text tokens）
    text_to_vision_attn = text_to_vision.mean(dim=(1, 2))  # (batch, n_vision)

    # Step 4: 传递给pruner
    soft_mask = pruner(vision_hidden, question_embeddings,
                      text_to_vision_attn=text_to_vision_attn)
    ...
```

### 3. 配置文件

**文件**: [configs/vision_token_pruning.yaml](configs/vision_token_pruning.yaml)

```yaml
method_settings:
  # Attention Residual配置
  use_attn_residual: false  # 是否启用（默认关闭，向后兼容）
  attn_residual_weight: 0.5  # Residual权重
  learnable_attn_weight: false  # 权重是否可学习
```

**启用方式**:
```yaml
use_attn_residual: true       # 开启功能
attn_residual_weight: 0.5     # 固定权重0.5
learnable_attn_weight: false  # 固定权重（不优化）
```

**可学习权重**:
```yaml
use_attn_residual: true
attn_residual_weight: 0.5     # 初始值
learnable_attn_weight: true   # 作为nn.Parameter训练
```

### 4. 训练/评估更新

**文件**:
- [method/training.py](method/training.py)
- [method/evaluation.py](method/evaluation.py)

```python
# 读取配置
use_attn_residual = config["method_settings"].get("use_attn_residual", False)

# 传递给hook注册
handles = register_multi_layer_hooks(
    backbone,
    layer_pruners,
    vision_positions,
    question_embeddings,
    mask_collector=pruning_masks,
    use_attn_residual=use_attn_residual  # 新增参数
)
```

## 测试结果

运行 `python test_attn_residual.py`:

```
✓ Test 1: VisionPrunerHead支持attention residual
  - 不使用residual: mean=0.8808
  - 使用residual: mean=0.8734
  - 差异: 0.040191（residual确实生效）

✓ Test 2: Text→vision attention提取逻辑正确
  - Input: (1, 32, 400, 400) attention weights
  - Output: (1, 288) averaged text→vision attention
  - 范围: [0.0024, 0.0026]

✓ Test 3: Hook机制能够捕获attention weights
  - 捕获前: storage['attn_weights'] is None
  - 捕获后: shape = (1, 8, 100, 100)
  - 捕获正确: True
```

所有测试通过✓

## 技术亮点

### 1. 向后兼容

- 默认`use_attn_residual=False`，行为与原实现完全一致
- 只有显式启用才会激活新功能

### 2. 零运行时开销（未启用时）

```python
if use_attn_residual:
    # 只有启用时才注册attention hook
    attn_handle = target_layer.self_attn.register_forward_hook(...)
else:
    # 不注册hook，零开销
```

### 3. 灵活的权重控制

- **固定权重**: `learnable_attn_weight=False` → 注册为buffer，不参与优化
- **可学习权重**: `learnable_attn_weight=True` → nn.Parameter，随pruner一起训练

### 4. 正确的梯度流

```python
# attention residual的梯度会反向传播到pruner
keep_logits = base_logits + weight * attention
soft_mask = gumbel_softmax(keep_logits)
loss.backward()  # ✓ 梯度会流向base_logits和weight（如果可学习）
```

## 使用建议

### 实验1: 固定权重探索

```yaml
use_attn_residual: true
attn_residual_weight: 0.3  # 尝试 0.1, 0.3, 0.5, 1.0
learnable_attn_weight: false
```

观察不同权重下的剪枝效果。

### 实验2: 可学习权重

```yaml
use_attn_residual: true
attn_residual_weight: 0.5  # 初始值
learnable_attn_weight: true  # 让模型自己学习最优权重
```

训练后检查`pruner.attn_residual_weight.item()`的最终值。

### 实验3: 与现有方法对比

1. **Baseline**: `use_attn_residual=false`
2. **+ Attention**: `use_attn_residual=true, attn_residual_weight=0.5`

比较准确率和token保留率。

## 理论依据

### 为什么这样做有用？

1. **Cross-attention的局限性**:
   - Pruner的cross-attention是**新学习的、独立的**attention
   - 可能与LLM内部的真实attention模式不一致

2. **真实attention的优势**:
   - 来自LLM当前层，反映**真实的信息流**
   - 如果某个vision token被text高度关注 → 说明LLM认为它重要
   - 这是一个**数据驱动的先验**

3. **Residual的作用**:
   - 将**学习的决策**（cross-attention）与**观测的信号**（真实attention）结合
   - 类似于人类决策：既有理性分析（cross-attn），也参考经验数据（real attn）

## 潜在改进方向

### 1. Layer-specific权重

每层使用不同的residual weight：

```python
self.attn_residual_weights = nn.Parameter(
    torch.tensor([0.5] * len(layer_indices))  # 每层一个权重
)
```

### 2. 非线性融合

当前是线性加权，可以尝试：

```python
# 门控机制
gate = sigmoid(learnable_gate)
keep_logits = gate * base_logits + (1 - gate) * attention
```

### 3. 多尺度attention

不仅用当前层，还用前几层的attention：

```python
# 捕获layer-1, layer-2的attention
attn_history = [attn_layer_i, attn_layer_i_minus_1, ...]
avg_attn = stack(attn_history).mean(dim=0)
```

## 文件修改清单

✓ [method/models/layer_pruner.py](method/models/layer_pruner.py)
  - VisionPrunerHead.__init__: 添加3个参数
  - VisionPrunerHead.forward: 添加text_to_vision_attn参数和residual逻辑
  - LayerSpecificPruner.__init__: 传递参数给所有pruner

✓ [method/utils.py](method/utils.py)
  - create_layer_pruning_modifier: 添加use_attn_residual参数，实现attention提取
  - register_multi_layer_hooks: 添加attention capture hook注册

✓ [method/training.py](method/training.py)
  - 读取use_attn_residual配置，传递给register_multi_layer_hooks

✓ [method/evaluation.py](method/evaluation.py)
  - 同training.py，确保评估时也能使用residual

✓ [main.py](main.py)
  - LayerSpecificPruner实例化时传递3个新参数

✓ [configs/vision_token_pruning.yaml](configs/vision_token_pruning.yaml)
  - 添加3个配置项：use_attn_residual, attn_residual_weight, learnable_attn_weight

✓ [test_attn_residual.py](test_attn_residual.py) (新增)
  - 完整的单元测试

## 总结

- **核心创新**: 将LLM内部真实attention作为pruner决策的额外信号
- **实现完整**: 包含可配置开关、固定/可学习权重、完整测试
- **向后兼容**: 默认关闭，不影响现有功能
- **测试通过**: 所有单元测试通过✓

**祝实验顺利！**

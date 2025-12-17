# Token Merge位置修正 - 技术总结

## 问题诊断

### 原始错误实现

**错误流程**（之前的代码）：
```
Image
  → CLIP Vision Tower (576×1024)
  → Multi-Modal Projector (576×1024 → 576×4096)
  → Token Merge (576×4096 → M×4096) ❌ 错误！在4096维空间merge
  → LLM Forward
```

**问题**：
1. Token Merge在**投影后的4096维空间**操作，而不是原始1024维CLIP特征空间
2. 从`original_embeddings`提取vision tokens（已经过projector）
3. Token Merger期望1024维输入，但实际接收4096维
4. 再次调用`multi_modal_projector`（输入已经是4096维）

### 正确实现

**正确流程**：
```
Image
  → CLIP Vision Tower (576×1024) - 原始视觉特征
  → Token Merge (576×1024 → M×1024) ✅ 在1024维CLIP空间merge
  → Multi-Modal Projector (M×1024 → M×4096) - 只投影一次
  → LLM Forward
```

**原理**：
- CLIP输出是**语义丰富的1024维视觉特征**，适合做token相似度计算和合并
- Multi-Modal Projector只是**线性投影层**，不改变语义，只改变维度以匹配LLM
- 在CLIP空间merge可以利用CLIP预训练的语义相似度

---

## 修改内容

### 1. Backbone修改 - 返回未投影的Vision Features

**文件**: [engine/backbones/impl/mllm/llava.py](engine/backbones/impl/mllm/llava.py:374-393)

```python
# 新增：直接调用vision_tower获取1024维CLIP输出
if hasattr(self.model, 'vision_tower'):
    vision_tower = self.model.vision_tower
    if hasattr(vision_tower, 'forward'):
        raw_vision_features = vision_tower(pixel_values)  # (1, 576, 1024)
    else:
        raw_vision_features = None
else:
    raw_vision_features = None

# 原有：get_image_features (已包含vision_tower + projector)
image_features_list = self.model.get_image_features(pixel_values=pixel_values)
```

**返回结果**：
```python
result = {
    "embeddings": full_embeddings,        # (1, seq_len, 4096) - 完整序列
    "raw_vision_features": raw_vision_features,  # (1, 576, 1024) - 未投影
    ...
}
```

---

### 2. Training流程修改

**文件**: [method/training.py](method/training.py:79-120)

**修改前**（错误）：
```python
# 从embeddings提取vision features (已经是4096维!)
vision_features_raw = original_embeddings[:, v_start:v_end+1, :]  # ❌ 4096维

# Token Merge
merge_result = token_merger(vision_features_raw, ...)  # ❌ 输入维度错误

# 再次投影（输入已经是4096维）
if merged_vision.shape[-1] != original_embeddings.shape[-1]:
    merged_vision = backbone.model.multi_modal_projector(merged_vision)  # ❌ 重复投影
```

**修改后**（正确）：
```python
# 1. 获取未投影的vision features
vision_features_raw = emb_info['raw_vision_features']  # ✅ (1, 576, 1024)

if vision_features_raw is None:
    raise ValueError("backbone未返回raw_vision_features")

# 2. Token Merge (在1024维空间)
merge_result = token_merger(vision_features_raw, question_embeddings, ...)
merged_vision = merge_result['merged_features']  # ✅ (1, M, 1024)

# 3. 投影到LLM维度 (只投影一次)
merged_vision = backbone.model.multi_modal_projector(merged_vision)  # ✅ (1, M, 4096)
```

---

### 3. Evaluation流程修改

**文件**: [method/evaluation.py](method/evaluation.py:84-112)

修改内容与training.py相同：
- 从`emb_info['raw_vision_features']`获取1024维CLIP输出
- Token Merge在1024维空间操作
- 使用`multi_modal_projector`投影到4096维（只投影一次）

---

### 4. Token Merger V3 - 固定输出实现

**文件**: [method/models/token_merger.py](method/models/token_merger.py:261-417)

新增`LearnableTokenMergerV3`类：

**核心设计**：
```python
class LearnableTokenMergerV3(nn.Module):
    """固定输出M个tokens的可学习池化Merger

    优势：
    1. 无top-k采样，训练/推理完全一致
    2. 梯度流畅通（无断点）
    3. 输出维度固定，易于集成

    架构：
    - M个可学习查询向量（learnable pooling slots）
    - Question-aware调制：用问题嵌入调制查询
    - Cross-attention池化：查询关注所有vision tokens
    """

    def __init__(self, d_vision=1024, ...):  # d_vision=1024 (CLIP输出)
        # M个可学习查询
        self.pool_queries = nn.Parameter(...)  # (M, d_internal)

        # Question调制器
        self.query_modulator = nn.Sequential(...)

        # Cross-attention池化
        self.pool_attention = nn.MultiheadAttention(...)

    def forward(self, vision_features, question_embeddings):
        # 1. 准备查询（可选question调制）
        queries = self.pool_queries + question_bias  # (batch, M, d_internal)

        # 2. Cross-attention池化
        pooled = self.pool_attention(
            query=queries,      # M个查询
            key=vision_features,  # 所有N个vision tokens
            value=vision_features
        )  # 输出: (batch, M, d_internal)

        # 3. 投影回vision维度
        return self.output_proj(pooled)  # (batch, M, d_vision=1024)
```

**与V1/V2的区别**：
- **V1/V2**: 使用Gumbel-Top-K选择tokens → 有随机性，梯度有断点
- **V3**: 固定M个可学习查询 → 确定性，梯度畅通

---

## 配置更新

**文件**: [configs/vision_token_pruning.yaml](configs/vision_token_pruning.yaml:151)

```yaml
method_settings:
  merger_type: "fixed_pooling"  # 使用V3版本
  merge_ratio: 0.5  # 保留50%的tokens (576 → 288)

backbone_settings:
  mllm_settings:
    vision_dim: 1024  # CLIP ViT-L/14输出维度
    hidden_dim: 4096  # LLaMA-7B Hidden Dim
```

---

## 验证测试

**测试脚本**: [test_token_merge_position.py](test_token_merge_position.py)

**测试内容**：
1. ✅ Backbone返回`raw_vision_features` (1024维)
2. ✅ Token Merger输入/输出维度正确 (1024→1024)
3. ✅ Multi-Modal Projector投影正确 (1024→4096)
4. ✅ Token数量正确 (576 → 288)

**运行测试**：
```bash
python test_token_merge_position.py
```

---

## 理论支持

### 为什么在CLIP空间merge更合理？

1. **语义丰富性**
   - CLIP features是经过对比学习训练的**多模态语义表示**
   - 相似的视觉内容在CLIP空间距离更近
   - 适合做token相似度计算和聚合

2. **维度效率**
   - 1024维 vs 4096维：计算量减少约4倍
   - 更小的参数量（merger的Q/K/V投影）

3. **Multi-Modal Projector的作用**
   - 只是**线性映射层**：将1024维投影到4096维
   - 不改变语义结构，只改变维度匹配LLM
   - 在merge后投影可以减少投影层的计算量（M < N）

### 类比

```
错误做法：先翻译后压缩
英文 → 翻译成中文 → 压缩中文句子 ❌

正确做法：先压缩后翻译
英文 → 压缩英文句子 → 翻译成中文 ✅
```

CLIP空间 = 英文（原始语义空间）
LLM空间 = 中文（目标语义空间）
Projector = 翻译器
Token Merge = 压缩器

---

## 影响

### 性能影响
- **计算量减少**: merge在1024维空间，Q/K/V投影参数量减少75%
- **语义保留**: 在CLIP预训练的语义空间merge，相似度计算更准确
- **投影效率**: merge后token数减少，projector计算量减少50%

### 训练影响
- **梯度流**: V3版本梯度畅通，无top-k断点
- **确定性**: 训练/推理行为一致，无Gumbel噪声
- **稳定性**: 固定M个输出，易于调试和优化

---

## 实现清单

- [x] 修改backbone返回raw_vision_features
- [x] 修改training.py使用1024维vision features
- [x] 修改evaluation.py使用1024维vision features
- [x] 实现Token Merger V3 (固定输出版本)
- [x] 更新配置文件
- [x] 创建验证测试脚本
- [x] 编写技术文档

---

## 总结

此次修改将Token Merge从**投影后的4096维LLM空间**移到**投影前的1024维CLIP空间**，这是正确的设计：

1. **语义层面**: CLIP空间更适合做视觉token的相似度计算
2. **效率层面**: 1024维比4096维计算量更小
3. **架构层面**: 先merge后投影可以减少projector的计算量

配合新的V3版本（固定输出M个tokens），整个系统更加稳定、高效、易于训练。

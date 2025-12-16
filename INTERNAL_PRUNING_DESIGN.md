# LLM 内部剪枝设计方案

## 📌 核心决策
- **剪枝时机**: 只在 Prefill 阶段剪枝
- **实现方式**: 使用 PyTorch Hooks
- **目标**: 在 LLM 某些层之后插入剪枝器，减少后续层的计算量

---

## 🔍 关键知识点（从源码分析得出）

### 1. Generate 流程
**位置**: `transformers/generation/utils.py:2219`

```python
# generate() 方法的核心循环
while not_finished:
    # 准备输入
    model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

    # 第一次迭代（Prefill）
    if is_prefill:
        outputs = self(**model_inputs, return_dict=True)
        is_prefill = False  # 标记完成
    # 后续迭代（Decode）
    else:
        outputs = model_forward(**model_inputs, return_dict=True)

    # 更新 KV cache
    model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, ...)

    # 提取 logits 并选择下一个 token
    next_token_logits = outputs.logits[:, -1, :]
    next_tokens = argmax(next_token_logits, dim=-1)

    # 拼接到序列
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

**关键点**:
- ✅ Prefill 和 Decode 都调用 `forward()`
- ✅ Prefill: `past_key_values=None`, `input_ids=[B, L]`
- ✅ Decode: `past_key_values=cache`, `input_ids=[B, 1]`
- ✅ `is_prefill` 标志在第一次 forward 后自动设为 `False`

---

### 2. LLaVA Forward 流程
**位置**: `transformers/models/llava/modeling_llava.py:367`

```python
# LlavaForConditionalGeneration.forward()
def forward(self, input_ids, pixel_values, past_key_values=None, ...):
    # 1. 调用 LlavaModel
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        past_key_values=past_key_values,
        ...
    )

    # 2. 计算 logits
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    return LlavaCausalLMOutputWithPast(
        logits=logits,
        past_key_values=outputs.past_key_values,  # 返回 KV cache
        ...
    )
```

**LlavaModel.forward() 内部**:
```python
# modeling_llava.py:243
def forward(self, input_ids, pixel_values, ...):
    # 1. 获取文本 embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. 如果有图像，处理并替换 <image> token
    if pixel_values is not None:
        # 通过 vision_tower 编码
        image_features = self.get_image_features(pixel_values, ...)

        # 找到 <image> token 位置并替换
        special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # 3. 调用 language_model（核心 Transformer 层）
    outputs = self.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,  # KV cache
        ...
    )

    return outputs
```

**关键点**:
- ✅ `pixel_values` 只在 Prefill 时传入（由 `prepare_inputs_for_generation` 控制）
- ✅ `language_model` 是真正的 Transformer 层堆栈
- ✅ Hook 应该注册在 `language_model.layers[i]` 上

---

### 3. Prepare Inputs for Generation
**位置**: `transformers/models/llava/modeling_llava.py:453`

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    pixel_values=None,
    cache_position=None,
    ...
):
    # 调用父类准备基础输入
    model_inputs = super().prepare_inputs_for_generation(...)

    # 关键判断：只在 Prefill 时传递 pixel_values
    if cache_position[0] == 0:
        model_inputs["pixel_values"] = pixel_values

    return model_inputs
```

**关键点**:
- ✅ `cache_position[0] == 0` 判断是否为 Prefill
- ✅ Decode 时 `pixel_values` 为 `None`，跳过图像处理

---

### 4. KV Cache 结构
**位置**: `transformers/cache_utils.py`

```python
class DynamicCache:
    """
    存储结构:
    - key_cache: List[Tensor]  # 每层一个: [batch, num_heads, seq_len, head_dim]
    - value_cache: List[Tensor]  # 每层一个: [batch, num_heads, seq_len, head_dim]
    """

    def update(self, key_states, value_states, layer_idx):
        """更新某一层的 cache"""
        if layer_idx == 0 or len(self.key_cache) <= layer_idx:
            # 第一次：直接保存
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 后续：拼接新 token
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
```

**关键点**:
- ✅ Cache 按层存储，每层独立
- ✅ Prefill: 保存完整序列的 KV
- ✅ Decode: 每次拼接新 token 的 KV
- ✅ **如果 Prefill 时剪枝了，cache 存储的是剪枝后的长度** ✅

---

## 🎯 内部剪枝方案（Hook-Based）

### 核心思路
1. **注册 Hooks**: 在 `language_model.layers[i]` 的某些层注册 `forward_hook`
2. **Prefill 时剪枝**: Hook 检测到 `past_key_values=None` 时应用剪枝
3. **Decode 时跳过**: Hook 检测到 `past_key_values` 存在时跳过剪枝
4. **修改 hidden_states**: 在 hook 中直接移除被剪枝的 token
5. **更新位置信息**: 记录新的 `vision_positions`，供后续层使用

### Hook 函数伪代码
```python
def pruning_hook(module, input, output):
    # 1. 检查是否是 Prefill（通过检查输入长度或 past_key_values）
    is_prefill = (input[1] is None)  # past_key_values 为 None

    if not is_prefill:
        return output  # Decode 阶段，跳过剪枝

    # 2. 提取 hidden_states
    hidden_states = output[0]  # [batch, seq_len, hidden_dim]

    # 3. 识别 vision token 位置
    vision_start, vision_end = get_vision_positions()
    vision_hidden = hidden_states[:, vision_start:vision_end+1, :]

    # 4. 应用剪枝器
    soft_mask, hard_mask = generator(vision_hidden, question_embedding)

    # 5. 移除被剪枝的 token（根据 hard_mask）
    kept_indices = torch.nonzero(hard_mask[:, :, 0]).squeeze()
    pruned_vision = vision_hidden[:, kept_indices, :]

    # 6. 重新拼接序列
    new_hidden = torch.cat([
        hidden_states[:, :vision_start, :],
        pruned_vision,
        hidden_states[:, vision_end+1:, :]
    ], dim=1)

    # 7. 更新输出
    output[0] = new_hidden
    return output
```

---

## ⚠️ 关键挑战与解决方案

### 挑战 1: Batch 内不同样本剪枝数量不同
**问题**: 不同样本可能保留不同数量的 token，导致序列长度不一致

**解决方案**:
```python
# 方案 A: Padding（推荐，简单）
max_kept = max(num_kept_per_sample)
for each sample:
    if num_kept < max_kept:
        pad with zeros
        update attention_mask to mask out padding

# 方案 B: 分别处理（复杂，但更高效）
# 将 batch 拆分，逐个样本处理
```

### 挑战 2: Attention Mask 更新
**问题**: 剪枝后序列长度变化，需要更新 `attention_mask`

**解决方案**:
```python
# Hook 需要同时修改 attention_mask
# 方法 1: 通过全局状态传递
self.pruner_manager.updated_attention_mask = new_mask

# 方法 2: 存储在 model_kwargs 中（需要在外部处理）
```

### 挑战 3: Position IDs
**问题**: 剪枝后位置编码需要调整

**解决方案**:
```python
# 保持原始位置编码，只移除 token
# 或者：重新生成连续的位置编码
new_position_ids = torch.arange(0, new_seq_len, device=device)
```

---

## 📊 FLOPs 计算

### 理论计算
```python
def calculate_flops_reduction(
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    original_seq_len=676,  # 100 text + 576 vision
    pruned_seq_len=376,     # 100 text + 276 vision (剪枝 300 个)
    pruning_layer=10        # 在第 10 层剪枝
):
    # 每层 Attention 的 FLOPs 主要取决于 seq_len^2
    # FLOPs_attention ≈ 4 * seq_len^2 * hidden_dim

    # 前 10 层（剪枝前）
    flops_before = pruning_layer * 4 * (original_seq_len ** 2) * hidden_dim

    # 后 22 层（剪枝后）
    flops_after = (num_layers - pruning_layer) * 4 * (pruned_seq_len ** 2) * hidden_dim

    # Baseline（不剪枝）
    flops_baseline = num_layers * 4 * (original_seq_len ** 2) * hidden_dim

    reduction = (flops_baseline - (flops_before + flops_after)) / flops_baseline

    return {
        "baseline": flops_baseline,
        "with_pruning": flops_before + flops_after,
        "reduction": f"{reduction * 100:.2f}%"
    }
```

**预期结果**（剪枝 300/576 个 token，在第 10 层）:
- 前 10 层: 100% FLOPs（未剪枝）
- 后 22 层: ~44% FLOPs（序列长度从 676 降到 376）
- **总体减少**: ~38% FLOPs

---

## 🚀 实现步骤

### Step 1: 创建 InternalPrunerManager
- [ ] 实现 Hook 注册逻辑
- [ ] 实现剪枝 Hook 函数
- [ ] 处理 Batch padding
- [ ] 记录剪枝统计

### Step 2: 集成到 Backbone
- [ ] 在 `__init__` 中初始化 pruner manager
- [ ] 修改 `forward_from_embeddings` 传递 vision_positions
- [ ] 添加 enable/disable 接口

### Step 3: 修改训练流程
- [ ] 在 `train_step` 中传递必要信息
- [ ] 收集剪枝统计
- [ ] 计算剪枝相关 loss

### Step 4: 评估与 FLOPs 计算
- [ ] 实现 FLOPs 计算工具
- [ ] 对比剪枝前后差异
- [ ] 评估生成质量

---

## 🔧 配置示例

```yaml
method_settings:
  use_internal_pruning: true
  internal_pruner_layers: [10]  # 在第 10 层后剪枝
  target_token_num: 276          # 目标保留 276 个 vision token

  # 原有参数保持不变
  gen_num_layers: 2
  gen_num_heads: 2
  ...
```

---

## 📚 参考文献

### Transformers 源码位置
1. **Generate 主流程**: `transformers/generation/utils.py:2219-2566`
2. **LLaVA Forward**: `transformers/models/llava/modeling_llava.py:367-451`
3. **Prepare Inputs**: `transformers/models/llava/modeling_llava.py:453-481`
4. **KV Cache**: `transformers/cache_utils.py`

### 关键代码行
- Prefill 判断: `modeling_llava.py:476` (`cache_position[0] == 0`)
- Generate 循环: `generation/utils.py:2764-2834`
- Image 处理: `modeling_llava.py:272-283`
- Language Model 调用: `modeling_llava.py:285-292`

---

## 💡 注意事项

1. **只在 Prefill 剪枝**: Decode 阶段不剪枝，使用 Prefill 生成的压缩 cache
2. **保持梯度流**: Hook 中的所有操作必须可微
3. **Attention Mask 一致性**: 剪枝后需要同步更新 mask
4. **Batch 处理**: 使用 padding 处理不同长度
5. **性能开销**: Hook 有一定开销，但相比剪枝收益可忽略

---

## 🎯 预期效果

- **计算量减少**: 30-40% FLOPs（取决于剪枝比例和层位置）
- **生成速度**: Prefill 阶段可能略慢（剪枝开销），Decode 阶段更快（序列短）
- **内存占用**: KV cache 减少（存储剪枝后的序列）
- **生成质量**: 取决于剪枝器的训练效果

---

*最后更新: 2025-12-14*
*作者: Claude Code*

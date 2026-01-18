# Attention Consistency Pruning 实现计划

## 源码分析

### LLaVA 代码结构 (transformers库)

```
LlavaForConditionalGeneration
├── model: LlavaModel
│   ├── vision_tower: ViT
│   ├── multi_modal_projector: MLP
│   └── language_model: LlamaModel
│       ├── embed_tokens
│       ├── layers: ModuleList[LlamaDecoderLayer]
│       │   └── LlamaDecoderLayer
│       │       ├── input_layernorm
│       │       ├── self_attn: LlamaAttention
│       │       ├── post_attention_layernorm
│       │       └── mlp
│       ├── norm: RMSNorm
│       └── rotary_emb
└── lm_head: Linear
```

### 关键发现

1. **LlamaAttention.forward()** 返回 `(attn_output, attn_weights)`
2. **LlamaDecoderLayer.forward()** 丢弃了 attn_weights，只返回 hidden_states
3. **LlamaModel.forward()** 遍历 layers 列表，但不收集 attn_weights
4. 需要修改 DecoderLayer 使其能够返回 attn_weights

## 实现策略

### 核心思路

**不修改 transformers 源码**，而是：
1. 继承 `LlavaForConditionalGeneration`
2. 替换 `language_model.layers` 中的特定层为可剪枝层
3. 重写 `LlamaModel.forward()` 以支持剪枝

### 为什么不用 Hook

Hook 的问题（来自之前的经验）：
1. 难以获取和修改 attention weights
2. 中间状态难以控制
3. 调试困难
4. backward 时的 hook 容易出问题

### 为什么不直接修改 transformers 源码

1. 升级 transformers 时会丢失修改
2. 维护困难
3. 难以和其他项目共享

## 详细设计

### 1. PrunableLlamaDecoderLayer

继承 `LlamaDecoderLayer`，在剪枝层重写 forward 以：
- 保留并返回 attention weights
- 保留并返回 value states
- 接受外部传入的 pruner

```python
class PrunableLlamaDecoderLayer(nn.Module):
    """
    包装原始的 LlamaDecoderLayer，添加剪枝功能。
    """

    def __init__(self, original_layer, layer_idx, pruner=None, discriminator=None):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.pruner = pruner
        self.discriminator = discriminator
        self.is_pruning_layer = pruner is not None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        # === 新增参数 ===
        vision_mask=None,         # (batch, seq) bool mask，True 表示是 vision token
        question_mask=None,       # (batch, seq) bool mask
        answer_mask=None,         # (batch, seq) bool mask
        return_pruning_info=False,
        **kwargs
    ):
        if not self.is_pruning_layer:
            # 非剪枝层：直接调用原始 layer
            return self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )

        # === 剪枝层的特殊处理 ===
        return self.forward_with_pruning(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            vision_mask=vision_mask,
            question_mask=question_mask,
            answer_mask=answer_mask,
            return_pruning_info=return_pruning_info,
            **kwargs
        )
```

### 2. 剪枝层的 Forward 逻辑

```python
def forward_with_pruning(self, hidden_states, ...):
    """
    核心剪枝逻辑：
    1. 手动执行 attention 计算以获取 weights 和 V
    2. 计算 h_real 和 h_fake
    3. 应用 mask 到 hidden_states
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    layer = self.original_layer

    # === Step 1: LayerNorm + Q/K/V 投影 ===
    residual = hidden_states
    hidden_states_normed = layer.input_layernorm(hidden_states)

    # 获取 attention 模块
    attn = layer.self_attn
    head_dim = attn.head_dim
    num_heads = attn.config.num_attention_heads

    # Q/K/V 投影
    query_states = attn.q_proj(hidden_states_normed)
    key_states = attn.k_proj(hidden_states_normed)
    value_states = attn.v_proj(hidden_states_normed)

    # Reshape: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # === Step 2: 计算 Attention Weights ===
    # attn_weights: (batch, heads, seq, seq)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)

    # 应用 causal mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # === Step 3: 提取 question→vision attention ===
    # question_mask: (batch, seq), vision_mask: (batch, seq)
    # q2v_attn: (batch, n_vision) - question 对每个 vision token 的平均 attention
    q_indices = question_mask.nonzero(as_tuple=True)
    v_indices = vision_mask.nonzero(as_tuple=True)

    # 简化：假设每个 batch 的 vision/question 位置相同
    n_vision = vision_mask[0].sum().item()
    n_question = question_mask[0].sum().item()

    # 获取 question→vision 的 attention 子矩阵
    # attn_weights: (batch, heads, seq, seq)
    # 我们需要: attn_weights[:, :, question_positions, vision_positions]
    v_start = vision_mask[0].nonzero()[0].item()
    v_end = v_start + n_vision
    q_start = question_mask[0].nonzero()[0].item()
    q_end = q_start + n_question

    q2v_attn = attn_weights[:, :, q_start:q_end, v_start:v_end]  # (batch, heads, n_q, n_v)
    q2v_attn_avg = q2v_attn.mean(dim=(1, 2))  # (batch, n_vision)

    # === Step 4: Pruner 生成 mask ===
    # 输入: q2v_attn_avg (batch, n_vision)
    # 输出: hard_mask (batch, n_vision)
    residual_importance = self.pruner(q2v_attn_avg)  # (batch, n_vision)
    importance = q2v_attn_avg + residual_importance  # (batch, n_vision)

    # Gumbel-Softmax
    hard_mask = self.pruner.gumbel_softmax(importance)  # (batch, n_vision), 0/1

    # === Step 5: 计算 h_real 和 h_fake ===
    # answer tokens 的聚合结果
    a_start = answer_mask[0].nonzero()[0].item()
    a_end = a_start + answer_mask[0].sum().item()

    # h_real: 完整 attention 聚合
    # attn_output = attn_weights @ V
    attn_output_real = torch.matmul(attn_weights, value_states)  # (batch, heads, seq, head_dim)
    h_real = attn_output_real[:, :, a_start:a_end, :]  # (batch, heads, n_ans, head_dim)

    # h_fake: 剪枝后的 attention 聚合
    # 1. 修改 attention weights，把被剪掉的 vision tokens 权重置零
    # 2. 重新归一化
    attn_weights_fake = attn_weights.clone()
    # hard_mask: (batch, n_vision) -> 扩展到 (batch, 1, 1, n_vision)
    mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_vision)
    # 只修改对 vision tokens 的 attention
    attn_weights_fake[:, :, :, v_start:v_end] = attn_weights_fake[:, :, :, v_start:v_end] * mask_expanded

    # 重新归一化（防止除零，加一个小常数）
    attn_weights_fake = attn_weights_fake / (attn_weights_fake.sum(dim=-1, keepdim=True) + 1e-8)

    attn_output_fake = torch.matmul(attn_weights_fake, value_states)  # (batch, heads, seq, head_dim)
    h_fake = attn_output_fake[:, :, a_start:a_end, :]  # (batch, heads, n_ans, head_dim)

    # === Step 6: 应用 mask 到 hidden_states ===
    # 被剪掉的 vision tokens 的 hidden_states 缩放
    # 使用 fake attention 计算的 output
    attn_output = attn_output_fake
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_output = attn.o_proj(attn_output)

    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    if return_pruning_info:
        return hidden_states, {
            'h_real': h_real,           # (batch, heads, n_ans, head_dim)
            'h_fake': h_fake,           # (batch, heads, n_ans, head_dim)
            'hard_mask': hard_mask,     # (batch, n_vision)
            'q2v_attn': q2v_attn_avg,   # (batch, n_vision)
            'importance': importance,   # (batch, n_vision)
        }
    return hidden_states
```

### 3. LayerPruner 设计

```python
class LayerPruner(nn.Module):
    """
    轻量级剪枝网络，输出对 LLM attention 的残差调整。

    设计理念：
    - 输入是 LLM 自己的 question→vision attention
    - 输出是残差调整（可正可负）
    - 初始化为零，初始时完全依赖 LLM attention
    """

    def __init__(self, d_internal=128, temperature=1.0):
        super().__init__()
        self.d_internal = d_internal
        self.temperature = temperature

        # MLP: 输入 attention 值，输出残差
        self.mlp = nn.Sequential(
            nn.Linear(1, d_internal),
            nn.LayerNorm(d_internal),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal, 1)
        )

        # 初始化最后一层为零，初始时输出残差为 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q2v_attn):
        """
        输入: q2v_attn (batch, n_vision) - LLM 的 question→vision attention
        输出: residual (batch, n_vision) - 残差调整
        """
        # (batch, n_vision) -> (batch, n_vision, 1) -> MLP -> (batch, n_vision, 1) -> (batch, n_vision)
        x = q2v_attn.unsqueeze(-1)
        residual = self.mlp(x).squeeze(-1)
        return residual

    def gumbel_softmax(self, importance):
        """
        将 importance score 转换为 0/1 hard mask。

        输入: importance (batch, n_vision) - 未归一化的重要性分数
        输出: hard_mask (batch, n_vision) - 0/1 mask
        """
        # 转换为二分类 logits: [drop_logit, keep_logit]
        # drop_logit 固定为 0，keep_logit 为 importance
        stacked = torch.stack([
            torch.zeros_like(importance),  # drop logit = 0
            importance                      # keep logit
        ], dim=-1)  # (batch, n_vision, 2)

        if self.training:
            y = F.gumbel_softmax(stacked, tau=self.temperature, hard=True, dim=-1)
            return y[..., 1]  # 取 keep 的概率
        else:
            # 推理时直接 argmax
            return (importance > 0).float()

    def set_temperature(self, temperature):
        self.temperature = temperature
```

### 4. LayerDiscriminator 设计

```python
class LayerDiscriminator(nn.Module):
    """
    判别单个 answer token 的聚合结果是 real 还是 fake。

    输入: h (batch, heads, head_dim) - 单个 answer token 从所有 positions 聚合的结果
    输出: logit (batch,) - real/fake 判断

    设计理念：
    - 每个 answer token 独立判别
    - 输入是 attention 聚合后的结果（不是 hidden states）
    - 轻量级网络，避免过强
    """

    def __init__(self, num_heads, head_dim, d_hidden=256, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        input_dim = num_heads * head_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )

    def forward(self, h):
        """
        输入: h (batch, heads, head_dim) - 单个 answer token 的聚合结果
        输出: logit (batch,) - real/fake 判断
        """
        # Flatten: (batch, heads, head_dim) -> (batch, heads * head_dim)
        h_flat = h.view(h.shape[0], -1)
        return self.net(h_flat).squeeze(-1)
```

### 5. PrunableLlavaForConditionalGeneration

```python
class PrunableLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    """
    可剪枝的 LLaVA 模型。

    通过替换特定层的 DecoderLayer 为 PrunableLlamaDecoderLayer 实现剪枝。
    """

    def __init__(self, config, pruning_layers=[4, 14, 24]):
        super().__init__(config)
        self.pruning_layers = pruning_layers

        # 获取 LLM 配置
        llm_config = config.text_config
        num_heads = llm_config.num_attention_heads
        head_dim = llm_config.hidden_size // num_heads

        # 创建 pruners 和 discriminators
        self.pruners = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

        for layer_idx in pruning_layers:
            self.pruners[str(layer_idx)] = LayerPruner()
            self.discriminators[str(layer_idx)] = LayerDiscriminator(
                num_heads=num_heads,
                head_dim=head_dim
            )

        # 替换剪枝层
        self._replace_pruning_layers()

    def _replace_pruning_layers(self):
        """替换特定层为可剪枝层"""
        llm = self.model.language_model
        for layer_idx in self.pruning_layers:
            original_layer = llm.layers[layer_idx]
            pruner = self.pruners[str(layer_idx)]
            discriminator = self.discriminators[str(layer_idx)]

            llm.layers[layer_idx] = PrunableLlamaDecoderLayer(
                original_layer=original_layer,
                layer_idx=layer_idx,
                pruner=pruner,
                discriminator=discriminator
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, pruning_layers=[4, 14, 24], **kwargs):
        """从预训练模型加载并设置剪枝层"""
        # 先加载原始模型
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 设置剪枝层
        model.pruning_layers = pruning_layers

        # 创建 pruners 和 discriminators
        llm_config = model.config.text_config
        num_heads = llm_config.num_attention_heads
        head_dim = llm_config.hidden_size // num_heads

        model.pruners = nn.ModuleDict()
        model.discriminators = nn.ModuleDict()

        for layer_idx in pruning_layers:
            model.pruners[str(layer_idx)] = LayerPruner()
            model.discriminators[str(layer_idx)] = LayerDiscriminator(
                num_heads=num_heads,
                head_dim=head_dim
            )

        # 替换剪枝层
        model._replace_pruning_layers()

        return model
```

## 训练流程

### Forward Pass

```python
def training_step(model, batch):
    """
    单个训练步骤：
    1. Forward pass（收集 h_real/h_fake）
    2. 计算 losses
    3. Backward
    """

    # 准备 masks
    vision_mask = batch['vision_mask']      # (batch, seq)
    question_mask = batch['question_mask']  # (batch, seq)
    answer_mask = batch['answer_mask']      # (batch, seq)

    # Forward pass
    # 需要修改 forward 使其在剪枝层收集信息
    outputs, pruning_infos = model.forward_with_pruning(
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        attention_mask=batch['attention_mask'],
        vision_mask=vision_mask,
        question_mask=question_mask,
        answer_mask=answer_mask,
        labels=batch['labels'],
    )

    # pruning_infos: Dict[layer_idx -> {'h_real', 'h_fake', 'hard_mask', ...}]

    # === 计算 Losses ===

    # 1. Task Loss (CE on answer tokens)
    task_loss = outputs.loss

    # 2. Adversarial Loss (每层，每个 answer token)
    adv_loss = 0
    for layer_idx, info in pruning_infos.items():
        h_real = info['h_real']  # (batch, heads, n_ans, head_dim)
        h_fake = info['h_fake']

        disc = model.discriminators[str(layer_idx)]

        # 每个 answer token 独立判别
        for ans_idx in range(h_fake.shape[2]):
            fake_pred = disc(h_fake[:, :, ans_idx, :])
            # Pruner 的目标：让 fake 被判为 real
            adv_loss = adv_loss + F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )

    # 3. Discriminator Loss
    disc_loss = 0
    for layer_idx, info in pruning_infos.items():
        h_real = info['h_real']
        h_fake = info['h_fake']

        disc = model.discriminators[str(layer_idx)]

        for ans_idx in range(h_real.shape[2]):
            real_pred = disc(h_real[:, :, ans_idx, :])
            fake_pred = disc(h_fake[:, :, ans_idx, :].detach())  # detach!

            disc_loss = disc_loss + F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            ) + F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

    # 4. Sparsity Loss
    sparsity_loss = 0
    for layer_idx, info in pruning_infos.items():
        hard_mask = info['hard_mask']  # (batch, n_vision)
        kept_ratio = hard_mask.mean()
        target_ratio = target_token_num / n_vision
        sparsity_loss = sparsity_loss + torch.abs(kept_ratio - target_ratio)

    # === 总损失 ===
    pruner_loss = task_loss * w_task + adv_loss * w_adv + sparsity_loss * w_sparsity
    disc_loss = disc_loss * w_disc

    return pruner_loss, disc_loss
```

### 交替训练

```python
# 1. 更新 Discriminators
disc_loss.backward()
disc_optimizer.step()
disc_optimizer.zero_grad()

# 2. 更新 Pruners
pruner_loss.backward()
pruner_optimizer.step()
pruner_optimizer.zero_grad()
```

## 文件结构

```
method/
├── models/
│   ├── prunable_llava.py          # PrunableLlavaForConditionalGeneration
│   ├── prunable_llama_layer.py    # PrunableLlamaDecoderLayer
│   ├── layer_pruner_new.py        # LayerPruner (新设计)
│   └── layer_discriminator.py     # LayerDiscriminator
├── training_acp.py                # Attention Consistency Pruning 训练逻辑
├── losses_acp.py                  # Loss 计算
└── utils.py                       # 工具函数
```

## 配置更新

```yaml
method_settings:
  # 剪枝层配置（可调整）
  pruning_layers: [4, 14, 24]

  # Pruner 配置
  pruner_d_internal: 128

  # Discriminator 配置
  disc_d_hidden: 256
  disc_dropout: 0.1

  # Temperature
  temperature: 1.0
  temperature_min: 0.5
  temperature_anneal_rate: 0.4

  # 目标保留率
  target_token_num: 144

  # Loss 权重
  task_loss_weight: 1.0
  adv_loss_weight: 0.5
  sparsity_weight: 0.2
  disc_loss_weight: 1.0
```

## 关键注意事项

1. **Position 计算**：需要正确识别 vision/question/answer tokens 的位置
2. **GQA (Grouped Query Attention)**：LLaMA 使用 GQA，需要正确处理 num_key_value_heads
3. **Causal Mask**：剪枝不能破坏因果性
4. **梯度流**：确保 Gumbel-Softmax 的梯度能传到 pruner
5. **内存**：h_real/h_fake 可能很大，考虑是否需要优化

## 下一步

1. 实现 `LayerPruner` 和 `LayerDiscriminator`
2. 实现 `PrunableLlamaDecoderLayer`
3. 实现 `PrunableLlavaForConditionalGeneration`
4. 集成到训练框架
5. 测试和调试

# Attention Consistency Pruning (ACP) - 项目文档

## 项目概述

基于对抗一致性学习的视觉 Token 剪枝框架，用于 LLaVA 等多模态大语言模型。核心思想是通过判别器约束剪枝后的 attention 聚合结果与原始结果的一致性，在减少计算量的同时保持模型性能。

## 目录结构

```
domain-adaptation-token-pruning/
├── main_acp_ddp.py          # DDP 分布式训练脚本（主入口）
├── main_acp.py              # 单卡训练脚本
├── eval_acp.py              # 评估脚本
├── configs/
│   └── vision_token_pruning.yaml  # 训练配置文件
├── method/models/           # 核心模型实现
│   ├── prunable_llava.py    # 可剪枝 LLaVA 包装器
│   ├── prunable_llama_layer.py  # 可剪枝 Decoder Layer
│   ├── layer_pruner_acp.py  # CrossAttention 剪枝器
│   ├── layer_discriminator.py  # Attention 聚合判别器
│   └── adapter.py           # 剪枝补偿 Adapter
└── engine/
    ├── configs/loader.py    # 配置加载器
    └── datas/               # 数据集加载
        ├── loader.py        # 数据集注册与加载
        └── impl/            # 各数据集实现
```

## 核心架构

### 1. 模型层级关系

```
PrunableLlavaForConditionalGeneration (method/models/prunable_llava.py)
├── base_model: LlavaForConditionalGeneration (冻结)
├── pruner_manager: LayerPrunerManager
│   └── pruners: {layer_idx: CrossAttentionPruner}
├── disc_manager: LayerDiscriminatorManager
│   └── discriminators: {layer_idx: LayerDiscriminator}
└── adapter_manager: AdapterManager
    └── adapters: {layer_idx: PruningAdapter}
```

### 2. 剪枝层结构 (`PrunableLlamaDecoderLayer`)

替换 LLaMA 的指定层（默认 [4, 14, 24]），在 self-attention 计算中：

1. 正常计算 Q/K/V 和 attention weights
2. 提取 question→vision 的 attention（q2v_attn）作为重要性 baseline
3. **Pruner** 根据 vision hidden states 和 q2v_attn 计算 keep_logits
4. Gumbel-Softmax 生成 hard_mask（训练可微，推理概率阈值）
5. 计算两种 attention 聚合：
   - `h_real = attn_weights @ V`（完整聚合）
   - `h_fake = (attn_weights * mask) @ V`（剪枝后聚合）
6. **Adapter** 对 h_fake 进行补偿修正
7. 最终使用 h_fake 作为输出

### 3. CrossAttentionPruner (`layer_pruner_acp.py:18`)

```python
# 核心计算流程
baseline = log(q2v_attn) - mean(log(q2v_attn))  # LLM attention 作为 baseline
v = vision_proj(vision_hidden)                   # 投影到内部维度
_, attn_weights = cross_attn(queries, v, v)      # 可学习 queries 的 cross-attention
delta = query_aggregator(attn_weights) + token_scorer(v)  # 学习的修正量
keep_logits = baseline + delta + keep_bias       # 残差设计
```

- 4 个可学习 pruning queries，通过 cross-attention 评估 token 重要性
- 训练时用 Gumbel-Softmax，推理时用概率阈值（sigmoid(keep_logits) > threshold）

### 4. LayerDiscriminator (`layer_discriminator.py:18`)

判别单个 answer token 的 attention 聚合结果是 real 还是 fake：

```python
# 输入：h = Σ attn[i] * V[i]，形状 (batch, heads, n_ans, head_dim)
h_normalized = F.normalize(h, p=2, dim=-1)  # L2 归一化
logit = MLP(h_flat)  # 4096 -> 512 -> 256 -> 1
```

支持三种损失类型：`bce`、`wgan`（带梯度惩罚）、`hinge`

## 损失函数

### 训练涉及的四个损失

| 损失 | 计算位置 | 作用 | 默认权重 |
|------|----------|------|----------|
| Task Loss | `main_acp_ddp.py:360` | Answer tokens 的 CrossEntropy | 1.0 |
| Adv Loss | `layer_discriminator.py:293` | 让 h_fake 被判为 real（训练 Pruner） | 0.5 |
| Disc Loss | `layer_discriminator.py:191` | 区分 h_real vs h_fake（训练 Discriminator） | 1.0 |
| Sparsity Loss | `main_acp_ddp.py:495` | 约束 token 保留率到目标值 | 0.2 |

### 优化器分离

```python
# Pruner + Adapter 共用一个优化器
pruner_total = task_loss * w_task + adv_loss * w_adv + sparsity_loss * w_sparsity
pruner_optimizer.step()

# Discriminator 单独优化
disc_loss.backward()
disc_optimizer.step()
```

### Sparsity Loss 计算

累积多层剪枝的加权平均保留率：

```python
# 剪枝是累积的：L4 mask × L14 mask × L24 mask
weighted_kept = Σ (n_affected_layers × cumulative_ratio)
avg_kept_ratio = weighted_kept / total_layers
sparsity_loss = |avg_kept_ratio - target_ratio|
```

## 关键配置项 (`configs/vision_token_pruning.yaml`)

```yaml
method_settings:
  pruning_layers: [4, 14, 24]      # 剪枝层索引
  target_token_num: 144            # 目标保留 token 数（原始 576）

  # Pruner
  pruner_d_internal: 512           # Cross-attention 内部维度
  pruner_n_heads: 4                # Cross-attention 头数
  pruner_thresholds:               # 推理阈值（概率空间 0-1）
    4: 0.4
    14: 0.4
    24: 0.4

  # Discriminator
  disc_d_d: 512                    # MLP 隐藏层维度
  disc_loss_type: hinge            # bce / wgan / hinge
  disc_use_spectral_norm: true     # 谱归一化

  # Temperature Annealing
  temperature: 1.0                 # Gumbel-Softmax 初始温度
  temperature_min: 0.1             # 最低温度
  temperature_anneal_rate: 0.8     # 前 80% 步数退火

  # Loss Weights
  task_loss_weight: 1.3
  adv_loss_weight: 0.45
  sparsity_weight: 0.2
```

## 运行方式

### DDP 分布式训练

```bash
torchrun --nproc_per_node=4 main_acp_ddp.py --config configs/vision_token_pruning.yaml
```

### 单卡训练

```bash
python main_acp.py --config configs/vision_token_pruning.yaml
```

## 推理模式

### Hard Pruning 推理 (`prunable_llava.py:569`)

`generate_with_hard_pruning()` 实现物理删除被剪枝的 tokens：

1. Prefill 阶段逐层处理，在剪枝层物理删除 tokens
2. 更新 position_ids、KV cache
3. Decode 阶段正常自回归生成

与训练时的软剪枝不同，推理时真正减少了计算量。

## 数据流

```
输入 → preprocess_batch() → model.forward() → pruning_infos
                                    ↓
                            {layer_idx: {
                                'h_real': List[(heads, n_ans, head_dim)],
                                'h_fake': List[(heads, n_ans, head_dim)],
                                'hard_mask': (batch, n_vision),
                                'keep_logits': (batch, n_vision)
                            }}
                                    ↓
                            compute losses → backward → step
```

## 支持的数据集

- `vqa-vqav2`: VQAv2
- `vqa-pope`: POPE
- `vqa-mme`: MME
- `vqa-gqa`: GQA
- `vqa-sqa`: ScienceQA
- `vqa-mmb`: MMBench
- `vqa-seed`: SEEDBench

## 关键文件快速索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 训练主循环 | `main_acp_ddp.py` | 769-1117 |
| 训练单步 | `main_acp_ddp.py` | 397-572 |
| Task Loss | `main_acp_ddp.py` | 360-390 |
| 模型 Forward | `prunable_llava.py` | 183-349 |
| 剪枝层 Forward | `prunable_llama_layer.py` | 全文 |
| Pruner 计算 | `layer_pruner_acp.py` | 98-170 |
| Discriminator Loss | `layer_discriminator.py` | 191-249 |
| Adversarial Loss | `layer_discriminator.py` | 293-336 |
| 配置加载 | `engine/configs/loader.py` | 全文 |

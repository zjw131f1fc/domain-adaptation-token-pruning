# Domain Adaptation Token Pruning - 实现文档

## 项目概述

本项目实现了一个**端到端可训练**的多模态大模型（LLaVA-1.5-7B）视觉token剪枝方法，包含两个核心组件：

1. **Vision Token Merge（LLM输入前）** - 可学习的token合并，减少输入到LLM的vision token数量
2. **Layer-wise Pruning（LLM内部3层）** - 在LLM的第10/20/31层进行逐层可训练剪枝

训练策略采用**GAN对抗训练**，通过Discriminator判别剪枝后的hidden states，确保剪枝模型的输出接近原始模型。

---

## 系统架构

```
输入: image + question + answer
    ↓
[1] LLaVA Vision Encoder (CLIP ViT)
    → vision_features (1, 576, 1024)
    ↓
[2] 【Learnable Token Merger (可训练)】
    → merged_vision_features (1, ~288, 1024)
    → importance_logits (用于sparsity loss)
    ↓
[3] Vision Projector + Text Embedding拼接
    → full_embeddings (1, seq_len, 4096)
    → 更新 vision_token_positions
    ↓
[4] LLM Forward with 【Layer-Specific Pruning (可训练)】
    → Layer 10: pruning_mask_1 (早期剪枝)
    → Layer 20: pruning_mask_2 (中期剪枝)
    → Layer 31: pruning_mask_3 (后期剪枝)
    ↓
[5] 提取目标层Hidden States
    ↓
[6] Discriminator判别 real vs pruned
    → 二分类: P(real | hidden_states)
    ↓
[7] 多目标Loss
    → Generator/Pruner: adv_loss + task_loss + sparsity_loss
    → Discriminator: real_loss + fake_loss
```

---

## 核心技术组件

### 1. Learnable Token Merger

**文件**: [method/models/token_merger.py](method/models/token_merger.py)

#### 1.1 基础版本 (`LearnableTokenMerger`)

**架构设计**:
```python
class LearnableTokenMerger(nn.Module):
    """
    可训练的Vision Token合并器

    方法: 学习token重要性，使用Gumbel-Top-K选择要保留的tokens，
         然后通过软分配将所有tokens合并到保留的cluster centers

    关键技术:
    1. Importance Scoring: 小型MLP预测每个token的重要性
    2. Gumbel-Top-K Sampling: 可微分的top-k选择（训练时加噪声探索）
    3. Soft Assignment: 通过学习的Q/K投影计算相似度，softmax得到合并权重
    4. Temperature Annealing: 训练初期软分配，后期硬分配
    """

    def __init__(self, d_model=1024, num_heads=4, merge_ratio=0.5):
        super().__init__()
        self.d_model = d_model
        self.merge_ratio = merge_ratio  # 保留的token比例

        # 1. Importance scorer: 预测每个token的重要性
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        # 2. Query/Key投影: 计算token之间的相似度
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # 3. Temperature (外部动态更新)
        self.temperature = 1.0

    def forward(self, vision_features, use_gumbel=True):
        """
        输入:
            vision_features: (batch, N, d_model) - CLIP输出，N=576
            use_gumbel: bool - 是否使用Gumbel noise（训练时True，推理时False）

        输出:
            {
                'merged_features': (batch, M, d_model) - 合并后的features，M ≈ N * merge_ratio
                'merge_indices': (batch, M) - 保留的token索引
                'merge_weights': (batch, N, M) - 软分配矩阵
                'importance_logits': (batch, N) - 重要性分数（用于sparsity loss）
            }
        """
        batch, N, d = vision_features.shape
        M = int(N * self.merge_ratio)  # 目标保留数量

        # === Step 1: 计算token重要性 ===
        importance_logits = self.importance_scorer(vision_features).squeeze(-1)  # (batch, N)

        # === Step 2: 使用Gumbel-Top-K选择保留的tokens ===
        if use_gumbel and self.training:
            # Gumbel-Max技巧: 添加噪声使采样可微分
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_logits) + 1e-8) + 1e-8)
            perturbed_logits = importance_logits + gumbel_noise
            _, top_k_indices = torch.topk(perturbed_logits, k=M, dim=-1)  # (batch, M)
        else:
            # 推理: 确定性top-K
            _, top_k_indices = torch.topk(importance_logits, k=M, dim=-1)

        # === Step 3: 提取保留的tokens作为cluster centers ===
        cluster_centers = torch.gather(
            vision_features, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, d)
        )  # (batch, M, d)

        # === Step 4: 计算所有tokens到cluster centers的相似度 ===
        Q = self.q_proj(vision_features)    # (batch, N, d) - 所有tokens作为query
        K = self.k_proj(cluster_centers)    # (batch, M, d) - cluster centers作为key

        # Scaled dot-product similarity
        similarity = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # (batch, N, M)

        # === Step 5: 软分配 (temperature控制的softmax) ===
        merge_weights = F.softmax(similarity / self.temperature, dim=-1)  # (batch, N, M)

        # === Step 6: 软合并 - 加权聚合所有tokens到cluster centers ===
        merged_features = torch.matmul(
            merge_weights.transpose(-2, -1),  # (batch, M, N)
            vision_features                    # (batch, N, d)
        )  # (batch, M, d)

        return {
            'merged_features': merged_features,
            'merge_indices': top_k_indices,
            'merge_weights': merge_weights,
            'importance_logits': importance_logits
        }

    def set_temperature(self, temperature):
        """外部更新temperature (用于annealing)"""
        self.temperature = temperature
```

#### 1.2 Question-Aware版本 (`LearnableTokenMergerV2`)

与基础版本的区别:
- 使用**Cross-Attention**让vision tokens关注question
- Question-aware的重要性评分
- 更适合VQA等需要问题引导的任务

**关键改进**:
```python
# Step 1: 投影到统一维度
V = self.vision_proj(vision_features)         # (batch, N, d_internal)
Q = self.text_proj(question_embeddings)       # (batch, n_text, d_internal)

# Step 2: Cross-Attention - vision关注question
attended_V, _ = self.cross_attn(
    query=V,     # vision tokens作为query
    key=Q,       # question作为key/value
    value=Q
)

# Step 3: 基于question-aware features计算重要性
importance_logits = self.importance_scorer(attended_V).squeeze(-1)
```

**配置选择**:
```yaml
method_settings:
  merger_type: "simple"  # 使用 LearnableTokenMerger (基础版)
  # merger_type: "question_aware"  # 使用 LearnableTokenMergerV2
```

---

### 2. Layer-Specific Pruner

**文件**: [method/models/layer_pruner.py](method/models/layer_pruner.py)

#### 2.1 架构设计

```python
class LayerSpecificPruner(nn.Module):
    """
    为多个LLM层学习不同的剪枝策略

    核心思想:
    - 早期层（Layer 10）: 去除明显不相关的vision tokens（如背景）
    - 中期层（Layer 20）: 进一步精炼，去除冗余细节
    - 后期层（Layer 31）: 只保留对最终预测最关键的tokens

    每层独立学习，实现渐进式剪枝
    """

    def __init__(self, d_model=4096, d_text=4096, layer_indices=[10, 20, 31]):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每个层创建独立的剪枝头
        self.pruners = nn.ModuleDict({
            str(layer_idx): VisionPrunerHead(d_model, d_text)
            for layer_idx in layer_indices
        })

    def get_pruner(self, layer_idx):
        """获取指定层的剪枝头"""
        if str(layer_idx) not in self.pruners:
            raise ValueError(f"No pruner for layer {layer_idx}")
        return self.pruners[str(layer_idx)]

    def get_all_layers(self):
        """返回所有剪枝层的索引"""
        return self.layer_indices


class VisionPrunerHead(nn.Module):
    """
    单层Vision Token剪枝头

    架构: Cross-Attention + MLP
    - Vision tokens关注question embeddings (cross-attention)
    - 生成每个vision token的keep/drop决策
    """

    def __init__(self, d_vision=4096, d_text=4096, d_internal=512, num_heads=4):
        super().__init__()
        self.d_internal = d_internal

        # === Feature投影 ===
        self.vision_proj = nn.Linear(d_vision, d_internal)
        self.text_proj = nn.Linear(d_text, d_internal)

        # === Cross-Attention: vision tokens关注question ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # === Mask预测头 ===
        self.mask_predictor = nn.Sequential(
            nn.Linear(d_internal, d_internal // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal // 2, 1)
        )

        # Temperature (外部动态更新)
        self.temperature = 1.0

    def forward(self, vision_hidden, question_embeddings, use_gumbel=True):
        """
        输入:
            vision_hidden: (batch, n_vision, d_vision) - 当前层的vision token hidden states
            question_embeddings: (batch, n_text, d_text) - question embeddings
            use_gumbel: bool - 是否使用Gumbel-Softmax

        输出:
            soft_mask: (batch, n_vision) - 每个token的保留概率，范围[0, 1]
        """
        # === Step 1: 投影到内部维度 ===
        V = self.vision_proj(vision_hidden)         # (batch, n_vision, d_internal)
        Q = self.text_proj(question_embeddings)     # (batch, n_text, d_internal)

        # === Step 2: Cross-attention - vision关注question ===
        attended_V, _ = self.cross_attn(
            query=V,
            key=Q,
            value=Q,
            need_weights=False
        )  # (batch, n_vision, d_internal)

        # === Step 3: 预测keep/drop logits ===
        keep_logits = self.mask_predictor(attended_V).squeeze(-1)  # (batch, n_vision)

        # === Step 4: Gumbel-Softmax (可微分的二分类) ===
        if use_gumbel and self.training:
            # 构建 [drop_logits, keep_logits]
            stacked_logits = torch.stack([
                torch.zeros_like(keep_logits),  # drop = 0
                keep_logits                      # keep = logits
            ], dim=-1)  # (batch, n_vision, 2)

            # 添加Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(keep_logits) + 1e-8) + 1e-8)
            gumbel_logits = (stacked_logits + gumbel_noise.unsqueeze(-1)) / self.temperature

            # Softmax + 提取keep概率
            probs = F.softmax(gumbel_logits, dim=-1)  # (batch, n_vision, 2)
            soft_mask = probs[..., 1]  # (batch, n_vision)
        else:
            # 推理: 使用sigmoid (确定性)
            soft_mask = torch.sigmoid(keep_logits / self.temperature)

        return soft_mask

    def set_temperature(self, temperature):
        """外部更新temperature"""
        self.temperature = temperature
```

#### 2.2 为什么需要Cross-Attention

```
Question: "What color is the car?"
Vision Tokens: [background, tree, car_1, car_2, sky, ...]

Cross-Attention后:
- car_1, car_2的表示与"color", "car"高度相关
- background, sky的表示与question相关度弱

→ Mask Predictor给car tokens更高的keep概率
```

#### 2.3 为什么每层需要独立Pruner

- **Layer 10 hidden states**: 仍保留许多低层视觉特征（边缘、纹理）
- **Layer 20 hidden states**: 开始抽象化（物体语义）
- **Layer 31 hidden states**: 高度抽象（与问题相关的语义）

每层对"重要性"的定义不同，因此需要独立学习。

---

### 3. 在LLM层注册Hooks

**文件**: [method/utils.py](method/utils.py)

#### 3.1 核心机制

使用PyTorch的`register_forward_pre_hook`在指定层注入剪枝逻辑。详细代码请参考 [method/utils.py](method/utils.py) 中的:
- `register_multi_layer_hooks()`
- `remove_hooks()`

#### 3.2 为什么缩放Hidden States等价于修改Attention Scores

```
原始Attention:
Q = hidden_states @ W_q
K = hidden_states @ W_k
V = hidden_states @ W_v
scores = (Q @ K.T) / sqrt(d_k)
attn_weights = softmax(scores)
output = attn_weights @ V

应用soft_mask后:
hidden_states_vision *= soft_mask  (逐元素乘法)

→ Q_vision = (hidden_states_vision * soft_mask) @ W_q = Q_original * soft_mask
→ K_vision = K_original * soft_mask
→ V_vision = V_original * soft_mask

→ scores_vision_to_vision = (Q_vis @ K_vis.T) = (Q * mask) @ (K * mask).T
                           = Q @ K.T * (mask ⊗ mask.T)  (外积)
→ scores_text_to_vision = Q_text @ K_vis.T = Q_text @ K.T * mask
→ scores_vision_to_text = Q_vis @ K_text.T = Q @ K_text.T * mask

→ 所有涉及vision tokens的attention scores都被soft_mask缩放
```

---

### 4. 训练流程

**文件**: [method/training.py](method/training.py)

训练流程包含以下关键步骤:

1. **Token Merge**: 在LLM输入前合并vision tokens
2. **Layer-wise Pruning**: 在Layer 10/20/31分别剪枝
3. **Discriminator判别**: 判断hidden states是否来自剪枝模型
4. **多目标Loss计算**:
   - Generator Loss: `adv_loss` + `task_loss` + `sparsity_loss` + `token_count_loss`
   - Discriminator Loss: `real_loss` + `fake_loss`
5. **Temperature Annealing**: 动态调整temperature

**关键点**:
- 所有loss累加使用**非inplace操作**（`a = a + b`而非`a += b`）
- 使用`mask_collector`收集每层的soft_mask用于sparsity loss
- Discriminator在判别fake时关闭梯度（用于generator），判别real时开启梯度

详细实现请参考 [method/training.py](method/training.py)。

---

### 5. 评估流程

**文件**: [method/evaluation.py](method/evaluation.py)

评估包含三种模式:
1. **origin**: 无剪枝（baseline）
2. **soft**: 使用soft mask（训练时的状态）
3. **hard**: 二值化mask（threshold=0.5，真实推理场景）

详细实现请参考 [method/evaluation.py](method/evaluation.py)。

---

## 配置文件

**文件**: [configs/vision_token_pruning.yaml](configs/vision_token_pruning.yaml)

关键配置项:

```yaml
method_settings:
  # Token Merge
  merger_type: "simple"  # "simple" 或 "question_aware"
  merge_ratio: 0.5       # 保留50%的tokens

  # Layer-wise Pruning
  pruning_layers: [10, 20, 31]
  target_sparsity: 0.3   # 每层剪掉30%

  # Temperature Annealing
  temperature: 1.0
  temperature_min: 0.1
  temperature_anneal_rate: 0.5

  # Loss权重
  adv_loss_weight: 1.0
  task_loss_weight: 25.0
  sparsity_weight: 1e-4
  token_count_loss_weight: 1e-3
```

---

## 关键知识点

### 1. Gumbel-Softmax

**目的**: 可微分的离散采样

**公式**:
```python
gumbel_noise = -log(-log(uniform(0,1)))
soft_sample = softmax((logits + gumbel_noise) / temperature)
```

### 2. Temperature Annealing

**策略**: 训练初期高温（soft, 探索），后期低温（hard, 确定性）

```python
if progress < anneal_rate:
    current_temp = initial_temp - (progress / anneal_rate) * (initial_temp - final_temp)
else:
    current_temp = final_temp
```

### 3. PyTorch避免Inplace操作

**错误**:
```python
loss += some_loss  # ❌
loss *= weight     # ❌
```

**正确**:
```python
loss = loss + some_loss  # ✅
loss = loss * weight     # ✅
```

**原因**: Inplace操作会修改张量版本号，破坏计算图。

---

## 预期结果

### Token数量变化

```
原始: 676 tokens (100 text + 576 vision)
  ↓ Token Merge (ratio=0.5)
  → ~400 tokens (100 text + ~288 vision)
  ↓ Layer 10/20/31 Pruning (sparsity=0.3)
  → ~248 tokens (100 text + ~99 vision)
```

**FLOPs减少**: 约60-70%

### 性能目标

- VQA Accuracy: 下降 < 3%
- BLEU Score: 下降 < 5%

---

## 故障排除

### Issue 1: Inplace操作错误

**症状**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

**解决**: 将所有`+=`, `*=`, `/=`改为`a = a + b`形式

### Issue 2: OOM

**解决**:
- 减少batch_size
- 使用`torch_dtype="float16"`
- 减少`max_vision_tokens`

### Issue 3: Discriminator过强

**解决**:
- 降低discriminator学习率
- 增加dropout
- 启用`disc_reinit_prob`

---

## 参考文献

1. Token Merging: Bolya et al., ICLR 2023
2. Gumbel-Softmax: Jang et al., ICLR 2017
3. LLaVA: Liu et al., NeurIPS 2023
4. Spectral Normalization: Miyato et al., ICLR 2018

---

## 实现清单

- [x] 创建`method/models/token_merger.py`
- [x] 创建`method/models/layer_pruner.py`
- [x] 实现`method/training.py`
- [x] 实现`method/evaluation.py`
- [x] 实现`method/utils.py`
- [x] 更新`configs/vision_token_pruning.yaml`
- [x] 修复所有inplace操作错误
- [x] 完成端到端测试

**祝实现顺利！**

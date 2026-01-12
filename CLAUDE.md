# Vision Token Pruning with GAN

基于 GAN 对抗训练的多模态大语言模型（MLLM）视觉 token 剪枝项目。

## 概述
本项目实现了针对 MLLM（如 LLaVA、Qwen）的视觉 token 剪枝方法，通过在多个 LLM 层（如第 5/15/25 层）进行渐进式剪枝，在保持模型性能的同时大幅减少视觉 token 数量，加速推理。

## 核心特性
- **基于 GAN 的对抗训练**：Discriminator 确保剪枝后的表示保持质量
- **Gumbel-Softmax 采样**：使用 PyTorch 原生的 F.gumbel_softmax（hard=True）实现可微二分类决策，配合温度退火
- **Layer-wise Pruning**：在 LLM 内部多层进行渐进式剪枝
- **问题感知剪枝**：Cross-attention 机制使剪枝决策考虑问题上下文

## 架构

### 核心组件

#### Layer-Specific Pruners（LLM 内部阶段）
- **VisionPrunerHead**：基于 cross-attention 的每层剪枝头
- **独立学习**：不同层（如第 5/15/25 层）独立学习剪枝策略
- **渐进式剪枝**：
  - 早期层（Layer 5）：移除明显不相关的 token（如背景）
  - 中期层（Layer 15）：进一步精炼，去除冗余细节
  - 后期层（Layer 25）：只保留对最终预测最关键的 token

#### Discriminator
- **多层隐藏状态判别器**：从多个 LLM 层提取特征进行判别
- **谱归一化支持**：提高 GAN 训练稳定性
- **随机重初始化机制**：防止判别器过强

### 训练流程
```
图像 → 视觉编码器 → 带剪枝 Hooks 的 LLM → 输出（Fake）
                           ↓
            不带剪枝的 LLM → 输出（Real）
                           ↓
            Discriminator 判断 Real vs Fake
```

## 损失函数

### Layer Pruners 损失
1. **Task Loss**：答案预测的交叉熵损失，保持任务性能
2. **Adversarial Loss**：GAN 损失，使剪枝后的表示能欺骗判别器
3. **Sparsity Loss**：约束 token 保留率，确保达到目标稀疏度
   - 基于所有层的平均保留率
   - 可选：仅在超过目标时惩罚（sparsity_loss_only_on_excess）
4. **Token Count Loss**：直接优化 token 数量，鼓励减少
5. **Binarization Loss**：鼓励 soft_mask 接近 0 或 1，减少模糊决策

### Discriminator 损失
- **Real Loss**：将真实样本判别为真
- **Fake Loss**：将剪枝样本判别为假

### 动态权重调度
- **Task weight**：从高开始（150.0），逐渐降低至 120.0（优先学习保留信息）
- **Adversarial weight**：从低开始（10.0），逐渐提高至 80.0（后期强化对抗）
- **Sparsity weight**：从低（10.0）warmup 到高（40.0），防止 token 数反弹
- **余弦调度**：平滑过渡，避免训练不稳定

## 配置

### 关键超参数
- `pruning_layers: [5, 15, 25]`：要剪枝的 LLM 层索引
- `target_token_num: 200`：目标保留的 token 数量
- `temperature: 0.8`：Gumbel-Softmax 初始温度
- `temperature_min: 0.2`：Gumbel-Softmax 最小温度
- `temperature_anneal_rate: 0.4`：温度退火比例（前 40% 步数）

完整配置选项请参见 `configs/vision_token_pruning.yaml`。

## 使用方法

### 训练
```bash
python main.py
```

配置从 `configs/vision_token_pruning.yaml` 加载。

### 评估
```bash
python evaluate_checkpoint.py --checkpoint_path outputs/checkpoints/...
```

### 超参数搜索（Optuna）
```bash
python optuna_search.py
```

配置搜索空间和策略，自动寻找最优超参数组合。

## 支持的模型
- **LLaVA-1.5-7B**：默认使用的模型
- **LLaVA-Next**：支持更高分辨率
- **Qwen-2.5-3B**：轻量级替代方案

## 支持的数据集
- **VQA v2**：视觉问答数据集（默认）
- **GQA**：基于场景图的视觉问答
- **MMBench**：多模态评测基准
- **POPE**：对象幻觉评估
- **ScienceQA**：科学问答
- **SEED-Bench**：多模态理解评测
- **MME**：综合性能评测

## 实现细节

### Gumbel-Softmax
使用 PyTorch 原生的 `F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)`：
- **hard=True**：前向传播使用 one-hot（0/1），反向传播使用软梯度（STE 变体）
- **温度退火**：从 0.8 开始，在前 40% 训练步数中退火至 0.2
- **优势**：
  - 数值稳定
  - 训练/推理一致性
  - 更好的梯度流

### 温度退火
```python
progress = current_step / total_steps
if progress < anneal_rate:
    current_temp = temp_max - (progress / anneal_rate) * (temp_max - temp_min)
else:
    current_temp = temp_min
```

### 渐进式剪枝
- **早期层**（如 Layer 5）：移除背景、无关区域的 token
- **中期层**（如 Layer 15）：进一步精炼，去除冗余细节
- **后期层**（如 Layer 25）：只保留对最终答案最关键的 token
- 每层独立学习，但 mask 累积乘法

## 文件结构
```
domain-adaptation-token-pruning/
├── main.py                    # 主训练脚本
├── evaluate_checkpoint.py     # 评估脚本
├── optuna_search.py          # 超参数搜索
├── configs/                   # 配置文件
│   └── vision_token_pruning.yaml
├── method/                    # 核心实现
│   ├── models/
│   │   ├── layer_pruner.py   # 逐层剪枝头
│   │   ├── token_merger.py   # Token 合并模块（遗留）
│   │   ├── discriminator.py  # GAN 判别器
│   │   └── generator.py      # 旧版 Generator（遗留）
│   ├── training.py           # 训练循环
│   ├── evaluation.py         # 评估逻辑
│   └── utils.py              # 辅助函数
└── engine/                    # 训练框架
    ├── backbones/            # MLLM 实现
    ├── datas/                # 数据集加载器
    ├── trainers/             # 训练器
    └── managers/             # 任务管理
```

## 训练技巧

### 超参数调优建议
1. **温度退火**：
   - 初始温度太高（>1.0）：模型探索过度，训练不稳定
   - 初始温度太低（<0.5）：模型探索不足，陷入局部最优
   - 推荐：0.5-0.8 开始，退火至 0.1-0.3

2. **损失权重平衡**：
   - Task weight 过高：准确率高但剪枝不足
   - Adversarial weight 过高：过度剪枝，准确率下降
   - Sparsity weight 过高：强制剪枝，可能损害性能
   - 使用 Optuna 自动搜索最优组合

3. **目标 token 数**：
   - 从较宽松的目标开始（如 300 tokens）
   - 逐步降低至目标（如 200 tokens）
   - 监控准确率下降

### 训练监控
关注以下指标：
- `avg_kept_ratio`：所有层的平均 token 保留率
- `final_kept_ratio`：最后一层的 token 保留率
- `disc_accuracy`：判别器准确率（应在 60-80%）
- `task_loss`：任务损失（应稳定下降）

## 性能指标

### 评估模式
- **origin**：无剪枝，baseline 性能
- **hard**：硬剪枝（threshold=0.5），实际部署性能

### 评分机制
```python
acc_drop = accuracy_baseline - accuracy_hard
if acc_drop > 0:
    acc_penalty = acc_drop * (1 + 50 * acc_drop)  # 准确率下降的指数惩罚

keep_ratio = hard_avg_keep_ratio
if keep_ratio < 0.10:
    keep_ratio_penalty = 10.0 * (0.10 - keep_ratio)  # 过度剪枝惩罚
elif keep_ratio > 0.60:
    keep_ratio_penalty = 10.0 * (keep_ratio - 0.60)  # 剪枝不足惩罚
else:
    keep_ratio_penalty = 0.1 * keep_ratio  # 轻微惩罚鼓励更低保留率

score = -acc_penalty - keep_ratio_penalty
```

目标：在保持准确率的同时，最大化 token 剪枝率。

## 常见问题

### Q: 准确率下降严重怎么办？
A:
1. 增加 `task_loss_weight`
2. 降低 `adv_loss_weight`
3. 放宽 `target_token_num`（增加保留 token 数）
4. 检查温度退火是否过快

### Q: Token 保留率不收敛？
A:
1. 增加 `sparsity_weight` 和 `sparsity_weight_max`
2. 检查 `sparsity_loss_only_on_excess` 设置
3. 增加 `token_count_loss_weight`

### Q: 判别器准确率过高/过低？
A:
- 过高（>90%）：判别器过强，增加 `disc_dropout` 或 `disc_reinit_prob`
- 过低（<50%）：判别器过弱，降低 `disc_dropout`，检查 `adv_loss_weight`

## 更新日志

### 最新改进
- **使用 PyTorch 原生 Gumbel-Softmax**：替换手动实现，提高数值稳定性和训练/推理一致性
- **优化损失权重调度**：余弦调度，更平滑的训练过程

## 参考文献
- Gumbel-Softmax: [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- LLaVA: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- Straight-Through Estimator: [Estimating or Propagating Gradients Through Stochastic Neurons](https://arxiv.org/abs/1308.3432)

## 引用
如果本项目对您的研究有帮助，请考虑引用：
```
[待添加发表信息]
```

## 许可
[待添加许可信息]

## 联系方式
[待添加联系信息]

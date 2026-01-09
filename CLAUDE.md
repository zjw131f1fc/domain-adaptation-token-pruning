# Vision Token Pruning - HF Trainer 迁移项目

## 任务目标

将现有的Vision Token Pruning训练流程迁移到Hugging Face Trainer框架，以支持FSDP多卡训练解决显存瓶颈问题。

### 核心挑战

1. **复杂的训练流程**：GAN对抗训练 + 两阶段token剪枝
2. **多组件架构**：Backbone (LLaVA-1.5-7B) + Token Merger + Layer Pruners + Discriminator
3. **多optimizer需求**：三组不同学习率的optimizer
4. **Hook机制**：动态注入layer-wise pruning逻辑
5. **显存限制**：单卡24GB RTX 3090，需要多卡并行

## 解决方案设计

### 方案概述

创建统一包装模型 `VisionTokenPruningModel`，将所有训练逻辑封装到模型的forward中：

1. **模型包装**：创建`VisionTokenPruningModel(nn.Module)`包含所有组件
2. **Hook替代**：通过继承LLaVA模型并覆写layers来实现剪枝，而非外部hook
3. **单Optimizer + Parameter Groups**：使用不同学习率的参数组替代多optimizer
4. **HF Trainer集成**：利用HF Trainer的FSDP支持实现多卡训练

### 架构设计

```
VisionTokenPruningModel (nn.Module)
├── backbone: LLaVAWithPruning (继承并覆写)
│   ├── vision_tower
│   ├── multi_modal_projector
│   └── language_model (覆写的layers)
│       └── layers[i]: PruningLayerWrapper (注入剪枝逻辑)
├── token_merger: TokenMerger
├── layer_pruners: LayerPrunerGroup
└── discriminator: Discriminator

Optimizer: Adam with Parameter Groups
├── token_merger params (lr=1e-5)
├── layer_pruners params (lr=1e-4)
└── discriminator params (lr=6e-4)
```

### 关键实现点

#### 1. 继承覆写替代Hook

**原方案**（使用hook）：
```python
handles = register_multi_layer_hooks(
    backbone, layer_pruners, vision_pos, question_embeddings
)
result = backbone.forward(...)
remove_hooks(handles)
```

**新方案**（继承覆写）：
```python
class LLaVAWithPruning(LlavaForConditionalGeneration):
    def __init__(self, original_model, layer_pruners):
        # 复用原模型权重
        self.__dict__ = original_model.__dict__.copy()
        self.layer_pruners = layer_pruners
        self._override_language_model()

    def _override_language_model(self):
        # 包装需要剪枝的layers
        for layer_idx in pruning_layers:
            self.language_model.model.layers[layer_idx] = \
                PruningLayerWrapper(original_layer, pruner)
```

#### 2. 统一模型Forward

将 `method/training.py:train_step()` 的逻辑移到模型forward中：

```python
class VisionTokenPruningModel(nn.Module):
    def forward(self, batch):
        # Phase 1: Preprocess
        emb_info = self.backbone.preprocess_batch(images, questions, answers)

        # Phase 2: Token Merge
        merged_vision = self.token_merger(vision_features, question_emb)

        # Phase 3: Pruning Forward (fake)
        self._freeze_discriminator()
        result_fake = self.backbone(embeddings=merged_embeddings, ...)

        # Phase 4: Real Forward (no pruning)
        with torch.no_grad():
            result_real = self.backbone_original(embeddings=original_embeddings, ...)

        # Phase 5: Discriminator Loss
        self._unfreeze_discriminator()
        disc_loss = self._compute_disc_loss(result_real, result_fake)

        # Phase 6: Total Loss
        total_loss = gen_loss + disc_loss
        return {"loss": total_loss, "metrics": {...}}
```

#### 3. Parameter Groups Optimizer

```python
optimizer = torch.optim.Adam([
    {'params': model.token_merger.parameters(), 'lr': 1e-5},
    {'params': model.layer_pruners.parameters(), 'lr': 1e-4},
    {'params': model.discriminator.parameters(), 'lr': 6e-4},
])

# 在HF Trainer中使用
trainer.build_trainer(
    model=model,
    trainer_kwargs={
        "optimizers": (optimizer, None)
    }
)
```

## 环境配置

- **Conda环境**: `rl-pruning`
- **Python**: 3.11
- **关键包位置**:
  - transformers: `/home/ubuntu/anaconda3/envs/rl-pruning/lib/python3.11/site-packages/transformers/`
  - torch: `/home/ubuntu/anaconda3/envs/rl-pruning/lib/python3.11/site-packages/torch/`
- **GPU**: 8x NVIDIA RTX 3090 (24GB each)
- **可用GPU**: [2, 5, 7] (配置中设置)

## 关键文件

### 现有实现
- `method/training.py`: 当前的train_step逻辑（已被统一到VisionTokenPruningModel）
- `method/utils.py`: 工具函数（hook注册、token替换等）
- `method/models/layer_pruner.py`: Layer Pruner实现
- `method/models/token_merger.py`: Token Merger实现
- `method/models/discriminator.py`: Discriminator实现
- `engine/trainers/impl/basic_pytorch.py`: 当前trainer
- `configs/vision_token_pruning.yaml`: 原训练配置

### 新增文件（✅ 已实现）
- `method/models/pruning_wrapper.py`: ✅ LLaVAWithPruning + PruningLayerWrapper
- `method/models/unified_model.py`: ✅ VisionTokenPruningModel
- `engine/trainers/impl/hf_trainer.py`: ✅ HFTrainerWrapper增强版（支持FSDP）
- `configs/vision_token_pruning_fsdp.yaml`: ✅ FSDP配置示例
- `test_vtp_hf_trainer.py`: ✅ 集成测试脚本

## 待验证问题

### 1. FSDP + Parameter Groups 兼容性
- [ ] 验证FSDP是否支持不同学习率的参数组
- [ ] 测试FSDP wrap策略对分组optimizer的影响

### 2. 继承覆写 + FSDP 兼容性
- [ ] 验证覆写的layers能否正确被FSDP wrap
- [ ] 测试模型序列化/反序列化

### 3. 显存优化效果
- [ ] 对比单卡vs多卡显存占用
- [ ] 测试gradient checkpointing对显存的影响
- [ ] 验证两次forward（real+fake）的显存峰值

### 4. 训练效率
- [ ] 对比HF Trainer vs 原basic-pytorch的训练速度
- [ ] 测试FSDP通信开销

## 实现计划

### Phase 1: Prototype验证（✅ 已完成）
1. [x] 添加HF Trainer到trainer loader
2. [x] 实现PruningLayerWrapper（method/models/pruning_wrapper.py）
3. [x] 实现LLaVAWithPruning包装模型（method/models/pruning_wrapper.py）
4. [x] 实现VisionTokenPruningModel统一模型（method/models/unified_model.py）
5. [x] 增强HFTrainerWrapper支持自定义optimizer和FSDP（engine/trainers/impl/hf_trainer.py）
6. [x] 创建配置示例（configs/vision_token_pruning_fsdp.yaml）
7. [x] 创建测试脚本（test_vtp_hf_trainer.py）
8. [ ] 单GPU测试验证（下一步）

### Phase 2: FSDP集成
1. [ ] 配置FSDP策略
2. [ ] 测试多卡训练
3. [ ] 调试显存和通信问题

### Phase 3: 完整迁移
1. [ ] 迁移完整train_step逻辑
2. [ ] 迁移所有loss计算
3. [ ] 迁移metrics统计
4. [ ] 端到端测试

### Phase 4: 优化和验证
1. [ ] 性能对比和优化
2. [ ] 准确性验证（对比原实现）
3. [ ] 文档和配置更新

## 源码查看技巧

### 查看transformers中的LLaVA实现
```bash
# 定位LLaVA模型文件
conda run -n rl-pruning python -c "
from transformers.models.llava import modeling_llava
print(modeling_llava.__file__)
"

# 查看源码
cat /home/ubuntu/anaconda3/envs/rl-pruning/lib/python3.11/site-packages/transformers/models/llava/modeling_llava.py
```

### 查看FSDP实现
```bash
# 定位FSDP文件
conda run -n rl-pruning python -c "
from torch.distributed.fsdp import FullyShardedDataParallel
import inspect
print(inspect.getfile(FullyShardedDataParallel))
"
```

### 查看HF Trainer optimizer处理
```bash
# 查看Trainer如何处理optimizer
cat /home/ubuntu/anaconda3/envs/rl-pruning/lib/python3.11/site-packages/transformers/trainer.py | grep -A 20 "def create_optimizer"
```

## 配置示例

### HF Trainer配置
```yaml
trainer_settings:
  type: "deep-learning"
  name: "hf-trainer"  # 切换到HF Trainer
  dl_settings:
    batch_size: 4
    epochs: 3
  hf_settings:
    # TrainingArguments参数
    gradient_accumulation_steps: 4
    bf16: true
    learning_rate: 1.0e-4  # 基础lr（parameter groups会覆盖）
    warmup_ratio: 0.1

    # FSDP配置
    fsdp: "full_shard"
    fsdp_config:
      fsdp_transformer_layer_cls_to_wrap: ["LlamaDecoderLayer"]
      fsdp_backward_prefetch: "backward_pre"
      fsdp_cpu_ram_efficient_loading: true
```

## 参考资料

- [HF Trainer文档](https://huggingface.co/docs/transformers/main_classes/trainer)
- [FSDP教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [LLaVA模型结构](https://huggingface.co/docs/transformers/model_doc/llava)
- [GAN训练最佳实践](https://github.com/soumith/ganhacks)

## 备注

- 优先使用Parameter Groups而非分离的optimizers
- 继承覆写优于外部hook（FSDP兼容性更好）
- 保持与原训练逻辑的一致性，逐步验证迁移

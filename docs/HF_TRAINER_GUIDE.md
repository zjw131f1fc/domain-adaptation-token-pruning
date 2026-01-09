# Vision Token Pruning - HF Trainer + FSDP 使用指南

## 快速开始

### 1. 运行测试（单GPU）

首先验证新架构是否正常工作：

```bash
conda activate rl-pruning
python test_vtp_hf_trainer.py
```

这个测试会：
- ✓ 创建VisionTokenPruningModel
- ✓ 验证optimizer groups
- ✓ 测试forward pass
- ✓ 构建HF Trainer

### 2. 单GPU训练

使用FSDP配置，但在单GPU上训练（自动禁用FSDP）：

```bash
conda activate rl-pruning
python main.py --config configs/vision_token_pruning_fsdp.yaml
```

### 3. 多GPU训练（FSDP）

使用HF Accelerate启动多卡训练：

```bash
conda activate rl-pruning

# 方式1: 使用accelerate launch
accelerate launch --config_file accelerate_config.yaml main.py \
    --config configs/vision_token_pruning_fsdp.yaml

# 方式2: 使用torchrun
torchrun --nproc_per_node=3 main.py \
    --config configs/vision_token_pruning_fsdp.yaml
```

## 架构说明

### 核心组件

```
VisionTokenPruningModel (统一模型)
├── backbone: LLaVAWithPruning
│   └── 集成pruning的LLaVA（覆写decoder layers）
├── layer_pruners: LayerSpecificPruner
├── discriminator: Discriminator
└── token_merger: TokenMerger (可选)

单个optimizer with parameter groups:
├── layer_pruners params (lr=1e-4)
├── discriminator params (lr=6e-4)
└── token_merger params (lr=1e-5)
```

### 关键特性

1. **无需外部Hook**
   - 通过继承LLaVA并覆写layers实现pruning
   - FSDP兼容性更好

2. **单Optimizer + Parameter Groups**
   - 不同组件使用不同学习率
   - HF Trainer原生支持

3. **统一Forward**
   - 所有训练逻辑（GAN、pruning、loss计算）都在model.forward()
   - 返回单个loss给HF Trainer

4. **FSDP多卡训练**
   - 完全分片（ZeRO-3）降低显存
   - 自动梯度同步和参数更新

## 配置说明

### 关键配置项

```yaml
trainer_settings:
  name: "hf-trainer"  # 使用HF Trainer

  dl_settings:
    batch_size: 4  # per-device batch size

    # optimizer配置（用于parameter groups）
    optimizers:
      layer_pruners:
        lr: 1.0e-4
      discriminator:
        lr: 6.0e-4

  hf_settings:
    # 混合精度
    bf16: true

    # Gradient accumulation
    gradient_accumulation_steps: 4

    # FSDP配置
    fsdp: "full_shard"  # ZeRO-3
    fsdp_config:
      fsdp_transformer_layer_cls_to_wrap: ["LlamaDecoderLayer"]
      fsdp_backward_prefetch: "backward_pre"
```

### 显存优化选项

如果仍然OOM，可以尝试：

1. **减小batch size**:
   ```yaml
   batch_size: 2
   gradient_accumulation_steps: 8  # 保持有效batch=16
   ```

2. **启用CPU offload**:
   ```yaml
   fsdp_config:
     fsdp_cpu_ram_efficient_loading: true
   ```

3. **Gradient checkpointing**（在backbone加载时）:
   ```yaml
   backbone_settings:
     mllm_settings:
       use_gradient_checkpointing: true
   ```

## 与原版对比

### 旧架构（basic-pytorch）
```python
# 使用外部hook注入pruning
handles = register_multi_layer_hooks(...)
result = backbone.forward(...)
remove_hooks(handles)

# 多个独立optimizer
optimizer_disc.step()
optimizer_pruner.step()
optimizer_merger.step()
```

### 新架构（hf-trainer + FSDP）
```python
# Pruning集成在模型内部
model = VisionTokenPruningModel(...)
outputs = model(batch)  # 自动应用pruning

# 单个optimizer with groups
optimizer = Adam(model.create_optimizer_groups())
optimizer.step()

# HF Trainer处理所有训练循环
trainer.train()
```

## 调试技巧

### 1. 查看模型结构
```python
from method.models.unified_model import VisionTokenPruningModel
model = VisionTokenPruningModel(...)
print(model)
```

### 2. 检查optimizer groups
```python
param_groups = model.create_optimizer_groups()
for i, group in enumerate(param_groups):
    print(f"Group {i}: lr={group['lr']}, {len(list(group['params']))} params")
```

### 3. 验证pruning是否生效
```python
model.eval()
outputs = model(batch)
print(f"Avg tokens after pruning: {outputs['avg_tokens']}")
```

### 4. 监控FSDP状态
```python
# 在训练中
trainer.hf_trainer.state.log_history  # 查看训练日志
```

## 常见问题

### Q: FSDP wrap错误
```
RuntimeError: Expected to have finished reduction in the prior iteration...
```
**解决**: 确保所有参数都被wrap，检查`fsdp_transformer_layer_cls_to_wrap`配置。

### Q: Parameter groups不生效
确认`optim="adam"`或`optim="adamw_torch"`，其他optimizer可能不支持groups。

### Q: 显存仍然不足
尝试：
1. 减小batch size
2. 启用CPU offload
3. 使用`fsdp: "shard_grad_op"` (ZeRO-2) 而非full_shard

## 后续优化

- [ ] Gradient checkpointing集成
- [ ] Mixed precision优化（FP8）
- [ ] Flash Attention集成
- [ ] 分布式evaluation

## 参考资料

- [HF Trainer文档](https://huggingface.co/docs/transformers/main_classes/trainer)
- [FSDP教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Accelerate文档](https://huggingface.co/docs/accelerate/)

# visualize_distribution_shift.py - Skip Adapter 使用说明

## 新增功能

脚本现在支持选择性加载 repair adapter，可以跳过指定层的 adapter。

## 命令行参数

- `--skip_repair_layers_a`: 指定 checkpoint A 要跳过的 repair adapter 层
- `--skip_repair_layers_b`: 指定 checkpoint B 要跳过的 repair adapter 层

## 使用方式

### 1. 跳过第一个 adapter（最靠近输入的层）

```bash
python scripts/visualize_distribution_shift.py \
    --checkpoint_real path/to/real.ckpt \
    --checkpoint_a path/to/a.ckpt \
    --checkpoint_b path/to/b.ckpt \
    --config_real configs/config.yaml \
    --config_a configs/config.yaml \
    --config_b configs/config.yaml \
    --skip_repair_layers_a first \
    --skip_repair_layers_b first
```

### 2. 跳过最后一个 adapter

```bash
python scripts/visualize_distribution_shift.py \
    ... \
    --skip_repair_layers_a last
```

### 3. 跳过指定层号的 adapter

假设 checkpoint 中有 3 个 adapter 在层 [13, 22, 29]，跳过层 13：

```bash
python scripts/visualize_distribution_shift.py \
    ... \
    --skip_repair_layers_a 13
```

### 4. 跳过多个 adapter

跳过层 13 和 22：

```bash
python scripts/visualize_distribution_shift.py \
    ... \
    --skip_repair_layers_a 13,22
```

### 5. 对 A 和 B 使用不同的跳过策略

```bash
python scripts/visualize_distribution_shift.py \
    ... \
    --skip_repair_layers_a first \
    --skip_repair_layers_b 13,22
```

## 工作原理

1. **自动检测**：脚本会自动从 checkpoint 中检测所有 adapter 层
2. **解析关键字**：
   - `first`: 自动跳过第一个（层号最小的）adapter
   - `last`: 自动跳过最后一个（层号最大的）adapter
   - 数字：直接指定要跳过的层号
3. **选择性加载**：只加载未被跳过的 adapter 权重
4. **模型构建**：使用过滤后的层列表构建模型

## 示例输出

```
=== Resolving skip_repair_layers ===
  Checkpoint A: will skip repair layers [13]
  Checkpoint B: will skip repair layers [13]

=== Load & run A (summary) ===
  Note: config repair_layers=[13, 22, 29] != checkpoint repair_layers=[13, 22, 29]; using checkpoint.
  Note: Skipping repair layers [13]. Using layers [22, 29] (original: [13, 22, 29])
  Loaded pruner_state_dict
  Loaded repair_adapter_state_dict (skipped layers: [13])
```

## 注意事项

1. 如果跳过所有 adapter，repair adapter 功能会被自动禁用
2. Real baseline 不受 skip_repair_layers 影响（始终使用 keep-all + no repair）
3. 跳过的层不会被加载到模型中，节省内存

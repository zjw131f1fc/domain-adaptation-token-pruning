# 简化版 Repair 分析脚本使用说明

## 功能

这是从 `analyze_repair_objective.py` 提取的简化版本，用于快速分析 repair adapter 的效果。

## 核心功能

- **三次前向传播对比**：
  - Teacher: keep_all 模式（不剪枝）
  - Student OFF: 正常剪枝，repair 禁用
  - Student ON: 正常剪枝，repair 启用

- **计算指标**：
  - Token-wise MSE（相对于 Teacher）
  - 每层的 Gain = MSE_OFF - MSE_ON

- **输出**：
  - CSV 表格：每层的 MSE 和 Gain
  - 两张图表：MSE 对比图、Gain 曲线图

## 使用方法

```bash
# 基本用法（只需要 checkpoint 参数）
python scripts/simple_repair_analysis.py \
    --checkpoint outputs/tasks/20260304-1405_vqa-vqav2_llava157b_6056/checkpoints/checkpoint_final.pt

# 跳过第一个 repair adapter（覆盖 deployed layer 设置）
python scripts/simple_repair_analysis.py \
    --checkpoint outputs/tasks/20260304-1405_vqa-vqav2_llava157b_6056/checkpoints/checkpoint_final.pt \
    --skip_first_adapter

# 或者使用 CUDA_VISIBLE_DEVICES 指定 GPU
CUDA_VISIBLE_DEVICES=4 python scripts/simple_repair_analysis.py \
    --checkpoint outputs/tasks/20260304-1405_vqa-vqav2_llava157b_6056/checkpoints/checkpoint_final.pt \
    --skip_first_adapter
```

## 参数说明

- `--checkpoint`: 训练好的检查点路径（必需）
- `--skip_first_adapter`: 跳过第一个 repair adapter 的加载（可选）
  - 启用此选项后，会从 repair_layers 中移除第一个层
  - 不加载第一个 adapter 的权重
  - 用于测试不同 adapter 配置的效果

## 固定参数（硬编码）

- 配置文件: `configs/vision_token_pruning.yaml`
- 模型路径: `llava-hf/llava-1.5-7b-hf`
- 样本数: 64
- Batch size: 1
- 数据集: test split
- 捕获层: 所有层

## 输出

输出目录自动生成：
- 默认: `outputs/visualizations/simple_repair_{checkpoint_name}/`
- 跳过第一个 adapter: `outputs/visualizations/simple_repair_{checkpoint_name}_skip1st/`

包含：
- `repair_analysis.csv`: 每层的详细指标
- `mse_comparison.png`: MSE 对比图（OFF vs ON）
- `gain.png`: Gain 曲线图

## 与原版的区别

| 特性 | 原版 | 简化版 |
|------|------|--------|
| 参数数量 | 20+ | 1 |
| 输出文件 | 10+ | 3 |
| 图表类型 | 8+ | 2 |
| PCA/C2ST | 支持 | 移除 |
| 损失类型 | 可配置 | 固定 MSE |
| 层排除 | 支持 | 移除 |

## 示例输出

```
加载模型: outputs/tasks/.../checkpoint_final.pt
模式: 跳过第一个 repair adapter
跳过第一个 adapter，使用 repair_layers: [15, 25]
已加载 pruner_state_dict
跳过加载权重: adapters.5.xxx
已加载 repair adapter（跳过第一个 adapter layer 5）
模型层数: 32
剪枝层: [4, 14, 24]
Repair 层: [15, 25]
目标 token 数: 64

加载数据: 64 个样本
开始评估...
已处理 10/64 个样本...
已处理 20/64 个样本...
...

=== 摘要 ===
平均 MSE (OFF): 0.001234
平均 MSE (ON):  0.000987
平均 Gain:      0.000247

Repair 层平均 Gain: 0.000312

所有输出已保存到: outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st
```

## 注意事项

1. 确保 checkpoint 路径正确
2. 确保配置文件 `configs/vision_token_pruning.yaml` 存在
3. 需要 GPU（自动使用 cuda:0）
4. 评估 64 个样本大约需要几分钟

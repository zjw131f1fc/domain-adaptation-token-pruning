# TODO

## 当前实现说明

### Adapter 设计
- **已移除 query 参数**：Adapter 现在只使用 `mask` 进行 FiLM 调制，不再使用 `query`
- **作用范围**：Adapter 对所有 token 起效
- **设计含义**：对所有直接接触到剪枝后结果的 token，都施加修复

### Adversarial Loss 设计
- **作用范围**：adv_loss 只应用于生成 answer token 的 token（即 answer 位置前一个位置的 hidden states）
- **设计含义**：关注生成 answer 时的表示质量，确保剪枝后的表示与完整表示在判别器看来无法区分

### 推理优化（待实现）
- **现状**：因为计算 hard_mask 的位置已经提前（在 attention 计算之后、Adapter 之前），理论上 evaluate 可以提前进行物理剪枝，省去更多的 FLOPs
- **当前实现**：仿照训练流程，使用 attention 重归一化的方式（软剪枝），然后在层末尾进行物理删除
- **潜在优化**：可以在计算出 hard_mask 后立即进行物理剪枝，跳过被剪掉 token 的后续计算

## 待办事项

- [ ] 优化推理流程：在 hard_mask 计算后立即物理剪枝，减少 FLOPs
- [ ] 评估移除 query 后的训练效果
- [ ] 考虑是否需要简化 LightweightAdapter（移除 query_proj 相关代码）

# Domain Adaptation Token Pruning (ACP) — Agent Notes

本仓库实现了基于 LLaVA (1.5) 的 **vision token pruning**（Attention Consistency Pruning, ACP）以及配套的
**delayed repair adapter**（部署口径/推理口径也在仓库中实现）。

> 重要：仓库内存在不少过时的文档/注释（尤其是 `TODO.md` 与部分 YAML 里的 TODO）。以 **实际代码路径** 为准。

---

## 1) “真相来源”（建议从这些文件读起）

**训练 / 评估入口**
- `main_acp_ddp.py`：torchrun 启动的分布式训练主脚本（但不是用 DDP wrapper；见下文“分布式”）。
- `eval_acp_ddp.py`：torchrun 启动的分布式评估脚本。
- `scripts/run_ddp.sh`：训练启动器（设置 GPU、torchrun、wrapper log）。
- `scripts/run_eval_ddp.sh`：评估启动器（同上）。
- `scripts/benchmark_latency.py`：延迟基准（包含 hard 物理剪枝 + deployed adapter 的速度口径）。

**核心模块**
- `method/models/prunable_llava.py`：PrunableLlavaForConditionalGeneration（forward / hard generate 两条推理路线）。
- `method/models/prunable_llama_layer.py`：PrunableLlamaDecoderLayer（剪枝层/非剪枝层的实际 masking 逻辑）。
- `method/models/layer_pruner_acp.py`：CrossAttentionPruner + LayerPrunerManager（pruner 本体）。
- `method/models/adapter.py`：RepairContextEncoder + LightweightAdapter + AdapterManager（delayed repair adapter）。
- `engine/train_utils.py`：train_step（task loss / sparsity / repair loss 等训练目标）。
- `engine/eval_utils.py`：evaluate（origin / hard_forward / hard 三种 eval 路由）。
- `engine/data_utils.py`：preprocess_batch（最关键的 “vision/question/answer 位置标注”）。
- `engine/datas/loader.py` + `engine/datas/impl/*`：数据集 registry（VQAv2/POPE/MME/…）。
- `engine/configs/loader.py`：配置合并 + 自动任务目录（`outputs/tasks/<tag>`）。

---

## 2) 仓库结构与职责边界

- `engine/`：训练/评估/数据/配置“基础设施层”。
- `method/`：模型改造与方法实现（pruner、prunable layer、adapter）。
- `configs/`：实验 YAML（注意：有些注释可能过时）。
- `scripts/`：启动脚本、分析脚本、可视化与 benchmark。
- `.history*`：历史/临时代码，不作为主路径；默认不要改、不要依赖。

---

## 3) 核心概念：Pruner（ACP）

### 3.1 CrossAttentionPruner（`method/models/layer_pruner_acp.py`）

输入：
- `vision_hidden`: (B, N_vision, D)
- `q2v_attn`: (B, N_vision) —— 来自 LLM 自注意力中 “question tokens → vision tokens” 的注意力均值
- `cumulative_vision_mask`: (B, N_vision) —— 累积保留 mask（1=保留，0=之前层已剪掉）
- 可选 `question_hidden/question_lengths`（当 `use_question_condition=true`）

核心：
- baseline：`log(q2v_attn)` 做中心化（且**只对当前保留位置**计算均值，避免被剪掉位置污染）。
- delta：learnable pruning queries cross-attend vision tokens + per-token scorer。
- `keep_logits = baseline + delta + keep_bias`
- mask：`gumbel_sigmoid_mask()`（训练可带 noise；推理/评估通常关闭 noise 变确定性阈值）。

### 3.2 累积剪枝约束（防止“复活”）

每个 pruning layer 会输出当前层决策 `current_mask`，并更新：
`new_cumulative_mask = cumulative_mask * current_mask`。

这条约束在所有路由里都必须成立，否则后续层可能把之前剪掉的 token “复活”。

---

## 4) 核心概念：PrunableLlamaDecoderLayer（训练/评估主路径）

文件：`method/models/prunable_llama_layer.py`

该层包装了原始 `LlamaDecoderLayer`，并分两类执行：

### 4.1 非剪枝层（但继承累计 mask）
- 若传入 `cumulative_vision_mask`，会走 “post-softmax masking + renorm”：
  - attention weights 先按 causal mask 正常 softmax
  - vision 区间乘以 `cumulative_vision_mask`
  - 再整体除以 sum 重新归一化

### 4.2 剪枝层
- 先算 `attn_weights`
- 从 `attn_weights[q_start:q_end, vision_start:vision_end]` 抽取 `q2v_attn_avg`
- 调用 pruner 产出 `current_mask`（支持 `pruning_mode=keep_all/topk_attn/normal`）
- 更新 `new_cumulative_mask`
- 在 attention 上用 `new_cumulative_mask` 做 post-softmax masking + renorm
- 返回 `pruning_info`（含 `cumulative_mask/q2v_attn/...`，以及 repair context embedding）

---

## 5) Delayed Repair Adapter（语言侧，仅修复 gen_answer tokens）

文件：`method/models/adapter.py` + 注入点 `method/models/prunable_llava.py`

### 5.1 RepairContextEncoder（pruning layer 侧缓存上下文）
输出是低维向量（便于缓存/部署）：
- `mask_emb`: 编码 cumulative vision mask（attention pooling 或 linear）
- `pruned_emb`: 聚合“被剪掉 token 的信息”（可选开关 `repair_use_pruned_info`）

### 5.2 LightweightAdapter（repair layer 侧注入修复）
FiLM 调制 + bottleneck MLP：
- 条件项可由 `mask_emb/query_emb/pruned_emb` 组成
- 输出是 residual delta：`x + alpha * up(film(down(x)))`

### 5.3 forward() 里的修复口径（训练/评估 hard_forward）
在 `PrunableLlavaForConditionalGeneration.forward()`：
- pruning layer：缓存 repair context（`mask_emb/pruned_emb`）
- repair layer：选取 repair_source_layers 或 “最近的 pruning layer” 作为上下文来源
- **仅对 gen_answer 区域**应用 `delta`
  - gen_answer 区域由 `answer_starts/answer_ends` 推导（见 `method/models/prunable_llava.py` 内的 `gen_mask_full`）
- `repair_detach_input` 控制梯度是否可经由 base hidden states 回流到 pruner

---

## 6) Deployed Adapter（hard 物理剪枝/部署口径）

文件：`method/models/prunable_llava.py`，入口 `generate_with_hard_pruning()`

hard 路线是“物理删除 + KV cache”的推理路径，用于速度/部署口径：
- pruning layer：仍会缓存 repair context（并用 padded 回 576 维保证与训练一致的 mask 语义）
- repair layer：默认只修复 **最后一个 token**（用于下一 token 预测；更贴近短生成部署）
- 物理删除发生在 pruning layer 之后，并裁剪 `past_key_values`

注意：hard 路线强调速度口径，并不保证与 forward() 数值完全一致。

---

## 7) 三条推理/评估路线（请明确你要的是哪条）

在 `engine/eval_utils.py` 中：
- `origin`：不剪枝，直接 `model.generate()`（基线）。
- `hard_forward`：用 `forward() + greedy decode` 做生成（不物理删 token）。
  - 优点：最稳；准确率能反映 delayed repair（gen_answer-only repair）的收益。
  - 缺点：慢（每步从头 forward）。
- `hard`：`generate_with_hard_pruning()`（物理删 token + KV cache）。
  - 优点：速度/部署口径；可用于 latency benchmark。
  - 风险：`engine/eval_utils.py` 里标注该路线可能存在已知 bug（以当前实现为准）。

---

## 8) 数据与 “位置语义”（经常是隐含 bug 来源）

`engine/data_utils.py:preprocess_batch()` 返回并约定：
- `n_vision` 固定为 576（与 LLaVA 1.5 默认 patch tokens 一致；hard 路线也按此 padded）
- `vision_start/vision_end`：通过 `<image>` token 定位（找不到则 fallback 到 1..576）
- `question_starts/question_ends`：默认 question 从 vision_end 开始；question_end 定位到 `\nASSISTANT:` 之后
- `answer_starts/answer_ends`：训练时 answer 区域用于 task loss / gen_answer mask

如果你改 prompt 模板或 tokenizer 行为，需要同步检查：
1) `assistant_ids` 匹配逻辑
2) `vision_start/vision_end` 与模型 placeholder mask 行为

---

## 9) 配置（YAML）里真正影响路由的关键开关

见 `configs/vision_token_pruning.yaml`（以及 sweeps 下的文件）：
- `method_settings.pruning_layers`
- `method_settings.target_token_num`
- `method_settings.gumbel_mode` + hybrid 参数（训练阶段调度）
- `method_settings.eval_temperature` / `method_settings.eval_pruning_threshold`
- `method_settings.use_question_condition`（会额外构造 question_hidden）
- `method_settings.use_repair_adapter` / `repair_layers` / `repair_source_layers` / `repair_detach_input`
- Ablations：
  - `ablation_w_o_pruner_topk_attn`
  - `ablation_w_o_adapter`
  - `ablation_w_o_repair_loss`
  - `ablation_repair_mean_only`
- `evaluation_settings.eval_mode`：`["origin", "hard_forward"]` 等

---

## 10) Checkpoint 格式（加载/保存时需要对齐）

训练保存（`main_acp_ddp.py`）与评估加载（`eval_acp_ddp.py`）使用的常见 key：
- `pruner_state_dict`
- `disc_state_dict`（当前训练 loss 可能不使用判别器，但状态仍会保存）
- `repair_context_encoder_state_dict`
- `repair_adapter_state_dict`
- 可选：optimizer/scheduler state、step、batch

---

## 11) 输出目录与日志

`engine/configs/loader.py` 默认会为每次运行创建独立任务目录：
- `./outputs/tasks/<experiment_tag>/checkpoints/`
- `./outputs/tasks/<experiment_tag>/logs/`

wrapper 脚本额外写：
- `logs/ddp_runs/train_*.log`
- `logs/eval_runs/eval_*.log`

---

## 12) 分布式训练（实现细节）

虽然脚本名是 DDP，但本仓库主路径不是用 `DistributedDataParallel(model)` 包裹整个模型。
取而代之的是：
- 冻结 base model
- 手动对 pruner/adapter/disc 的梯度做 all-reduce（`engine/distributed.py:sync_gradients`）
- 用 `broadcast_model_params()` 保证初始参数一致

动机：避免冻结主干 + 条件分支导致的 DDP 不一致/死锁问题。

---

## 13) 给后续 agent 的工作约定（避免被“过时信息”带偏）

建议遵循：
1) **先用 `rg` 在代码里找调用链**，再看文档/注释。
2) 默认忽略 `.history/`、`.history_basic_trainer/` 中的实现（除非明确要回溯旧版本）。
3) 评估准确率优先用 `hard_forward`，速度/部署口径优先用 `hard` + `scripts/benchmark_latency.py`。
4) 修改剪枝/repair 行为时，务必同时检查：
   - forward 路线（`method/models/prunable_llava.py`）
   - hard 路线（同文件 `generate_with_hard_pruning()`）
   - 位置标注（`engine/data_utils.py`）


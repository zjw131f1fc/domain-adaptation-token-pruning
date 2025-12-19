# 调试修复记录

## 问题描述

评估时发现：即使 pruner 输出全1（不剪枝），准确率仍然从 0.863 下降到 0.79（约7%）。

## 调试发现

### 问题1: raw_vision_features 包含 CLS token

**现象**:
```
vision_in_emb shape: torch.Size([1, 576, 4096])    # embeddings中576个vision tokens
vision_features_raw: torch.Size([1, 577, 1024])   # raw features有577个（多了1个CLS）
```

**原因**:
`vision_tower(pixel_values).last_hidden_state` 返回577个token（含CLS），但 LLaVA 的 `get_image_features` 方法会去掉 CLS token。

### 问题2: 两次调用 vision_tower 导致数值不一致

**现象**:
```
Diff between vision_in_emb and vision_reprojected:
  max: 2.858785
  mean: 0.178843
  torch.allclose (atol=1e-5): False
```

**原因**:
原代码在 `preprocess` 函数中两次调用 `vision_tower`：
1. 第一次：获取 `raw_vision_features`
2. 第二次：通过 `get_image_features` 获取 `image_token_embeds`

即使在 `torch.no_grad()` 下，两次前向传播的结果也可能有微小差异，导致 `raw_vision_features` 重新投影后与原始 embeddings 不一致。

---

## 最终修复方案

### 修改文件: `engine/backbones/impl/mllm/llava.py`

**位置**: `preprocess` 函数中的 vision feature 获取逻辑

**修改前** (两次调用 vision_tower):
```python
# 2) 获取raw vision features
raw_vision_features = vision_tower(pixel_values).last_hidden_state  # 第一次调用

# 3) 通过 model.get_image_features 获取图像特征
image_features_list = self.model.get_image_features(pixel_values=pixel_values)  # 第二次调用
image_token_embeds = torch.cat(image_features_list, dim=0).unsqueeze(0)
```

**修改后** (只调用一次 vision_tower):
```python
# 2) 只调用一次 vision_tower，同时获取 raw 和 projected features
# 避免两次调用导致的数值不一致
raw_vision_features = None
image_token_embeds = None

if hasattr(self.model, 'vision_tower') and hasattr(self.model, 'multi_modal_projector'):
    vision_tower = self.model.vision_tower
    projector = self.model.multi_modal_projector

    # 调用 vision_tower 获取原始 features（只调用一次！）
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)

    # 获取指定层的 hidden states（默认使用 vision_feature_layer）
    vision_feature_layer = self.model.config.vision_feature_layer
    if isinstance(vision_feature_layer, int):
        selected_features = vision_outputs.hidden_states[vision_feature_layer]
    else:
        # 多层拼接的情况
        selected_features = torch.cat(
            [vision_outputs.hidden_states[idx] for idx in vision_feature_layer],
            dim=-1
        )

    # 根据 vision_feature_select_strategy 决定是否去掉 CLS
    vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
    if vision_feature_select_strategy == "default":
        selected_features = selected_features[:, 1:]  # 去掉 CLS token

    # raw_vision_features: 未投影的 CLIP 输出 (1, 576, 1024)
    raw_vision_features = selected_features

    # 通过 projector 投影到 LLM hidden dim (1, 576, 4096)
    image_token_embeds = projector(selected_features)
    image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
else:
    # Fallback: 使用 get_image_features（会额外调用一次 vision_tower）
    image_features_list = self.model.get_image_features(pixel_values=pixel_values)
    image_token_embeds = torch.cat(image_features_list, dim=0).unsqueeze(0)
    image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
```

---

## 调试用的临时修改（需要还原）

以下文件在调试过程中被临时修改，需要用 git 还原：

1. **configs/vision_token_pruning.yaml**
   - `epochs: 0` (原来是2)
   - `batch_size: 5` (原来是10)
   - `debug_all_ones: true` (新增，需要删除或改为false)
   - `train: 100` (原来是600)
   - `test: 50` (原来是200)

2. **method/models/layer_pruner.py**
   - 添加了 `debug_all_ones` 参数
   - 添加了全1输出的调试逻辑

3. **method/evaluation.py**
   - 完全重写为调试版本，需要还原

4. **main.py**
   - 添加了 `debug_all_ones` 参数传递

---

## 还原命令

```bash
# 还原所有调试修改
git checkout -- configs/vision_token_pruning.yaml
git checkout -- method/models/layer_pruner.py
git checkout -- method/evaluation.py
git checkout -- main.py

# llava.py 的修改需要保留！不要还原！
# 如果不小心还原了，需要手动重新修改 preprocess 函数
```

---

## 验证修复

修复后，运行调试评估应该看到：

1. `vision_features_raw` shape 为 `(1, 576, 1024)`（与 vision_in_emb 的576一致）
2. `vision_in_emb` 和 `vision_reprojected` 的 diff 为 **0**：
   ```
   Diff between vision_in_emb and vision_reprojected:
     max: 0.000000
     mean: 0.000000
     torch.allclose (atol=1e-5): True
   ```
3. baseline、emb_direct、emb_replaced、soft_pruning 的准确率应该**完全一致**

---

## 关键要点

1. **CLS token 处理**: LLaVA 默认使用 `vision_feature_select_strategy="default"`，会去掉 CLS token
2. **单次前向传播**: 确保 `raw_vision_features` 和 `image_token_embeds` 来自同一次 vision_tower 调用
3. **正确的层选择**: 使用 `vision_feature_layer` 配置选择正确的 hidden states 层

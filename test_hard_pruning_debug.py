"""调试hard pruning问题的测试脚本"""
import torch
from PIL import Image
import requests
from io import BytesIO

# 加载配置和模型
from engine.configs.loader import get_config_by_name
from engine.base_backbone import BackboneRegistry

# 加载配置
config = get_config_by_name("vision_token_pruning")

# 覆盖一些设置
config["model_settings"]["device_map"] = "auto"
config["model_settings"]["torch_dtype"] = "float16"
config["method_settings"]["enable_token_merger"] = False  # 禁用merger简化调试

# 加载backbone
backbone = BackboneRegistry.create(
    config["model_settings"]["backbone"],
    config["model_settings"]
)

# 创建测试图片和问题
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))
question = "What is in this image?"

print("=" * 60)
print("测试1: Baseline生成（无任何hook）")
print("=" * 60)
with torch.no_grad():
    pred_baseline = backbone.generate(image, question, max_new_tokens=20)
print(f"Baseline预测: {pred_baseline}")

# 准备评估所需的embeddings
with torch.no_grad():
    emb_info = backbone.preprocess(image, question, None)
    original_embeddings = emb_info['embeddings']
    original_vision_pos = emb_info['vision_token_positions']
    original_attention_mask = emb_info['attention_mask']

print(f"\n序列信息:")
print(f"  总tokens: {original_embeddings.shape[1]}")
print(f"  Vision位置: {original_vision_pos}")
print(f"  Vision tokens数: {original_vision_pos[1] - original_vision_pos[0] + 1}")

print("\n" + "=" * 60)
print("测试2: 使用embeddings直接生成（无hook）")
print("=" * 60)
with torch.no_grad():
    pred_emb = backbone.generate(
        embeddings=original_embeddings.clone(),
        attention_mask=original_attention_mask.clone(),
        max_new_tokens=20
    )
print(f"Embeddings预测: {pred_emb}")

# 加载layer pruners
from method.models.layer_pruner import LayerSpecificPruner
layer_pruners = LayerSpecificPruner(
    d_model=config["method_settings"]["d_model"],
    d_text=config["method_settings"]["d_text"],
    layer_indices=config["method_settings"]["pruning_layers"]
).to(backbone.device)
layer_pruners.eval()

# 提取question embeddings
v_start, v_end = original_vision_pos
question_embeddings = original_embeddings[:, v_end+1:, :]

print("\n" + "=" * 60)
print("测试3: Hard Pruning（使用register_hard_pruning_at_model_level）")
print("=" * 60)

from method.utils import register_hard_pruning_at_model_level

restore_fn, hard_context = register_hard_pruning_at_model_level(
    backbone,
    layer_pruners,
    original_vision_pos,
    question_embeddings,
    threshold=0.5
)

try:
    with torch.no_grad():
        pred_hard = backbone.generate(
            embeddings=original_embeddings.clone(),
            attention_mask=original_attention_mask.clone(),
            max_new_tokens=20
        )
    print(f"Hard预测: {pred_hard}")

    # 获取统计信息
    final_v_start, final_v_end = hard_context.get_positions()
    final_tokens = final_v_end - final_v_start + 1
    print(f"\nHard pruning统计:")
    print(f"  最终vision位置: [{final_v_start}, {final_v_end}]")
    print(f"  最终vision tokens: {final_tokens}")

    for stat in hard_context.get_stats():
        print(f"  Layer {stat['layer_idx']}: {stat['original_count']} → {stat['kept_count']} (keep_ratio={stat['keep_ratio']:.2%})")

finally:
    restore_fn()

print("\n" + "=" * 60)
print("测试4: 再次用Baseline生成（验证恢复）")
print("=" * 60)
with torch.no_grad():
    pred_after = backbone.generate(image, question, max_new_tokens=20)
print(f"恢复后预测: {pred_after}")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"Baseline: {pred_baseline}")
print(f"Embeddings: {pred_emb}")
print(f"Hard: {pred_hard}")
print(f"恢复后: {pred_after}")
print(f"Baseline == Embeddings: {pred_baseline == pred_emb}")
print(f"Baseline == Hard: {pred_baseline == pred_hard}")

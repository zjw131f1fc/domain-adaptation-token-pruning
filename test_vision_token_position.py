"""测试获取 vision token 位置的不同方法是否得到相同结果"""

import os

HF_HOME = "/data/users/zjw/huggingface_cache"
HF_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def method_1_current(model, input_ids):
    """当前使用的方法：手动查找"""
    image_token_id = model.config.image_token_index
    image_token_indices = torch.where(input_ids[0] == image_token_id)[0]

    if len(image_token_indices) == 0:
        raise ValueError("未找到图像占位 token")

    img_token_start_idx = int(image_token_indices[0])
    img_token_end_idx = int(image_token_indices[-1])

    return img_token_start_idx, img_token_end_idx, image_token_indices


def method_2_simple(model, input_ids):
    """简化方法：一行代码"""
    vision_mask = input_ids == model.config.image_token_index
    vision_indices = torch.where(vision_mask[0])[0]

    if len(vision_indices) == 0:
        raise ValueError("未找到图像占位 token")

    vision_start = int(vision_indices[0])
    vision_end = int(vision_indices[-1])

    return vision_start, vision_end, vision_indices


def method_3_library(model, input_ids, inputs_embeds, image_features):
    """使用库提供的 get_placeholder_mask 方法"""
    # 获取 mask
    special_image_mask = model.model.get_placeholder_mask(
        input_ids, inputs_embeds, image_features
    )

    # 从 mask 中提取位置（需要压缩到 2D）
    mask_2d = special_image_mask.any(dim=-1)  # (batch, seq_len)
    vision_indices = torch.where(mask_2d[0])[0]

    if len(vision_indices) == 0:
        raise ValueError("未找到图像占位 token")

    vision_start = int(vision_indices[0])
    vision_end = int(vision_indices[-1])

    return vision_start, vision_end, vision_indices


def main():
    print("=" * 60)
    print("测试不同方法获取 vision token 位置")
    print("=" * 60)

    # 加载模型和处理器
    print("\n[1] 加载 LLaVA 模型...")
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"   模型加载完成")
    print(f"   image_token_index: {model.config.image_token_index}")

    # 准备测试数据
    print("\n[2] 准备测试数据...")
    # 创建一个简单的测试图像
    image = Image.new('RGB', (224, 224), color='red')
    question = "What is in this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    input_ids = inputs['input_ids']
    pixel_values = inputs['pixel_values']

    print(f"   input_ids shape: {input_ids.shape}")
    print(f"   序列长度: {input_ids.shape[1]}")

    # 获取 embeddings 用于 method_3
    with torch.no_grad():
        text_token_embeds = model.get_input_embeddings()(input_ids)
        image_features_list = model.get_image_features(pixel_values=pixel_values)
        image_token_embeds = torch.cat(image_features_list, dim=0).unsqueeze(0)
        image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)

    print(f"   image_features shape: {image_token_embeds.shape}")
    print(f"   vision token 数量: {image_token_embeds.shape[1]}")

    # 测试方法 1
    print("\n[3] 测试方法 1（当前方法）...")
    start_1, end_1, indices_1 = method_1_current(model, input_ids)
    print(f"   起始位置: {start_1}")
    print(f"   结束位置: {end_1}")
    print(f"   vision token 数量: {len(indices_1)}")
    print(f"   前5个索引: {indices_1[:5].tolist()}")
    print(f"   后5个索引: {indices_1[-5:].tolist()}")

    # 测试方法 2
    print("\n[4] 测试方法 2（简化方法）...")
    start_2, end_2, indices_2 = method_2_simple(model, input_ids)
    print(f"   起始位置: {start_2}")
    print(f"   结束位置: {end_2}")
    print(f"   vision token 数量: {len(indices_2)}")
    print(f"   前5个索引: {indices_2[:5].tolist()}")
    print(f"   后5个索引: {indices_2[-5:].tolist()}")

    # 测试方法 3
    print("\n[5] 测试方法 3（使用库方法）...")
    start_3, end_3, indices_3 = method_3_library(
        model, input_ids, text_token_embeds, image_token_embeds
    )
    print(f"   起始位置: {start_3}")
    print(f"   结束位置: {end_3}")
    print(f"   vision token 数量: {len(indices_3)}")
    print(f"   前5个索引: {indices_3[:5].tolist()}")
    print(f"   后5个索引: {indices_3[-5:].tolist()}")

    # 对比结果
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    # 检查起始和结束位置
    positions_match = (start_1 == start_2 == start_3) and (end_1 == end_2 == end_3)
    print(f"\n起始/结束位置是否一致: {positions_match}")
    if positions_match:
        print(f"  ✓ 所有方法得到相同的起始位置: {start_1}")
        print(f"  ✓ 所有方法得到相同的结束位置: {end_1}")
    else:
        print(f"  ✗ 方法 1: start={start_1}, end={end_1}")
        print(f"  ✗ 方法 2: start={start_2}, end={end_2}")
        print(f"  ✗ 方法 3: start={start_3}, end={end_3}")

    # 检查索引列表
    indices_match_1_2 = torch.equal(indices_1, indices_2)
    indices_match_2_3 = torch.equal(indices_2, indices_3)
    indices_match_1_3 = torch.equal(indices_1, indices_3)

    print(f"\n索引列表是否完全一致:")
    print(f"  方法1 vs 方法2: {indices_match_1_2}")
    print(f"  方法2 vs 方法3: {indices_match_2_3}")
    print(f"  方法1 vs 方法3: {indices_match_1_3}")

    all_match = indices_match_1_2 and indices_match_2_3 and indices_match_1_3

    # 最终结论
    print("\n" + "=" * 60)
    if all_match and positions_match:
        print("✓ 测试通过！所有方法得到完全相同的结果")
        print("  推荐使用方法2（最简洁）")
    else:
        print("✗ 测试失败！不同方法得到了不同的结果")
        print("  需要进一步检查")
    print("=" * 60)

    # 额外信息：显示 input_ids 的结构
    print(f"\n[附加信息] input_ids 结构:")
    print(f"  完整序列: {input_ids[0].tolist()}")
    print(f"  image_token_index 的值: {model.config.image_token_index}")

    # 统计 image_token_index 出现的次数
    num_image_tokens = (input_ids == model.config.image_token_index).sum().item()
    print(f"  image_token 出现次数: {num_image_tokens}")


if __name__ == "__main__":
    main()

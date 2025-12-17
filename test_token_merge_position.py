"""验证Token Merge在正确位置（CLIP输出后，LLM投影前）

正确流程：
Image → CLIP Vision Tower (1024维)
      → Token Merge (1024维 → 1024维)
      → Multi-Modal Projector (1024维 → 4096维)
      → LLM Forward

错误流程（之前的实现）：
Image → CLIP Vision Tower (1024维)
      → Multi-Modal Projector (1024维 → 4096维)
      → Token Merge (4096维 → 4096维) ❌ 错误！
"""

import torch
from PIL import Image
import numpy as np

def test_token_merge_position():
    """测试Token Merge是否在正确位置"""
    print("="*80)
    print("验证Token Merge位置修正")
    print("="*80)

    # 加载配置
    from engine.configs.loader import load_config
    config = load_config(override_file="configs/vision_token_pruning.yaml")

    # 简化配置用于测试
    config["backbone_settings"]["mllm_settings"]["device_map"] = "cpu"
    config["global_settings"]["device"] = "cpu"

    print("\n[1] 加载Backbone...")
    from engine.backbones.loader import load_backbone
    backbone = load_backbone(config)

    print(f"✓ Backbone加载完成: {config['backbone_settings']['name']}")
    print(f"✓ Vision dim: {config['backbone_settings']['mllm_settings']['vision_dim']}")
    print(f"✓ Hidden dim: {config['backbone_settings']['mllm_settings']['hidden_dim']}")

    # 检查模型组件
    print("\n[2] 检查模型组件...")
    has_vision_tower = hasattr(backbone.model, 'vision_tower')
    has_projector = hasattr(backbone.model, 'multi_modal_projector')

    print(f"✓ vision_tower: {has_vision_tower}")
    print(f"✓ multi_modal_projector: {has_projector}")

    if not has_vision_tower or not has_projector:
        print("❌ 模型缺少必要组件")
        return False

    # 创建Token Merger
    print("\n[3] 创建Token Merger V3...")
    from method.models.token_merger import LearnableTokenMergerV3

    merger = LearnableTokenMergerV3(
        d_vision=config['backbone_settings']['mllm_settings']['vision_dim'],  # 1024
        d_text=config['backbone_settings']['mllm_settings']['hidden_dim'],    # 4096
        d_internal=512,
        num_heads=8,
        merge_ratio=0.5,
        use_question=True
    )

    print(f"✓ Token Merger创建完成")
    print(f"  - d_vision: {merger.d_vision}")
    print(f"  - d_internal: {merger.d_internal}")
    print(f"  - merge_ratio: {merger.merge_ratio}")

    # 准备测试数据
    print("\n[4] 准备测试数据...")
    # 创建一个简单的测试图像
    test_image = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
    test_question = "What is in the image?"
    test_answer = "A test image."

    print("✓ 测试数据准备完成")

    # 测试完整流程
    print("\n[5] 测试完整流程...")
    with torch.no_grad():
        # Step 1: Backbone preprocess
        emb_info = backbone.preprocess(test_image, test_question, test_answer)

        print(f"✓ Preprocess完成:")
        print(f"  - embeddings shape: {emb_info['embeddings'].shape}")
        print(f"  - vision_token_positions: {emb_info['vision_token_positions']}")

        # Step 2: 检查raw_vision_features
        if 'raw_vision_features' not in emb_info:
            print("❌ 错误：preprocess未返回raw_vision_features")
            return False

        raw_vision = emb_info['raw_vision_features']
        if raw_vision is None:
            print("❌ 错误：raw_vision_features为None")
            return False

        print(f"✓ raw_vision_features存在: {raw_vision.shape}")

        # 验证维度
        expected_vision_dim = config['backbone_settings']['mllm_settings']['vision_dim']
        if raw_vision.shape[-1] != expected_vision_dim:
            print(f"❌ 错误：raw_vision_features维度不正确")
            print(f"  期望: (..., {expected_vision_dim})")
            print(f"  实际: {raw_vision.shape}")
            return False

        print(f"✓ raw_vision_features维度正确: {raw_vision.shape[-1]}维（CLIP输出）")

        # Step 3: Token Merge (在1024维空间)
        v_start, v_end = emb_info['vision_token_positions']
        question_emb = emb_info['embeddings'][:, v_end+1:-10, :]  # 简化提取question

        merge_result = merger(raw_vision, question_emb)
        merged_vision = merge_result['merged_features']

        print(f"✓ Token Merge完成:")
        print(f"  - 输入: {raw_vision.shape} (1024维)")
        print(f"  - 输出: {merged_vision.shape} (1024维)")

        # 验证merge输出维度
        if merged_vision.shape[-1] != expected_vision_dim:
            print(f"❌ 错误：Token Merge输出维度不正确")
            return False

        expected_merged_tokens = int(raw_vision.shape[1] * merger.merge_ratio)
        if merged_vision.shape[1] != expected_merged_tokens:
            print(f"❌ 错误：Token Merge输出token数不正确")
            print(f"  期望: {expected_merged_tokens}")
            print(f"  实际: {merged_vision.shape[1]}")
            return False

        print(f"✓ Token数量正确: {raw_vision.shape[1]} → {merged_vision.shape[1]}")

        # Step 4: 投影到LLM维度
        projected_vision = backbone.model.multi_modal_projector(merged_vision)

        print(f"✓ Multi-Modal Projector完成:")
        print(f"  - 输入: {merged_vision.shape} (1024维)")
        print(f"  - 输出: {projected_vision.shape} (4096维)")

        # 验证投影输出维度
        expected_hidden_dim = config['backbone_settings']['mllm_settings']['hidden_dim']
        if projected_vision.shape[-1] != expected_hidden_dim:
            print(f"❌ 错误：Projector输出维度不正确")
            return False

        print(f"✓ 投影维度正确: {projected_vision.shape[-1]}维（LLM Hidden Dim）")

    print("\n" + "="*80)
    print("✅ 所有测试通过！Token Merge在正确位置：")
    print("   Image → CLIP (1024维) → Token Merge (1024维) → Projector (4096维)")
    print("="*80)

    return True


if __name__ == "__main__":
    try:
        success = test_token_merge_position()
        if not success:
            print("\n❌ 测试失败")
            exit(1)
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

"""测试Attention Residual功能 - 新方案

新方案：使用 Pre-hook保存输入 + Post-hook计算attention并剪枝
这是最稳健的方案，避免复杂的时序问题。
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/users/zjw/projects/domain-adaptation-token-pruning')


def test_vision_pruner_head():
    """测试VisionPrunerHead的attention residual功能"""
    from method.models.layer_pruner import VisionPrunerHead

    print("=" * 60)
    print("测试1: VisionPrunerHead with Attention Residual")
    print("=" * 60)

    # 创建pruner（启用attention residual）
    pruner = VisionPrunerHead(
        d_vision=4096,
        d_text=4096,
        d_internal=512,
        num_heads=4,
        use_attn_residual=True,
        attn_residual_weight=0.5,
        learnable_attn_weight=False
    )

    # 模拟输入
    batch_size = 1
    n_vision = 288
    n_text = 50
    d_vision = 4096
    d_text = 4096

    vision_hidden = torch.randn(batch_size, n_vision, d_vision)
    question_embeddings = torch.randn(batch_size, n_text, d_text)
    text_to_vision_attn = torch.randn(batch_size, n_vision)  # 模拟attention

    # 测试forward（不使用attention）
    print("\n[测试1.1] Forward without attention residual:")
    mask1 = pruner(vision_hidden, question_embeddings, use_gumbel=False, text_to_vision_attn=None)
    print(f"  Output shape: {mask1.shape}")
    print(f"  Output range: [{mask1.min().item():.4f}, {mask1.max().item():.4f}]")
    print(f"  Mean: {mask1.mean().item():.4f}")

    # 测试forward（使用attention）
    print("\n[测试1.2] Forward with attention residual:")
    mask2 = pruner(vision_hidden, question_embeddings, use_gumbel=False, text_to_vision_attn=text_to_vision_attn)
    print(f"  Output shape: {mask2.shape}")
    print(f"  Output range: [{mask2.min().item():.4f}, {mask2.max().item():.4f}]")
    print(f"  Mean: {mask2.mean().item():.4f}")

    # 验证residual确实有效果
    print("\n[测试1.3] Verify residual effect:")
    diff = (mask2 - mask1).abs().mean()
    print(f"  Mean absolute difference: {diff.item():.6f}")
    print(f"  ✓ Residual is {'ACTIVE' if diff > 1e-6 else 'INACTIVE'}")

    print("\n✓ Test 1 PASSED\n")


def load_backbone_for_test():
    """使用正式的配置加载backbone"""
    from engine.configs.loader import load_config
    from engine.backbones.loader import load_backbone

    # load_config 第一个参数是 override_dict，第二个是 override_file
    config = load_config(override_file="configs/vision_token_pruning.yaml")
    backbone = load_backbone(config)

    # 冻结参数
    if hasattr(backbone, "model"):
        for param in backbone.model.parameters():
            param.requires_grad = False

    return backbone, config


def test_combined_pre_post_hook():
    """测试组合方案：pre-hook保存输入，post-hook应用剪枝"""
    from method.models.layer_pruner import LayerSpecificPruner

    print("\n" + "=" * 60)
    print("测试2: Pre-hook保存输入 + Post-hook计算attention并剪枝")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1] 加载LLaVA模型...")
    backbone, config = load_backbone_for_test()

    # 2. 准备测试输入
    print("\n[2] 准备测试输入...")
    from PIL import Image

    image = Image.new('RGB', (224, 224), color='red')
    question = "What color is this image?"
    answer = "Red"

    emb_info = backbone.preprocess(image, question, answer)
    embeddings = emb_info['embeddings']
    attention_mask = emb_info['attention_mask']
    vision_pos = emb_info['vision_token_positions']
    v_start, v_end = vision_pos

    print(f"   embeddings shape: {embeddings.shape}")
    print(f"   vision_pos: {vision_pos}")

    # 3. 创建pruner
    print("\n[3] 创建Layer Pruner...")
    device = config["global_settings"]["device"]
    layer_pruners = LayerSpecificPruner(
        d_model=4096,
        d_text=4096,
        layer_indices=[10],
        use_attn_residual=True,
        attn_residual_weight=0.5
    ).to(device)

    # 提取question embeddings
    answer_pos = emb_info['answer_token_positions']
    if answer_pos[0] < 0:
        answer_start_abs = embeddings.shape[1] + answer_pos[0]
    else:
        answer_start_abs = answer_pos[0]
    question_embeddings = embeddings[:, v_end+1:answer_start_abs, :]

    print(f"   question_embeddings shape: {question_embeddings.shape}")

    # 4. 设置hooks
    print("\n[4] 设置combined hooks...")

    target_layer_idx = 10
    target_layer = backbone.model.model.language_model.layers[target_layer_idx]
    self_attn = target_layer.self_attn
    pruner = layer_pruners.get_pruner(target_layer_idx)

    # 上下文存储
    context = {
        'input_hidden_states': None,
        'text_to_vision_attn': None,
        'mask_collected': None
    }

    stats = {'pre_called': False, 'post_called': False, 'mask_mean': None, 'attn_computed': False}

    def layer_pre_hook(module, args, kwargs):
        """保存输入hidden_states"""
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get('hidden_states')

        context['input_hidden_states'] = hidden_states
        stats['pre_called'] = True
        print(f"\n   [Layer Pre-Hook] Saved input: {hidden_states.shape}")

        return args, kwargs

    def layer_post_hook(module, args, kwargs, output):
        """计算attention并应用剪枝"""
        hidden_states_out = output  # Layer输出
        hidden_states_in = context['input_hidden_states']  # Layer输入

        print(f"\n   [Layer Post-Hook] Processing...")
        print(f"      Input shape: {hidden_states_in.shape}")
        print(f"      Output shape: {hidden_states_out.shape}")

        # === 手动计算attention weights ===
        # 使用Layer输入（经过input_layernorm后的）来计算
        normed_input = target_layer.input_layernorm(hidden_states_in)

        batch, seq_len, d_model = normed_input.shape
        num_heads = self_attn.config.num_attention_heads
        head_dim = self_attn.head_dim

        # 计算Q, K
        Q = self_attn.q_proj(normed_input)
        K = self_attn.k_proj(normed_input)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

        # 计算attention scores (不加mask，只看原始相似度)
        scaling = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scaling
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # (batch, num_heads, seq_len, seq_len)

        print(f"      Computed attn_weights: {attn_weights.shape}")
        stats['attn_computed'] = True

        # === 提取text→vision的attention ===
        text_start = v_end + 1
        text_end = answer_start_abs
        text_indices = list(range(text_start, text_end))
        vision_indices = list(range(v_start, v_end + 1))

        if len(text_indices) > 0 and len(vision_indices) > 0:
            text_to_vision = attn_weights[:, :, text_indices, :][:, :, :, vision_indices]
            text_to_vision_attn = text_to_vision.mean(dim=(1, 2))  # (batch, n_vision)

            print(f"      text_to_vision_attn shape: {text_to_vision_attn.shape}")
            print(f"      text_to_vision_attn range: [{text_to_vision_attn.min().item():.6f}, {text_to_vision_attn.max().item():.6f}]")

            context['text_to_vision_attn'] = text_to_vision_attn
        else:
            text_to_vision_attn = None

        # === 提取vision hidden states并调用pruner ===
        vision_hidden = hidden_states_out[:, v_start:v_end+1, :]

        with torch.enable_grad():
            soft_mask = pruner(
                vision_hidden,
                question_embeddings,
                text_to_vision_attn=text_to_vision_attn
            )

        print(f"      soft_mask shape: {soft_mask.shape}")
        print(f"      soft_mask mean: {soft_mask.mean().item():.4f}")

        stats['post_called'] = True
        stats['mask_mean'] = soft_mask.mean().item()
        context['mask_collected'] = soft_mask

        # === 应用mask ===
        soft_mask = soft_mask.to(hidden_states_out.dtype)
        new_hidden = hidden_states_out.clone()
        new_hidden[:, v_start:v_end+1, :] = vision_hidden * soft_mask.unsqueeze(-1)

        return new_hidden

    # 注册hooks
    pre_handle = target_layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True)
    post_handle = target_layer.register_forward_hook(layer_post_hook, with_kwargs=True)

    try:
        print("\n[5] 执行forward...")
        with torch.no_grad():
            result = backbone.forward(
                embeddings=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        print("\n[6] 检查结果...")
        print(f"   Pre-hook called: {stats['pre_called']}")
        print(f"   Post-hook called: {stats['post_called']}")
        print(f"   Attention computed: {stats['attn_computed']}")
        if stats['mask_mean'] is not None:
            print(f"   Mask mean: {stats['mask_mean']:.4f}")

        if stats['pre_called'] and stats['post_called'] and stats['attn_computed']:
            print("\n   ✓ 组合方案成功!")
            return True
        else:
            print("\n   ✗ 组合方案有问题")
            return False

    finally:
        pre_handle.remove()
        post_handle.remove()


def test_new_hook_implementation():
    """测试新的hook实现（utils.py中的register_multi_layer_hooks_v2）"""
    from method.models.layer_pruner import LayerSpecificPruner

    print("\n" + "=" * 60)
    print("测试3: 新的Hook实现（register_multi_layer_hooks_v2）")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1] 加载LLaVA模型...")
    backbone, config = load_backbone_for_test()

    # 2. 准备测试输入
    print("\n[2] 准备测试输入...")
    from PIL import Image

    image = Image.new('RGB', (224, 224), color='red')
    question = "What color is this image?"
    answer = "Red"

    emb_info = backbone.preprocess(image, question, answer)
    embeddings = emb_info['embeddings']
    attention_mask = emb_info['attention_mask']
    vision_pos = emb_info['vision_token_positions']
    v_start, v_end = vision_pos

    # 提取question embeddings
    answer_pos = emb_info['answer_token_positions']
    if answer_pos[0] < 0:
        answer_start_abs = embeddings.shape[1] + answer_pos[0]
    else:
        answer_start_abs = answer_pos[0]
    question_embeddings = embeddings[:, v_end+1:answer_start_abs, :]

    # 3. 创建pruner
    print("\n[3] 创建Layer Pruner...")
    device = config["global_settings"]["device"]
    layer_pruners = LayerSpecificPruner(
        d_model=4096,
        d_text=4096,
        layer_indices=[10, 20, 31],
        use_attn_residual=True,
        attn_residual_weight=0.5
    ).to(device)

    # 4. 使用新的hook注册函数
    print("\n[4] 使用新的hook注册函数...")

    from method.utils import register_multi_layer_hooks_v2, remove_hooks

    mask_collector = []
    handles = register_multi_layer_hooks_v2(
        backbone=backbone,
        layer_pruners=layer_pruners,
        vision_positions=vision_pos,
        question_embeddings=question_embeddings,
        mask_collector=mask_collector,
        use_attn_residual=True
    )

    try:
        print("\n[5] 执行forward...")
        with torch.no_grad():
            result = backbone.forward(
                embeddings=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        print("\n[6] 检查结果...")
        print(f"   Collected {len(mask_collector)} masks")
        for i, mask in enumerate(mask_collector):
            print(f"   Mask {i}: shape={mask.shape}, mean={mask.mean().item():.4f}")

        if len(mask_collector) == 3:
            print("\n   ✓ 新实现成功!")
            return True
        else:
            print("\n   ✗ 新实现有问题")
            return False

    finally:
        remove_hooks(handles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, choices=[0, 1, 2, 3],
                       help="0: all, 1: pruner head, 2: combined hook, 3: new implementation")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("开始测试Attention Residual实现")
    print("="*60 + "\n")

    try:
        if args.test == 0 or args.test == 1:
            test_vision_pruner_head()

        if args.test == 0 or args.test == 2:
            test_combined_pre_post_hook()

        if args.test == 0 or args.test == 3:
            test_new_hook_implementation()

        print("\n" + "="*60)
        print("✓✓✓ 测试完成！ ✓✓✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗✗✗ 测试失败: {e} ✗✗✗")
        import traceback
        traceback.print_exc()

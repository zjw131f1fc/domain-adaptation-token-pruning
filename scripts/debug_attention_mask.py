#!/usr/bin/env python
"""验证 attention mask 是否真的屏蔽了 token

测试方法：
1. 构建一个简单的 attention 计算
2. 对比有 mask 和无 mask 的输出
3. 验证被 mask 的 token 不参与 attention
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F


def test_attention_mask_effect():
    """测试 attention mask 的效果"""
    print("=" * 60)
    print("Testing attention mask effect")
    print("=" * 60)

    # 模拟参数
    batch_size = 1
    seq_len = 10
    n_heads = 1
    head_dim = 4
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建随机 Q, K, V
    torch.manual_seed(42)
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    # 计算 attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"\nAttention scores shape: {attn_scores.shape}")
    print(f"Attention scores:\n{attn_scores[0, 0]}")

    # 方法1: 无 mask
    attn_weights_no_mask = F.softmax(attn_scores, dim=-1)
    output_no_mask = torch.matmul(attn_weights_no_mask, V)
    print(f"\n--- Without mask ---")
    print(f"Attention weights:\n{attn_weights_no_mask[0, 0]}")
    print(f"Output:\n{output_no_mask[0, 0]}")

    # 方法2: 用 -inf mask 屏蔽位置 3, 5, 7
    mask_positions = [3, 5, 7]
    attn_mask = torch.zeros(batch_size, n_heads, seq_len, seq_len, device=device, dtype=dtype)
    for pos in mask_positions:
        attn_mask[:, :, :, pos] = float('-inf')

    attn_scores_masked = attn_scores + attn_mask
    attn_weights_masked = F.softmax(attn_scores_masked, dim=-1)
    output_masked = torch.matmul(attn_weights_masked, V)
    print(f"\n--- With mask (positions {mask_positions} masked) ---")
    print(f"Attention weights:\n{attn_weights_masked[0, 0]}")
    print(f"Output:\n{output_masked[0, 0]}")

    # 验证被 mask 的位置权重为 0
    print(f"\n--- Verification ---")
    for pos in mask_positions:
        weight_at_pos = attn_weights_masked[0, 0, :, pos]
        print(f"Weights at position {pos}: {weight_at_pos}")
        assert torch.allclose(weight_at_pos, torch.zeros_like(weight_at_pos), atol=1e-6), \
            f"Position {pos} should have zero weight!"
    print("✓ All masked positions have zero attention weight!")

    # 方法3: 物理删除位置 3, 5, 7
    keep_positions = [i for i in range(seq_len) if i not in mask_positions]
    K_deleted = K[:, :, keep_positions, :]
    V_deleted = V[:, :, keep_positions, :]

    attn_scores_deleted = torch.matmul(Q, K_deleted.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights_deleted = F.softmax(attn_scores_deleted, dim=-1)
    output_deleted = torch.matmul(attn_weights_deleted, V_deleted)
    print(f"\n--- With physical deletion (positions {mask_positions} removed) ---")
    print(f"K shape after deletion: {K_deleted.shape}")
    print(f"Attention weights:\n{attn_weights_deleted[0, 0]}")
    print(f"Output:\n{output_deleted[0, 0]}")

    # 比较 mask 和物理删除的输出
    print(f"\n--- Comparing mask vs physical deletion ---")
    diff = (output_masked - output_deleted).abs()
    print(f"Max diff: {diff.max().item():.10f}")
    print(f"Mean diff: {diff.mean().item():.10f}")

    if diff.max().item() < 1e-5:
        print("✓ Mask and physical deletion produce IDENTICAL outputs!")
    else:
        print("✗ Mask and physical deletion produce DIFFERENT outputs!")

    return diff.max().item() < 1e-5


def test_build_vision_pruning_attention_mask():
    """测试 build_vision_pruning_attention_mask 函数"""
    print("\n" + "=" * 60)
    print("Testing build_vision_pruning_attention_mask")
    print("=" * 60)

    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from method.models.prunable_llava import build_vision_pruning_attention_mask

    # 模拟参数
    batch_size = 2
    n_vision = 576
    seq_len = 605
    vision_start = 5
    vision_end = vision_start + n_vision
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建 cumulative_vision_mask
    # 样本 0: 保留前 400 个 vision tokens
    # 样本 1: 保留前 300 个 vision tokens
    cumulative_vision_mask = torch.zeros(batch_size, n_vision, device=device, dtype=dtype)
    cumulative_vision_mask[0, :400] = 1.0
    cumulative_vision_mask[1, :300] = 1.0

    print(f"cumulative_vision_mask shape: {cumulative_vision_mask.shape}")
    print(f"Sample 0 kept: {cumulative_vision_mask[0].sum().item()}")
    print(f"Sample 1 kept: {cumulative_vision_mask[1].sum().item()}")

    # 构建 attention mask
    attn_mask = build_vision_pruning_attention_mask(
        cumulative_vision_mask, vision_start, vision_end, seq_len, dtype, device
    )

    print(f"\nAttention mask shape: {attn_mask.shape}")
    print(f"Expected: (batch={batch_size}, 1, seq={seq_len}, seq={seq_len})")

    # 验证 mask 值
    min_val = torch.finfo(dtype).min

    # 检查样本 0
    print(f"\n--- Sample 0 ---")
    # 被剪掉的 vision tokens (400-576) 应该是 -inf
    pruned_region_0 = attn_mask[0, 0, 0, vision_start + 400:vision_end]
    kept_region_0 = attn_mask[0, 0, 0, vision_start:vision_start + 400]
    print(f"Pruned region (400-576) values: min={pruned_region_0.min().item()}, max={pruned_region_0.max().item()}")
    print(f"Kept region (0-400) values: min={kept_region_0.min().item()}, max={kept_region_0.max().item()}")

    # 检查样本 1
    print(f"\n--- Sample 1 ---")
    pruned_region_1 = attn_mask[1, 0, 0, vision_start + 300:vision_end]
    kept_region_1 = attn_mask[1, 0, 0, vision_start:vision_start + 300]
    print(f"Pruned region (300-576) values: min={pruned_region_1.min().item()}, max={pruned_region_1.max().item()}")
    print(f"Kept region (0-300) values: min={kept_region_1.min().item()}, max={kept_region_1.max().item()}")

    # 验证
    assert pruned_region_0.max().item() < -1e30, "Sample 0 pruned region should be -inf"
    assert pruned_region_1.max().item() < -1e30, "Sample 1 pruned region should be -inf"
    print("\n✓ Attention mask correctly marks pruned tokens as -inf!")

    # 验证 per-sample mask 不同
    diff_between_samples = (attn_mask[0] - attn_mask[1]).abs().sum().item()
    print(f"\nDiff between sample 0 and sample 1: {diff_between_samples}")
    assert diff_between_samples > 0, "Per-sample masks should be different!"
    print("✓ Per-sample masks are correctly different!")


if __name__ == "__main__":
    test_attention_mask_effect()
    test_build_vision_pruning_attention_mask()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

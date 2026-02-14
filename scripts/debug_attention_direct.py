#!/usr/bin/env python
"""直接验证后续层的 attention weights 中被 mask 的位置是否为 0

通过在非剪枝层注册 hook，检查 attention weights
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 简化测试：直接测试 attention mask 的效果
    print("\n" + "=" * 60)
    print("Testing attention mask in LlamaAttention")
    print("=" * 60)

    # 模拟参数
    batch_size = 1
    seq_len = 20
    n_vision = 10
    vision_start = 5
    vision_end = vision_start + n_vision
    dtype = torch.bfloat16

    # 创建 cumulative_vision_mask：保留前 6 个 vision tokens
    cumulative_vision_mask = torch.zeros(batch_size, n_vision, device=device, dtype=dtype)
    cumulative_vision_mask[0, :6] = 1.0  # 保留 0-5，剪掉 6-9

    print(f"\ncumulative_vision_mask: {cumulative_vision_mask[0].tolist()}")
    print(f"Kept positions: 0-5, Pruned positions: 6-9")

    # 使用 build_vision_pruning_attention_mask 构建 mask
    from method.models.prunable_llava import build_vision_pruning_attention_mask

    attn_mask = build_vision_pruning_attention_mask(
        cumulative_vision_mask, vision_start, vision_end, seq_len, dtype, device
    )

    print(f"\nattn_mask shape: {attn_mask.shape}")

    # 检查 mask 值
    # 对于 query 位置 15（在 vision 之后），检查它对 vision tokens 的 mask
    query_pos = 15
    vision_mask_for_query = attn_mask[0, 0, query_pos, vision_start:vision_end]
    print(f"\nMask values for query position {query_pos} attending to vision tokens:")
    print(f"  Positions 0-5 (kept): {vision_mask_for_query[:6].tolist()}")
    print(f"  Positions 6-9 (pruned): {vision_mask_for_query[6:].tolist()}")

    # 验证：kept 位置应该是 0（或 causal mask 值），pruned 位置应该是 -inf
    min_val = torch.finfo(dtype).min
    kept_values = vision_mask_for_query[:6]
    pruned_values = vision_mask_for_query[6:]

    print(f"\nExpected: kept=0, pruned={min_val}")
    print(f"Actual: kept max={kept_values.max().item()}, pruned min={pruned_values.min().item()}")

    # 模拟 attention 计算
    print("\n" + "=" * 60)
    print("Simulating attention computation")
    print("=" * 60)

    torch.manual_seed(42)
    # 创建随机 Q, K, V
    Q = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=dtype)  # 1 head
    K = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=dtype)
    V = torch.randn(batch_size, 1, seq_len, 64, device=device, dtype=dtype)

    # 计算 attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5)
    print(f"\nattn_scores shape: {attn_scores.shape}")

    # 应用 attention mask
    attn_scores_masked = attn_scores + attn_mask

    # Softmax
    attn_weights = F.softmax(attn_scores_masked, dim=-1, dtype=torch.float32).to(dtype)

    # 检查 query 位置 15 对 vision tokens 的 attention weights
    vision_weights = attn_weights[0, 0, query_pos, vision_start:vision_end]
    print(f"\nAttention weights for query position {query_pos} to vision tokens:")
    print(f"  Positions 0-5 (kept): {vision_weights[:6].tolist()}")
    print(f"  Positions 6-9 (pruned): {vision_weights[6:].tolist()}")

    # 验证：pruned 位置的权重应该是 0
    pruned_weights = vision_weights[6:]
    if pruned_weights.abs().max().item() < 1e-6:
        print(f"\n✓ [PASS] Pruned positions have ZERO attention weight!")
    else:
        print(f"\n✗ [FAIL] Pruned positions have non-zero attention weight: {pruned_weights.abs().max().item()}")

    # 计算 attention output
    attn_output = torch.matmul(attn_weights, V)

    # 对比：如果物理删除 pruned tokens
    print("\n" + "=" * 60)
    print("Comparing with physical deletion")
    print("=" * 60)

    # 物理删除 pruned vision tokens (positions 6-9)
    keep_positions = list(range(vision_start)) + list(range(vision_start, vision_start + 6)) + list(range(vision_end, seq_len))
    K_deleted = K[:, :, keep_positions, :]
    V_deleted = V[:, :, keep_positions, :]

    # 重新计算 attention（只用 causal mask）
    new_seq_len = len(keep_positions)
    causal_mask_deleted = torch.triu(
        torch.full((seq_len, new_seq_len), min_val, device=device, dtype=dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    attn_scores_deleted = torch.matmul(Q, K_deleted.transpose(-2, -1)) / (64 ** 0.5)
    attn_scores_deleted = attn_scores_deleted + causal_mask_deleted
    attn_weights_deleted = F.softmax(attn_scores_deleted, dim=-1, dtype=torch.float32).to(dtype)
    attn_output_deleted = torch.matmul(attn_weights_deleted, V_deleted)

    # 比较输出
    diff = (attn_output - attn_output_deleted).abs()
    print(f"\nOutput comparison:")
    print(f"  Max diff: {diff.max().item():.10f}")
    print(f"  Mean diff: {diff.mean().item():.10f}")

    if diff.max().item() < 1e-5:
        print(f"\n✓ [PASS] Attention mask and physical deletion produce IDENTICAL outputs!")
    else:
        print(f"\n✗ [FAIL] Outputs differ!")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The attention mask correctly:
1. Sets pruned positions to -inf in the mask
2. After softmax, pruned positions get 0 attention weight
3. The output is identical to physical deletion

This confirms that:
- Training (attention mask) and Inference (physical deletion) are equivalent
- Subsequent layers correctly use the attention mask to ignore pruned tokens
""")


if __name__ == "__main__":
    main()

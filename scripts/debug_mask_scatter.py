#!/usr/bin/env python
"""调试 mask scatter 逻辑 - 对比 bug 版本和修复版本"""

import torch

def test_bug_version():
    """复现 union mask 导致 L8 > L2 的 bug"""
    print("=" * 60)
    print("BUG 版本: cumulative_vision_mask = union_mask")
    print("=" * 60)

    batch_size = 2
    n_vision_orig = 10
    device = torch.device("cpu")
    dtype = torch.float32

    # === L2 (第一个剪枝层) ===
    print("\n--- L2 ---")
    cumulative_vision_mask = torch.ones(batch_size, n_vision_orig, device=device, dtype=dtype)

    # 样本 0: 保留位置 [0, 1, 2]
    # 样本 1: 保留位置 [0, 1, 2, 3, 4]
    hard_mask_L2 = torch.zeros(batch_size, n_vision_orig, device=device, dtype=dtype)
    hard_mask_L2[0, :3] = 1
    hard_mask_L2[1, :5] = 1

    print(f"hard_mask_L2[0]: {hard_mask_L2[0].tolist()}")
    print(f"hard_mask_L2[1]: {hard_mask_L2[1].tolist()}")
    print(f"L2 平均保留率: {hard_mask_L2.mean().item():.4f}")

    # BUG: 使用 union mask
    union_mask_L2 = (hard_mask_L2.sum(dim=0) > 0).float()
    n_vision_after_L2 = int(union_mask_L2.sum().item())
    cumulative_vision_mask = union_mask_L2.unsqueeze(0).expand(batch_size, -1)

    # === L8 ===
    print("\n--- L8 ---")
    hard_mask_L8_current = torch.ones(batch_size, n_vision_after_L2, device=device, dtype=dtype)

    cumulative_vision_mask_clean = (cumulative_vision_mask > 0.5)
    hard_mask_full_list = []
    for b in range(batch_size):
        kept_indices_b = cumulative_vision_mask_clean[b].nonzero(as_tuple=True)[0]
        hm_b = torch.zeros(n_vision_orig, device=device, dtype=dtype)
        hm_b = hm_b.scatter(0, kept_indices_b, hard_mask_L8_current[b])
        hard_mask_full_list.append(hm_b)

    hard_mask_full_L8 = torch.stack(hard_mask_full_list, dim=0)
    print(f"L8 平均保留率: {hard_mask_full_L8.mean().item():.4f}")

    if hard_mask_full_L8.mean() > hard_mask_L2.mean():
        print(f"\n[BUG!] L8 ({hard_mask_full_L8.mean().item():.4f}) > L2 ({hard_mask_L2.mean().item():.4f})")
    return hard_mask_L2.mean().item(), hard_mask_full_L8.mean().item()


def test_fixed_version():
    """修复版本: cumulative_vision_mask = per-sample mask"""
    print("\n" + "=" * 60)
    print("修复版本: cumulative_vision_mask = per-sample mask")
    print("=" * 60)

    batch_size = 2
    n_vision_orig = 10
    device = torch.device("cpu")
    dtype = torch.float32

    # === L2 ===
    print("\n--- L2 ---")
    cumulative_vision_mask = torch.ones(batch_size, n_vision_orig, device=device, dtype=dtype)

    hard_mask_L2 = torch.zeros(batch_size, n_vision_orig, device=device, dtype=dtype)
    hard_mask_L2[0, :3] = 1
    hard_mask_L2[1, :5] = 1

    print(f"hard_mask_L2[0]: {hard_mask_L2[0].tolist()}")
    print(f"hard_mask_L2[1]: {hard_mask_L2[1].tolist()}")
    print(f"L2 平均保留率: {hard_mask_L2.mean().item():.4f}")

    # 物理删除用 union mask
    union_mask_L2 = (hard_mask_L2.sum(dim=0) > 0).float()
    n_vision_after_L2 = int(union_mask_L2.sum().item())

    # 修复: cumulative_vision_mask 保持 per-sample
    cumulative_vision_mask = hard_mask_L2.clone()
    print(f"cumulative_vision_mask[0]: {cumulative_vision_mask[0].tolist()}")
    print(f"cumulative_vision_mask[1]: {cumulative_vision_mask[1].tolist()}")

    # === L8 ===
    print("\n--- L8 ---")
    hard_mask_L8_current = torch.ones(batch_size, n_vision_after_L2, device=device, dtype=dtype)

    cumulative_vision_mask_clean = (cumulative_vision_mask > 0.5)
    hard_mask_full_list = []
    for b in range(batch_size):
        kept_indices_b = cumulative_vision_mask_clean[b].nonzero(as_tuple=True)[0]
        print(f"样本 {b} 的 kept_indices_b (per-sample): {kept_indices_b.tolist()}")
        hm_b = torch.zeros(n_vision_orig, device=device, dtype=dtype)
        # 只 scatter 该样本实际保留的位置数量
        n_kept_b = len(kept_indices_b)
        hm_b = hm_b.scatter(0, kept_indices_b, hard_mask_L8_current[b, :n_kept_b])
        hard_mask_full_list.append(hm_b)
        print(f"样本 {b} 的 hard_mask_full_L8: {hm_b.tolist()}")

    hard_mask_full_L8 = torch.stack(hard_mask_full_list, dim=0)
    print(f"\nL8 平均保留率: {hard_mask_full_L8.mean().item():.4f}")

    if hard_mask_full_L8.mean() <= hard_mask_L2.mean():
        print(f"\n[OK] L8 ({hard_mask_full_L8.mean().item():.4f}) <= L2 ({hard_mask_L2.mean().item():.4f})")
    return hard_mask_L2.mean().item(), hard_mask_full_L8.mean().item()


if __name__ == "__main__":
    l2_bug, l8_bug = test_bug_version()
    l2_fix, l8_fix = test_fixed_version()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"BUG 版本: L2={l2_bug:.4f}, L8={l8_bug:.4f} {'[BUG!]' if l8_bug > l2_bug else '[OK]'}")
    print(f"修复版本: L2={l2_fix:.4f}, L8={l8_fix:.4f} {'[BUG!]' if l8_fix > l2_fix else '[OK]'}")

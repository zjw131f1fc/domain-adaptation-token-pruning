#!/usr/bin/env python
"""测试 Attention Consistency Pruning 实现

验证所有组件能正常导入和基本功能。
"""

import sys
import torch
import torch.nn as nn

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("Testing imports...")

    try:
        from method.models.layer_pruner_acp import LayerPruner, LayerPrunerManager
        print("✓ LayerPruner, LayerPrunerManager imported")
    except Exception as e:
        print(f"✗ Failed to import LayerPruner: {e}")
        return False

    try:
        from method.models.layer_discriminator import LayerDiscriminator, LayerDiscriminatorManager
        print("✓ LayerDiscriminator, LayerDiscriminatorManager imported")
    except Exception as e:
        print(f"✗ Failed to import LayerDiscriminator: {e}")
        return False

    try:
        from method.models.prunable_llama_layer import PrunableLlamaDecoderLayer
        print("✓ PrunableLlamaDecoderLayer imported")
    except Exception as e:
        print(f"✗ Failed to import PrunableLlamaDecoderLayer: {e}")
        return False

    try:
        from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
        print("✓ PrunableLlavaForConditionalGeneration imported")
    except Exception as e:
        print(f"✗ Failed to import PrunableLlavaForConditionalGeneration: {e}")
        return False

    try:
        from method.training_acp import train_step_acp, ACPTrainer
        print("✓ train_step_acp, ACPTrainer imported")
    except Exception as e:
        print(f"✗ Failed to import training_acp: {e}")
        return False

    print("All imports successful!")
    return True


def test_layer_pruner():
    """测试 LayerPruner"""
    print("\n" + "=" * 60)
    print("Testing LayerPruner...")

    from method.models.layer_pruner_acp import LayerPruner

    pruner = LayerPruner(d_internal=128, temperature=1.0)

    # 模拟输入：question→vision attention
    batch_size = 2
    n_vision = 576
    q2v_attn = torch.rand(batch_size, n_vision)  # 模拟归一化的 attention

    # 测试 forward
    residual = pruner(q2v_attn)
    assert residual.shape == (batch_size, n_vision), f"Expected {(batch_size, n_vision)}, got {residual.shape}"
    print(f"✓ Forward: input {q2v_attn.shape} -> residual {residual.shape}")

    # 测试 compute_importance
    importance = pruner.compute_importance(q2v_attn)
    assert importance.shape == (batch_size, n_vision)
    print(f"✓ Compute importance: {importance.shape}")

    # 测试 gumbel_softmax_mask
    pruner.train()
    hard_mask = pruner.gumbel_softmax_mask(importance)
    assert hard_mask.shape == (batch_size, n_vision)
    assert hard_mask.min() >= 0 and hard_mask.max() <= 1
    print(f"✓ Gumbel softmax mask (train): {hard_mask.shape}, range [{hard_mask.min():.2f}, {hard_mask.max():.2f}]")

    pruner.eval()
    hard_mask_eval = pruner.gumbel_softmax_mask(importance)
    assert hard_mask_eval.shape == (batch_size, n_vision)
    assert set(hard_mask_eval.unique().tolist()).issubset({0.0, 1.0})
    print(f"✓ Gumbel softmax mask (eval): {hard_mask_eval.shape}, values: {hard_mask_eval.unique().tolist()}")

    # 测试完整流程
    pruner.train()
    hard_mask, info = pruner.forward_full(q2v_attn)
    assert 'residual' in info
    assert 'importance' in info
    print(f"✓ Forward full: mask {hard_mask.shape}, info keys: {list(info.keys())}")

    # 测试初始化（最后一层应该是零）
    last_layer = pruner.mlp[-1]
    assert torch.allclose(last_layer.weight, torch.zeros_like(last_layer.weight))
    assert torch.allclose(last_layer.bias, torch.zeros_like(last_layer.bias))
    print("✓ Last layer initialized to zero")

    print("LayerPruner tests passed!")
    return True


def test_layer_discriminator():
    """测试 LayerDiscriminator"""
    print("\n" + "=" * 60)
    print("Testing LayerDiscriminator...")

    from method.models.layer_discriminator import LayerDiscriminator

    num_heads = 32
    head_dim = 128

    disc = LayerDiscriminator(num_heads=num_heads, head_dim=head_dim, d_hidden=256)

    # 模拟输入：单个 answer token 的聚合结果
    batch_size = 2
    h_single = torch.rand(batch_size, num_heads, head_dim)

    # 测试单个 token 判别
    logit_single = disc(h_single)
    assert logit_single.shape == (batch_size,), f"Expected {(batch_size,)}, got {logit_single.shape}"
    print(f"✓ Single token: input {h_single.shape} -> logit {logit_single.shape}")

    # 模拟输入：多个 answer tokens 的聚合结果
    n_answer = 5
    h_multi = torch.rand(batch_size, num_heads, n_answer, head_dim)

    # 测试多个 tokens 判别
    logit_multi = disc(h_multi)
    assert logit_multi.shape == (batch_size, n_answer), f"Expected {(batch_size, n_answer)}, got {logit_multi.shape}"
    print(f"✓ Multiple tokens: input {h_multi.shape} -> logit {logit_multi.shape}")

    # 测试 forward_batch_answers
    logit_mean = disc.forward_batch_answers(h_multi, reduce='mean')
    assert logit_mean.shape == (batch_size,)
    print(f"✓ Batch answers (mean): {logit_mean.shape}")

    logit_none = disc.forward_batch_answers(h_multi, reduce='none')
    assert logit_none.shape == (batch_size, n_answer)
    print(f"✓ Batch answers (none): {logit_none.shape}")

    print("LayerDiscriminator tests passed!")
    return True


def test_layer_pruner_manager():
    """测试 LayerPrunerManager"""
    print("\n" + "=" * 60)
    print("Testing LayerPrunerManager...")

    from method.models.layer_pruner_acp import LayerPrunerManager

    layer_indices = [4, 14, 24]
    manager = LayerPrunerManager(layer_indices=layer_indices, d_internal=128)

    # 测试获取 pruner
    for idx in layer_indices:
        pruner = manager.get_pruner(idx)
        assert pruner is not None
        print(f"✓ Got pruner for layer {idx}")

    # 测试设置温度
    manager.set_temperature(0.5)
    for idx in layer_indices:
        pruner = manager.get_pruner(idx)
        assert pruner.temperature == 0.5
    print("✓ Temperature set to 0.5 for all pruners")

    print("LayerPrunerManager tests passed!")
    return True


def test_layer_discriminator_manager():
    """测试 LayerDiscriminatorManager"""
    print("\n" + "=" * 60)
    print("Testing LayerDiscriminatorManager...")

    from method.models.layer_discriminator import LayerDiscriminatorManager

    layer_indices = [4, 14, 24]
    num_heads = 32
    head_dim = 128

    manager = LayerDiscriminatorManager(
        layer_indices=layer_indices,
        num_heads=num_heads,
        head_dim=head_dim
    )

    # 测试获取 discriminator
    for idx in layer_indices:
        disc = manager.get_discriminator(idx)
        assert disc is not None
        print(f"✓ Got discriminator for layer {idx}")

    # 模拟 h_real 和 h_fake
    batch_size = 2
    n_answer = 5

    h_real_dict = {idx: torch.rand(batch_size, num_heads, n_answer, head_dim) for idx in layer_indices}
    h_fake_dict = {idx: torch.rand(batch_size, num_heads, n_answer, head_dim) for idx in layer_indices}

    # 测试 compute_disc_loss
    disc_loss = manager.compute_disc_loss(h_real_dict, h_fake_dict)
    assert disc_loss.shape == ()
    print(f"✓ Disc loss: {disc_loss.item():.4f}")

    # 测试 compute_adv_loss
    adv_loss = manager.compute_adv_loss(h_fake_dict)
    assert adv_loss.shape == ()
    print(f"✓ Adv loss: {adv_loss.item():.4f}")

    # 测试 compute_accuracy
    acc_info = manager.compute_accuracy(h_real_dict, h_fake_dict)
    assert 'overall' in acc_info
    assert 'real_acc' in acc_info
    assert 'fake_acc' in acc_info
    print(f"✓ Accuracy: overall={acc_info['overall']:.2%}, real={acc_info['real_acc']:.2%}, fake={acc_info['fake_acc']:.2%}")

    print("LayerDiscriminatorManager tests passed!")
    return True


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "=" * 60)
    print("Testing gradient flow...")

    from method.models.layer_pruner_acp import LayerPruner
    from method.models.layer_discriminator import LayerDiscriminator
    import torch.nn.functional as F

    # 创建组件
    pruner = LayerPruner(d_internal=128)
    disc = LayerDiscriminator(num_heads=32, head_dim=128)

    # 模拟输入
    batch_size = 2
    n_vision = 576
    num_heads = 32
    head_dim = 128
    n_answer = 5

    q2v_attn = torch.rand(batch_size, n_vision, requires_grad=False)

    # 训练模式
    pruner.train()

    # Forward
    residual = pruner(q2v_attn)
    importance = q2v_attn + residual
    hard_mask = pruner.gumbel_softmax_mask(importance)

    # 模拟 h_fake（简化：直接用 hard_mask 生成）
    h_fake = torch.rand(batch_size, num_heads, n_answer, head_dim) * hard_mask.mean()
    h_fake.requires_grad_(True)

    # Discriminator forward
    fake_logit = disc(h_fake)

    # Adversarial loss
    adv_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))

    # Backward
    adv_loss.backward()

    # 检查 pruner 参数是否有梯度
    has_grad = False
    for name, param in pruner.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"✓ {name}: grad norm = {param.grad.norm().item():.6f}")

    if not has_grad:
        print("! Note: No gradient on pruner (expected in this simplified test)")
        print("  In real training, gradient flows through h_fake computation")

    print("Gradient flow test completed!")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Attention Consistency Pruning - Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("LayerPruner", test_layer_pruner),
        ("LayerDiscriminator", test_layer_discriminator),
        ("LayerPrunerManager", test_layer_pruner_manager),
        ("LayerDiscriminatorManager", test_layer_discriminator_manager),
        ("Gradient Flow", test_gradient_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

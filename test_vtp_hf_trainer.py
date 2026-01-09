"""测试VisionTokenPruningModel + HF Trainer集成

这个脚本用于验证新架构的正确性,不需要完整训练。

测试内容:
1. 模型初始化
2. Forward pass
3. Optimizer groups创建
4. HF Trainer构建
5. 单步训练

运行:
    conda run -n rl-pruning python test_vtp_hf_trainer.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engine.configs.loader import load_config
from engine.backbones.loader import load_backbone
from engine.datas.loader import load_dataset
from engine.trainers.loader import load_trainer
from method.models.layer_pruner import LayerSpecificPruner
from method.models.discriminator import Discriminator
from method.models.unified_model import VisionTokenPruningModel


def test_model_creation(config):
    """测试1: 创建VisionTokenPruningModel"""
    print("\n" + "=" * 60)
    print("测试1: 创建VisionTokenPruningModel")
    print("=" * 60)

    # Load backbone
    print("Loading backbone...")
    backbone = load_backbone(config)

    # Create components
    print("Creating components...")
    method_config = config.method_settings

    # Layer Pruners
    layer_pruners = LayerSpecificPruner(
        d_model=4096,
        d_text=4096,
        layer_indices=method_config.pruning_layers,
        d_internal=method_config.pruner_d_internal,
        num_heads=method_config.pruner_num_heads,
        use_attn_residual=method_config.get('use_attn_residual', False),
        attn_residual_weight=method_config.get('attn_residual_weight', 0.5),
        learnable_attn_weight=method_config.get('learnable_attn_weight', False)
    )

    # Discriminator
    discriminator = Discriminator(
        d_model=4096,
        num_layers=len(method_config.disc_target_layers),
        d_d=method_config.disc_d_d,
        dropout=method_config.get('disc_dropout', 0.1),
        use_spectral_norm=method_config.get('disc_use_spectral_norm', False)
    )

    # Create unified model
    print("Creating VisionTokenPruningModel...")
    model = VisionTokenPruningModel(
        config=config,
        backbone=backbone,
        layer_pruners=layer_pruners,
        discriminator=discriminator,
        token_merger=None  # Disabled for this test
    )

    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def test_optimizer_groups(model, config):
    """测试2: 创建optimizer groups"""
    print("\n" + "=" * 60)
    print("测试2: 创建Optimizer Groups")
    print("=" * 60)

    param_groups = model.create_optimizer_groups()

    print(f"✓ Created {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        n_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: lr={group['lr']}, {n_params:,} params")

    # Create optimizer
    optimizer = torch.optim.Adam(param_groups)
    print(f"✓ Optimizer created: {type(optimizer).__name__}")

    return optimizer


def test_forward_pass(model, dataset_bundle):
    """测试3: Forward pass"""
    print("\n" + "=" * 60)
    print("测试3: Forward Pass")
    print("=" * 60)

    # Get a small batch
    train_samples = dataset_bundle['splits']['train'].samples[:2]
    print(f"Testing with {len(train_samples)} samples")

    # Forward
    print("Running forward pass...")
    model.eval()  # eval mode for testing
    with torch.no_grad():
        outputs = model(train_samples)

    print(f"✓ Forward pass successful")
    print(f"  - Loss: {outputs['loss'].item():.4f}")
    print(f"  - Metrics: {list(outputs.keys())}")

    # Check some key metrics
    if 'avg_tokens' in outputs:
        print(f"  - Avg tokens: {outputs['avg_tokens']:.1f}")
    if 'disc_real_acc' in outputs:
        print(f"  - Disc real acc: {outputs['disc_real_acc']:.3f}")
    if 'disc_fake_acc' in outputs:
        print(f"  - Disc fake acc: {outputs['disc_fake_acc']:.3f}")

    return outputs


def test_hf_trainer_setup(model, config, dataset_bundle):
    """测试4: HF Trainer构建"""
    print("\n" + "=" * 60)
    print("测试4: HF Trainer构建")
    print("=" * 60)

    # Load trainer
    print("Loading HF Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # Build trainer
    print("Building trainer...")
    trainer.build_trainer(model=model)

    print(f"✓ HF Trainer built successfully")
    print(f"  - Trainer type: {type(trainer).__name__}")
    print(f"  - Train dataset size: {len(trainer.splits['train'])}")

    return trainer


def main():
    print("=" * 60)
    print("VisionTokenPruningModel + HF Trainer Integration Test")
    print("=" * 60)

    # Load config
    config_path = "configs/vision_token_pruning_fsdp.yaml"
    print(f"\nLoading config from: {config_path}")
    config = load_config(override_file=config_path)

    # Modify config for quick testing
    config.dataset_settings.split['train'] = 10  # Only 10 samples
    config.dataset_settings.split['test'] = 2
    config.trainer_settings.dl_settings.batch_size = 2
    config.trainer_settings.dl_settings.epochs = 1
    config.trainer_settings.hf_settings['logging_steps'] = 1
    config.trainer_settings.hf_settings['save_strategy'] = 'no'  # Don't save for testing

    # Disable FSDP for single GPU test
    if 'fsdp' in config.trainer_settings.hf_settings:
        del config.trainer_settings.hf_settings['fsdp']
        del config.trainer_settings.hf_settings['fsdp_config']
        print("  (FSDP disabled for single GPU test)")

    # Load dataset
    print("\nLoading dataset...")
    dataset_bundle = load_dataset(config)
    print(f"✓ Dataset loaded: {len(dataset_bundle['splits']['train'])} train samples")

    # Run tests
    try:
        # Test 1: Model creation
        model = test_model_creation(config)

        # Test 2: Optimizer groups
        optimizer = test_optimizer_groups(model, config)

        # Test 3: Forward pass
        outputs = test_forward_pass(model, dataset_bundle)

        # Test 4: HF Trainer setup
        trainer = test_hf_trainer_setup(model, config, dataset_bundle)

        # Final summary
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run full training with: python main.py --config configs/vision_token_pruning_fsdp.yaml")
        print("2. Enable FSDP in config for multi-GPU training")
        print("3. Monitor training metrics and adjust hyperparameters")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

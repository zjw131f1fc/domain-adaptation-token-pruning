#!/usr/bin/env python
"""Attention Consistency Pruning - 主入口

使用新的 Attention Consistency Pruning 架构进行训练。

特点：
1. 直接继承 LlavaForConditionalGeneration，不使用 hook
2. 在剪枝层计算 h_real 和 h_fake，不需要完整的 real forward
3. 每个 answer token 独立判别
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str = "configs/vision_token_pruning.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config: dict):
    """设置环境"""
    global_settings = config.get('global_settings', {})

    # 设置随机种子
    seed = global_settings.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设置 CUDA 内存分配
    cuda_alloc_conf = global_settings.get('pytorch_cuda_alloc_conf')
    if cuda_alloc_conf:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cuda_alloc_conf

    # 设置缓存目录
    cache_dir = global_settings.get('hf_cache_dir')
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir


def create_model(config: dict, device: torch.device):
    """创建可剪枝的 LLaVA 模型"""
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

    method_config = config.get('method_settings', {})

    # 获取剪枝层配置
    pruning_layers = method_config.get('pruning_layers', [4, 14, 24])
    pruner_d_internal = method_config.get('pruner_d_internal', 128)
    disc_d_hidden = method_config.get('disc_d_hidden', 256)
    temperature = method_config.get('temperature', 1.0)
    dropout = method_config.get('pruner_dropout', 0.1)
    disc_spectral_norm = method_config.get('disc_use_spectral_norm', False)

    # 获取模型路径
    backbone_config = config.get('backbone_settings', {})
    model_name = backbone_config.get('name', 'llava-1.5-7b')

    # 映射模型名到 HuggingFace 路径
    model_mapping = {
        'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
    }
    model_path = model_mapping.get(model_name, model_name)

    print(f"Loading model from {model_path}...")

    # 创建可剪枝模型
    model = PrunableLlavaForConditionalGeneration.from_pretrained(
        model_path,
        pruning_layers=pruning_layers,
        pruner_d_internal=pruner_d_internal,
        disc_d_hidden=disc_d_hidden,
        temperature=temperature,
        dropout=dropout,
        disc_use_spectral_norm=disc_spectral_norm,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    # 冻结基础模型
    model.freeze_base_model()

    print(f"Model loaded. Pruning layers: {pruning_layers}")
    print(f"Trainable parameters:")
    print(f"  - Pruners: {sum(p.numel() for p in model.get_pruner_parameters()):,}")
    print(f"  - Discriminators: {sum(p.numel() for p in model.get_discriminator_parameters()):,}")

    return model


def main():
    """主函数"""
    print("=" * 60)
    print("Attention Consistency Pruning - Main")
    print("=" * 60)

    # 加载配置
    config = load_config()
    setup_environment(config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建模型
    model = create_model(config, device)

    # 创建训练器
    from method.training_acp import ACPTrainer

    trainer = ACPTrainer(
        model=model,
        config=config,
        device=device
    )

    print("\nModel and trainer created successfully!")
    print("\nTo run training, you need to:")
    print("1. Load your dataset (e.g., VQAv2)")
    print("2. Create data loader")
    print("3. Call trainer.train_step(batch) in your training loop")
    print("\nExample:")
    print("  for batch in data_loader:")
    print("      losses = trainer.train_step(batch)")
    print("      print(losses['metrics'])")


if __name__ == "__main__":
    main()

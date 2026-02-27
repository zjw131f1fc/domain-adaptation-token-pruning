#!/usr/bin/env python
"""可视化 h_real, h_fake, h_corrected 的分布偏移

用法:
    python scripts/visualize_distribution_shift.py --checkpoint <path>

分析内容:
1. Global Mean Distance - 中心点距离
2. Per-Sample Distance - 样本级距离
3. Distribution Shape - 方差、标准差
4. MMD - Maximum Mean Discrepancy
5. PCA / t-SNE / 1D projection 可视化
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="configs/vision_token_pruning.yaml", help="Config file path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory")
    return parser.parse_args()


def load_model_and_processor(checkpoint_path, config_path, device):
    """加载模型和 checkpoint"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    # 加载配置
    config = load_config(override_file=config_path)
    method_cfg = config['method_settings']

    model_path = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading base model from {model_path}...")

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"

    # 创建可剪枝模型（从配置读取参数）
    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=method_cfg.get('pruning_layers', [4, 14, 24]),
        pruner_d_internal=method_cfg.get('pruner_d_internal', 512),
        pruner_n_heads=method_cfg.get('pruner_n_heads', 4),
        pruner_n_queries=method_cfg.get('pruner_n_queries', 32),
        pruner_query_dropout=0.0,  # 分析时关闭 dropout
        use_adapter=method_cfg.get('use_adapter', True),
        adapter_bottleneck=method_cfg.get('adapter_bottleneck', 512),
        adapter_type=method_cfg.get('adapter_type', 'lightweight'),
        use_separated_adapters=method_cfg.get('use_separated_adapters', False),
        vision_adapter_bottleneck=method_cfg.get('vision_adapter_bottleneck', 512),
        text_adapter_bottleneck=method_cfg.get('text_adapter_bottleneck', 512),
        mask_encoder_type=method_cfg.get('mask_encoder_type', 'attention'),
        temperature=method_cfg.get('eval_temperature', 0.1),
        dropout=0.0,  # 分析时关闭 dropout
        adapter_dropout=0.0,  # 分析时关闭 dropout
        use_gumbel_noise=False,  # 分析时关闭 Gumbel noise
        pruning_threshold=method_cfg.get('eval_pruning_threshold', 0.5),
        use_question_condition=method_cfg.get('use_question_condition', False),
    )

    model.freeze_base_model()

    # 加载 checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        print("  Loaded pruner_state_dict")

    # 根据 checkpoint 中的 key 判断 adapter 类型
    if 'separated_adapter_state_dict' in checkpoint and model.use_adapter:
        model.separated_adapter_manager.load_state_dict(checkpoint['separated_adapter_state_dict'])
        print("  Loaded separated_adapter_state_dict")
    elif 'adapter_state_dict' in checkpoint and model.use_adapter:
        model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
        print("  Loaded adapter_state_dict")

    model.eval()
    print("Model loaded.")

    return model, processor


def load_samples(num_samples, config_path):
    """加载样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    config = load_config(override_file=config_path)
    config['dataset_settings']['split'] = {'train': num_samples * 2, 'test': num_samples}

    data_bundle = load_dataset(config)
    test_dataset = data_bundle['splits']['test']

    return list(test_dataset)[:num_samples]


def preprocess_sample(sample, processor, device):
    """预处理样本"""
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer.capitalize()}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    # 找 vision tokens 位置
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    n_vision_tokens = 576

    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = vision_start + n_vision_tokens
    else:
        vision_start = 1
        vision_end = vision_start + n_vision_tokens

    # 找 ASSISTANT: 位置
    assistant_ids = processor.tokenizer.encode("\nASSISTANT:", add_special_tokens=False)
    if assistant_ids[0] == 29871:
        assistant_ids = assistant_ids[1:]

    ids = input_ids[0].tolist()
    assistant_pos = None
    for j in range(len(ids) - len(assistant_ids) + 1):
        if ids[j:j+len(assistant_ids)] == assistant_ids:
            assistant_pos = j + len(assistant_ids)
            break

    if assistant_pos is None:
        return None

    question_starts = [vision_end]
    question_ends = [assistant_pos]
    answer_starts = [assistant_pos]

    pad_token_id = processor.tokenizer.pad_token_id
    answer_end = seq_len
    for j in range(assistant_pos, seq_len):
        if ids[j] == pad_token_id:
            answer_end = j
            break
    answer_ends = [answer_end]

    return {
        'inputs': inputs,
        'vision_start': vision_start,
        'vision_end': vision_end,
        'question_starts': question_starts,
        'question_ends': question_ends,
        'answer_starts': answer_starts,
        'answer_ends': answer_ends,
        'seq_len': seq_len,
    }


def compute_mmd(X, Y, gamma=1.0):
    """计算 Maximum Mean Discrepancy"""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K_XX = np.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_YY = np.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    K_XY = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))

    m = X.shape[0]
    n = Y.shape[0]

    mmd = (K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n))
    return mmd


def cohens_d(x, y):
    """计算 Cohen's d 效应量"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)


def collect_h_vectors(model, processor, samples, device):
    """收集所有样本的 h_real, h_fake, h_corrected"""
    h_real_all = defaultdict(list)
    h_fake_all = defaultdict(list)
    h_corrected_all = defaultdict(list)

    print("Collecting h vectors...")
    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        prep = preprocess_sample(sample, processor, device)
        if prep is None:
            continue

        inputs = prep['inputs']

        with torch.no_grad():
            output = model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                vision_start=prep['vision_start'],
                vision_end=prep['vision_end'],
                question_starts=prep['question_starts'],
                question_ends=prep['question_ends'],
                answer_starts=prep['answer_starts'],
                answer_ends=prep['answer_ends'],
                return_pruning_info=True,
            )

        if hasattr(output, 'pruning_info') and output.pruning_info:
            for layer_idx, info in output.pruning_info.items():
                if 'h_real' in info and 'h_fake' in info:
                    for h_real, h_fake in zip(info['h_real'], info['h_fake']):
                        # h_real, h_fake: (heads, n_ans, head_dim)
                        # 展平为 (n_ans, heads * head_dim)
                        h_real_flat = h_real.permute(1, 0, 2).reshape(h_real.shape[1], -1)
                        h_fake_flat = h_fake.permute(1, 0, 2).reshape(h_fake.shape[1], -1)
                        h_real_all[layer_idx].append(h_real_flat.float().cpu().numpy())
                        h_fake_all[layer_idx].append(h_fake_flat.float().cpu().numpy())

                if 'h_corrected' in info:
                    for h_corr in info['h_corrected']:
                        h_corr_flat = h_corr.permute(1, 0, 2).reshape(h_corr.shape[1], -1)
                        h_corrected_all[layer_idx].append(h_corr_flat.float().cpu().numpy())

    # 合并所有样本
    result = {}
    for layer_idx in h_real_all.keys():
        result[layer_idx] = {
            'h_real': np.concatenate(h_real_all[layer_idx], axis=0),
            'h_fake': np.concatenate(h_fake_all[layer_idx], axis=0),
            'h_corrected': np.concatenate(h_corrected_all[layer_idx], axis=0) if h_corrected_all[layer_idx] else None,
        }

    return result


def analyze_and_visualize(h_data, layer_idx, output_dir):
    """分析并可视化单层的分布偏移"""
    h_real = h_data['h_real']
    h_fake = h_data['h_fake']
    h_corrected = h_data['h_corrected']

    n_samples = h_real.shape[0]
    print(f"  Layer {layer_idx}: h_real={n_samples}, h_fake={h_fake.shape[0]}, h_corrected={h_corrected.shape[0] if h_corrected is not None else 0}")

    # ==================== [1] Global Mean Distance ====================
    print(f"    [1] Global Mean Distance (原有度量 - 可能掩盖样本级差异):")

    center_real = h_real.mean(axis=0)
    center_fake = h_fake.mean(axis=0)
    center_corrected = h_corrected.mean(axis=0) if h_corrected is not None else None

    l2_fake = np.linalg.norm(center_fake - center_real)
    l2_corrected = np.linalg.norm(center_corrected - center_real) if center_corrected is not None else 0

    cos_fake = 1 - np.dot(center_fake, center_real) / (np.linalg.norm(center_fake) * np.linalg.norm(center_real) + 1e-8)
    cos_corrected = 1 - np.dot(center_corrected, center_real) / (np.linalg.norm(center_corrected) * np.linalg.norm(center_real) + 1e-8) if center_corrected is not None else 0

    l2_improvement = (l2_fake - l2_corrected) / (l2_fake + 1e-8) * 100
    cos_improvement = (cos_fake - cos_corrected) / (cos_fake + 1e-8) * 100

    print(f"      Center L2 (h_fake -> h_real): {l2_fake:.4f}")
    print(f"      Center L2 (h_corrected -> h_real): {l2_corrected:.4f}")
    print(f"      L2 improvement: {l2_improvement:.1f}%")
    print(f"      Center cosine dist (h_fake -> h_real): {cos_fake:.4f}")
    print(f"      Center cosine dist (h_corrected -> h_real): {cos_corrected:.4f}")
    print(f"      Cosine improvement: {cos_improvement:.1f}%")

    # ==================== [2] Per-Sample Distance ====================
    print(f"\n    [2] Per-Sample Distance (判别器看到的是样本级差异):")

    per_sample_l2_fake = np.linalg.norm(h_fake - h_real, axis=1)
    per_sample_l2_corrected = np.linalg.norm(h_corrected - h_real, axis=1) if h_corrected is not None else np.zeros(n_samples)

    per_sample_cos_fake = 1 - np.sum(h_fake * h_real, axis=1) / (np.linalg.norm(h_fake, axis=1) * np.linalg.norm(h_real, axis=1) + 1e-8)
    per_sample_cos_corrected = 1 - np.sum(h_corrected * h_real, axis=1) / (np.linalg.norm(h_corrected, axis=1) * np.linalg.norm(h_real, axis=1) + 1e-8) if h_corrected is not None else np.zeros(n_samples)

    l2_imp = (per_sample_l2_fake.mean() - per_sample_l2_corrected.mean()) / (per_sample_l2_fake.mean() + 1e-8) * 100
    cos_imp = (per_sample_cos_fake.mean() - per_sample_cos_corrected.mean()) / (per_sample_cos_fake.mean() + 1e-8) * 100

    print(f"      Per-sample L2 (h_fake -> h_real): mean={per_sample_l2_fake.mean():.4f}, std={per_sample_l2_fake.std():.4f}")
    print(f"      Per-sample L2 (h_corrected -> h_real): mean={per_sample_l2_corrected.mean():.4f}, std={per_sample_l2_corrected.std():.4f}")
    print(f"      Per-sample L2 improvement: {l2_imp:.1f}%")
    print(f"      Per-sample cosine (h_fake -> h_real): mean={per_sample_cos_fake.mean():.4f}, std={per_sample_cos_fake.std():.4f}")
    print(f"      Per-sample cosine (h_corrected -> h_real): mean={per_sample_cos_corrected.mean():.4f}, std={per_sample_cos_corrected.std():.4f}")
    print(f"      Per-sample cosine improvement: {cos_imp:.1f}%")

    # ==================== [3] Distribution Shape ====================
    print(f"\n    [3] Distribution Shape (方差、标准差 - 关键诊断):")

    std_real = h_real.std(axis=1).mean()
    std_fake = h_fake.std(axis=1).mean()
    std_corrected = h_corrected.std(axis=1).mean() if h_corrected is not None else 0

    var_real = h_real.var(axis=0).mean()
    var_fake = h_fake.var(axis=0).mean()
    var_corrected = h_corrected.var(axis=0).mean() if h_corrected is not None else 0

    print(f"      Mean std across samples (h_real): {std_real:.4f}")
    print(f"      Mean std across samples (h_fake): {std_fake:.4f}")
    print(f"      Mean std across samples (h_corrected): {std_corrected:.4f}")
    print(f"      Std ratio (fake/real): {std_fake / (std_real + 1e-8):.4f}")
    print(f"      Std ratio (corrected/real): {std_corrected / (std_real + 1e-8):.4f}")
    print(f"      Mean var across dimensions (h_real): {var_real:.4f}")
    print(f"      Mean var across dimensions (h_fake): {var_fake:.4f}")
    print(f"      Mean var across dimensions (h_corrected): {var_corrected:.4f}")

    # ==================== [4] MMD ====================
    print(f"\n    [4] MMD (Maximum Mean Discrepancy - 更敏感的分布距离):")

    # 使用子采样计算 MMD（避免内存问题）
    max_mmd_samples = min(500, n_samples)
    idx = np.random.choice(n_samples, max_mmd_samples, replace=False)

    mmd_fake = compute_mmd(h_fake[idx], h_real[idx])
    mmd_corrected = compute_mmd(h_corrected[idx], h_real[idx]) if h_corrected is not None else 0

    mmd_improvement = (mmd_fake - mmd_corrected) / (mmd_fake + 1e-8) * 100

    print(f"      MMD (h_fake vs h_real): {mmd_fake:.6f}")
    print(f"      MMD (h_corrected vs h_real): {mmd_corrected:.6f}")
    print(f"      MMD improvement: {mmd_improvement:.1f}%")

    # ==================== 可视化 ====================
    os.makedirs(output_dir, exist_ok=True)

    # 子采样用于可视化
    max_vis_samples = min(1000, n_samples)
    vis_idx = np.random.choice(n_samples, max_vis_samples, replace=False)

    h_real_vis = h_real[vis_idx]
    h_fake_vis = h_fake[vis_idx]
    h_corrected_vis = h_corrected[vis_idx] if h_corrected is not None else None

    # ----- PCA -----
    print("    Running PCA...")
    pca = PCA(n_components=2)
    all_data = np.vstack([h_real_vis, h_fake_vis] + ([h_corrected_vis] if h_corrected_vis is not None else []))
    pca.fit(all_data)

    real_pca = pca.transform(h_real_vis)
    fake_pca = pca.transform(h_fake_vis)
    corrected_pca = pca.transform(h_corrected_vis) if h_corrected_vis is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='h_real', s=10)
    ax.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_pca is not None:
        ax.scatter(corrected_pca[:, 0], corrected_pca[:, 1], alpha=0.5, label='h_corrected', s=10)
    ax.legend()
    ax.set_title(f'Layer {layer_idx} - PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_pca.png'), dpi=150)
    plt.close()
    print(f"    Saved to {output_dir}/distribution_shift_layer{layer_idx}_pca.png")

    # ----- Per-sample distance plot -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(per_sample_l2_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
    axes[0].hist(per_sample_l2_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
    axes[0].legend()
    axes[0].set_title(f'Layer {layer_idx} - Per-sample L2 Distance')
    axes[0].set_xlabel('L2 Distance')

    axes[1].hist(per_sample_cos_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
    axes[1].hist(per_sample_cos_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
    axes[1].legend()
    axes[1].set_title(f'Layer {layer_idx} - Per-sample Cosine Distance')
    axes[1].set_xlabel('Cosine Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'per_sample_distances_layer{layer_idx}.png'), dpi=150)
    plt.close()
    print(f"    Saved per-sample distance plot to {output_dir}/per_sample_distances_layer{layer_idx}.png")

    # ----- 1D Projection (LDA) -----
    # 用 LDA 找到最能区分 h_real 和 h_fake 的方向
    labels = np.array([0] * len(h_real_vis) + [1] * len(h_fake_vis))
    X_for_lda = np.vstack([h_real_vis, h_fake_vis])

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_for_lda, labels)

    proj_real = lda.transform(h_real_vis).flatten()
    proj_fake = lda.transform(h_fake_vis).flatten()
    proj_corrected = lda.transform(h_corrected_vis).flatten() if h_corrected_vis is not None else None

    # Cohen's d
    d_fake = abs(cohens_d(proj_fake, proj_real))
    d_corrected = abs(cohens_d(proj_corrected, proj_real)) if proj_corrected is not None else 0

    print(f"    1D projection (lda) effect sizes:")
    print(f"      Cohen's d (h_fake vs h_real): {d_fake:.3f}")
    print(f"      Cohen's d (h_corrected vs h_real): {d_corrected:.3f}")

    # 方差分析
    var_proj_real = proj_real.var()
    var_proj_fake = proj_fake.var()
    var_proj_corrected = proj_corrected.var() if proj_corrected is not None else 0

    print(f"    1D projection variance (解释平滑度):")
    print(f"      Var(proj_real): {var_proj_real:.4f}")
    print(f"      Var(proj_fake): {var_proj_fake:.4f}")
    print(f"      Var(proj_corrected): {var_proj_corrected:.4f}")
    print(f"      Std(proj_real): {np.sqrt(var_proj_real):.4f}")
    print(f"      Std(proj_fake): {np.sqrt(var_proj_fake):.4f}")
    print(f"      Std(proj_corrected): {np.sqrt(var_proj_corrected):.4f}")
    print(f"      Var ratio (corrected/real): {var_proj_corrected / (var_proj_real + 1e-8):.2f}x")
    print(f"      Var ratio (corrected/fake): {var_proj_corrected / (var_proj_fake + 1e-8):.2f}x")

    # ===== 修正量分析 =====
    if proj_corrected is not None:
        # 计算每个样本的修正量和方向
        mean_real = proj_real.mean()
        mean_fake = proj_fake.mean()
        mean_corrected = proj_corrected.mean()

        # 修正量（在 LDA 方向上）
        correction = proj_corrected - proj_fake  # 每个样本的修正量
        total_gap = mean_real - mean_fake  # fake 到 real 的总距离

        # 修正方向：正值表示向 real 方向移动
        correction_toward_real = correction * np.sign(total_gap)

        # 统计
        mean_correction = correction.mean()
        mean_correction_toward_real = correction_toward_real.mean()
        pct_toward_real = (correction_toward_real > 0).mean() * 100
        correction_ratio = mean_correction_toward_real / (abs(total_gap) + 1e-8) * 100

        print(f"    Adapter 修正分析 (1D LDA 方向):")
        print(f"      h_real 中心: {mean_real:.3f}")
        print(f"      h_fake 中心: {mean_fake:.3f}")
        print(f"      h_corrected 中心: {mean_corrected:.3f}")
        print(f"      fake->real 距离: {total_gap:.3f}")
        print(f"      fake->corrected 移动: {mean_correction:.3f}")
        print(f"      向 real 方向移动: {mean_correction_toward_real:.3f} ({correction_ratio:.1f}% of gap)")
        print(f"      向 real 方向移动的样本比例: {pct_toward_real:.1f}%")

    # 1D projection 可视化（增强版）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：分布直方图
    ax = axes[0]
    ax.hist(proj_real, bins=50, alpha=0.5, label=r'$h_{real}$ (Full Model)', density=True, color='gray')
    ax.hist(proj_fake, bins=50, alpha=0.5, label=r'$h_{fake}$ (Pruned w/o Adapter)', density=True, color='orange')
    if proj_corrected is not None:
        ax.hist(proj_corrected, bins=50, alpha=0.5, label=r'$h_{corrected}$ (ACP Final)', density=True, color='green')

    # 添加中心线
    ax.axvline(proj_real.mean(), color='gray', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(proj_fake.mean(), color='orange', linestyle='--', linewidth=2, alpha=0.8)
    if proj_corrected is not None:
        ax.axvline(proj_corrected.mean(), color='green', linestyle='--', linewidth=2, alpha=0.8)

    ax.legend(fontsize=10)
    ax.set_title(f'Layer {layer_idx}: 1D Projection (Real vs Fake)', fontsize=14)
    ax.set_xlabel('1D Projection', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    # 右图：修正方向分析
    ax2 = axes[1]
    if proj_corrected is not None:
        # 绘制修正量分布
        ax2.hist(correction_toward_real, bins=50, alpha=0.7, color='blue', density=True)
        ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='No correction')
        ax2.axvline(correction_toward_real.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {correction_toward_real.mean():.3f}')

        # 获取当前y轴范围
        y_max = ax2.get_ylim()[1]

        ax2.legend(fontsize=10)
        ax2.set_title(f'Layer {layer_idx}: Adapter Correction Direction\n(Positive = Toward Real)', fontsize=14)
        ax2.set_xlabel('Correction Amount (toward real)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)

        # 添加文字说明
        textstr = f'Gap (fake→real): {total_gap:.2f}\nCorrection: {mean_correction_toward_real:.2f} ({correction_ratio:.1f}%)\nSamples toward real: {pct_toward_real:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_proj1d.png'), dpi=150)
    plt.close()
    print(f"    Saved to {output_dir}/distribution_shift_layer{layer_idx}_proj1d.png")

    # ----- t-SNE -----
    print("    Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    all_vis = np.vstack([h_real_vis, h_fake_vis] + ([h_corrected_vis] if h_corrected_vis is not None else []))
    all_tsne = tsne.fit_transform(all_vis)

    n_real = len(h_real_vis)
    n_fake = len(h_fake_vis)
    real_tsne = all_tsne[:n_real]
    fake_tsne = all_tsne[n_real:n_real + n_fake]
    corrected_tsne = all_tsne[n_real + n_fake:] if h_corrected_vis is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='h_real', s=10)
    ax.scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_tsne is not None:
        ax.scatter(corrected_tsne[:, 0], corrected_tsne[:, 1], alpha=0.5, label='h_corrected', s=10)
    ax.legend()
    ax.set_title(f'Layer {layer_idx} - t-SNE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}.png'), dpi=150)
    plt.close()
    print(f"    Saved t-SNE to {output_dir}/distribution_shift_layer{layer_idx}.png")

    # ----- DA comparison (Domain Adaptation 视角) -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA
    axes[0, 0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='h_real', s=10)
    axes[0, 0].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_pca is not None:
        axes[0, 0].scatter(corrected_pca[:, 0], corrected_pca[:, 1], alpha=0.5, label='h_corrected', s=10)
    axes[0, 0].legend()
    axes[0, 0].set_title('PCA')

    # t-SNE
    axes[0, 1].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='h_real', s=10)
    axes[0, 1].scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_tsne is not None:
        axes[0, 1].scatter(corrected_tsne[:, 0], corrected_tsne[:, 1], alpha=0.5, label='h_corrected', s=10)
    axes[0, 1].legend()
    axes[0, 1].set_title('t-SNE')

    # 1D projection
    axes[1, 0].hist(proj_real, bins=50, alpha=0.5, label='h_real', density=True)
    axes[1, 0].hist(proj_fake, bins=50, alpha=0.5, label='h_fake', density=True)
    if proj_corrected is not None:
        axes[1, 0].hist(proj_corrected, bins=50, alpha=0.5, label='h_corrected', density=True)
    axes[1, 0].legend()
    axes[1, 0].set_title(f'1D LDA Projection (d_fake={d_fake:.2f}, d_corr={d_corrected:.2f})')

    # Per-sample L2 distance
    axes[1, 1].hist(per_sample_l2_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
    axes[1, 1].hist(per_sample_l2_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
    axes[1, 1].legend()
    axes[1, 1].set_title('Per-sample L2 Distance')

    plt.suptitle(f'Layer {layer_idx} - Distribution Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_da.png'), dpi=150)
    plt.close()
    print(f"    Saved DA comparison to {output_dir}/distribution_shift_layer{layer_idx}_da.png")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using config: {args.config}")

    # 加载模型
    model, processor = load_model_and_processor(args.checkpoint, args.config, device)

    # 加载样本
    print(f"\nLoading {args.num_samples} samples...")
    samples = load_samples(args.num_samples, args.config)
    print(f"Loaded {len(samples)} samples")

    # 收集 h 向量
    h_data = collect_h_vectors(model, processor, samples, device)

    # 分析和可视化
    print("\nGenerating visualizations...")
    for layer_idx in sorted(h_data.keys()):
        analyze_and_visualize(h_data[layer_idx], layer_idx, args.output_dir)

    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

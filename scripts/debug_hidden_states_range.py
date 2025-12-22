"""调试脚本：查看LLM hidden states的数值范围

用途：帮助确定合适的噪声强度 (disc_noise_scale)
运行：python scripts/debug_hidden_states_range.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration


def analyze_hidden_states():
    """分析hidden states的数值范围"""

    print("=" * 80)
    print("Hidden States 数值范围分析")
    print("=" * 80)

    # 1. 加载模型
    print("\n[1/4] 加载LLaVA模型...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"  模型: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    device = model.device
    print(f"  设备: {device}")

    # 2. 准备测试数据（简单的示例）
    print("\n[2/4] 准备测试数据...")

    # 创建3个测试样本
    test_samples = [
        {
            "image": Image.new('RGB', (336, 336), color='red'),
            "question": "What color is this image?",
            "answer": "red"
        },
        {
            "image": Image.new('RGB', (336, 336), color='blue'),
            "question": "Describe the content of this image in detail.",
            "answer": "This is a solid blue colored image"
        },
        {
            "image": Image.new('RGB', (336, 336), color='green'),
            "question": "What objects can you see?",
            "answer": "green background"
        }
    ]

    print(f"  创建了 {len(test_samples)} 个测试样本")

    # 3. Forward获取hidden states
    print("\n[3/4] Forward获取hidden states...")

    disc_target_layers = [-1, -3, -5]  # 从配置文件读取
    all_stats = []

    for sample_idx, sample in enumerate(test_samples):
        print(f"\n  样本 {sample_idx + 1}/{len(test_samples)}:")

        # 构造prompt
        prompt = f"USER: <image>\n{sample['question']}\nASSISTANT: {sample['answer']}"

        # Preprocess
        inputs = processor(
            text=prompt,
            images=sample['image'],
            return_tensors="pt"
        ).to(device, torch.float16)

        with torch.no_grad():
            # Forward with output_hidden_states
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # 获取所有层的hidden states
            all_hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

            # 找到vision tokens的位置
            # LLaVA的image token id通常是特殊的
            input_ids = inputs['input_ids'][0]
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

            # 简单起见，假设vision tokens在中间某个位置
            # 实际上LLaVA-1.5会用576个tokens表示图像
            # 这里我们提取所有text tokens（排除image tokens的粗略估计）
            seq_len = all_hidden_states[0].shape[1]

            # 粗略估计：假设前10个是prompt开始，后面是图像，再后面是question+answer
            # 实际应该更精确地检测，但这里只是为了分析数值范围
            v_start = 10
            v_end = v_start + 576  # LLaVA-1.5的vision tokens通常是576个

            # 如果v_end超过序列长度，调整
            if v_end >= seq_len:
                v_end = seq_len - 20  # 留一些给后面的text

            # 提取目标层的hidden states
            for layer_idx in disc_target_layers:
                hidden = all_hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

                # 提取text部分（排除vision）
                text_before = hidden[:, :v_start, :]
                text_after = hidden[:, v_end+1:, :]

                if text_after.shape[1] > 0:
                    text_hidden = torch.cat([text_before, text_after], dim=1)
                else:
                    text_hidden = text_before

                # 统计数值范围（转换为float32以支持quantile）
                text_hidden_f32 = text_hidden.float()
                stats = {
                    'layer_idx': layer_idx,
                    'sample_idx': sample_idx,
                    'mean': text_hidden_f32.mean().item(),
                    'std': text_hidden_f32.std().item(),
                    'min': text_hidden_f32.min().item(),
                    'max': text_hidden_f32.max().item(),
                    'abs_max': text_hidden_f32.abs().max().item(),
                    'median': text_hidden_f32.median().item(),
                    'q25': text_hidden_f32.quantile(0.25).item(),
                    'q75': text_hidden_f32.quantile(0.75).item(),
                }

                all_stats.append(stats)

                print(f"    Layer {layer_idx:3d}: "
                      f"mean={stats['mean']:7.2f}, "
                      f"std={stats['std']:6.2f}, "
                      f"range=[{stats['min']:7.2f}, {stats['max']:7.2f}], "
                      f"abs_max={stats['abs_max']:7.2f}")

    # 4. 分析并给出建议
    print("\n" + "=" * 80)
    print("[4/4] 统计汇总与建议")
    print("=" * 80)

    # 汇总统计
    all_means = [s['mean'] for s in all_stats]
    all_stds = [s['std'] for s in all_stats]
    all_abs_maxs = [s['abs_max'] for s in all_stats]

    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)
    avg_abs_max = np.mean(all_abs_maxs)
    max_abs_max = np.max(all_abs_maxs)

    print(f"\n跨所有样本和层的统计:")
    print(f"  平均 mean:    {avg_mean:7.2f}")
    print(f"  平均 std:     {avg_std:7.2f}")
    print(f"  平均 abs_max: {avg_abs_max:7.2f}")
    print(f"  最大 abs_max: {max_abs_max:7.2f}")

    # 5. 测试加权融合后的数值范围
    print("\n" + "=" * 80)
    print("加权融合后的数值范围分析")
    print("=" * 80)

    pooled_stats = []

    for sample_idx, sample in enumerate(test_samples):
        # 构造prompt
        prompt = f"USER: <image>\n{sample['question']}\nASSISTANT: {sample['answer']}"

        # Preprocess
        inputs = processor(
            text=prompt,
            images=sample['image'],
            return_tensors="pt"
        ).to(device, torch.float16)

        with torch.no_grad():
            # Forward
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            all_hidden_states = outputs.hidden_states
            seq_len = all_hidden_states[0].shape[1]

            # 粗略估计vision位置
            v_start = 10
            v_end = v_start + 576
            if v_end >= seq_len:
                v_end = seq_len - 20

            # 提取text hidden states
            text_hidden_list = []
            for layer_idx in disc_target_layers:
                hidden = all_hidden_states[layer_idx]
                text_before = hidden[:, :v_start, :]
                text_after = hidden[:, v_end+1:, :]

                if text_after.shape[1] > 0:
                    text_hidden = torch.cat([text_before, text_after], dim=1)
                else:
                    text_hidden = text_before

                text_hidden_list.append(text_hidden)

            # 加权融合
            pooled_list = []
            for text_hidden in text_hidden_list:
                batch_size, text_len, hidden_dim = text_hidden.shape

                # 创建线性递增的权重
                start_weight, end_weight = 0.5, 1.0
                weights = torch.linspace(start_weight, end_weight, text_len, device=text_hidden.device)
                weights = weights.view(1, -1, 1)

                # 加权平均
                weighted_hidden = text_hidden * weights
                pooled = weighted_hidden.sum(dim=1) / weights.sum()
                pooled_list.append(pooled)

            # 统计pooled后的数值范围
            for i, pooled in enumerate(pooled_list):
                stats = {
                    'layer_idx': disc_target_layers[i],
                    'sample_idx': sample_idx,
                    'mean': pooled.mean().item(),
                    'std': pooled.std().item(),
                    'min': pooled.min().item(),
                    'max': pooled.max().item(),
                    'abs_max': pooled.abs().max().item(),
                }
                pooled_stats.append(stats)

                print(f"  样本{sample_idx+1} Layer{disc_target_layers[i]:3d} (pooled): "
                      f"mean={stats['mean']:7.2f}, "
                      f"std={stats['std']:6.2f}, "
                      f"abs_max={stats['abs_max']:7.2f}")

    # 汇总pooled统计
    pooled_abs_maxs = [s['abs_max'] for s in pooled_stats]
    pooled_stds = [s['std'] for s in pooled_stats]

    avg_pooled_abs_max = np.mean(pooled_abs_maxs)
    avg_pooled_std = np.mean(pooled_stds)
    max_pooled_abs_max = np.max(pooled_abs_maxs)

    print(f"\nPooled hidden states统计:")
    print(f"  平均 abs_max: {avg_pooled_abs_max:7.2f}")
    print(f"  最大 abs_max: {max_pooled_abs_max:7.2f}")
    print(f"  平均 std:     {avg_pooled_std:7.2f}")

    # 6. 给出噪声强度建议
    print("\n" + "=" * 80)
    print("噪声强度建议 (disc_noise_scale)")
    print("=" * 80)

    # 默认配置值
    current_disc_noise_scale = 0.01

    print(f"\n当前配置: disc_noise_scale = {current_disc_noise_scale}")

    print("\n基于hidden states数值范围的建议:")

    # 方案1: 基于std
    noise_0_5_pct = avg_pooled_std * 0.005  # 0.5% std
    noise_1_pct = avg_pooled_std * 0.01     # 1% std
    noise_2_pct = avg_pooled_std * 0.02     # 2% std
    noise_5_pct = avg_pooled_std * 0.05     # 5% std

    print(f"\n  方案A - 基于pooled std ({avg_pooled_std:.2f}):")
    print(f"    0.5% std: disc_noise_scale = {noise_0_5_pct:.4f}  (极保守)")
    print(f"    1.0% std: disc_noise_scale = {noise_1_pct:.4f}  (保守)")
    print(f"    2.0% std: disc_noise_scale = {noise_2_pct:.4f}  (推荐)")
    print(f"    5.0% std: disc_noise_scale = {noise_5_pct:.4f}  (激进)")

    # 方案2: 基于abs_max
    noise_0_5_pct_max = avg_pooled_abs_max * 0.005
    noise_1_pct_max = avg_pooled_abs_max * 0.01
    noise_2_pct_max = avg_pooled_abs_max * 0.02

    print(f"\n  方案B - 基于pooled abs_max ({avg_pooled_abs_max:.2f}):")
    print(f"    0.5% max: disc_noise_scale = {noise_0_5_pct_max:.4f}")
    print(f"    1.0% max: disc_noise_scale = {noise_1_pct_max:.4f}")
    print(f"    2.0% max: disc_noise_scale = {noise_2_pct_max:.4f}")

    # 综合推荐
    recommended = noise_2_pct  # 2% std
    conservative = noise_1_pct  # 1% std
    aggressive = noise_5_pct    # 5% std

    print(f"\n  【综合推荐】:")
    print(f"    保守方案: disc_noise_scale = {conservative:.4f}")
    print(f"    推荐方案: disc_noise_scale = {recommended:.4f}  ⭐")
    print(f"    激进方案: disc_noise_scale = {aggressive:.4f}")

    print(f"\n  说明:")
    print(f"    - 如果判别器学习太快 (accuracy>95%)，增大噪声")
    print(f"    - 如果判别器学习太慢 (accuracy<60%)，减小噪声")
    print(f"    - 建议从推荐方案开始，根据训练情况调整")

    # 7. 对比当前配置
    relative_noise_current = (current_disc_noise_scale / avg_pooled_std) * 100

    print(f"\n  当前配置分析:")
    print(f"    disc_noise_scale = {current_disc_noise_scale}")
    print(f"    相对强度 = {relative_noise_current:.2f}% of pooled std")

    if relative_noise_current < 0.5:
        print(f"    ⚠️  噪声太小，几乎无正则化效果！建议增大到 {recommended:.4f}")
    elif relative_noise_current < 1.0:
        print(f"    ✓  噪声适中（保守），可以尝试增大到 {recommended:.4f}")
    elif relative_noise_current < 3.0:
        print(f"    ✓✓ 噪声强度合理")
    else:
        print(f"    ⚠️  噪声可能过大，可能影响判别器学习")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    return {
        'avg_pooled_std': avg_pooled_std,
        'avg_pooled_abs_max': avg_pooled_abs_max,
        'recommended_noise': recommended,
        'conservative_noise': conservative,
        'aggressive_noise': aggressive,
    }


if __name__ == "__main__":
    print("开始分析LLM hidden states数值范围...")
    results = analyze_hidden_states()
    print(f"\n建议将配置文件中的 disc_noise_scale 修改为: {results['recommended_noise']:.4f}")

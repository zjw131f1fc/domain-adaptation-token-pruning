#!/usr/bin/env python
"""分析 Adapter 在 vision token 和 text token 上的效果差异

用法:
    python scripts/analyze_adapter_effect.py --checkpoint <path>

分析内容:
1. Adapter 在不同位置类型的输出变化量
2. h_real vs h_fake 在不同位置的差异
3. FiLM 调制参数 (gamma, beta) 在不同位置的分布
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
import torch.nn.functional as F
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to analyze")
    parser.add_argument("--device", type=str, default="cuda:7", help="Device to use")
    return parser.parse_args()


def load_model_and_processor(checkpoint_path, device):
    """加载模型和 checkpoint"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

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

    # 创建可剪枝模型
    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=[4, 14, 24],
        pruner_d_internal=512,
        pruner_n_heads=4,
        adapter_bottleneck=512,
        adapter_type='lightweight',
        temperature=1.0,
        dropout=0.0,  # 分析时关闭 dropout
    )

    model.freeze_base_model()

    # 加载 checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        print("  Loaded pruner_state_dict")
    if 'adapter_state_dict' in checkpoint:
        model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
        print("  Loaded adapter_state_dict")

    model.eval()
    print("Model loaded.")

    return model, processor


def load_samples(num_samples):
    """加载样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    config = load_config(override_file="configs/vision_token_pruning.yaml")
    config.dataset_settings['split'] = {'train': num_samples * 2, 'test': num_samples}

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


class AdapterAnalysisHook:
    """Hook 用于捕获 Adapter 的中间结果"""

    def __init__(self):
        self.captured = {}

    def clear(self):
        self.captured = {}

    def create_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            x = args[0] if len(args) > 0 else kwargs.get('x')
            mask = kwargs.get('mask')
            query = kwargs.get('query')

            # 计算 adapter 的增量（output - x）
            delta = output - x

            self.captured[layer_idx] = {
                'x': x.detach().clone(),
                'output': output.detach().clone(),
                'delta': delta.detach().clone(),
                'mask': mask.detach().clone() if mask is not None else None,
                'query': query.detach().clone() if query is not None else None,
            }

        return hook


def analyze_single_sample(model, processor, sample, device, hook):
    """分析单个样本"""
    prep = preprocess_sample(sample, processor, device)
    if prep is None:
        return None

    inputs = prep['inputs']
    hook.clear()

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

    # 提取各位置类型的统计
    results = {}
    for layer_idx, data in hook.captured.items():
        delta = data['delta'][0]  # (seq, hidden)
        x = data['x'][0]
        output_h = data['output'][0]

        # 计算各位置的 delta 范数
        delta_norm = delta.float().norm(dim=-1)  # (seq,)
        x_norm = x.float().norm(dim=-1)
        relative_change = delta_norm / (x_norm + 1e-8)

        # 分位置类型统计
        vs, ve = prep['vision_start'], prep['vision_end']
        qs, qe = prep['question_starts'][0], prep['question_ends'][0]
        ans_s, ans_e = prep['answer_starts'][0], prep['answer_ends'][0]

        results[layer_idx] = {
            'vision': {
                'delta_norm_mean': delta_norm[vs:ve].mean().item(),
                'delta_norm_std': delta_norm[vs:ve].std().item(),
                'relative_change_mean': relative_change[vs:ve].mean().item(),
            },
            'question': {
                'delta_norm_mean': delta_norm[qs:qe].mean().item(),
                'delta_norm_std': delta_norm[qs:qe].std().item(),
                'relative_change_mean': relative_change[qs:qe].mean().item(),
            },
            'answer': {
                'delta_norm_mean': delta_norm[ans_s:ans_e].mean().item(),
                'delta_norm_std': delta_norm[ans_s:ans_e].std().item(),
                'relative_change_mean': relative_change[ans_s:ans_e].mean().item(),
            },
            'gen_answer': {
                # 生成 answer 的位置（ans_s-1 到 ans_e-1）
                'delta_norm_mean': delta_norm[ans_s-1:ans_e-1].mean().item(),
                'delta_norm_std': delta_norm[ans_s-1:ans_e-1].std().item(),
                'relative_change_mean': relative_change[ans_s-1:ans_e-1].mean().item(),
            },
        }

    # 提取 h_real vs h_fake 差异
    pruning_infos = getattr(output, 'pruning_infos', None) or getattr(output, 'pruning_info', None)
    if pruning_infos:
        for layer_idx, info in pruning_infos.items():
            if 'h_real' in info and 'h_fake' in info:
                h_real_list = info['h_real']
                h_fake_list = info['h_fake']

                # 计算差异
                diffs = []
                for h_real, h_fake in zip(h_real_list, h_fake_list):
                    diff = (h_real.float() - h_fake.float()).abs().mean().item()
                    diffs.append(diff)

                if layer_idx in results:
                    results[layer_idx]['h_diff'] = np.mean(diffs)

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 加载模型
    model, processor = load_model_and_processor(args.checkpoint, device)

    # 加载样本
    print(f"\nLoading {args.num_samples} samples...")
    samples = load_samples(args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # 创建 hook
    hook = AdapterAnalysisHook()

    # 注册 hook 到所有 adapter
    handles = []
    for layer_idx in model.pruning_layers:
        adapter = model.adapter_manager.get_adapter(layer_idx)
        handle = adapter.register_forward_hook(hook.create_hook(layer_idx), with_kwargs=True)
        handles.append(handle)

    # 收集统计
    all_results = defaultdict(lambda: defaultdict(list))

    print("\nAnalyzing samples...")
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        result = analyze_single_sample(model, processor, sample, device, hook)
        if result is None:
            continue

        for layer_idx, layer_result in result.items():
            for pos_type in ['vision', 'question', 'answer', 'gen_answer']:
                if pos_type in layer_result:
                    for metric, value in layer_result[pos_type].items():
                        all_results[layer_idx][f"{pos_type}_{metric}"].append(value)

            if 'h_diff' in layer_result:
                all_results[layer_idx]['h_diff'].append(layer_result['h_diff'])

    # 清理 hooks
    for handle in handles:
        handle.remove()

    # 打印结果
    print("\n" + "=" * 80)
    print("ADAPTER EFFECT ANALYSIS")
    print("=" * 80)

    for layer_idx in sorted(all_results.keys()):
        print(f"\n{'='*40}")
        print(f"Layer {layer_idx}")
        print(f"{'='*40}")

        layer_data = all_results[layer_idx]

        # 按位置类型打印
        for pos_type in ['vision', 'question', 'answer', 'gen_answer']:
            delta_key = f"{pos_type}_delta_norm_mean"
            rel_key = f"{pos_type}_relative_change_mean"

            if delta_key in layer_data:
                delta_vals = layer_data[delta_key]
                rel_vals = layer_data[rel_key]

                print(f"\n  {pos_type.upper():12s}:")
                print(f"    delta_norm:      mean={np.mean(delta_vals):.4f}, std={np.std(delta_vals):.4f}")
                print(f"    relative_change: mean={np.mean(rel_vals):.4f}, std={np.std(rel_vals):.4f}")

        # h_real vs h_fake 差异
        if 'h_diff' in layer_data:
            h_diffs = layer_data['h_diff']
            print(f"\n  h_real vs h_fake diff: mean={np.mean(h_diffs):.4f}, std={np.std(h_diffs):.4f}")

    # 打印总结
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nAdapter 对不同位置类型的调整幅度（relative_change）：")
    print(f"{'Layer':<8} {'Vision':<12} {'Question':<12} {'Answer':<12} {'GenAnswer':<12}")
    print("-" * 56)

    for layer_idx in sorted(all_results.keys()):
        layer_data = all_results[layer_idx]
        row = f"{layer_idx:<8}"
        for pos_type in ['vision', 'question', 'answer', 'gen_answer']:
            key = f"{pos_type}_relative_change_mean"
            if key in layer_data:
                val = np.mean(layer_data[key])
                row += f" {val:<11.4f}"
            else:
                row += f" {'N/A':<11}"
        print(row)

    print("\n分析结论：")
    print("- 如果 Vision 和 Text (Question/Answer) 的 relative_change 差异很大，")
    print("  说明 Adapter 对不同位置类型的调整策略不同")
    print("- 如果 GenAnswer 位置的调整幅度最大，说明 Adapter 学会了重点关注生成位置")
    print("- h_real vs h_fake diff 反映了剪枝造成的信息损失程度")


if __name__ == "__main__":
    main()

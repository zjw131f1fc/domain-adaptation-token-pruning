#!/usr/bin/env python
"""对比训练时 eval 和推理时的保留数量是否一致

测试内容：
1. 训练模式（Gumbel-Softmax）的保留数量
2. Eval 模式（Threshold）的保留数量
3. 推理模式（generate_with_hard_pruning）的保留数量

用法:
    python scripts/debug_kept_count_compare.py
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path

import torch
from collections import defaultdict

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model_and_processor(device, checkpoint_path=None):
    """加载模型"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

    model_path = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading model from {model_path}...")

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
        dropout=0.15,
    )

    model.freeze_base_model()

    # 加载 checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'pruner_state_dict' in ckpt:
            model.pruner_manager.load_state_dict(ckpt['pruner_state_dict'])
        if 'adapter_state_dict' in ckpt:
            model.adapter_manager.load_state_dict(ckpt['adapter_state_dict'])
        print("Checkpoint loaded.")

    print("Model loaded.")
    return model, processor


def load_samples(n_samples=10):
    """加载测试样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    config = load_config(override_file="configs/vision_token_pruning.yaml")
    config.dataset_settings['split'] = {'train': n_samples * 2, 'test': n_samples * 2}

    data_bundle = load_dataset(config)
    test_dataset = data_bundle['splits']['test']

    samples = [test_dataset[i] for i in range(min(n_samples, len(test_dataset)))]
    print(f"Loaded {len(samples)} samples.")
    return samples


def preprocess_sample(sample, processor, device, mode="train"):
    """预处理样本"""
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    if mode == "train":
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer.capitalize()}"
    else:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

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
        raise ValueError("Cannot find ASSISTANT: in prompt")

    question_starts = [vision_end]
    question_ends = [assistant_pos]
    answer_starts = [assistant_pos]

    # 找 answer 结束位置
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


def get_kept_counts_from_pruning_infos(pruning_infos, pruning_layers):
    """从 pruning_infos 中提取每层的保留数量"""
    counts = {}
    for layer_idx in pruning_layers:
        if layer_idx in pruning_infos:
            hard_mask = pruning_infos[layer_idx]['hard_mask']
            counts[layer_idx] = (hard_mask > 0.5).sum().item()
    return counts


def get_kept_counts_from_stats(stats, pruning_layers):
    """从 stats 中提取每层的保留数量"""
    counts = {}
    for layer_idx in pruning_layers:
        key = f'L{layer_idx}_n_kept'
        if key in stats:
            counts[layer_idx] = stats[key]
    return counts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载配置获取 checkpoint 路径
    from engine.configs.loader import load_config
    config = load_config(override_file="configs/vision_token_pruning.yaml")
    checkpoint_path = config.global_settings.get('checkpoint')

    # 加载模型
    model, processor = load_model_and_processor(device, checkpoint_path)
    pruning_layers = [4, 14, 24]

    # 加载样本
    samples = load_samples(n_samples=5)

    print("\n" + "=" * 80)
    print("对比训练模式、Eval 模式、推理模式的保留数量")
    print("=" * 80)

    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n--- Sample {idx + 1} ---")
        print(f"Question: {sample['question'][:50]}...")

        # 预处理
        prep_train = preprocess_sample(sample, processor, device, mode="train")
        prep_infer = preprocess_sample(sample, processor, device, mode="inference")

        results = {'sample_idx': idx}

        # ========== 1. 训练模式（Gumbel-Softmax）==========
        model.train()
        train_counts_list = []
        n_runs = 10  # 多次运行取平均

        for _ in range(n_runs):
            with torch.no_grad():
                output = model(
                    input_ids=prep_train['inputs']['input_ids'],
                    pixel_values=prep_train['inputs']['pixel_values'],
                    attention_mask=prep_train['inputs']['attention_mask'],
                    vision_start=prep_train['vision_start'],
                    vision_end=prep_train['vision_end'],
                    question_starts=prep_train['question_starts'],
                    question_ends=prep_train['question_ends'],
                    answer_starts=prep_train['answer_starts'],
                    answer_ends=prep_train['answer_ends'],
                    return_pruning_info=True,
                )
            counts = get_kept_counts_from_pruning_infos(output.pruning_infos, pruning_layers)
            train_counts_list.append(counts)

        # 计算平均和范围
        train_avg = {}
        train_min = {}
        train_max = {}
        for layer_idx in pruning_layers:
            values = [c[layer_idx] for c in train_counts_list]
            train_avg[layer_idx] = sum(values) / len(values)
            train_min[layer_idx] = min(values)
            train_max[layer_idx] = max(values)

        results['train'] = {'avg': train_avg, 'min': train_min, 'max': train_max}

        # ========== 2. Eval 模式（Threshold）==========
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=prep_train['inputs']['input_ids'],
                pixel_values=prep_train['inputs']['pixel_values'],
                attention_mask=prep_train['inputs']['attention_mask'],
                vision_start=prep_train['vision_start'],
                vision_end=prep_train['vision_end'],
                question_starts=prep_train['question_starts'],
                question_ends=prep_train['question_ends'],
                answer_starts=prep_train['answer_starts'],
                answer_ends=prep_train['answer_ends'],
                return_pruning_info=True,
            )
        eval_counts = get_kept_counts_from_pruning_infos(output.pruning_infos, pruning_layers)
        results['eval'] = eval_counts

        # ========== 3. 推理模式（generate_with_hard_pruning）==========
        model.eval()
        with torch.no_grad():
            _, stats = model.generate_with_hard_pruning(
                input_ids=prep_infer['inputs']['input_ids'],
                pixel_values=prep_infer['inputs']['pixel_values'],
                attention_mask=prep_infer['inputs'].get('attention_mask'),
                vision_start=prep_infer['vision_start'],
                vision_end=prep_infer['vision_end'],
                question_starts=prep_infer['question_starts'],
                question_ends=prep_infer['question_ends'],
                max_new_tokens=1,
            )
        infer_counts = get_kept_counts_from_stats(stats, pruning_layers)
        results['infer'] = infer_counts

        all_results.append(results)

        # 打印结果
        print(f"\n{'Layer':<8} {'Train (Gumbel)':<25} {'Eval (Thresh)':<15} {'Infer':<10} {'Eval-Infer':<10}")
        print("-" * 70)
        for layer_idx in pruning_layers:
            train_str = f"{train_avg[layer_idx]:.1f} [{train_min[layer_idx]}-{train_max[layer_idx]}]"
            eval_val = eval_counts.get(layer_idx, 'N/A')
            infer_val = infer_counts.get(layer_idx, 'N/A')
            if isinstance(eval_val, (int, float)) and isinstance(infer_val, (int, float)):
                diff = eval_val - infer_val
                diff_str = f"{diff:+.0f}"
            else:
                diff_str = "N/A"
            print(f"L{layer_idx:<7} {train_str:<25} {eval_val:<15} {infer_val:<10} {diff_str:<10}")

    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)

    for layer_idx in pruning_layers:
        eval_vals = [r['eval'].get(layer_idx, 0) for r in all_results]
        infer_vals = [r['infer'].get(layer_idx, 0) for r in all_results]
        diffs = [e - i for e, i in zip(eval_vals, infer_vals)]

        print(f"\nLayer {layer_idx}:")
        print(f"  Eval  - mean: {sum(eval_vals)/len(eval_vals):.1f}, range: [{min(eval_vals)}, {max(eval_vals)}]")
        print(f"  Infer - mean: {sum(infer_vals)/len(infer_vals):.1f}, range: [{min(infer_vals)}, {max(infer_vals)}]")
        print(f"  Diff  - mean: {sum(diffs)/len(diffs):.1f}, range: [{min(diffs)}, {max(diffs)}]")

    print("\n" + "=" * 80)
    print("说明：")
    print("- Train (Gumbel): 训练模式下 Gumbel-Softmax 的保留数量（多次运行）")
    print("- Eval (Thresh): Eval 模式下 Threshold 的保留数量（使用训练输入）")
    print("- Infer: 推理模式 generate_with_hard_pruning 的保留数量（使用推理输入）")
    print("- Eval-Infer: Eval 和 Infer 的差异（应该接近 0）")
    print("=" * 80)


if __name__ == "__main__":
    main()

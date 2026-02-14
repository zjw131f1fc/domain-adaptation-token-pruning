#!/usr/bin/env python
"""对比训练和推理时的行为是否一致

用法:
    python scripts/debug_query_compare.py                    # 两个模式都跑
    python scripts/debug_query_compare.py --mode query       # 对比 adapter query
    python scripts/debug_query_compare.py --mode retention   # 对比保留 token 和 logits

模式说明:
    - query: 对比训练和推理时 adapter 收到的 query 是否一致
    - retention: 对比训练 (mask) 和推理 (eval) 模式下保留的 token 和 logits 是否一致
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
from pathlib import Path

# 检查实际加载的包路径
import tokenizers
import transformers
print(f"tokenizers version: {tokenizers.__version__}, path: {tokenizers.__file__}")
print(f"transformers version: {transformers.__version__}, path: {transformers.__file__}")

import torch
import torch.nn.functional as F
from collections import defaultdict

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_model_and_processor(device, checkpoint_path=None, pruning_threshold=0.5, use_gumbel_noise=True, temperature=1.0):
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
        temperature=temperature,
        dropout=0.15,
        use_gumbel_noise=use_gumbel_noise,
        pruning_threshold=pruning_threshold,
    )

    model.freeze_base_model()

    # 加载 checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'pruner_state_dict' in checkpoint:
            model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
            print("  Loaded pruner_manager state")

        if 'adapter_state_dict' in checkpoint:
            model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
            print("  Loaded adapter_manager state")

        if 'separated_adapter_state_dict' in checkpoint:
            if hasattr(model, 'separated_adapter_manager') and model.separated_adapter_manager is not None:
                model.separated_adapter_manager.load_state_dict(checkpoint['separated_adapter_state_dict'])
                print("  Loaded separated_adapter_manager state")

        print(f"  Checkpoint step: {checkpoint.get('step', 'unknown')}")

    print("Model loaded.")

    return model, processor


def load_single_sample():
    """加载一条单 token 答案的样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    # 加载配置
    config = load_config(override_file="configs/vision_token_pruning.yaml")

    # 临时修改配置，只加载少量数据
    config.dataset_settings['split'] = {'train': 100, 'test': 100}

    data_bundle = load_dataset(config)
    train_dataset = data_bundle['splits']['train']

    # 找一条单 token 答案的样本
    # VQAv2 的答案通常是 "yes", "no", 数字, 颜色等
    single_token_answers = ['yes', 'no', '1', '2', '3', 'red', 'blue', 'green', 'white', 'black']

    for sample in train_dataset:
        answer = sample['answer'].lower().strip()
        if answer in single_token_answers:
            print(f"Found single-token sample: question='{sample['question']}', answer='{answer}'")
            return sample

    # 如果没找到，返回第一个样本
    print(f"No single-token sample found, using first sample: answer='{train_dataset[0]['answer']}'")
    return train_dataset[0]


def preprocess_sample(sample, processor, device, mode="train"):
    """预处理样本（复用 main_acp_ddp.py 的逻辑）"""
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
        'answer': answer,
        'seq_len': seq_len,
    }


class AdapterHook:
    """Hook 用于捕获 adapter 的输入"""

    def __init__(self):
        self.captured = {}  # {layer_idx: {'x': tensor, 'mask': tensor, 'query': tensor}}

    def clear(self):
        self.captured = {}

    def create_hook(self, layer_idx):
        def hook(module, args, kwargs):
            # adapter.forward(x, mask=mask, query=query)
            x = args[0] if len(args) > 0 else kwargs.get('x')
            mask = kwargs.get('mask')
            query = kwargs.get('query')

            self.captured[layer_idx] = {
                'x': x.detach().clone() if x is not None else None,
                'mask': mask.detach().clone() if mask is not None else None,
                'query': query.detach().clone() if query is not None else None,
            }

        return hook


def compare_tensors(name, t1, t2):
    """对比两个 tensor"""
    if t1 is None and t2 is None:
        print(f"  {name}: both None")
        return

    if t1 is None or t2 is None:
        print(f"  {name}: one is None! train={t1 is not None}, infer={t2 is not None}")
        return

    print(f"  {name}:")
    print(f"    shape: train={t1.shape}, infer={t2.shape}")
    print(f"    dtype: train={t1.dtype}, infer={t2.dtype}")
    print(f"    train - mean={t1.float().mean().item():.6f}, std={t1.float().std().item():.6f}")
    print(f"    infer - mean={t2.float().mean().item():.6f}, std={t2.float().std().item():.6f}")

    # 如果 shape 相同，计算差异
    if t1.shape == t2.shape:
        diff = (t1.float() - t2.float()).abs()
        print(f"    diff  - max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

        # 找到 max diff 的位置
        max_idx = diff.argmax().item()
        max_pos = []
        temp = max_idx
        for dim in reversed(t1.shape):
            max_pos.insert(0, temp % dim)
            temp //= dim
        print(f"    max_diff at position: {max_pos}")
        print(f"    train value at max_pos: {t1[tuple(max_pos)].item()}")
        print(f"    infer value at max_pos: {t2[tuple(max_pos)].item()}")

        # 计算相对误差
        rel_diff = diff / (t1.float().abs() + 1e-8)
        print(f"    rel_diff - max={rel_diff.max().item():.6f}, mean={rel_diff.mean().item():.6f}")

        # 判断是否一致
        if diff.max().item() < 1e-4:
            print(f"    [MATCH] (max_diff < 1e-4)")
        else:
            print(f"    [MISMATCH!]")
    else:
        # shape 不同，尝试对比重叠部分
        min_seq = min(t1.shape[1], t2.shape[1])
        t1_part = t1[:, :min_seq, :]
        t2_part = t2[:, :min_seq, :]
        diff = (t1_part.float() - t2_part.float()).abs()
        print(f"    (comparing first {min_seq} positions)")
        print(f"    diff  - max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

        # 找到 max diff 的位置
        max_idx = diff.argmax().item()
        max_pos = []
        temp = max_idx
        for dim in reversed(t1_part.shape):
            max_pos.insert(0, temp % dim)
            temp //= dim
        print(f"    max_diff at position: {max_pos}")
        print(f"    train value at max_pos: {t1_part[tuple(max_pos)].item()}")
        print(f"    infer value at max_pos: {t2_part[tuple(max_pos)].item()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model, processor = load_model_and_processor(device)

    # 加载样本
    sample = load_single_sample()

    # 创建 hook
    hook = AdapterHook()

    # 注册 hook 到所有 adapter
    handles = []
    for layer_idx in model.pruning_layers:
        adapter = model.adapter_manager.get_adapter(layer_idx)
        handle = adapter.register_forward_pre_hook(hook.create_hook(layer_idx), with_kwargs=True)
        handles.append(handle)

    print("\n" + "=" * 60)
    print("Running TRAIN mode (with eval for fair comparison)...")
    print("=" * 60)

    # 预处理（训练模式，包含答案）
    prep_train = preprocess_sample(sample, processor, device, mode="train")
    inputs_train = prep_train['inputs']

    print(f"\n*** TRAIN: seq={prep_train['seq_len']}, question=[{prep_train['question_starts'][0]},{prep_train['question_ends'][0]}) ***\n")

    # 使用 eval 模式进行公平对比（排除 Gumbel 噪声影响）
    # 这样训练和推理都用 threshold 逻辑生成 mask
    model.eval()
    hook.clear()

    with torch.no_grad():  # 不需要梯度，只是对比
        output_train = model(
            input_ids=inputs_train['input_ids'],
            pixel_values=inputs_train['pixel_values'],
            attention_mask=inputs_train['attention_mask'],
            vision_start=prep_train['vision_start'],
            vision_end=prep_train['vision_end'],
            question_starts=prep_train['question_starts'],
            question_ends=prep_train['question_ends'],
            answer_starts=prep_train['answer_starts'],
            answer_ends=prep_train['answer_ends'],
            return_pruning_info=True,
        )

    train_captured = {k: v.copy() for k, v in hook.captured.items()}

    print("\nCaptured in TRAIN mode:")
    for layer_idx, data in train_captured.items():
        print(f"  Layer {layer_idx}:")
        print(f"    x: {data['x'].shape if data['x'] is not None else None}")
        print(f"    mask: {data['mask'].shape if data['mask'] is not None else None}")
        print(f"    query: {data['query'].shape if data['query'] is not None else None}")
        if data['query'] is not None:
            q = data['query']
            print(f"    query stats: mean={q.float().mean().item():.6f}, std={q.float().std().item():.6f}")

    print("\n" + "=" * 60)
    print("Running INFERENCE mode...")
    print("=" * 60)

    # 预处理（推理模式，不包含答案）
    prep_infer = preprocess_sample(sample, processor, device, mode="inference")
    inputs_infer = prep_infer['inputs']

    print(f"\n*** INFER: seq={prep_infer['seq_len']}, question=[{prep_infer['question_starts'][0]},{prep_infer['question_ends'][0]}) ***\n")

    # 推理模式
    model.eval()
    hook.clear()

    with torch.no_grad():
        output_ids, kept_stats = model.generate_with_hard_pruning(
            input_ids=inputs_infer['input_ids'],
            pixel_values=inputs_infer['pixel_values'],
            attention_mask=inputs_infer.get('attention_mask'),
            vision_start=prep_infer['vision_start'],
            vision_end=prep_infer['vision_end'],
            question_starts=prep_infer['question_starts'],
            question_ends=prep_infer['question_ends'],
            max_new_tokens=1,  # 只生成一个 token
        )

    infer_captured = {k: v.copy() for k, v in hook.captured.items()}

    print("\nCaptured in INFERENCE mode:")
    for layer_idx, data in infer_captured.items():
        print(f"  Layer {layer_idx}:")
        print(f"    x: {data['x'].shape if data['x'] is not None else None}")
        print(f"    mask: {data['mask'].shape if data['mask'] is not None else None}")
        print(f"    query: {data['query'].shape if data['query'] is not None else None}")
        if data['query'] is not None:
            q = data['query']
            print(f"    query stats: mean={q.float().mean().item():.6f}, std={q.float().std().item():.6f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # 注意：训练模式的序列包含答案，推理模式不包含
    # 所以 shape 会不同，但前面的部分（prompt）应该一致
    print(f"\nNote: train seq_len includes answer, infer seq_len does not")
    print(f"  train input_ids shape: {inputs_train['input_ids'].shape}")
    print(f"  infer input_ids shape: {inputs_infer['input_ids'].shape}")

    for layer_idx in model.pruning_layers:
        print(f"\n--- Layer {layer_idx} ---")

        train_data = train_captured.get(layer_idx, {})
        infer_data = infer_captured.get(layer_idx, {})

        compare_tensors("x", train_data.get('x'), infer_data.get('x'))
        compare_tensors("mask", train_data.get('mask'), infer_data.get('mask'))
        compare_tensors("query", train_data.get('query'), infer_data.get('query'))

    # 清理 hooks
    for handle in handles:
        handle.remove()

    # 比较最终生成的 token
    print(f"\n" + "=" * 60)
    print("COMPARING GENERATED TOKENS")
    print("=" * 60)

    # 训练模式：取 answer_start 位置的 logits，argmax 得到预测 token
    answer_start = prep_train['answer_starts'][0]
    train_logits = output_train.logits[0, answer_start - 1, :]  # 预测 answer_start 位置的 token
    train_pred_token = train_logits.argmax().item()
    train_pred_text = processor.tokenizer.decode([train_pred_token])

    # 推理模式：生成的第一个新 token
    # output_ids 包含整个序列（prompt + generated）
    infer_pred_token = output_ids[0, -1].item()  # 最后一个 token 是生成的
    infer_pred_text = processor.tokenizer.decode([infer_pred_token])

    print(f"Expected answer: {sample['answer']}")
    print(f"Train mode prediction (argmax at answer_start-1):")
    print(f"  token_id={train_pred_token}, text='{train_pred_text}'")
    print(f"Inference mode prediction (generated token):")
    print(f"  token_id={infer_pred_token}, text='{infer_pred_text}'")

    if train_pred_token == infer_pred_token:
        print(f"\n[TOKEN MATCH] Train and inference produce the same token!")
    else:
        print(f"\n[TOKEN MISMATCH!] Train and inference produce different tokens!")
        # 打印 top-5 对比
        print(f"\nTrain mode top-5:")
        train_top5 = train_logits.topk(5)
        for i, (val, idx) in enumerate(zip(train_top5.values, train_top5.indices)):
            print(f"  {i+1}. token_id={idx.item()}, text='{processor.tokenizer.decode([idx.item()])}', logit={val.item():.4f}")

        # 推理模式也获取 logits 来对比
        print(f"\nNote: To compare inference logits, need to modify generate to return logits")

    # 打印完整生成结果
    generated = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nFull generated text: {generated}")
    if "ASSISTANT:" in generated:
        pred = generated.split("ASSISTANT:")[-1].strip()
        print(f"Predicted answer: {pred}")


def main_retention(args, model=None, processor=None, train_dataset=None):
    """对比训练第三阶段和推理时的保留率和保留的具体 token"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Pruning threshold: {args.pruning_threshold}")

    # 加载模型（如果没有传入）
    if model is None or processor is None:
        model, processor = load_model_and_processor(
            device,
            checkpoint_path=args.checkpoint,
            pruning_threshold=args.pruning_threshold,
            use_gumbel_noise=False,  # 第三阶段不使用 noise
            temperature=0.1,
        )

    # 加载样本（如果没有传入）
    if train_dataset is None:
        from engine.configs.loader import load_config
        from engine.datas.loader import load_dataset

        config = load_config(override_file="configs/vision_token_pruning.yaml")
        config.dataset_settings['split'] = {'train': 100, 'test': 100}
        data_bundle = load_dataset(config)
        train_dataset = data_bundle['splits']['train']

    print("\n" + "=" * 70)
    print("对比两次 eval 模式运行的 mask 决策和 logits 一致性")
    print("=" * 70)

    all_train_ratios = {layer: [] for layer in model.pruning_layers}
    all_infer_ratios = {layer: [] for layer in model.pruning_layers}
    all_mask_match = []
    all_logits_diff = []

    for sample_idx in range(min(args.num_samples, len(train_dataset))):
        sample = train_dataset[sample_idx]
        print(f"\n--- Sample {sample_idx + 1}: Q='{sample['question'][:50]}...', A='{sample['answer']}' ---")

        # 预处理（推理模式，不包含答案，两边都用这个）
        prep = preprocess_sample(sample, processor, device, mode="inference")
        inputs = prep['inputs']

        # ========== 训练模式 (mask, 不做物理删除) ==========
        # 注意：用 eval 模式消除 dropout 影响，只测试 mask 决策逻辑
        model.eval()
        model.pruner_manager.set_use_gumbel_noise(False)  # 关闭 noise
        model.pruner_manager.set_temperature(0.1)

        with torch.no_grad():
            output_train = model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs['attention_mask'],
                vision_start=prep['vision_start'],
                vision_end=prep['vision_end'],
                question_starts=prep['question_starts'],
                question_ends=prep['question_ends'],
                answer_starts=prep['answer_starts'] if 'answer_starts' in prep else [prep['question_ends'][0]],
                answer_ends=prep['answer_ends'] if 'answer_ends' in prep else [prep['question_ends'][0] + 1],
                return_pruning_info=True,
            )

        # 收集训练模式的 mask
        train_masks = {}
        train_ratios = {}
        for layer_idx in model.pruning_layers:
            if layer_idx in output_train.pruning_infos:
                hard_mask = output_train.pruning_infos[layer_idx]['hard_mask']
                train_masks[layer_idx] = hard_mask.clone()
                ratio = hard_mask.float().mean().item()
                train_ratios[layer_idx] = ratio
                all_train_ratios[layer_idx].append(ratio)

        # 训练模式的 logits（最后一个位置，用于预测下一个 token）
        train_logits = output_train.logits[0, -1, :].clone()

        # ========== 第二次运行 (验证确定性) ==========
        # 两边都用 eval 模式，验证 mask 决策是否确定性
        model.eval()

        with torch.no_grad():
            output_infer = model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask'),
                vision_start=prep['vision_start'],
                vision_end=prep['vision_end'],
                question_starts=prep['question_starts'],
                question_ends=prep['question_ends'],
                answer_starts=prep['answer_starts'] if 'answer_starts' in prep else [prep['question_ends'][0]],
                answer_ends=prep['answer_ends'] if 'answer_ends' in prep else [prep['question_ends'][0] + 1],
                return_pruning_info=True,
            )

        # 收集推理模式的 mask
        infer_masks = {}
        infer_ratios = {}
        for layer_idx in model.pruning_layers:
            if layer_idx in output_infer.pruning_infos:
                hard_mask = output_infer.pruning_infos[layer_idx]['hard_mask']
                infer_masks[layer_idx] = hard_mask.clone()
                ratio = hard_mask.float().mean().item()
                infer_ratios[layer_idx] = ratio
                all_infer_ratios[layer_idx].append(ratio)

        # 推理模式的 logits
        infer_logits = output_infer.logits[0, -1, :].clone()

        # ========== 对比 ==========
        print(f"\n  [保留率对比]")
        print(f"  {'Layer':<8} {'Train':>12} {'Infer':>12} {'Diff':>12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        for layer_idx in model.pruning_layers:
            t_ratio = train_ratios.get(layer_idx, 0)
            i_ratio = infer_ratios.get(layer_idx, 0)
            diff = abs(t_ratio - i_ratio)
            match = "OK" if diff < 0.001 else "!!"
            print(f"  L{layer_idx:<6} {t_ratio:>11.2%} {i_ratio:>11.2%} {diff:>11.6f} {match}")

        # 对比具体保留的 token
        print(f"\n  [保留 token 对比]")
        sample_mask_match = True
        for layer_idx in model.pruning_layers:
            t_mask = train_masks.get(layer_idx)
            i_mask = infer_masks.get(layer_idx)
            if t_mask is not None and i_mask is not None:
                # 对比 mask 是否完全一致
                mask_equal = torch.equal(t_mask, i_mask)
                if mask_equal:
                    print(f"  L{layer_idx}: mask 完全一致 ✓")
                else:
                    sample_mask_match = False
                    diff_count = (t_mask != i_mask).sum().item()
                    print(f"  L{layer_idx}: mask 不一致! 差异数量: {diff_count}")
                    # 打印前几个不一致的位置
                    diff_indices = (t_mask[0] != i_mask[0]).nonzero(as_tuple=True)[0][:5]
                    for idx in diff_indices:
                        print(f"    位置 {idx.item()}: train={t_mask[0, idx].item()}, infer={i_mask[0, idx].item()}")
        all_mask_match.append(sample_mask_match)

        # 对比 logits
        print(f"\n  [Logits 对比]")
        logits_diff = (train_logits.float() - infer_logits.float()).abs()
        max_diff = logits_diff.max().item()
        mean_diff = logits_diff.mean().item()
        all_logits_diff.append(max_diff)

        print(f"  max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")

        # 对比 top-5 预测
        train_top5 = train_logits.topk(5)
        infer_top5 = infer_logits.topk(5)

        print(f"  Train top-5: {[processor.tokenizer.decode([idx.item()]) for idx in train_top5.indices]}")
        print(f"  Infer top-5: {[processor.tokenizer.decode([idx.item()]) for idx in infer_top5.indices]}")

        if train_top5.indices[0] == infer_top5.indices[0]:
            print(f"  Top-1 预测一致 ✓")
        else:
            print(f"  Top-1 预测不一致!")

        if max_diff < 1e-3:
            print(f"  Logits 数值一致 (max_diff < 1e-3) ✓")
        elif max_diff < 1e-1:
            print(f"  Logits 数值接近 (max_diff < 0.1)")
        else:
            print(f"  Logits 数值差异较大!")

    # ========== 汇总统计 ==========
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    print(f"\n[保留率]")
    print(f"{'Layer':<8} {'Train Mean':>12} {'Infer Mean':>12} {'Diff':>12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for layer_idx in model.pruning_layers:
        t_mean = sum(all_train_ratios[layer_idx]) / len(all_train_ratios[layer_idx]) if all_train_ratios[layer_idx] else 0
        i_mean = sum(all_infer_ratios[layer_idx]) / len(all_infer_ratios[layer_idx]) if all_infer_ratios[layer_idx] else 0
        diff = abs(t_mean - i_mean)
        match = "OK" if diff < 0.001 else "!!"
        print(f"L{layer_idx:<6} {t_mean:>11.2%} {i_mean:>11.2%} {diff:>11.6f} {match}")

    print(f"\n[Mask 一致性]")
    match_count = sum(all_mask_match)
    print(f"  {match_count}/{len(all_mask_match)} 样本的 mask 完全一致")

    print(f"\n[Logits 差异]")
    print(f"  max_diff 平均: {sum(all_logits_diff)/len(all_logits_diff):.6f}")
    print(f"  max_diff 最大: {max(all_logits_diff):.6f}")

    if all(all_mask_match) and max(all_logits_diff) < 1e-3:
        print(f"\n✓ 两次运行的 mask 决策和 logits 完全一致!")
    else:
        print(f"\n✗ 两次运行存在差异，可能有非确定性因素!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比训练和推理")
    parser.add_argument("--mode", type=str, default="both", choices=["query", "retention", "both"],
                        help="模式: query=对比adapter query, retention=对比保留率, both=两个都跑")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/tasks/20260215-0233_vqa-vqav2_llava157b_dcc0/checkpoints/checkpoint_final.pt",
                        help="Checkpoint 路径")
    parser.add_argument("--pruning_threshold", type=float, default=0.5,
                        help="Sigmoid 阈值 (默认 0.5)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="测试样本数量")
    args = parser.parse_args()

    # 共享数据加载
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset

    print("加载数据集...")
    config = load_config(override_file="configs/vision_token_pruning.yaml")
    config.dataset_settings['split'] = {'train': 100, 'test': 100}
    data_bundle = load_dataset(config)
    train_dataset = data_bundle['splits']['train']
    print(f"数据集加载完成，共 {len(train_dataset)} 条样本")

    if args.mode in ["query", "both"]:
        print("\n" + "=" * 70)
        print("模式 1: 对比 Adapter Query")
        print("=" * 70)
        main()

    if args.mode in ["retention", "both"]:
        print("\n" + "=" * 70)
        print("模式 2: 验证 eval 模式下 mask 决策的确定性")
        print("=" * 70)
        main_retention(args, train_dataset=train_dataset)

#!/usr/bin/env python
"""对比训练和推理时 adapter 收到的 query 是否一致

用法:
    python scripts/debug_query_compare.py
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
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


def load_model_and_processor(device):
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


if __name__ == "__main__":
    main()

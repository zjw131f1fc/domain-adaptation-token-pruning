#!/usr/bin/env python
"""收集并导出 h_real/h_fake/h_corrected（以及可选的 attn 版本）

用法:
    python scripts/visualize_distribution_shift_v2.py --checkpoint <path>

说明:
- 这个 v2 脚本只负责“捕获 gap”的原始特征，不再输出旧版的距离指标、投影可视化等。
- 输出为 npz 文件，便于后续在 notebook/脚本里自由做统计与可视化。
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import types

import torch
import numpy as np

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
    parser.add_argument("--include_attn", action="store_true", help="Also export h_real_attn/h_fake_attn if available")
    parser.add_argument("--force_no_adapter", action="store_true", help="Disable adapter regardless of config/checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="export_h",
        choices=["export_h", "gap_curve", "gap_impact"],
        help=(
            "export_h: dump h vectors per pruning layer; "
            "gap_curve: compute layerwise pruned-vs-unpruned gap curves; "
            "gap_impact: correlate per-layer gap with answer NLL / confidence (teacher-forcing)."
        ),
    )
    parser.add_argument("--proj_dim", type=int, default=64, help="Projection dim used for gap_curve (smaller=cheaper)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projection (gap_curve)")
    parser.add_argument(
        "--single_prune_layer",
        type=int,
        default=-1,
        help="Only apply pruning at this layer index (others keep-all). For --mode gap_curve/gap_impact.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Tokenizer max_length (gap_impact uses engine.data_utils.preprocess_batch).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="In gap_impact: print top-k worst/best samples by delta_nll.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="",
        help="Optional: write a human-readable gap_impact report to this path.",
    )
    return parser.parse_args()


def load_model_and_processor(checkpoint_path, config_path, device, force_no_adapter: bool = False):
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
    use_adapter = False if force_no_adapter else method_cfg.get('use_adapter', True)
    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=method_cfg.get('pruning_layers', [4, 14, 24]),
        pruner_d_internal=method_cfg.get('pruner_d_internal', 512),
        pruner_n_heads=method_cfg.get('pruner_n_heads', 4),
        pruner_n_queries=method_cfg.get('pruner_n_queries', 32),
        pruner_query_dropout=0.0,  # 分析时关闭 dropout
        use_adapter=use_adapter,
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

    data_bundle = load_dataset(config)
    test_dataset = data_bundle['splits']['train']

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

def cohens_d(x, y):
    """计算 Cohen's d 效应量"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)


def _flatten_h(h: torch.Tensor) -> np.ndarray:
    # h: (heads, n_ans, head_dim) -> (n_ans, heads*head_dim)
    return h.permute(1, 0, 2).reshape(h.shape[1], -1).float().cpu().numpy()


def collect_h_vectors(model, processor, samples, device, include_attn: bool = False):
    """收集所有样本的 h_real, h_fake, h_corrected（以及可选的 attn 版本）"""
    h_real_all = defaultdict(list)
    h_fake_all = defaultdict(list)
    h_corrected_all = defaultdict(list)
    h_real_attn_all = defaultdict(list)
    h_fake_attn_all = defaultdict(list)

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

        pruning_infos = getattr(output, 'pruning_infos', None) or getattr(output, 'pruning_info', None)
        if pruning_infos:
            for layer_idx, info in pruning_infos.items():
                if 'h_real' in info and 'h_fake' in info:
                    for h_real, h_fake in zip(info['h_real'], info['h_fake']):
                        h_real_all[layer_idx].append(_flatten_h(h_real))
                        h_fake_all[layer_idx].append(_flatten_h(h_fake))

                if 'h_corrected' in info:
                    for h_corr in info['h_corrected']:
                        h_corrected_all[layer_idx].append(_flatten_h(h_corr))

                if include_attn and ('h_real_attn' in info and 'h_fake_attn' in info):
                    for h_real_attn, h_fake_attn in zip(info['h_real_attn'], info['h_fake_attn']):
                        h_real_attn_all[layer_idx].append(_flatten_h(h_real_attn))
                        h_fake_attn_all[layer_idx].append(_flatten_h(h_fake_attn))

    # 合并所有样本
    result = {}
    for layer_idx in h_real_all.keys():
        result[layer_idx] = {
            'h_real': np.concatenate(h_real_all[layer_idx], axis=0),
            'h_fake': np.concatenate(h_fake_all[layer_idx], axis=0),
            'h_corrected': np.concatenate(h_corrected_all[layer_idx], axis=0) if h_corrected_all[layer_idx] else None,
        }
        if include_attn and h_real_attn_all[layer_idx]:
            result[layer_idx]['h_real_attn'] = np.concatenate(h_real_attn_all[layer_idx], axis=0)
            result[layer_idx]['h_fake_attn'] = np.concatenate(h_fake_attn_all[layer_idx], axis=0)

    return result


def export_h_vectors(h_data: dict, output_dir: str, prefix: str = "h_vectors"):
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx in sorted(h_data.keys()):
        payload = h_data[layer_idx]
        out_path = os.path.join(output_dir, f"{prefix}_layer{layer_idx}.npz")
        np.savez_compressed(out_path, **payload)
        keys = ", ".join(sorted(payload.keys()))
        n_real = payload["h_real"].shape[0]
        n_fake = payload["h_fake"].shape[0]
        n_corr = payload["h_corrected"].shape[0] if payload.get("h_corrected") is not None else 0
        print(f"  Saved layer {layer_idx} -> {out_path} (h_real={n_real}, h_fake={n_fake}, h_corrected={n_corr}; keys={keys})")


def _get_llm_layers(model):
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise ValueError("Expected PrunableLlavaForConditionalGeneration-like model with .base_model")
    llm = base_model.model.language_model
    return llm.layers


@contextmanager
def _force_pruners_keep_all(model, allow_layer_idx: int | None = None):
    """Monkeypatch pruners to always return all-ones mask.

    If allow_layer_idx is not None, that layer's pruner is left untouched.
    """
    base_model = model.module if hasattr(model, "module") else model
    pruner_manager = getattr(base_model, "pruner_manager", None)
    if pruner_manager is None:
        yield
        return

    saved = {}

    def _keep_all_forward_full(self, vision_hidden, q2v_attn_avg, cumulative_vision_mask=None, **kwargs):
        batch_size = vision_hidden.shape[0]
        n_vision = vision_hidden.shape[1]
        device = vision_hidden.device
        dtype = vision_hidden.dtype
        current_mask = torch.ones(batch_size, n_vision, device=device, dtype=dtype)
        pruner_info = {'keep_logits': None}
        return current_mask, pruner_info

    for k, pruner in pruner_manager.pruners.items():
        if allow_layer_idx is not None and int(k) == int(allow_layer_idx):
            continue
        saved[k] = pruner.forward_full
        pruner.forward_full = types.MethodType(_keep_all_forward_full, pruner)

    try:
        yield
    finally:
        for k, pruner in pruner_manager.pruners.items():
            if k in saved:
                pruner.forward_full = saved[k]


def _token_scopes(prep):
    vs, ve = prep['vision_start'], prep['vision_end']
    qs, qe = prep['question_starts'][0], prep['question_ends'][0]
    ans_s, ans_e = prep['answer_starts'][0], prep['answer_ends'][0]
    gen_s = ans_s - 1
    gen_e = ans_e - 1
    return {
        "vision": (vs, ve),
        "question": (qs, qe),
        "answer": (ans_s, ans_e),
        "gen_answer": (gen_s, gen_e),
    }


def _capture_layer_projections(model, prep, proj_mat):
    """Run one forward and capture per-layer projected hidden states for multiple token scopes.

    Returns:
        (layer_proj, pruning_infos, logits)
            layer_proj: {layer_idx: {scope: np.ndarray(n_tokens, proj_dim)}}
            pruning_infos: model output pruning_infos (or None)
            logits: output logits (or None)
    """
    layers = _get_llm_layers(model)
    scopes = _token_scopes(prep)
    layer_proj = {}

    def _make_hook(layer_idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            hidden = hidden[0]  # batch=1 -> (seq, hidden)
            per_scope = {}
            for scope_name, (s, e) in scopes.items():
                if s is None or e is None or e <= s:
                    continue
                slice_h = hidden[s:e, :].float()
                per_scope[scope_name] = (slice_h @ proj_mat).cpu().numpy()
            layer_proj[layer_idx] = per_scope
        return hook

    handles = []
    for idx in range(len(layers)):
        handles.append(layers[idx].register_forward_hook(_make_hook(idx)))

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
            # Important: in this codebase pruning/cumulative masks depend on pruning_infos being returned.
            return_pruning_info=True,
        )

    for h in handles:
        h.remove()

    for idx in range(len(layers)):
        layer_proj.setdefault(idx, {})
    pruning_infos = getattr(output, 'pruning_infos', None) or getattr(output, 'pruning_info', None)
    logits = getattr(output, "logits", None)
    return layer_proj, pruning_infos, logits


def _preprocess_single_train(sample, processor, device, max_length: int = 1024):
    """用项目内 preprocess_batch 保证 prompt/token span 与训练一致。"""
    from engine.data_utils import preprocess_batch

    preprocessed = preprocess_batch(
        batch=[sample],
        processor=processor,
        device=device,
        max_length=max_length,
        mode="train",
    )
    # 补齐 gap_curve 需要的字段名风格
    return {
        "inputs": preprocessed["inputs"],
        "vision_start": preprocessed["vision_start"],
        "vision_end": preprocessed["vision_end"],
        "question_starts": [preprocessed["question_starts"][0]],
        "question_ends": [preprocessed["question_ends"][0]],
        "answer_starts": [preprocessed["answer_starts"][0]],
        "answer_ends": [preprocessed["answer_ends"][0]],
    }


def _answer_region_nll_and_entropy(logits, input_ids, answer_start: int, answer_end: int):
    """Teacher-forcing 下，计算 answer 区间的平均 NLL 与 entropy。

    - logits: (1, seq, vocab) 或 (seq, vocab)
    - input_ids: (1, seq)
    - answer_start/end: token index in the original sequence (same as preprocess_batch)
    """
    if logits is None:
        return float("nan"), float("nan")
    if logits.dim() == 3:
        logits = logits[0]
    ids = input_ids[0]

    # next-token prediction: logits[t] predicts ids[t+1]
    shift_logits = logits[:-1, :]
    shift_labels = ids[1:]

    # label positions corresponding to tokens in [answer_start, answer_end)
    # token at position p is predicted by logits[p-1]
    s = max(int(answer_start) - 1, 0)
    e = max(int(answer_end) - 1, s)
    if e <= s:
        return float("nan"), float("nan")

    sel_logits = shift_logits[s:e, :].float()
    sel_labels = shift_labels[s:e]

    logp = torch.log_softmax(sel_logits, dim=-1)
    nll = -logp.gather(dim=-1, index=sel_labels.unsqueeze(-1)).squeeze(-1)  # (n_tok,)
    nll_mean = float(nll.mean().item())

    p = torch.softmax(sel_logits, dim=-1)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    entropy_mean = float(entropy.mean().item())

    return nll_mean, entropy_mean


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float((x @ y) / denom)


def compute_and_export_gap_curves(
    model,
    processor,
    samples,
    device,
    output_dir,
    proj_dim=64,
    seed=42,
    single_prune_layer: int | None = None,
):
    """Compute layerwise gap curves: pruned vs unpruned baseline."""
    layers = _get_llm_layers(model)
    n_layers = len(layers)
    rng = np.random.default_rng(seed)

    hidden_size = getattr(model, "hidden_size", None)
    if not isinstance(hidden_size, int):
        hidden_size = model.base_model.config.text_config.hidden_size

    proj = rng.normal(size=(hidden_size, proj_dim)).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
    proj_mat = torch.from_numpy(proj).to(device=device, dtype=torch.float32)

    scopes = ["vision", "question", "answer", "gen_answer"]
    gap_lists = {scope: {i: [] for i in range(n_layers)} for scope in scopes}
    kept_ratio_lists = defaultdict(list)  # {layer_idx: [ratio, ...]}

    print("Computing gap curves (pruned vs unpruned)...")
    if single_prune_layer is not None:
        print(f"Single prune layer: {single_prune_layer}")
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        prep = preprocess_sample(sample, processor, device)
        if prep is None:
            continue

        with _force_pruners_keep_all(model):
            base_proj, _, _ = _capture_layer_projections(model, prep, proj_mat)
        if single_prune_layer is not None:
            with _force_pruners_keep_all(model, allow_layer_idx=single_prune_layer):
                pruned_proj, pruning_infos, _ = _capture_layer_projections(model, prep, proj_mat)
        else:
            pruned_proj, pruning_infos, _ = _capture_layer_projections(model, prep, proj_mat)

        if pruning_infos:
            for layer_idx, info in pruning_infos.items():
                cm = info.get("cumulative_mask")
                if cm is not None:
                    kept_ratio_lists[int(layer_idx)].append(float(cm.float().mean().item()))

        for layer_idx in range(n_layers):
            for scope in scopes:
                a = base_proj.get(layer_idx, {}).get(scope)
                b = pruned_proj.get(layer_idx, {}).get(scope)
                if a is None or b is None:
                    continue
                if a.shape != b.shape:
                    continue
                diff = b - a
                val = np.linalg.norm(diff, axis=1).mean()
                gap_lists[scope][layer_idx].append(float(val))

    os.makedirs(output_dir, exist_ok=True)
    out_name = "gap_curves.npz" if single_prune_layer is None else f"gap_curves_single_L{single_prune_layer}.npz"
    out_path = os.path.join(output_dir, out_name)
    payload = {}
    for scope, per_layer in gap_lists.items():
        means = np.array([np.mean(per_layer[i]) if per_layer[i] else np.nan for i in range(n_layers)], dtype=np.float32)
        stds = np.array([np.std(per_layer[i]) if per_layer[i] else np.nan for i in range(n_layers)], dtype=np.float32)
        counts = np.array([len(per_layer[i]) for i in range(n_layers)], dtype=np.int32)
        payload[f"{scope}_mean"] = means
        payload[f"{scope}_std"] = stds
        payload[f"{scope}_count"] = counts

    payload["proj_dim"] = np.array([proj_dim], dtype=np.int32)
    payload["seed"] = np.array([seed], dtype=np.int32)
    np.savez_compressed(out_path, **payload)

    print(f"Saved gap curves -> {out_path}")
    print("Quick readout (mean gap, projected space):")
    for scope in ["gen_answer", "answer", "question", "vision"]:
        arr = payload[f"{scope}_mean"]
        peak = int(np.nanargmax(arr)) if np.any(np.isfinite(arr)) else -1
        peak_val = float(arr[peak]) if peak >= 0 else float("nan")
        print(f"  {scope}: peak layer={peak}, peak={peak_val:.6f}")
    if kept_ratio_lists:
        kept_str = ", ".join(
            f"L{layer_idx}={np.mean(vals):.2%}" for layer_idx, vals in sorted(kept_ratio_lists.items())
        )
        print(f"Kept ratio (avg, from pruning_infos): {kept_str}")


def compute_and_export_gap_impact(
    model,
    processor,
    samples,
    device,
    output_dir,
    proj_dim=64,
    seed=42,
    max_length: int = 1024,
    single_prune_layer: int | None = None,
    topk: int = 10,
    report_path: str = "",
):
    """在 gap_curve 的基础上，加上 teacher-forcing 下的行为指标（NLL/entropy）。

    核心目标：回答“gap 是否真的影响模型行为/置信度？哪一层的 gap 最相关？”
    """
    layers = _get_llm_layers(model)
    n_layers = len(layers)
    rng = np.random.default_rng(seed)

    hidden_size = getattr(model, "hidden_size", None)
    if not isinstance(hidden_size, int):
        hidden_size = model.base_model.config.text_config.hidden_size

    proj = rng.normal(size=(hidden_size, proj_dim)).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8)
    proj_mat = torch.from_numpy(proj).to(device=device, dtype=torch.float32)

    scopes = ["vision", "question", "answer", "gen_answer"]
    gap_by_scope = {scope: [] for scope in scopes}  # list of (n_layers,) per sample
    nll_base_list, nll_pruned_list = [], []
    ent_base_list, ent_pruned_list = [], []
    kept_ratio_lists = defaultdict(list)
    meta_list = []

    print("Computing gap impact (per-sample gap vs teacher-forcing NLL)...")
    if single_prune_layer is not None:
        print(f"Single prune layer: {single_prune_layer}")

    n_used = 0
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        prep = _preprocess_single_train(sample, processor, device, max_length=max_length)
        inputs = prep["inputs"]

        with _force_pruners_keep_all(model):
            base_proj, _, base_logits = _capture_layer_projections(model, prep, proj_mat)

        if single_prune_layer is not None:
            with _force_pruners_keep_all(model, allow_layer_idx=single_prune_layer):
                pruned_proj, pruning_infos, pruned_logits = _capture_layer_projections(model, prep, proj_mat)
        else:
            pruned_proj, pruning_infos, pruned_logits = _capture_layer_projections(model, prep, proj_mat)

        # 行为指标：answer 区间的 teacher-forcing NLL / entropy
        ans_s = prep["answer_starts"][0]
        ans_e = prep["answer_ends"][0]
        nll_base, ent_base = _answer_region_nll_and_entropy(base_logits, inputs["input_ids"], ans_s, ans_e)
        nll_pruned, ent_pruned = _answer_region_nll_and_entropy(pruned_logits, inputs["input_ids"], ans_s, ans_e)

        # gap：每层、每 scope 一个平均 token L2（投影空间）
        per_scope_gaps = {scope: np.full((n_layers,), np.nan, dtype=np.float32) for scope in scopes}
        for layer_idx in range(n_layers):
            for scope in scopes:
                a = base_proj.get(layer_idx, {}).get(scope)
                b = pruned_proj.get(layer_idx, {}).get(scope)
                if a is None or b is None:
                    continue
                if a.shape != b.shape:
                    continue
                diff = b - a
                per_scope_gaps[scope][layer_idx] = float(np.linalg.norm(diff, axis=1).mean())

        for scope in scopes:
            gap_by_scope[scope].append(per_scope_gaps[scope])

        if pruning_infos:
            for layer_idx, info in pruning_infos.items():
                cm = info.get("cumulative_mask")
                if cm is not None:
                    kept_ratio_lists[int(layer_idx)].append(float(cm.float().mean().item()))

        nll_base_list.append(nll_base)
        nll_pruned_list.append(nll_pruned)
        ent_base_list.append(ent_base)
        ent_pruned_list.append(ent_pruned)
        qid = sample.get("question_id", -1)
        if isinstance(qid, (int, np.integer)):
            qid_int = int(qid)
        else:
            qid_s = str(qid)
            qid_int = int(qid_s) if qid_s.isdigit() else -1
        meta_list.append(
            {
                "sample_idx": int(i),
                "question_id": qid_int,
                "question": str(sample.get("question", "")),
                "answer": str(sample.get("answer", "")),
            }
        )
        n_used += 1

    if n_used == 0:
        raise ValueError("No usable samples for gap_impact.")

    nll_base_arr = np.array(nll_base_list, dtype=np.float32)
    nll_pruned_arr = np.array(nll_pruned_list, dtype=np.float32)
    ent_base_arr = np.array(ent_base_list, dtype=np.float32)
    ent_pruned_arr = np.array(ent_pruned_list, dtype=np.float32)

    payload = {
        "n_used": np.array([n_used], dtype=np.int32),
        "proj_dim": np.array([proj_dim], dtype=np.int32),
        "seed": np.array([seed], dtype=np.int32),
        "sample_idx": np.array([m["sample_idx"] for m in meta_list], dtype=np.int32),
        "question_id": np.array([m["question_id"] for m in meta_list], dtype=np.int64),
        "nll_base": nll_base_arr,
        "nll_pruned": nll_pruned_arr,
        "delta_nll": (nll_pruned_arr - nll_base_arr),
        "entropy_base": ent_base_arr,
        "entropy_pruned": ent_pruned_arr,
        "delta_entropy": (ent_pruned_arr - ent_base_arr),
    }

    for scope in scopes:
        payload[f"{scope}_gap_per_sample"] = np.stack(gap_by_scope[scope], axis=0)  # (n, n_layers)
        payload[f"{scope}_gap_mean"] = np.nanmean(payload[f"{scope}_gap_per_sample"], axis=0).astype(np.float32)
        payload[f"{scope}_gap_std"] = np.nanstd(payload[f"{scope}_gap_per_sample"], axis=0).astype(np.float32)

    if kept_ratio_lists:
        for layer_idx, vals in sorted(kept_ratio_lists.items()):
            payload[f"kept_ratio_L{layer_idx}"] = np.array(vals, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)
    out_name = "gap_impact.npz" if single_prune_layer is None else f"gap_impact_single_L{single_prune_layer}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(out_path, **payload)

    # quick readout
    delta_nll = payload["delta_nll"]
    delta_entropy = payload["delta_entropy"]
    print(f"Saved gap impact -> {out_path}")
    print("Quick readout (teacher-forcing, answer region):")
    print(f"  NLL base : mean={float(np.nanmean(nll_base_arr)):.4f}")
    print(f"  NLL pruned: mean={float(np.nanmean(nll_pruned_arr)):.4f}")
    print(f"  delta NLL (pruned-base): mean={float(np.nanmean(delta_nll)):.4f} ; worse@pruned={(delta_nll > 0).mean():.2%}")
    print(f"  delta entropy: mean={float(np.nanmean(delta_entropy)):.4f}")

    # quantiles help diagnose whether mean is dominated by outliers
    def _quantiles(arr: np.ndarray, qs=None):
        if qs is None:
            qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        arr = np.asarray(arr, dtype=np.float64)
        vals = np.nanpercentile(arr, qs)
        return qs, vals

    def _print_quantiles(name: str, arr: np.ndarray):
        qs, vals = _quantiles(arr)
        parts = ", ".join([f"p{q}={v:.4f}" for q, v in zip(qs, vals)])
        print(f"  {name}: {parts}")
        return parts

    print("Quantiles (answer region):")
    q_delta_nll = _print_quantiles("delta_nll", delta_nll)
    q_delta_entropy = _print_quantiles("delta_entropy", delta_entropy)

    # correlation: per-layer gap vs delta_nll / delta_entropy for each scope
    print("Correlation (Pearson r) between per-layer gap and delta metrics:")
    corr_tables = {}
    for scope in scopes:
        gaps = payload[f"{scope}_gap_per_sample"]  # (n, n_layers)
        corrs_nll = np.array([_pearsonr(gaps[:, l], delta_nll) for l in range(n_layers)], dtype=np.float32)
        corrs_ent = np.array([_pearsonr(gaps[:, l], delta_entropy) for l in range(n_layers)], dtype=np.float32)
        corr_tables[scope] = {"delta_nll": corrs_nll, "delta_entropy": corrs_ent}

        best_nll = int(np.nanargmax(np.abs(corrs_nll))) if np.any(np.isfinite(corrs_nll)) else -1
        best_ent = int(np.nanargmax(np.abs(corrs_ent))) if np.any(np.isfinite(corrs_ent)) else -1
        if best_nll >= 0:
            print(f"  {scope:10s} vs delta_nll    : best layer={best_nll}, r={float(corrs_nll[best_nll]):.4f}")
        if best_ent >= 0:
            print(f"  {scope:10s} vs delta_entropy: best layer={best_ent}, r={float(corrs_ent[best_ent]):.4f}")

    # top layers for gen_answer vs delta_nll (actionable for choosing where to repair)
    gen_corr = corr_tables["gen_answer"]["delta_nll"]
    finite = np.isfinite(gen_corr)
    if finite.any():
        layers_f = np.arange(n_layers)[finite]
        order = np.argsort(-np.abs(gen_corr[finite]))
        topn = min(5, len(order))
        print("Top layers by |r| (gen_answer gap vs delta_nll):")
        for j in range(topn):
            l = int(layers_f[order[j]])
            print(f"  #{j+1}: layer={l}, r={float(gen_corr[l]):.4f}")

    # top-k worst/best samples by delta_nll
    k = max(0, int(topk))
    worst_idx = None
    best_idx = None
    if k > 0:
        worst_idx = np.argsort(-delta_nll)[:k]
        best_idx = np.argsort(delta_nll)[:k]
        print(f"Top-{k} worst samples by delta_nll (pruned-base):")
        for rank, idx in enumerate(worst_idx, start=1):
            m = meta_list[int(idx)]
            print(
                f"  #{rank}: idx={m['sample_idx']} qid={m['question_id']} "
                f"delta_nll={float(delta_nll[idx]):.4f} nll_base={float(nll_base_arr[idx]):.4f} nll_pruned={float(nll_pruned_arr[idx]):.4f} "
                f"delta_ent={float(delta_entropy[idx]):.4f}"
            )
        print(f"Top-{k} best samples by delta_nll (most improved under pruning):")
        for rank, idx in enumerate(best_idx, start=1):
            m = meta_list[int(idx)]
            print(
                f"  #{rank}: idx={m['sample_idx']} qid={m['question_id']} "
                f"delta_nll={float(delta_nll[idx]):.4f} nll_base={float(nll_base_arr[idx]):.4f} nll_pruned={float(nll_pruned_arr[idx]):.4f} "
                f"delta_ent={float(delta_entropy[idx]):.4f}"
            )
    if kept_ratio_lists:
        kept_str = ", ".join(
            f"L{layer_idx}={np.mean(vals):.2%}" for layer_idx, vals in sorted(kept_ratio_lists.items())
        )
        print(f"Kept ratio (avg, from pruning_infos): {kept_str}")

    # optional human-readable report file
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("gap_impact report\n")
            f.write(f"n_used={n_used} proj_dim={proj_dim} seed={seed}\n")
            if single_prune_layer is not None:
                f.write(f"single_prune_layer={single_prune_layer}\n")
            f.write(f"checkpoint_mode=pruned_vs_keepall (teacher-forcing)\n")
            f.write("\nQuick readout (answer region):\n")
            f.write(f"nll_base_mean={float(np.nanmean(nll_base_arr)):.8f}\n")
            f.write(f"nll_pruned_mean={float(np.nanmean(nll_pruned_arr)):.8f}\n")
            f.write(f"delta_nll_mean={float(np.nanmean(delta_nll)):.8f}\n")
            f.write(f"worse_at_pruned={(delta_nll > 0).mean():.8f}\n")
            f.write(f"delta_entropy_mean={float(np.nanmean(delta_entropy)):.8f}\n")
            f.write("\nQuantiles:\n")
            f.write(f"delta_nll: {q_delta_nll}\n")
            f.write(f"delta_entropy: {q_delta_entropy}\n")
            f.write("\nCorrelation (best layer, Pearson r):\n")
            for scope in scopes:
                cn = corr_tables[scope]["delta_nll"]
                ce = corr_tables[scope]["delta_entropy"]
                bn = int(np.nanargmax(np.abs(cn))) if np.any(np.isfinite(cn)) else -1
                be = int(np.nanargmax(np.abs(ce))) if np.any(np.isfinite(ce)) else -1
                f.write(f"{scope:10s} vs delta_nll: layer={bn} r={float(cn[bn]) if bn>=0 else float('nan'):.8f}\n")
                f.write(f"{scope:10s} vs delta_entropy: layer={be} r={float(ce[be]) if be>=0 else float('nan'):.8f}\n")
            if kept_ratio_lists:
                f.write(f"\nKept ratio (avg): {kept_str}\n")

            if k > 0 and worst_idx is not None and best_idx is not None:
                f.write(f"\nTop-{k} worst samples by delta_nll:\n")
                for rank, idx in enumerate(worst_idx, start=1):
                    m = meta_list[int(idx)]
                    f.write(
                        f"#{rank} sample_idx={m['sample_idx']} question_id={m['question_id']} "
                        f"delta_nll={float(delta_nll[idx]):.8f} nll_base={float(nll_base_arr[idx]):.8f} nll_pruned={float(nll_pruned_arr[idx]):.8f} "
                        f"delta_entropy={float(delta_entropy[idx]):.8f}\n"
                    )
                    q = m["question"].replace("\n", "\\n")
                    a = m["answer"].replace("\n", "\\n")
                    f.write(f"  Q: {q}\n")
                    f.write(f"  A: {a}\n")
                f.write(f"\nTop-{k} best samples by delta_nll:\n")
                for rank, idx in enumerate(best_idx, start=1):
                    m = meta_list[int(idx)]
                    f.write(
                        f"#{rank} sample_idx={m['sample_idx']} question_id={m['question_id']} "
                        f"delta_nll={float(delta_nll[idx]):.8f} nll_base={float(nll_base_arr[idx]):.8f} nll_pruned={float(nll_pruned_arr[idx]):.8f} "
                        f"delta_entropy={float(delta_entropy[idx]):.8f}\n"
                    )
                    q = m["question"].replace("\n", "\\n")
                    a = m["answer"].replace("\n", "\\n")
                    f.write(f"  Q: {q}\n")
                    f.write(f"  A: {a}\n")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using config: {args.config}")
    print(f"Include attn features: {args.include_attn}")
    print(f"Mode: {args.mode}")
    print(f"Force no adapter: {args.force_no_adapter}")
    if args.mode in ("gap_curve", "gap_impact"):
        print(f"Single prune layer: {args.single_prune_layer}")

    # 加载模型
    model, processor = load_model_and_processor(args.checkpoint, args.config, device, force_no_adapter=args.force_no_adapter)

    # 加载样本
    print(f"\nLoading {args.num_samples} samples...")
    samples = load_samples(args.num_samples, args.config)
    print(f"Loaded {len(samples)} samples")

    if args.mode == "export_h":
        # 收集 h 向量（仅 pruning layers 的 pruning_infos）
        h_data = collect_h_vectors(model, processor, samples, device, include_attn=args.include_attn)
        print("\nExporting npz files...")
        export_h_vectors(h_data, args.output_dir)
        print(f"\nDone! Exported to {args.output_dir}")
    elif args.mode == "gap_curve":
        compute_and_export_gap_curves(
            model=model,
            processor=processor,
            samples=samples,
            device=device,
            output_dir=args.output_dir,
            proj_dim=args.proj_dim,
            seed=args.seed,
            single_prune_layer=(None if args.single_prune_layer < 0 else args.single_prune_layer),
        )
    else:
        compute_and_export_gap_impact(
            model=model,
            processor=processor,
            samples=samples,
            device=device,
            output_dir=args.output_dir,
            proj_dim=args.proj_dim,
            seed=args.seed,
            max_length=args.max_length,
            single_prune_layer=(None if args.single_prune_layer < 0 else args.single_prune_layer),
            topk=args.topk,
            report_path=args.report_path,
        )


if __name__ == "__main__":
    main()

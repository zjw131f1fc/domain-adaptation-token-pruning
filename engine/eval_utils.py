"""评估相关工具函数"""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Callable, Tuple
from tqdm import tqdm

from engine.distributed import is_main_process
from engine.data_utils import preprocess_batch, preprocess_batch_qwen2vl


def _normalize_eos_token_id(eos_token_id) -> Optional[int]:
    """将 eos_token_id 规范化为 int（兼容 list/tuple/None）。"""
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple)):
        return int(eos_token_id[0]) if len(eos_token_id) > 0 else None
    return int(eos_token_id)


def _get_eos_token_id(model, processor) -> Optional[int]:
    """尽量从 model/config/tokenizer 中拿到 eos_token_id。"""
    eos = None
    if getattr(model, "base_model", None) is not None:
        eos = getattr(getattr(model.base_model, "config", None), "eos_token_id", None)
    if eos is None and getattr(model, "config", None) is not None:
        eos = getattr(model.config, "eos_token_id", None)
    if eos is None and getattr(processor, "tokenizer", None) is not None:
        eos = getattr(processor.tokenizer, "eos_token_id", None)
    return _normalize_eos_token_id(eos)


def _extract_kept_stats_from_pruning_infos(
    pruning_infos: Optional[Dict[int, Dict[str, Any]]],
    pruning_layers: List[int],
    total_layers: int,
) -> Dict[str, float]:
    """从 forward() 的 pruning_infos 提取 kept ratio 统计，格式尽量对齐硬剪枝的 stats."""
    if not pruning_infos:
        return {}

    stats: Dict[str, float] = {}
    weighted_kept = 0.0

    # 第一个剪枝层之前的层是 100% 保留率
    if pruning_layers:
        first_pruning_layer = pruning_layers[0]
        weighted_kept += float(first_pruning_layer) * 1.0

    for i, layer_idx in enumerate(pruning_layers):
        info = pruning_infos.get(layer_idx)
        if not info:
            continue
        cumulative_mask = info.get("cumulative_mask", None)
        if cumulative_mask is None:
            continue

        # cumulative_mask: (batch, n_vision), 1=保留, 0=剪掉
        ratio = float(cumulative_mask.float().mean().item())
        n_kept = int((cumulative_mask > 0.5).sum().item())

        stats[f"L{layer_idx}_kept"] = ratio
        stats[f"L{layer_idx}_n_kept"] = n_kept

        if i < len(pruning_layers) - 1:
            n_affected = pruning_layers[i + 1] - layer_idx
        else:
            n_affected = total_layers - layer_idx
        weighted_kept += float(n_affected) * ratio

    if total_layers > 0:
        stats["avg_kept_ratio"] = weighted_kept / float(total_layers)

    return stats


@torch.no_grad()
def _generate_with_forward_pruning(
    *,
    model,
    processor,
    input_ids: torch.LongTensor,
    pixel_values: Optional[torch.FloatTensor],
    attention_mask: Optional[torch.Tensor],
    vision_start: int,
    vision_end: int,
    question_starts: List[int],
    question_ends: List[int],
    max_new_tokens: int = 32,
    image_grid_thw: Optional[torch.LongTensor] = None,
    pruning_layers: Optional[List[int]] = None,
) -> Tuple[torch.LongTensor, Dict[str, float]]:
    """用 forward() + greedy decode 做评估用生成（不走物理剪枝）。

    目的：
    - acc 评估时避开 generate_with_hard_pruning 的物理剪枝 bug
    - 复用 forward() 内部逻辑（包含 delayed repair adapter），让指标反映新 adapter 的收益
    """
    model.eval()
    eos_token_id = _get_eos_token_id(model, processor)

    # batch=1 的评估路径；如果未来想扩展 batch>1，需要更复杂的 finished mask
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"forward eval only supports batch_size=1, got input_ids shape={tuple(input_ids.shape)}")

    generated_ids = input_ids

    # answer_start：ASSISTANT: 后的第一个 token 位置（预处理里 question_ends 就是这个）
    answer_start = int(question_ends[0])

    kept_stats: Dict[str, float] = {}
    # 先跑一次 prompt-only 的 forward，用于统计 kept ratio（同时也能提前暴露 forward 路径的崩溃）
    if pruning_layers is None:
        pruning_layers = []
    total_layers = 0
    if getattr(model, "base_model", None) is not None:
        total_layers = len(model.base_model.model.language_model.layers)

    prompt_attention_mask = attention_mask
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)

    prompt_forward_kwargs = {
        "input_ids": generated_ids,
        "pixel_values": pixel_values,
        "attention_mask": prompt_attention_mask,
        "vision_start": vision_start,
        "vision_end": vision_end,
        "question_starts": question_starts,
        "question_ends": question_ends,
        "answer_starts": [answer_start],
        # 用 +1 让 gen_answer 覆盖到 prompt 的最后一个 token（用于预测第一个 answer token）
        "answer_ends": [int(generated_ids.shape[1]) + 1],
        "return_pruning_info": True,
        # apply_repair=None -> forward 内部按 self.use_repair_adapter 自动决定
    }
    if image_grid_thw is not None:
        prompt_forward_kwargs["image_grid_thw"] = image_grid_thw

    out_prompt = model(**prompt_forward_kwargs)
    kept_stats = _extract_kept_stats_from_pruning_infos(
        getattr(out_prompt, "pruning_infos", None),
        pruning_layers=pruning_layers,
        total_layers=total_layers,
    )

    # greedy decode（每一步都从头 forward，慢但最稳）
    for _ in range(int(max_new_tokens)):
        cur_len = int(generated_ids.shape[1])

        cur_attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)
        forward_kwargs = {
            "input_ids": generated_ids,
            "pixel_values": pixel_values,
            "attention_mask": cur_attention_mask,
            "vision_start": vision_start,
            "vision_end": vision_end,
            "question_starts": question_starts,
            "question_ends": question_ends,
            "answer_starts": [answer_start],
            # +1 让 gen_answer 覆盖到最后一个 token，从而影响下一 token 的 logits
            "answer_ends": [cur_len + 1],
            "return_pruning_info": False,
        }
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        out = model(**forward_kwargs)
        logits = out.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if eos_token_id is not None:
            if int(next_token.item()) == int(eos_token_id):
                break

    return generated_ids, kept_stats


@torch.no_grad()
def evaluate(
    model,
    processor,
    dataset,
    judge,
    config,
    device: torch.device,
    max_samples: int = 500,
    mode: str = "origin",
    distributed: bool = False,
    aggregate_judge: Optional[Callable] = None,
    requires_aggregate_eval: bool = False,
) -> Dict[str, float]:
    """评估模型

    Args:
        model: 可剪枝模型
        processor: LLaVA processor
        dataset: 评估数据集
        judge: 评估函数
        config: 配置对象
        device: 设备
        max_samples: 最大评估样本数
        mode: 评估模式：
            - 'origin': 不剪枝，直接 generate
            - 'hard': 物理剪枝推理（generate_with_hard_pruning），可能更快但存在已知 bug
            - 'hard_forward': 不做物理删除，用 forward() + greedy decode 生成（更稳，且适配 delayed repair adapter）
        distributed: 是否使用分布式评估（所有 rank 参与）
        aggregate_judge: 聚合评估函数（用于 MME/GQA 等需要全量评估的数据集）
        requires_aggregate_eval: 是否需要聚合评估

    Returns:
        评估结果字典
    """
    model.eval()

    # 设置评估时的温度和阈值
    method_cfg = config.method_settings
    eval_temp = method_cfg.get('eval_temperature', method_cfg.get('temperature_min', 0.1))
    eval_threshold = method_cfg.get('eval_pruning_threshold', 0.5)  # 评估时的剪枝阈值
    model.set_temperature(eval_temp)
    model.set_pruning_threshold(eval_threshold)
    model.set_use_gumbel_noise(False)  # 评估时不使用 Gumbel noise

    # 获取 max_length 配置
    max_length = config.trainer_settings.get('dl_settings', {}).get('max_length', 2048)

    n_samples = min(len(dataset), max_samples)

    # 分布式评估：每个 rank 处理一部分数据
    if distributed and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        indices = list(range(n_samples))
        # 每个 rank 处理自己的分片
        local_indices = indices[rank::world_size]
    else:
        world_size = 1
        local_indices = list(range(n_samples))

    predictions = []
    references = []
    samples_for_aggregate = []  # 用于聚合评估
    kept_ratios = []
    layer_kept_ratios = {}

    pruning_layers = config.method_settings.get('pruning_layers', [4, 14, 24])
    desc = f"Evaluating ({mode})"

    # 只在主进程显示进度条
    show_progress = is_main_process()

    # 中间统计日志间隔（按全局步数计算）
    # 确保 local_log_interval 不超过每卡实际处理的样本数，否则日志永不触发
    log_interval = 200
    local_samples = len(local_indices)
    if distributed and dist.is_initialized():
        # 每个 rank 处理 local_log_interval 个样本时，全局约处理 log_interval 个
        # 同时确保至少打印 4 次中间日志（如果样本数足够）
        local_log_interval = max(1, min(log_interval // world_size, local_samples // 4))
    else:
        local_log_interval = max(1, min(log_interval, local_samples // 4))

    for step_idx, i in enumerate(tqdm(local_indices, desc=desc, disable=not show_progress), start=1):
        sample = dataset[i]

        # 根据模型类型选择预处理函数
        backbone_name = config.backbone_settings.get('name', 'llava-1.5-7b')
        is_qwen2vl = 'qwen2-vl' in backbone_name.lower()

        if mode in ("hard", "hard_forward"):
            if is_qwen2vl:
                preprocessed = preprocess_batch_qwen2vl(
                    batch=[sample],
                    processor=processor,
                    device=device,
                    max_length=max_length,
                    mode="inference"
                )
            else:
                preprocessed = preprocess_batch(
                    batch=[sample],
                    processor=processor,
                    device=device,
                    max_length=max_length,
                    mode="inference"
                )
            inputs = preprocessed['inputs']

            if mode == "hard_forward":
                output_ids, stats = _generate_with_forward_pruning(
                    model=model,
                    processor=processor,
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask"),
                    vision_start=preprocessed["vision_start"],
                    vision_end=preprocessed["vision_end"],
                    question_starts=preprocessed["question_starts"],
                    question_ends=preprocessed["question_ends"],
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    max_new_tokens=32,
                    pruning_layers=pruning_layers,
                )
            else:
                # ========== 物理剪枝推理（旧 hard 模式）==========
                # 构建 forward 参数（仅用于 debug 计算训练路径的 kept ratio）
                forward_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                    "vision_start": preprocessed["vision_start"],
                    "vision_end": preprocessed["vision_end"],
                    "question_starts": preprocessed["question_starts"],
                    "question_ends": preprocessed["question_ends"],
                    "answer_starts": [preprocessed["question_ends"][0]],
                    "answer_ends": [preprocessed["question_ends"][0] + 1],
                    "return_pruning_info": True,
                }
                if "image_grid_thw" in inputs:
                    forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

                debug_train_ratios = {}
                if step_idx <= 5 and is_main_process():
                    model.eval()
                    output_train = model(**forward_kwargs)
                    for layer_idx in pruning_layers:
                        if output_train.pruning_infos and (layer_idx in output_train.pruning_infos):
                            cumulative_mask = output_train.pruning_infos[layer_idx]["cumulative_mask"]
                            debug_train_ratios[layer_idx] = cumulative_mask.float().mean().item()

                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "pixel_values": inputs.get("pixel_values"),
                    "attention_mask": inputs.get("attention_mask"),
                    "vision_start": preprocessed["vision_start"],
                    "vision_end": preprocessed["vision_end"],
                    "question_starts": preprocessed["question_starts"],
                    "question_ends": preprocessed["question_ends"],
                    "max_new_tokens": 32,
                    "debug_generate": (step_idx <= 3 and is_main_process()),
                }
                if "image_grid_thw" in inputs:
                    generate_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

                if hasattr(model, "generate_with_hard_pruning"):
                    output_ids, stats = model.generate_with_hard_pruning(**generate_kwargs)
                elif hasattr(model, "generate_with_pruning"):
                    # Qwen2-VL 路径（无物理删除，返回 dict）
                    gen_out = model.generate_with_pruning(
                        **generate_kwargs,
                        return_dict_in_generate=True,
                    )
                    output_ids = gen_out["sequences"]
                    stats = gen_out.get("pruning_stats", {})
                else:
                    raise AttributeError("Model does not implement generate_with_hard_pruning/generate_with_pruning")

                # Debug: 对比训练路径和推理路径的保留率
                if step_idx <= 5 and is_main_process() and debug_train_ratios and stats:
                    print(f"[Debug Eval {step_idx}] 训练路径 vs 推理路径:")
                    for layer_idx in pruning_layers:
                        train_ratio = debug_train_ratios.get(layer_idx, 0)
                        infer_ratio = stats.get(f"L{layer_idx}_kept", 0)
                        diff = abs(train_ratio - infer_ratio)
                        print(
                            f"  L{layer_idx}: train={train_ratio:.2%}, "
                            f"infer={infer_ratio:.2%}, diff={diff:.4f}"
                        )

            if "avg_kept_ratio" in stats:
                kept_ratios.append(stats["avg_kept_ratio"])
            for key, value in stats.items():
                if key.startswith("L") and "_kept" in key:
                    layer_idx = int(key[1:].split("_")[0])
                    if key.endswith("_n_kept"):
                        if f"{layer_idx}_n_kept" not in layer_kept_ratios:
                            layer_kept_ratios[f"{layer_idx}_n_kept"] = []
                        layer_kept_ratios[f"{layer_idx}_n_kept"].append(value)
                    elif key == f"L{layer_idx}_kept":
                        if layer_idx not in layer_kept_ratios:
                            layer_kept_ratios[layer_idx] = []
                        layer_kept_ratios[layer_idx].append(value)
        elif mode == "origin":
            # origin 模式：使用原始模型生成
            if is_qwen2vl:
                # Qwen2-VL 格式
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "placeholder"},
                            {"type": "text", "text": sample['question']},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=prompt,
                    images=sample['image'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)
            else:
                # LLaVA 格式
                prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
                inputs = processor(
                    text=prompt,
                    images=sample['image'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        else:
            raise ValueError(f"Unknown eval mode: {mode!r}")

        generated = processor.decode(output_ids[0], skip_special_tokens=True)

        # 根据模型类型提取预测结果
        if is_qwen2vl:
            # Qwen2-VL: 提取 assistant 回复
            if "assistant\n" in generated.lower():
                pred = generated.lower().split("assistant\n")[-1].strip()
            else:
                pred = generated.strip()
        else:
            # LLaVA
            if "ASSISTANT:" in generated:
                pred = generated.split("ASSISTANT:")[-1].strip()
            else:
                pred = generated.strip()

        predictions.append(pred)

        if 'answers' in sample:
            references.append(sample['answers'])
            gt = sample['answers']
        else:
            references.append(sample['answer'])
            gt = sample['answer']

        # 打印前 10 个样本的预测和 ground truth
        if step_idx <= 10 and is_main_process():
            print(f"[Eval {step_idx}] Pred: {pred!r} | GT: {gt!r}")

        # 聚合评估需要保留样本信息（只保留必要字段，不保留图像以避免显存累积）
        if requires_aggregate_eval:
            sample_info = {
                'answer': sample.get('answer'),
                'category': sample.get('category'),  # MME 需要
                'question_id': sample.get('question_id'),  # MME 需要（配对同图问题）
            }
            samples_for_aggregate.append(sample_info)

        # 每 local_log_interval 步打印中间统计
        if step_idx % local_log_interval == 0:
            _print_interim_stats(
                step_idx, predictions, references, kept_ratios, layer_kept_ratios,
                judge, distributed, world_size, requires_aggregate_eval
            )

    # 分布式评估：收集所有 rank 的结果
    if distributed and dist.is_initialized():
        predictions, references, kept_ratios, layer_kept_ratios, samples_for_aggregate = \
            _gather_distributed_results(
                predictions, references, kept_ratios, layer_kept_ratios,
                samples_for_aggregate, requires_aggregate_eval
            )

    # 根据是否需要聚合评估调用不同的 judge
    if requires_aggregate_eval and aggregate_judge is not None:
        result = aggregate_judge(predictions, references, samples_for_aggregate)
    else:
        result = judge(predictions, references)

    # 构建返回结果
    eval_result = {
        'mode': mode,
    }

    # 合并 judge 返回的所有字段
    eval_result.update(result)

    # 兼容旧接口：如果没有 accuracy 字段但有其他主指标，添加 accuracy 别名
    if 'accuracy' not in eval_result:
        if 'balanced_accuracy' in eval_result:
            eval_result['accuracy'] = eval_result['balanced_accuracy']
        elif 'total_score' in eval_result:
            # MME: 将 total_score 归一化为 0-1 范围作为 accuracy（假设满分 1400）
            eval_result['accuracy'] = eval_result['total_score'] / 1400.0

    if kept_ratios:
        eval_result['avg_kept_ratio'] = sum(kept_ratios) / len(kept_ratios)

    for key, values in layer_kept_ratios.items():
        if isinstance(key, int):
            eval_result[f'L{key}_kept'] = sum(values) / len(values)
        elif isinstance(key, str) and key.endswith('_n_kept'):
            layer_idx = key.split('_')[0]
            eval_result[f'L{layer_idx}_n_kept'] = sum(values) / len(values)

    return eval_result


def _print_interim_stats(
    step_idx, predictions, references, kept_ratios, layer_kept_ratios,
    judge, distributed, world_size, requires_aggregate_eval
):
    """打印中间统计信息"""
    if distributed and dist.is_initialized():
        # 分布式模式：收集所有 rank 的数据
        all_predictions = [None] * world_size
        all_references = [None] * world_size
        all_kept_ratios = [None] * world_size

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_references, references)
        dist.all_gather_object(all_kept_ratios, kept_ratios)

        # 合并所有 rank 的数据
        merged_preds = []
        merged_refs = []
        merged_kept = []
        for p, r, k in zip(all_predictions, all_references, all_kept_ratios):
            merged_preds.extend(p)
            merged_refs.extend(r)
            merged_kept.extend(k)

        if is_main_process():
            interim_total = len(merged_preds)
            if requires_aggregate_eval:
                # 聚合评估模式：只打印进度和 kept ratio（无法增量计算 accuracy）
                if merged_kept:
                    interim_kept = sum(merged_kept) / len(merged_kept)
                    print(f"\n[Step {interim_total}] Processed: {interim_total}, Kept: {interim_kept:.2%}")
                else:
                    print(f"\n[Step {interim_total}] Processed: {interim_total}")
            else:
                # 普通模式：打印 accuracy
                interim_result = judge(merged_preds, merged_refs)
                interim_acc = interim_result['accuracy']
                interim_correct = interim_result['correct']

                if merged_kept:
                    interim_kept = sum(merged_kept) / len(merged_kept)
                    # 打印每层保留率
                    layer_str = ""
                    if layer_kept_ratios:
                        layer_parts = []
                        for layer_idx in sorted([k for k in layer_kept_ratios.keys() if isinstance(k, int)]):
                            if layer_kept_ratios[layer_idx]:
                                avg_ratio = sum(layer_kept_ratios[layer_idx]) / len(layer_kept_ratios[layer_idx])
                                layer_parts.append(f"L{layer_idx}={avg_ratio:.2%}")
                        if layer_parts:
                            layer_str = f" [{', '.join(layer_parts)}]"
                    print(f"\n[Step {interim_total}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total}), Kept: {interim_kept:.2%}{layer_str}")
                else:
                    print(f"\n[Step {interim_total}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total})")
    else:
        # 单卡模式
        interim_total = len(predictions)
        if requires_aggregate_eval:
            # 聚合评估模式：只打印进度和 kept ratio
            if kept_ratios:
                interim_kept = sum(kept_ratios) / len(kept_ratios)
                print(f"\n[Step {interim_total}] Processed: {interim_total}, Kept: {interim_kept:.2%}")
            else:
                print(f"\n[Step {interim_total}] Processed: {interim_total}")
        else:
            # 普通模式：打印 accuracy
            interim_result = judge(predictions, references)
            interim_acc = interim_result['accuracy']
            interim_correct = interim_result['correct']

            if kept_ratios:
                interim_kept = sum(kept_ratios) / len(kept_ratios)
                # 打印每层保留率
                layer_str = ""
                if layer_kept_ratios:
                    layer_parts = []
                    for layer_idx in sorted([k for k in layer_kept_ratios.keys() if isinstance(k, int)]):
                        if layer_kept_ratios[layer_idx]:
                            avg_ratio = sum(layer_kept_ratios[layer_idx]) / len(layer_kept_ratios[layer_idx])
                            layer_parts.append(f"L{layer_idx}={avg_ratio:.2%}")
                    if layer_parts:
                        layer_str = f" [{', '.join(layer_parts)}]"
                print(f"\n[Step {step_idx}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total}), Kept: {interim_kept:.2%}{layer_str}")
            else:
                print(f"\n[Step {step_idx}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total})")


def _gather_distributed_results(
    predictions, references, kept_ratios, layer_kept_ratios,
    samples_for_aggregate, requires_aggregate_eval
):
    """收集分布式评估的结果"""
    # 收集所有 rank 的 predictions 和 references
    all_predictions = [None] * dist.get_world_size()
    all_references = [None] * dist.get_world_size()
    dist.all_gather_object(all_predictions, predictions)
    dist.all_gather_object(all_references, references)

    # 收集 kept_ratios
    all_kept_ratios = [None] * dist.get_world_size()
    dist.all_gather_object(all_kept_ratios, kept_ratios)

    # 收集 layer_kept_ratios
    all_layer_kept_ratios = [None] * dist.get_world_size()
    dist.all_gather_object(all_layer_kept_ratios, layer_kept_ratios)

    # 收集 samples_for_aggregate（如果需要聚合评估）
    if requires_aggregate_eval:
        all_samples = [None] * dist.get_world_size()
        dist.all_gather_object(all_samples, samples_for_aggregate)

    # 在所有 rank 上合并结果（保证一致性）
    merged_predictions = []
    merged_references = []
    merged_kept_ratios = []
    merged_layer_kept_ratios = {}

    for preds, refs in zip(all_predictions, all_references):
        merged_predictions.extend(preds)
        merged_references.extend(refs)

    for ratios in all_kept_ratios:
        merged_kept_ratios.extend(ratios)

    for layer_ratios in all_layer_kept_ratios:
        # 兼容异常/旧数据：某些 rank 可能返回非 dict（例如空 list）
        if not isinstance(layer_ratios, dict):
            continue
        for key, values in layer_ratios.items():
            if key not in merged_layer_kept_ratios:
                merged_layer_kept_ratios[key] = []
            merged_layer_kept_ratios[key].extend(values)

    # 合并 samples_for_aggregate
    merged_samples = []
    if requires_aggregate_eval:
        for samples in all_samples:
            merged_samples.extend(samples)

    return merged_predictions, merged_references, merged_kept_ratios, merged_layer_kept_ratios, merged_samples

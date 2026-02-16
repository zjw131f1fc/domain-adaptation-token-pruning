"""评估相关工具函数"""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Callable
from tqdm import tqdm

from engine.distributed import is_main_process
from engine.data_utils import preprocess_batch


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
        mode: 评估模式 ('origin' 或 'hard')
        distributed: 是否使用分布式评估（所有 rank 参与）
        aggregate_judge: 聚合评估函数（用于 MME/GQA 等需要全量评估的数据集）
        requires_aggregate_eval: 是否需要聚合评估

    Returns:
        评估结果字典
    """
    model.eval()

    # 设置评估时的温度
    method_cfg = config.method_settings
    eval_temp = method_cfg.get('eval_temperature', method_cfg.get('temperature_min', 0.1))
    model.set_temperature(eval_temp)
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

        if mode == "hard":
            preprocessed = preprocess_batch(
                batch=[sample],
                processor=processor,
                device=device,
                max_length=max_length,
                mode="inference"
            )
            inputs = preprocessed['inputs']

            # Debug: 同时用训练路径计算保留率
            debug_train_ratios = {}
            if step_idx <= 5 and is_main_process():
                # 用训练路径（model()）计算保留率
                model.eval()
                with torch.no_grad():
                    output_train = model(
                        input_ids=inputs['input_ids'],
                        pixel_values=inputs['pixel_values'],
                        attention_mask=inputs['attention_mask'],
                        vision_start=preprocessed['vision_start'],
                        vision_end=preprocessed['vision_end'],
                        question_starts=preprocessed['question_starts'],
                        question_ends=preprocessed['question_ends'],
                        answer_starts=[preprocessed['question_ends'][0]],
                        answer_ends=[preprocessed['question_ends'][0] + 1],
                        return_pruning_info=True,
                    )
                for layer_idx in pruning_layers:
                    if layer_idx in output_train.pruning_infos:
                        cumulative_mask = output_train.pruning_infos[layer_idx]['cumulative_mask']
                        debug_train_ratios[layer_idx] = cumulative_mask.float().mean().item()

            output_ids, stats = model.generate_with_hard_pruning(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask'),
                vision_start=preprocessed['vision_start'],
                vision_end=preprocessed['vision_end'],
                question_starts=preprocessed['question_starts'],
                question_ends=preprocessed['question_ends'],
                max_new_tokens=32,
                debug_generate=(step_idx <= 3 and is_main_process()),  # 前 3 个样本打印 debug
            )

            # Debug: 对比训练路径和推理路径的保留率
            if step_idx <= 5 and is_main_process() and debug_train_ratios:
                print(f"[Debug Eval {step_idx}] 训练路径 vs 推理路径:")
                for layer_idx in pruning_layers:
                    train_ratio = debug_train_ratios.get(layer_idx, 0)
                    infer_ratio = stats.get(f'L{layer_idx}_kept', 0)
                    diff = abs(train_ratio - infer_ratio)
                    print(f"  L{layer_idx}: train={train_ratio:.2%}, infer={infer_ratio:.2%}, diff={diff:.4f}")

            if 'avg_kept_ratio' in stats:
                kept_ratios.append(stats['avg_kept_ratio'])
            for key, value in stats.items():
                if key.startswith('L') and '_kept' in key:
                    layer_idx = int(key[1:].split('_')[0])
                    if key.endswith('_n_kept'):
                        if f'{layer_idx}_n_kept' not in layer_kept_ratios:
                            layer_kept_ratios[f'{layer_idx}_n_kept'] = []
                        layer_kept_ratios[f'{layer_idx}_n_kept'].append(value)
                    elif key == f'L{layer_idx}_kept':
                        if layer_idx not in layer_kept_ratios:
                            layer_kept_ratios[layer_idx] = []
                        layer_kept_ratios[layer_idx].append(value)
        else:
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

        generated = processor.decode(output_ids[0], skip_special_tokens=True)

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

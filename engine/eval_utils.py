"""评估相关工具函数"""

import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional, Callable, Tuple
from tqdm import tqdm

from engine.distributed import is_main_process
from engine.data_utils import preprocess_batch


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

    pruning_layers = sorted([int(x) for x in (pruning_layers or [])])
    if not pruning_layers or total_layers <= 0:
        return {}

    # 关键：剪枝层之前的 layer 没有 pruning_infos，它们默认是 100% 保留。
    # 旧实现会漏算这段，导致 avg_kept_ratio 偏小（尤其当第一个剪枝层较深时）。
    first_prune_layer = int(pruning_layers[0])
    if first_prune_layer > 0:
        weighted_kept += float(first_prune_layer) * 1.0

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

    stats["avg_kept_ratio"] = weighted_kept / float(total_layers)

    return stats


def _resolve_answer_text(sample: Dict[str, Any]) -> Optional[str]:
    """Resolve a single answer string from a dataset sample.

    Many datasets provide:
      - sample["answer"]: a training/official answer (string)
      - sample["answers"]: a list of acceptable answers for judging
    For representation-alignment metrics (teacher-forcing), we need a single string.
    """
    ans = sample.get("answer", None)
    if ans is not None:
        s = str(ans).strip()
        return s if s else None
    answers = sample.get("answers", None)
    if isinstance(answers, (list, tuple)) and len(answers) > 0:
        s = str(answers[0]).strip()
        return s if s else None
    return None


def _extract_prediction_text(generated: str) -> str:
    """提取用于判分的预测文本，避免把后续对话和多行补充一并送入 judge。"""
    text = str(generated).strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    if "USER:" in text:
        text = text.split("USER:")[0].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0]
    return text


def _init_w2_accumulators(
    *,
    layers: List[int],
    hidden_size: int,
    device: torch.device,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Initialize streaming moment accumulators for diagonal-Gaussian W2^2.

    Each layer keeps:
      sum_s, sum_s2, sum_t, sum_t2: (D,)
      count: (1,)
    """
    accs: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        accs[int(layer)] = {
            "sum_s": torch.zeros(hidden_size, device=device, dtype=torch.float32),
            "sum_s2": torch.zeros(hidden_size, device=device, dtype=torch.float32),
            "sum_t": torch.zeros(hidden_size, device=device, dtype=torch.float32),
            "sum_t2": torch.zeros(hidden_size, device=device, dtype=torch.float32),
            "count": torch.zeros(1, device=device, dtype=torch.float32),
        }
    return accs


def _update_w2_accumulator(
    acc: Dict[str, torch.Tensor],
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    mask: torch.Tensor,
):
    """Update moment accumulator with masked tokens.

    Args:
        student_h/teacher_h: (b, L, D)
        mask: (b, L) 0/1 (float or bool). Only masked positions contribute.
    """
    if student_h is None or teacher_h is None or mask is None:
        return
    if student_h.shape != teacher_h.shape:
        return
    if student_h.dim() != 3 or mask.dim() != 2:
        return

    h_s = student_h.float()
    h_t = teacher_h.float()
    m = mask.to(dtype=h_s.dtype)
    m_exp = m.unsqueeze(-1)  # (b, L, 1)

    acc["sum_s"] += (h_s * m_exp).sum(dim=(0, 1))
    acc["sum_s2"] += ((h_s * h_s) * m_exp).sum(dim=(0, 1))
    acc["sum_t"] += (h_t * m_exp).sum(dim=(0, 1))
    acc["sum_t2"] += ((h_t * h_t) * m_exp).sum(dim=(0, 1))
    acc["count"] += m.sum().reshape_as(acc["count"])


def _finalize_w2_accumulator(
    acc: Dict[str, torch.Tensor],
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute diagonal-Gaussian W2^2 proxy metrics from accumulated moments."""
    count = float(acc["count"].detach().item())
    if count <= 0:
        return {
            "count": 0.0,
            "mean_mse": float("nan"),
            "std_mse": float("nan"),
            "w2_sq": float("nan"),
        }

    denom = acc["count"].clamp(min=1.0)
    inv = 1.0 / denom

    ms = acc["sum_s"] * inv
    mt = acc["sum_t"] * inv

    vs = acc["sum_s2"] * inv - ms * ms
    vt = acc["sum_t2"] * inv - mt * mt
    vs = vs.clamp(min=0.0)
    vt = vt.clamp(min=0.0)

    std_s = torch.sqrt(vs + float(eps))
    std_t = torch.sqrt(vt + float(eps))

    mean_mse = torch.mean((ms - mt) ** 2).detach().item()
    std_mse = torch.mean((std_s - std_t) ** 2).detach().item()
    w2_sq = float(mean_mse + std_mse)

    return {
        "count": count,
        "mean_mse": float(mean_mse),
        "std_mse": float(std_mse),
        "w2_sq": float(w2_sq),
    }


def _compute_w2_from_masked_tokens(
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute diagonal-Gaussian W2^2 surrogate from masked tokens (no distributed reduce).

    This mirrors the training-side "mean + std" surrogate:
      W2^2 = mean((mu_s - mu_t)^2) + mean((sigma_s - sigma_t)^2)

    Returns:
        Dict with scalar tensors (float32): count, mean_mse, std_mse, w2_sq.
        Returns None if inputs are invalid or mask has zero count.
    """
    if student_h is None or teacher_h is None or mask is None:
        return None
    if student_h.shape != teacher_h.shape:
        return None
    if student_h.dim() != 3 or mask.dim() != 2:
        return None

    h_s = student_h.float()
    h_t = teacher_h.float()
    m = mask.to(dtype=h_s.dtype)
    count = m.sum()
    if float(count.detach().item()) <= 0:
        return None

    m_exp = m.unsqueeze(-1)
    denom = count.clamp(min=1.0)
    inv = 1.0 / denom

    sum_s = (h_s * m_exp).sum(dim=(0, 1))
    sum_s2 = ((h_s * h_s) * m_exp).sum(dim=(0, 1))
    sum_t = (h_t * m_exp).sum(dim=(0, 1))
    sum_t2 = ((h_t * h_t) * m_exp).sum(dim=(0, 1))

    ms = sum_s * inv
    mt = sum_t * inv
    vs = sum_s2 * inv - ms * ms
    vt = sum_t2 * inv - mt * mt
    vs = vs.clamp(min=0.0)
    vt = vt.clamp(min=0.0)

    std_s = torch.sqrt(vs + float(eps))
    std_t = torch.sqrt(vt + float(eps))

    mean_mse = torch.mean((ms - mt) ** 2)
    std_mse = torch.mean((std_s - std_t) ** 2)
    w2_sq = mean_mse + std_mse

    return {
        "count": count.detach().to(dtype=torch.float32),
        "mean_mse": mean_mse.detach().to(dtype=torch.float32),
        "std_mse": std_mse.detach().to(dtype=torch.float32),
        "w2_sq": w2_sq.detach().to(dtype=torch.float32),
    }


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
    pruning_layers: Optional[List[int]] = None,
    pruning_mode: str = "normal",
    target_token_num: Optional[int] = None,
    apply_repair: Optional[bool] = None,
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
        "pruning_mode": pruning_mode,
        "target_token_num": target_token_num,
        "apply_repair": apply_repair,
    }
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
            "pruning_mode": pruning_mode,
            "target_token_num": target_token_num,
            "apply_repair": apply_repair,
        }
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

    # Ablations（与 train_step 对齐）
    method_cfg = config.method_settings
    ab_w_o_pruner_topk = bool(method_cfg.get("ablation_w_o_pruner_topk_attn", False))
    ab_w_o_adapter = bool(method_cfg.get("ablation_w_o_adapter", False))

    eval_pruning_mode = "topk_attn" if ab_w_o_pruner_topk else "normal"
    eval_target_token_num = method_cfg.get("target_token_num", None)
    # 是否在评估时应用 delayed repair adapter：
    # - hard_forward: 作用于 gen_answer 区域（训练口径）
    # - hard: deployed adapter 口径（默认只修复最后一个 token）
    #
    # 可通过 evaluation_settings.apply_repair 控制：
    # - "auto"/None: 跟随 checkpoint/config（若启用 adapter 则自动应用）
    # - true/false: 强制开/关
    apply_repair_cfg = None
    if getattr(config, "evaluation_settings", None) is not None:
        apply_repair_cfg = config.evaluation_settings.get("apply_repair", "auto")

    if apply_repair_cfg is None:
        eval_apply_repair = None
    elif isinstance(apply_repair_cfg, bool):
        eval_apply_repair = apply_repair_cfg
    elif isinstance(apply_repair_cfg, str):
        s = apply_repair_cfg.strip().lower()
        if s in {"", "auto", "none", "null"}:
            eval_apply_repair = None
        elif s in {"true", "1", "yes", "y", "on"}:
            eval_apply_repair = True
        elif s in {"false", "0", "no", "n", "off"}:
            eval_apply_repair = False
        else:
            raise ValueError(f"Invalid evaluation_settings.apply_repair={apply_repair_cfg!r}, expected auto/true/false.")
    else:
        eval_apply_repair = bool(apply_repair_cfg)

    # Ablation has highest priority: force-disable repair.
    if ab_w_o_adapter:
        eval_apply_repair = False

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

    # ===== Representation drift metric: eval-average diagonal-Gaussian W2^2 (teacher-forcing) =====
    # This is a diagnostic metric aligned with the "distribution alignment" motivation:
    # compare the student route (pruned, optionally repaired) to a keep-all teacher route.
    #
    # Default behavior:
    # - enabled for hard/hard_forward modes (can be disabled via evaluation_settings.report_w2=false)
    report_w2 = False
    w2_layers: List[int] = []
    w2_accs: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
    w2_sample_sum: Optional[torch.Tensor] = None
    w2_sample_sum_pre: Optional[torch.Tensor] = None
    w2_sample_sum_gain: Optional[torch.Tensor] = None
    w2_sample_count: Optional[torch.Tensor] = None
    w2_layer_sum: Optional[torch.Tensor] = None
    w2_layer_sum_pre: Optional[torch.Tensor] = None
    w2_layer_sum_gain: Optional[torch.Tensor] = None
    w2_layer_count: Optional[torch.Tensor] = None

    pruning_layers = config.method_settings.get('pruning_layers', [4, 14, 24])
    desc = f"Evaluating ({mode})"

    # 只在主进程显示进度条
    show_progress = is_main_process()

    # 中间统计日志间隔（按全局步数计算）
    # 确保 local_log_interval 不超过每卡实际处理的样本数，否则日志永不触发
    log_interval = 200
    local_samples = len(local_indices)
    min_local_samples = local_samples
    if distributed and dist.is_initialized():
        # 关键：某些数据集大小无法整除 world_size，会导致各 rank 的 local_samples 相差 1。
        # 如果仍按各自 local_samples 触发中间 all_gather（collective），可能出现：
        # - 某些 rank 在最后一步触发 interim gather
        # - 其他 rank 已经进入最终 gather
        # 从而 collective 顺序不一致导致卡死（典型 DDP deadlock）。
        #
        # 解决：用全局 min_local_samples 约束中间日志，只在所有 rank 都能达到的步数上触发。
        min_tensor = torch.tensor(int(local_samples), device=device, dtype=torch.int64)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        min_local_samples = int(min_tensor.item())

    if distributed and dist.is_initialized():
        # 每个 rank 处理 local_log_interval 个样本时，全局约处理 log_interval 个
        # 同时确保至少打印 4 次中间日志（如果样本数足够）
        safe_div = max(1, (min_local_samples // 4))
        local_log_interval = max(1, min(log_interval // world_size, safe_div))
    else:
        safe_div = max(1, (local_samples // 4))
        local_log_interval = max(1, min(log_interval, safe_div))

    # 只允许触发到所有 rank 都能达到的最后一个同步步
    last_sync_step = 0
    if local_log_interval > 0:
        if distributed and dist.is_initialized():
            last_sync_step = (min_local_samples // local_log_interval) * local_log_interval
        else:
            last_sync_step = (local_samples // local_log_interval) * local_log_interval

    if getattr(config, "evaluation_settings", None) is not None:
        report_w2 = bool(config.evaluation_settings.get("report_w2", mode in ("hard", "hard_forward")))
    else:
        report_w2 = bool(mode in ("hard", "hard_forward"))

    if report_w2:
        w2_layers = [int(x) for x in (config.method_settings.get("repair_layers", []) or [])]
        if not w2_layers:
            report_w2 = False
        else:
            hidden_size = getattr(model, "hidden_size", None)
            if hidden_size is None and getattr(model, "base_model", None) is not None:
                hidden_size = getattr(model.base_model.model.language_model.config, "hidden_size", None)
            if hidden_size is None:
                raise RuntimeError("Cannot infer model hidden_size for W2 accumulator.")
            w2_accs = _init_w2_accumulators(layers=w2_layers, hidden_size=int(hidden_size), device=device)
            # Dataset average over samples (each sample contributes one scalar = mean over layers).
            w2_sample_sum = torch.zeros(1, device=device, dtype=torch.float32)
            w2_sample_sum_pre = torch.zeros(1, device=device, dtype=torch.float32)
            w2_sample_sum_gain = torch.zeros(1, device=device, dtype=torch.float32)
            w2_sample_count = torch.zeros(1, device=device, dtype=torch.float32)
            # Per-layer sample-average accumulators (aligned with "per adapter" analysis).
            w2_layer_sum = torch.zeros(len(w2_layers), device=device, dtype=torch.float32)
            w2_layer_sum_pre = torch.zeros(len(w2_layers), device=device, dtype=torch.float32)
            w2_layer_sum_gain = torch.zeros(len(w2_layers), device=device, dtype=torch.float32)
            w2_layer_count = torch.zeros(len(w2_layers), device=device, dtype=torch.float32)

    for step_idx, i in enumerate(tqdm(local_indices, desc=desc, disable=not show_progress), start=1):
        sample = dataset[i]

        if mode in ("hard", "hard_forward"):
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
                    max_new_tokens=32,
                    pruning_layers=pruning_layers,
                    pruning_mode=eval_pruning_mode,
                    target_token_num=eval_target_token_num,
                    apply_repair=eval_apply_repair,
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
                    "apply_repair": eval_apply_repair,
                }

                if hasattr(model, "generate_with_hard_pruning"):
                    output_ids, stats = model.generate_with_hard_pruning(**generate_kwargs)
                elif hasattr(model, "generate_with_pruning"):
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

            # Representation drift metric (teacher-forcing, uses GT answer)
            if report_w2 and w2_accs is not None:
                answer_text = _resolve_answer_text(sample)
                if answer_text is not None:
                    w2_sample = dict(sample)
                    w2_sample["answer"] = answer_text
                    w2_prep = preprocess_batch(
                        batch=[w2_sample],
                        processor=processor,
                        device=device,
                        max_length=max_length,
                        mode="train",
                    )
                    w2_inputs = w2_prep["inputs"]
                    student_kwargs = {
                        "input_ids": w2_inputs["input_ids"],
                        "pixel_values": w2_inputs.get("pixel_values"),
                        "attention_mask": w2_inputs.get("attention_mask"),
                        "vision_start": w2_prep["vision_start"],
                        "vision_end": w2_prep["vision_end"],
                        "question_starts": w2_prep["question_starts"],
                        "question_ends": w2_prep["question_ends"],
                        "answer_starts": w2_prep["answer_starts"],
                        "answer_ends": w2_prep["answer_ends"],
                        "return_pruning_info": False,
                        "pruning_mode": eval_pruning_mode,
                        "target_token_num": eval_target_token_num,
                        "apply_repair": eval_apply_repair,
                        "capture_layers": w2_layers,
                    }
                    teacher_kwargs = dict(student_kwargs)
                    teacher_kwargs["pruning_mode"] = "keep_all"
                    teacher_kwargs["target_token_num"] = None
                    teacher_kwargs["apply_repair"] = False

                    out_teacher = model(**teacher_kwargs)
                    out_student = model(**student_kwargs)

                    teacher_caps = getattr(out_teacher, "captured", None) or {}
                    student_caps = (
                        getattr(out_student, "captured_for_repair", None)
                        or getattr(out_student, "captured", None)
                        or {}
                    )
                    student_caps_pre = getattr(out_student, "captured_pre_repair", None) or {}

                    per_sample_w2 = []
                    per_sample_w2_pre = []
                    per_sample_w2_gain = []
                    for layer_pos, layer_idx in enumerate(w2_layers):
                        if layer_idx not in w2_accs:
                            continue
                        if layer_idx not in teacher_caps or layer_idx not in student_caps:
                            continue
                        t = teacher_caps[layer_idx]
                        s = student_caps[layer_idx]
                        m = s["mask"] * t["mask"]
                        _update_w2_accumulator(w2_accs[layer_idx], s["h"], t["h"], m)
                        details_post = _compute_w2_from_masked_tokens(s["h"], t["h"], m)

                        # per-adapter breakdown: pre-repair vs post-repair at the same layer
                        details_pre = None
                        if layer_idx in student_caps_pre:
                            sp = student_caps_pre[layer_idx]
                            mp = sp["mask"] * t["mask"]
                            details_pre = _compute_w2_from_masked_tokens(sp["h"], t["h"], mp)

                        if details_post is not None:
                            per_sample_w2.append(details_post["w2_sq"])
                            if w2_layer_sum is not None and w2_layer_count is not None:
                                w2_layer_sum[layer_pos] += details_post["w2_sq"].reshape_as(w2_layer_sum[layer_pos])
                                w2_layer_count[layer_pos] += 1.0

                        if details_pre is not None:
                            per_sample_w2_pre.append(details_pre["w2_sq"])
                            if w2_layer_sum_pre is not None:
                                w2_layer_sum_pre[layer_pos] += details_pre["w2_sq"].reshape_as(w2_layer_sum_pre[layer_pos])

                        if (details_pre is not None) and (details_post is not None):
                            gain = (details_pre["w2_sq"] - details_post["w2_sq"]).reshape_as(details_post["w2_sq"])
                            per_sample_w2_gain.append(gain)
                            if w2_layer_sum_gain is not None:
                                w2_layer_sum_gain[layer_pos] += gain.reshape_as(w2_layer_sum_gain[layer_pos])

                    if per_sample_w2 and (w2_sample_sum is not None) and (w2_sample_count is not None):
                        # 训练时 repair loss 是 "mean over layers"，这里保持一致：每个样本先对 layer 求均值。
                        w2_sample_sum += torch.stack(per_sample_w2).mean().reshape_as(w2_sample_sum)
                        w2_sample_count += 1.0
                        if per_sample_w2_pre and (w2_sample_sum_pre is not None):
                            w2_sample_sum_pre += torch.stack(per_sample_w2_pre).mean().reshape_as(w2_sample_sum_pre)
                        if per_sample_w2_gain and (w2_sample_sum_gain is not None):
                            w2_sample_sum_gain += torch.stack(per_sample_w2_gain).mean().reshape_as(w2_sample_sum_gain)

                    del out_teacher, out_student

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

        pred = _extract_prediction_text(generated)

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
        if (step_idx % local_log_interval == 0) and (step_idx <= last_sync_step):
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

    # Finalize W2^2 drift metrics (reduce across ranks once).
    w2_metrics: Dict[str, float] = {}
    if report_w2 and w2_accs is not None and w2_layers:
        if distributed and dist.is_initialized():
            # All ranks must reduce in the same order to avoid deadlocks.
            for layer_idx in w2_layers:
                acc = w2_accs[int(layer_idx)]
                for k in ("sum_s", "sum_s2", "sum_t", "sum_t2", "count"):
                    dist.all_reduce(acc[k], op=dist.ReduceOp.SUM)
            if w2_sample_sum is not None and w2_sample_count is not None:
                dist.all_reduce(w2_sample_sum, op=dist.ReduceOp.SUM)
            if w2_sample_sum_pre is not None:
                dist.all_reduce(w2_sample_sum_pre, op=dist.ReduceOp.SUM)
            if w2_sample_sum_gain is not None:
                dist.all_reduce(w2_sample_sum_gain, op=dist.ReduceOp.SUM)
            if w2_sample_count is not None:
                dist.all_reduce(w2_sample_count, op=dist.ReduceOp.SUM)
            for t in (w2_layer_sum, w2_layer_sum_pre, w2_layer_sum_gain, w2_layer_count):
                if t is not None:
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)

        per_layer_w2 = []
        per_layer_mean = []
        per_layer_std = []
        for layer_idx in w2_layers:
            acc = w2_accs[int(layer_idx)]
            m = _finalize_w2_accumulator(acc)
            if m["count"] <= 0:
                continue
            w2_metrics[f"L{int(layer_idx)}_w2_sq"] = m["w2_sq"]
            per_layer_w2.append(m["w2_sq"])
            per_layer_mean.append(m["mean_mse"])
            per_layer_std.append(m["std_mse"])

        if per_layer_w2:
            w2_metrics["avg_w2_sq"] = float(sum(per_layer_w2) / len(per_layer_w2))
            w2_metrics["avg_w2_mean_mse"] = float(sum(per_layer_mean) / len(per_layer_mean))
            w2_metrics["avg_w2_std_mse"] = float(sum(per_layer_std) / len(per_layer_std))

        # Dataset average over samples (each sample contributes one number).
        if w2_sample_sum is not None and w2_sample_count is not None:
            c = float(w2_sample_count.detach().item())
            if c > 0:
                w2_metrics["avg_w2_sq_sample"] = float((w2_sample_sum / w2_sample_count).detach().item())
                if w2_sample_sum_pre is not None:
                    w2_metrics["avg_w2_sq_sample_pre"] = float((w2_sample_sum_pre / w2_sample_count).detach().item())
                if w2_sample_sum_gain is not None:
                    w2_metrics["avg_w2_sq_sample_gain"] = float((w2_sample_sum_gain / w2_sample_count).detach().item())

        # Per-layer sample averages (adapter-wise breakdown).
        if w2_layer_sum is not None and w2_layer_count is not None:
            for layer_pos, layer_idx in enumerate(w2_layers):
                cnt = float(w2_layer_count[layer_pos].detach().item())
                if cnt <= 0:
                    continue
                w2_metrics[f"L{int(layer_idx)}_w2_sq_sample"] = float((w2_layer_sum[layer_pos] / w2_layer_count[layer_pos]).detach().item())
                if w2_layer_sum_pre is not None:
                    w2_metrics[f"L{int(layer_idx)}_w2_sq_sample_pre"] = float(
                        (w2_layer_sum_pre[layer_pos] / w2_layer_count[layer_pos]).detach().item()
                    )
                if w2_layer_sum_gain is not None:
                    w2_metrics[f"L{int(layer_idx)}_w2_sq_sample_gain"] = float(
                        (w2_layer_sum_gain[layer_pos] / w2_layer_count[layer_pos]).detach().item()
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
    eval_result.update(w2_metrics)

    # 兼容旧接口：如果没有 accuracy 字段但有其他主指标，添加 accuracy 别名
    if 'accuracy' not in eval_result:
        if 'balanced_accuracy' in eval_result:
            eval_result['accuracy'] = eval_result['balanced_accuracy']
        elif 'total_score' in eval_result:
            # MME: 按实际参与评估的类别数归一化；若缺失则回退到满分 1400。
            num_categories = eval_result.get('num_categories', None)
            if isinstance(num_categories, (int, float)) and float(num_categories) > 0:
                max_total_score = float(num_categories) * 200.0
            else:
                max_total_score = 1400.0
            if max_total_score > 0:
                eval_result['accuracy'] = max(0.0, min(1.0, eval_result['total_score'] / max_total_score))
            else:
                eval_result['accuracy'] = 0.0

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

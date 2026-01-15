"""Vision Token Pruning - 工具函数

包含embedding处理、mask应用、hook注册等通用辅助函数。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Callable


class ScaleVisionWithClippedGrad(torch.autograd.Function):
    """自定义autograd函数：vision_hidden * soft_mask

    保持梯度流通，不做裁剪。
    LLM 是 frozen 的，所以只需要 soft_mask 的梯度。
    """

    @staticmethod
    def forward(ctx, vision_hidden, soft_mask, grad_clip_value=None):
        """
        vision_hidden: (batch, n_vision, d_model)
        soft_mask: (batch, n_vision)
        """
        # 关键：detach vision_hidden 避免保存整个 LLM 计算图
        # 因为 vision_hidden 是 hidden_states 的 view，不 detach 会保存整个 base tensor
        # 我们不需要 vision_hidden 的梯度（LLM 是 frozen 的）
        vision_hidden_detached = vision_hidden.detach()
        ctx.save_for_backward(vision_hidden_detached, soft_mask)
        # 前向计算仍然使用原始 vision_hidden（保持计算图连接到 scaled_vision）
        return vision_hidden * soft_mask.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (batch, n_vision, d_model) - d_loss/d_scaled_vision
        """
        vision_hidden_detached, soft_mask = ctx.saved_tensors

        # 对vision_hidden的梯度：不需要（LLM是frozen的）
        grad_vision = None

        # 对soft_mask的梯度（不裁剪，让全局 grad_clip 处理）
        # grad_soft_mask = sum(grad_output * vision_hidden, dim=-1)
        grad_soft_mask = (grad_output * vision_hidden_detached).sum(dim=-1)

        return grad_vision, grad_soft_mask, None


def extract_target_hidden_states_batch(
    all_hidden_states: tuple,
    answer_positions_list: List[Tuple[int, int]],
    target_layer_indices: List[int],
    batch_size: int,
    attention_mask: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    """批量提取指定层和位置的hidden states

    参数:
        all_hidden_states: tuple of (batch, seq_len, hidden_dim)
        answer_positions_list: List of (start, end) for each sample
        target_layer_indices: 目标层索引列表
        batch_size: batch大小
        attention_mask: (batch, seq_len) attention mask，用于计算每个样本的实际长度

    返回:
        List of tensors, each (batch, max_answer_len, hidden_dim)
    """
    selected_hidden_states = []
    seq_len = all_hidden_states[0].shape[1]

    # 计算每个样本的实际长度（用于转换负索引）
    if attention_mask is not None:
        actual_lengths = attention_mask.sum(dim=1).tolist()  # List[int]
    else:
        actual_lengths = [seq_len] * batch_size  # 没有mask时假设全部有效

    # 计算每个样本的answer位置（转换负索引）
    answer_ranges = []
    max_answer_len = 0
    for i, (start, end) in enumerate(answer_positions_list):
        if start is None:
            answer_ranges.append((0, 0))
            continue
        # 转换负索引：使用每个样本的实际长度
        sample_len = int(actual_lengths[i])
        if start < 0:
            start = sample_len + start
        if end < 0:
            end = sample_len + end
        answer_len = end - start + 1
        max_answer_len = max(max_answer_len, answer_len)
        answer_ranges.append((start, end))

    for layer_idx in target_layer_indices:
        hidden = all_hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
        hidden_dim = hidden.shape[-1]

        # 使用 torch.cat 保持梯度流（不用 torch.zeros + in-place 赋值）
        slices = []
        for i, (start, end) in enumerate(answer_ranges):
            if start == end == 0:
                # 无效样本：用零填充（不需要梯度）
                slices.append(torch.zeros(1, max_answer_len, hidden_dim,
                                         device=hidden.device, dtype=hidden.dtype))
            else:
                answer_len = end - start + 1
                # 提取切片（保持梯度）
                slice_tensor = hidden[i:i+1, start:end+1, :]
                # 如果需要padding
                if answer_len < max_answer_len:
                    padding = torch.zeros(1, max_answer_len - answer_len, hidden_dim,
                                         device=hidden.device, dtype=hidden.dtype)
                    slice_tensor = torch.cat([slice_tensor, padding], dim=1)
                slices.append(slice_tensor)

        batch_hidden = torch.cat(slices, dim=0)
        selected_hidden_states.append(batch_hidden)

    return selected_hidden_states


def compute_task_loss_batch(
    logits: torch.Tensor,
    answer_positions_list: List[Tuple[int, int]],
    answers: List[str],
    processor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """批量计算任务损失

    参数:
        logits: (batch, seq_len, vocab_size) - 模型输出的logits
        answer_positions_list: List of (start, end) for each sample
        answers: List of answer strings
        processor: tokenizer所在的processor
        attention_mask: (batch, seq_len) attention mask，用于计算每个样本的实际长度

    返回:
        task_loss: torch.Tensor - 平均交叉熵损失
    """
    # print(processor.tokenizer.decode(logits.argmax(dim=-1)[0]))
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    vocab_size = logits.shape[-1]

    total_loss = torch.tensor(0.0, device=logits.device)
    valid_samples = 0

    # 计算每个样本的实际长度（用于转换负索引）
    if attention_mask is not None:
        actual_lengths = attention_mask.sum(dim=1).tolist()  # List[int]
    else:
        actual_lengths = [seq_len] * batch_size  # 没有mask时假设全部有效

    # DEBUG: 打印整体信息
    #print(f"[DEBUG task_loss] batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    #print(f"[DEBUG task_loss] answer_positions_list={answer_positions_list}")
    #print(f"[DEBUG task_loss] answers={answers}")

    for i in range(batch_size):
        answer = answers[i].capitalize()  # 对齐LLM输出格式（首字母大写）
        answer_start, answer_end = answer_positions_list[i]

        #print(f"[DEBUG task_loss] Sample {i}: answer='{answer}', pos=({answer_start}, {answer_end})")

        if answer_start is None:
            #print(f"[DEBUG task_loss] Sample {i}: SKIP - answer_start is None")
            continue

        # 转换负索引：使用每个样本的实际长度
        orig_start, orig_end = answer_start, answer_end
        sample_len = int(actual_lengths[i])
        if answer_start < 0:
            answer_start = sample_len + answer_start
        if answer_end < 0:
            answer_end = sample_len + answer_end

        if orig_start != answer_start or orig_end != answer_end:
            pass
            #print(f"[DEBUG task_loss] Sample {i}: converted pos ({orig_start}, {orig_end}) -> ({answer_start}, {answer_end}) [sample_len={sample_len}]")

        # 验证位置有效性
        if answer_start < 0 or answer_end >= seq_len or answer_start > answer_end:
            #print(f"[DEBUG task_loss] Sample {i}: SKIP - invalid pos: start={answer_start}, end={answer_end}, seq_len={seq_len}")
            continue

        answer_token_ids_list = processor.tokenizer.encode(answer, add_special_tokens=False)
        if len(answer_token_ids_list) == 0:
            #print(f"[DEBUG task_loss] Sample {i}: SKIP - empty tokenized answer")
            continue

        max_id = max(answer_token_ids_list)
        min_id = min(answer_token_ids_list)
        if max_id >= vocab_size or min_id < 0:
            #print(f"[DEBUG task_loss] Sample {i}: SKIP - token id out of range: min={min_id}, max={max_id}, vocab_size={vocab_size}")
            continue

        answer_token_ids = torch.tensor(answer_token_ids_list, device=logits.device, dtype=torch.long)

        # 从 answer_start-1 开始取logits (因为要预测下一个token)
        logits_for_answer = logits[i:i+1, answer_start-1:answer_end, :]

        expected_len = len(answer_token_ids)
        actual_len = logits_for_answer.shape[1]

        #print(f"[DEBUG task_loss] Sample {i}: token_ids={answer_token_ids_list}, expected_len={expected_len}, actual_len={actual_len}")
        #print(f"[DEBUG task_loss] Sample {i}: logits slice [{answer_start-1}:{answer_end}], shape={logits_for_answer.shape}")

        if actual_len != expected_len:
            #print(f"[DEBUG task_loss] Sample {i}: SKIP - length mismatch: expected={expected_len}, actual={actual_len}")
            continue

        loss = F.cross_entropy(
            logits_for_answer.reshape(-1, vocab_size),
            answer_token_ids,
            reduction='mean'
        )

        total_loss = total_loss + loss
        valid_samples += 1

    #print(f"[DEBUG task_loss] valid_samples={valid_samples}, total_loss={total_loss.item():.4f}")

    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        #print(f"[DEBUG task_loss] WARNING: No valid samples! Returning zero loss.")
        return total_loss


def get_current_sparsity_weight(config: Dict, current_step: int, total_steps: int) -> float:
    """根据训练进度获取当前稀疏权重"""
    sparsity_weight = config["method_settings"]["sparsity_weight"]

    # 检查是否启用warmup
    sparsity_warmup_enable = config["method_settings"]["sparsity_warmup_enable"]
    if not sparsity_warmup_enable:
        return sparsity_weight

    sparsity_weight_max = config["method_settings"]["sparsity_weight_max"]
    sparsity_warmup_ratio = config["method_settings"]["sparsity_warmup_ratio"]

    if total_steps == 0:
        return sparsity_weight

    progress = current_step / total_steps

    if progress < sparsity_warmup_ratio:
        warmup_progress = progress / sparsity_warmup_ratio
        current_weight = sparsity_weight + warmup_progress * (sparsity_weight_max - sparsity_weight)
    else:
        current_weight = sparsity_weight_max

    return current_weight


# ==================== Multi-Layer Hook工具函数 ====================

def create_layer_pruning_modifier(
    pruner,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False,
    question_mask: Optional[torch.Tensor] = None
) -> Callable:
    """创建层剪枝的modifier函数（用于hook）

    参数:
        pruner: VisionPrunerHead实例（该层的剪枝器）
        vision_positions: (start, end) - vision tokens在序列中的位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        mask_collector: 可选的列表，用于收集soft_mask（用于计算sparsity loss）
        use_attn_residual: 是否启用attention residual
        question_mask: (batch, n_text) - True表示有效token，False表示padding

    返回:
        modifier函数，签名为 (hidden_states, attention_mask) -> (new_hidden, new_mask)
    """

    # 用于存储attention weights的容器
    attention_storage = {'attn_weights': None}

    def modifier(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Hook函数，在layer执行前调用"""
        # === Step 1: 提取vision token hidden states ===
        v_start, v_end = vision_positions
        vision_hidden = hidden_states[:, v_start:v_end+1, :]  # (batch, n_vision, d_model)

        # === Step 2: 计算text→vision attention（如果启用） ===
        text_to_vision_attn = None
        if use_attn_residual:
            if attention_storage['attn_weights'] is not None:
                attn_weights = attention_storage['attn_weights']

                # 提取text positions (排除vision部分)
                seq_len = hidden_states.shape[1]
                text_indices = list(range(0, v_start)) + list(range(v_end+1, seq_len))
                vision_indices = list(range(v_start, v_end+1))

                if len(text_indices) > 0:
                    text_to_vision = attn_weights[:, :, text_indices, :][:, :, :, vision_indices]
                    text_to_vision_attn = text_to_vision.mean(dim=(1, 2))

                attention_storage['attn_weights'] = None

        # === Step 3: 调用pruner生成soft_mask ===
        # 将question_mask转换为key_padding_mask格式（True表示要mask掉的位置）
        key_padding_mask = None
        if question_mask is not None:
            key_padding_mask = ~question_mask  # 反转：False变True（padding位置要mask）

        with torch.enable_grad():
            soft_mask = pruner(vision_hidden, question_embeddings,
                             text_to_vision_attn=text_to_vision_attn,
                             key_padding_mask=key_padding_mask)

        # === Step 4: 收集mask ===
        if mask_collector is not None:
            mask_collector.append(soft_mask)

        # === Step 5: 应用mask ===
        soft_mask = soft_mask.to(vision_hidden.dtype)

        # 使用自定义autograd函数，在backward时裁剪soft_mask的梯度
        # 这样可以避免vision_hidden的大值（~200）导致梯度爆炸
        scaled_vision = ScaleVisionWithClippedGrad.apply(vision_hidden, soft_mask, 1.0)

        # === Step 6: 用 torch.cat 构建 new_hidden（保持梯度流）===
        # 不能用 clone + in-place 赋值，那样会断梯度
        new_hidden = torch.cat([
            hidden_states[:, :v_start, :],
            scaled_vision,
            hidden_states[:, v_end+1:, :]
        ], dim=1)

        # === DEBUG: 检查梯度流 ===
        # 注意：这个调试输出可以通过环境变量 DEBUG_ADV_GRAD=1 启用
        import os
        if os.environ.get("DEBUG_ADV_GRAD") == "1":
            print(f"\n[DEBUG hook] === Layer Pruning Hook 梯度检查 ===")
            print(f"[DEBUG hook] soft_mask.requires_grad: {soft_mask.requires_grad}")
            print(f"[DEBUG hook] soft_mask.grad_fn: {soft_mask.grad_fn}")
            print(f"[DEBUG hook] scaled_vision.requires_grad: {scaled_vision.requires_grad}")
            print(f"[DEBUG hook] scaled_vision.grad_fn: {scaled_vision.grad_fn}")
            print(f"[DEBUG hook] new_hidden.requires_grad: {new_hidden.requires_grad}")
            print(f"[DEBUG hook] new_hidden.grad_fn: {new_hidden.grad_fn}")

        return new_hidden, attention_mask

    if use_attn_residual:
        return modifier, attention_storage
    else:
        return modifier, None


def register_multi_layer_hooks(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False,
    question_mask: Optional[torch.Tensor] = None
) -> List[Any]:
    """在多个LLM层注册剪枝hooks

    参数:
        backbone: LLaVA backbone实例
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (start, end) - vision tokens位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        mask_collector: 可选的列表，用于收集soft_mask
        use_attn_residual: 是否启用attention residual
        question_mask: (batch, n_text) - True表示有效token，False表示padding

    返回:
        handles: hook handle列表
    """
    if use_attn_residual:
        return register_multi_layer_hooks_v2(
            backbone, layer_pruners, vision_positions, question_embeddings,
            mask_collector, use_attn_residual
        )

    handles = []

    for layer_idx in layer_pruners.get_all_layers():
        pruner = layer_pruners.get_pruner(layer_idx)
        modifier, _ = create_layer_pruning_modifier(
            pruner, vision_positions, question_embeddings, mask_collector,
            use_attn_residual=False, question_mask=question_mask
        )
        target_layer = backbone.model.model.language_model.layers[layer_idx]

        def hook_fn(module, args, mod=modifier):
            hidden_states = args[0]
            attention_mask = args[1] if len(args) > 1 else None
            new_hidden, new_mask = mod(hidden_states, attention_mask)

            new_args = list(args)
            new_args[0] = new_hidden
            if len(new_args) > 1:
                new_args[1] = new_mask
            return tuple(new_args)

        handle = target_layer.register_forward_pre_hook(hook_fn)
        handles.append(handle)

    return handles


def register_multi_layer_hooks_batch(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False,
    question_mask: Optional[torch.Tensor] = None
) -> List[Any]:
    """批量版本的多层剪枝hooks注册

    与单样本版本相同，因为hooks本身就支持batch处理。
    """
    return register_multi_layer_hooks(
        backbone, layer_pruners, vision_positions, question_embeddings,
        mask_collector, use_attn_residual, question_mask
    )


def register_multi_layer_hooks_v2(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False
) -> List[Any]:
    """在多个LLM层注册剪枝hooks（V2版本 - 更稳健）"""
    handles = []
    v_start, v_end = vision_positions

    attn_impl = backbone.model.model.language_model.config._attn_implementation
    use_eager_attn = (attn_impl == "eager")

    for layer_idx in layer_pruners.get_all_layers():
        pruner = layer_pruners.get_pruner(layer_idx)
        target_layer = backbone.model.model.language_model.layers[layer_idx]
        self_attn = target_layer.self_attn

        layer_context = {
            'attn_weights': None,
            'input_hidden_states': None
        }

        if use_attn_residual:
            if use_eager_attn:
                def create_attn_post_hook(ctx):
                    def attn_post_hook(module, args, kwargs, output):
                        attn_output, attn_weights = output
                        ctx['attn_weights'] = attn_weights
                        return output
                    return attn_post_hook

                attn_handle = self_attn.register_forward_hook(
                    create_attn_post_hook(layer_context),
                    with_kwargs=True
                )
                handles.append(attn_handle)
            else:
                def create_pre_hook(ctx):
                    def pre_hook(module, args, kwargs):
                        if len(args) > 0:
                            hidden_states = args[0]
                        else:
                            hidden_states = kwargs.get('hidden_states')
                        ctx['input_hidden_states'] = hidden_states
                        return args, kwargs
                    return pre_hook

                pre_handle = target_layer.register_forward_pre_hook(
                    create_pre_hook(layer_context),
                    with_kwargs=True
                )
                handles.append(pre_handle)

        def create_layer_post_hook(ctx, pruner_ref, layer_ref, attn_ref, layer_idx_ref, collector_ref, use_attn_ref, is_eager):
            def post_hook(module, args, kwargs, output):
                hidden_states_out = output
                vision_hidden = hidden_states_out[:, v_start:v_end+1, :]

                text_to_vision_attn = None
                if use_attn_ref:
                    attn_weights = None

                    if is_eager and ctx['attn_weights'] is not None:
                        attn_weights = ctx['attn_weights']
                        ctx['attn_weights'] = None
                    elif not is_eager and ctx['input_hidden_states'] is not None:
                        with torch.no_grad():
                            hidden_states_in = ctx['input_hidden_states']
                            normed_input = layer_ref.input_layernorm(hidden_states_in)

                            batch, seq_len, d_model = normed_input.shape
                            num_heads = attn_ref.config.num_attention_heads
                            head_dim = attn_ref.head_dim

                            Q = attn_ref.q_proj(normed_input)
                            K = attn_ref.k_proj(normed_input)

                            Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
                            K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

                            scaling = 1.0 / (head_dim ** 0.5)
                            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scaling
                            attn_weights = torch.softmax(attn_weights, dim=-1)

                        ctx['input_hidden_states'] = None

                    if attn_weights is not None:
                        seq_len = attn_weights.shape[-1]
                        q_start = v_end + 1
                        if q_start < seq_len:
                            q_to_v = attn_weights[:, :, q_start:, v_start:v_end+1]
                            text_to_vision_attn = q_to_v.mean(dim=(1, 2))

                with torch.enable_grad():
                    soft_mask = pruner_ref(
                        vision_hidden,
                        question_embeddings,
                        text_to_vision_attn=text_to_vision_attn
                    )

                if collector_ref is not None:
                    collector_ref.append(soft_mask)

                soft_mask = soft_mask.to(hidden_states_out.dtype)
                # 使用自定义autograd函数，在backward时裁剪soft_mask的梯度
                scaled_vision = ScaleVisionWithClippedGrad.apply(vision_hidden, soft_mask, 1.0)
                # 用 torch.cat 构建 new_hidden（保持梯度流）
                new_hidden = torch.cat([
                    hidden_states_out[:, :v_start, :],
                    scaled_vision,
                    hidden_states_out[:, v_end+1:, :]
                ], dim=1)

                return new_hidden

            return post_hook

        layer_handle = target_layer.register_forward_hook(
            create_layer_post_hook(
                layer_context, pruner, target_layer, self_attn,
                layer_idx, mask_collector, use_attn_residual, use_eager_attn
            ),
            with_kwargs=True
        )
        handles.append(layer_handle)

    return handles


def remove_hooks(handles: List[Any]):
    """移除所有注册的hooks"""
    for handle in handles:
        handle.remove()


def replace_vision_tokens_in_embeddings(
    full_embeddings: torch.Tensor,
    original_vision_pos: Tuple[int, int],
    merged_vision_features: torch.Tensor,
    original_attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
    """将合并后的vision features替换回完整embeddings

    参数:
        full_embeddings: (1, seq_len, d) - 原始完整序列
        original_vision_pos: (start, end) - 原始vision位置
        merged_vision_features: (1, M, d) - 合并后的vision features

    返回:
        new_embeddings: (1, new_seq_len, d)
        new_vision_pos: (new_start, new_end)
        new_attention_mask: (1, new_seq_len)
    """
    v_start, v_end = original_vision_pos

    new_embeddings = torch.cat([
        full_embeddings[:, :v_start, :],
        merged_vision_features,
        full_embeddings[:, v_end+1:, :]
    ], dim=1)

    new_v_start = v_start
    new_v_end = v_start + merged_vision_features.shape[1] - 1

    before_mask = original_attention_mask[:, :v_start]
    after_mask = original_attention_mask[:, v_end+1:]
    vision_mask = torch.ones(
        merged_vision_features.shape[:2],
        device=merged_vision_features.device,
        dtype=original_attention_mask.dtype
    )
    new_attention_mask = torch.cat([before_mask, vision_mask, after_mask], dim=1)

    return new_embeddings, (new_v_start, new_v_end), new_attention_mask


# ==================== Hard Pruning Hook工具函数 ====================

class HardPruningContext:
    """Hard Pruning的上下文对象"""

    def __init__(self, initial_vision_positions: Tuple[int, int]):
        self.vision_positions = initial_vision_positions
        self.pruning_stats = []
        self._is_decode_mode = False
        self.layer_pruners = None
        self.question_embeddings = None
        self.threshold = 0.5

    def update_positions(self, new_positions: Tuple[int, int], layer_idx: int,
                         original_count: int, kept_count: int):
        self.vision_positions = new_positions
        self.pruning_stats.append({
            'layer_idx': layer_idx,
            'original_count': original_count,
            'kept_count': kept_count,
            'pruned_count': original_count - kept_count,
            'keep_ratio': kept_count / original_count if original_count > 0 else 0.0
        })

    def get_positions(self) -> Tuple[int, int]:
        return self.vision_positions

    def get_stats(self) -> List[Dict]:
        return self.pruning_stats

    def set_decode_mode(self, is_decode: bool):
        self._is_decode_mode = is_decode

    def is_decode_mode(self) -> bool:
        return self._is_decode_mode

    def should_prune_layer(self, layer_idx: int) -> bool:
        if self.layer_pruners is None:
            return False
        return layer_idx in self.layer_pruners.get_all_layers()


def register_hard_pruning_at_model_level(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[Callable, HardPruningContext]:
    """在LlamaModel层面注册hard pruning"""
    context = HardPruningContext(vision_positions)
    context.layer_pruners = layer_pruners
    context.question_embeddings = question_embeddings
    context.threshold = threshold

    llama_model = backbone.model.model.language_model
    original_forward = llama_model.forward

    def wrapped_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        **kwargs
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        seq_len = inputs_embeds.shape[1]
        if seq_len < 50:
            context.set_decode_mode(True)
        else:
            context.set_decode_mode(False)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if not context.is_decode_mode() and context.should_prune_layer(layer_idx):
                old_seq_len = hidden_states.shape[1]

                hidden_states, kept_position_indices = apply_hard_pruning_to_hidden_states(
                    hidden_states,
                    context,
                    layer_idx
                )

                new_seq_len = hidden_states.shape[1]

                if new_seq_len != old_seq_len and kept_position_indices is not None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    original_position_ids = position_ids[0]

                    if kept_position_indices.device != original_position_ids.device:
                        kept_position_indices = kept_position_indices.to(original_position_ids.device)

                    kept_position_ids = original_position_ids[kept_position_indices]
                    position_ids = kept_position_ids.unsqueeze(0)
                    cache_position = kept_position_ids
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    if attention_mask is not None:
                        new_attention_mask = attention_mask[:, kept_position_indices]
                        attention_mask = new_attention_mask

                    temp_inputs_embeds = torch.zeros(1, new_seq_len, hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)

                    causal_mask = create_causal_mask(
                        config=self.config,
                        input_embeds=temp_inputs_embeds,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                    )

        hidden_states = self.norm(hidden_states)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    import types
    llama_model.forward = types.MethodType(wrapped_forward, llama_model)

    def restore_fn():
        llama_model.forward = original_forward

    return restore_fn, context


def apply_hard_pruning_to_hidden_states(
    hidden_states: torch.Tensor,
    context: HardPruningContext,
    layer_idx: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """对hidden_states应用hard pruning"""
    v_start, v_end = context.get_positions()
    seq_len = hidden_states.shape[1]

    if v_start >= seq_len or v_end >= seq_len:
        return hidden_states, None

    vision_hidden = hidden_states[:, v_start:v_end+1, :]
    n_vision = vision_hidden.shape[1]

    if n_vision == 0:
        return hidden_states, None

    pruner = context.layer_pruners.get_pruner(layer_idx)

    with torch.no_grad():
        soft_mask = pruner(vision_hidden, context.question_embeddings, use_gumbel=False)

    hard_mask = (soft_mask > context.threshold).float()
    kept_indices = torch.nonzero(hard_mask[0] > 0.5).squeeze(-1)

    if len(kept_indices) == 0:
        max_idx = soft_mask[0].argmax()
        kept_indices = max_idx.unsqueeze(0)

    kept_vision = vision_hidden[:, kept_indices, :]

    new_hidden = torch.cat([
        hidden_states[:, :v_start, :],
        kept_vision,
        hidden_states[:, v_end+1:, :]
    ], dim=1)

    device = hidden_states.device

    if kept_indices.device != device:
        kept_indices = kept_indices.to(device)

    text_before_indices = torch.arange(0, v_start, device=device)
    vision_kept_indices = v_start + kept_indices
    text_after_indices = torch.arange(v_end+1, seq_len, device=device)

    kept_position_indices = torch.cat([
        text_before_indices,
        vision_kept_indices,
        text_after_indices
    ])

    n_kept = len(kept_indices)
    new_v_start = v_start
    new_v_end = v_start + n_kept - 1
    context.update_positions((new_v_start, new_v_end), layer_idx, n_vision, n_kept)

    return new_hidden, kept_position_indices

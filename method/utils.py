"""Vision Token Pruning - 工具函数

包含embedding处理、mask应用、hook注册等通用辅助函数。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Callable
from PIL import Image


def get_vision_features_and_question_embedding(
    backbone,
    image: Image.Image,
    question: str,
    answer: Optional[str] = None,
    max_vision_tokens: int = 1800
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """获取vision features和question embeddings
    
    返回:
        vision_features: (1, num_vision_tokens, d_v) 或 None
        question_embeddings: (1, num_question_tokens, d_t) 或 None
        vision_positions: (start, end) 或 None
        answer_positions: (start, end) 或 None
    """
    with torch.no_grad():
        # 使用新API: preprocess 替代 get_embeddings
        emb_info = backbone.preprocess(image, question, answer)
        embeddings = emb_info['embeddings']
        vision_start, vision_end = emb_info['vision_token_positions']
        answer_positions = emb_info.get('answer_token_positions', None)
        
        num_vision_tokens = vision_end - vision_start + 1
        if num_vision_tokens > max_vision_tokens:
            del embeddings
            return None, None, None, None
        
        # 不再强制移动设备，保持在 embeddings 的原始设备上
        vision_features = embeddings[:, vision_start:vision_end+1, :]
        question_embeddings = embeddings[:, :vision_start, :]
        
        # 立即释放原始embeddings
        del embeddings
        torch.cuda.empty_cache()
        
        return vision_features, question_embeddings, (vision_start, vision_end), answer_positions


def apply_mask_to_embeddings(
    embeddings: torch.Tensor,
    vision_positions: Tuple[int, int],
    soft_mask: torch.Tensor
) -> torch.Tensor:
    """应用soft mask到embeddings
    
    参数:
        embeddings: (1, seq_len, hidden_dim)
        vision_positions: (start, end) 
        soft_mask: (1, num_vision_tokens) 或 (num_vision_tokens,)
    """
    vision_start, vision_end = vision_positions
    num_vision_tokens = vision_end - vision_start + 1
    
    # 确保soft_mask的形状正确
    if soft_mask.dim() == 1:
        soft_mask = soft_mask.unsqueeze(0)  # (num_vision_tokens,) -> (1, num_vision_tokens)
    
    # 检查维度是否匹配
    assert soft_mask.shape[1] == num_vision_tokens, \
        f"soft_mask shape {soft_mask.shape} doesn't match num_vision_tokens {num_vision_tokens}"
    
    masked_embeddings = embeddings.clone()
    mask_expanded = soft_mask.to(embeddings.device).unsqueeze(-1)  # (1, N, 1)

    # 应用 mask 缩放 (非 inplace 操作)
    masked_embeddings[:, vision_start:vision_end+1] = masked_embeddings[:, vision_start:vision_end+1] * mask_expanded

    return masked_embeddings


def batch_forward_with_masks(
    backbone,
    image: Image.Image,
    question: str,
    answer: str,
    masks: List[torch.Tensor],
    get_hidden_states: bool = True
) -> Dict[str, Any]:
    """使用多个masks进行batch forward
    
    参数:
        backbone: Backbone模型
        image: PIL Image
        question: 问题文本
        answer: 答案文本
        masks: mask列表
        get_hidden_states: 是否获取hidden states
    """
    # 重新获取embeddings（用完会立即释放）
    # 使用新API: preprocess 替代 get_embeddings
    emb_info = backbone.preprocess(image, question, answer)
    base_embeddings = emb_info['embeddings']
    vision_positions = emb_info['vision_token_positions']
    answer_positions = emb_info['answer_token_positions']
    
    seq_len = base_embeddings.shape[1]
    attention_mask = torch.ones(len(masks), seq_len, device=backbone.device, dtype=torch.long)
    
    batch_embeddings = []
    for i, mask in enumerate(masks):
        masked_emb = apply_mask_to_embeddings(
            base_embeddings,
            vision_positions,
            mask.to(backbone.device)
        )
        batch_embeddings.append(masked_emb)
    
    batch_embeddings = torch.cat(batch_embeddings, dim=0)

    # 使用新API: forward 替代 forward_from_embeddings
    result = backbone.forward(
        embeddings=batch_embeddings,
        attention_mask=attention_mask,
        output_hidden_states=get_hidden_states
    )
    
    return {
        'logits': result['logits'],
        'all_hidden_states': result.get('all_hidden_states', None),
        'vision_positions': vision_positions,
        'answer_positions': answer_positions
    }


def extract_target_hidden_states(
    all_hidden_states: tuple,
    target_positions: Tuple[int, int],
    target_layer_indices: List[int],
    batch_size: int = 1
) -> List[torch.Tensor]:
    """提取指定层和位置的hidden states"""
    start, end = target_positions
    selected_hidden_states = []
    
    for layer_idx in target_layer_indices:
        hidden = all_hidden_states[layer_idx]
        # 不再强制移动设备，保持在原始设备上
        hidden_slice = hidden[:, start:None if end+1==0 else end+1, :]
        selected_hidden_states.append(hidden_slice)
    
    return selected_hidden_states


def compute_task_loss(
    logits: torch.Tensor,
    answer_positions: Tuple[int, int],
    answer: str,
    processor
) -> torch.Tensor:
    """计算任务损失（预测answer的交叉熵损失）

    注意：answer_positions 使用负索引表示相对于序列末尾的位置。
    例如 (-5, -2) 表示从倒数第5个到倒数第2个token。

    参数:
        logits: (batch, seq_len, vocab_size) - 模型输出的logits
        answer_positions: (start, end) - answer在序列中的位置（支持负索引）
        answer: str - 答案文本
        processor: tokenizer所在的processor

    返回:
        task_loss: torch.Tensor - 交叉熵损失

    抛出:
        ValueError: 当answer为空、token id越界、或位置不匹配时
    """
    answer_start, answer_end = answer_positions
    seq_len = logits.shape[1]

    # 将负索引转换为正索引
    if answer_start < 0:
        answer_start = seq_len + answer_start
    if answer_end < 0:
        answer_end = seq_len + answer_end

    # 验证位置有效性
    if answer_start < 0 or answer_end >= seq_len or answer_start > answer_end:
        raise ValueError(
            f"answer位置无效: start={answer_start}, end={answer_end}, seq_len={seq_len}。"
            f"原始位置: {answer_positions}"
        )

    answer_token_ids_list = processor.tokenizer.encode(answer, add_special_tokens=False)
    if len(answer_token_ids_list) == 0:
        raise ValueError(f"answer '{answer}' 被分词后长度为0，无法计算task loss")

    vocab_size = logits.shape[-1]
    max_id = max(answer_token_ids_list)
    min_id = min(answer_token_ids_list)
    if max_id >= vocab_size or min_id < 0:
        # 防止越界标签导致CUDA断言
        raise ValueError(
            f"answer '{answer}' 的token id越界，范围[{min_id}, {max_id}]，"
            f"但vocab size={vocab_size}"
        )

    answer_token_ids = torch.tensor(answer_token_ids_list, device=logits.device, dtype=torch.long)

    # 从 answer_start-1 开始取logits，因为我们需要用第i-1个位置的logits预测第i个token
    # 例如：要预测answer的第一个token，需要用answer_start-1位置的logits
    logits_for_answer = logits[:, answer_start-1:answer_end, :]

    expected_len = len(answer_token_ids)
    actual_len = logits_for_answer.shape[1]
    if actual_len != expected_len:
        raise ValueError(
            f"logits长度与answer token数不匹配: "
            f"logits_for_answer.shape[1]={actual_len}, len(answer_token_ids)={expected_len}。"
            f"answer='{answer}', answer_positions=({answer_start}, {answer_end}), seq_len={seq_len}"
        )

    loss = F.cross_entropy(
        logits_for_answer.reshape(-1, logits_for_answer.shape[-1]),
        answer_token_ids.repeat(logits_for_answer.shape[0]),
        reduction='mean'
    )

    return loss


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


def update_generator_temperature(generator, config: Dict, current_step: int, total_steps: int):
    """更新Generator的temperature"""
    if not config["method_settings"]["gen_temperature_anneal"] or total_steps == 0:
        return
    
    gen_temperature = config["method_settings"]["gen_temperature"]
    gen_temperature_min = config["method_settings"]["gen_temperature_min"]
    gen_temperature_anneal_rate = config["method_settings"]["gen_temperature_anneal_rate"]
    
    progress = current_step / total_steps
    
    if progress < gen_temperature_anneal_rate:
        anneal_progress = progress / gen_temperature_anneal_rate
        current_temp = gen_temperature - anneal_progress * (gen_temperature - gen_temperature_min)
    else:
        current_temp = gen_temperature_min
    
    generator.set_temperature(current_temp)


def get_target_token_num(config: Dict, total_tokens: int) -> int:
    """计算目标token数"""
    if config["method_settings"]["use_token_num_target"]:
        return config["method_settings"]["target_token_num"]
    else:
        target_sparsity = config["method_settings"]["target_sparsity"]
        return int(total_tokens * (1.0 - target_sparsity))


# ==================== Multi-Layer Hook工具函数 ====================

def create_layer_pruning_modifier(
    pruner,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False
) -> Callable:
    """创建层剪枝的modifier函数（用于hook）

    参数:
        pruner: VisionPrunerHead实例（该层的剪枝器）
        vision_positions: (start, end) - vision tokens在序列中的位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        mask_collector: 可选的列表，用于收集soft_mask（用于计算sparsity loss）
        use_attn_residual: 是否启用attention residual

    返回:
        modifier函数，签名为 (hidden_states, attention_mask) -> (new_hidden, new_mask)
    """

    # 用于存储attention weights的容器
    attention_storage = {'attn_weights': None}

    def modifier(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Hook函数，在layer执行前调用

        参数:
            hidden_states: (batch, seq_len, d_model) - 输入到当前层的hidden states
            attention_mask: (batch, seq_len) - attention mask

        返回:
            (new_hidden_states, new_attention_mask)
        """
        # === Step 1: 提取vision token hidden states ===
        v_start, v_end = vision_positions
        vision_hidden = hidden_states[:, v_start:v_end+1, :]  # (batch, n_vision, d_model)

        # === Step 2: 计算text→vision attention（如果启用） ===
        text_to_vision_attn = None
        if use_attn_residual:
            if attention_storage['attn_weights'] is not None:
                attn_weights = attention_storage['attn_weights']  # (batch, num_heads, seq_len, seq_len)

                print(f"[DEBUG] Attention captured! Shape: {attn_weights.shape}")

                # 提取text positions (排除vision部分)
                seq_len = hidden_states.shape[1]
                # text indices: [0, v_start) + (v_end, seq_len)
                text_indices = list(range(0, v_start)) + list(range(v_end+1, seq_len))
                vision_indices = list(range(v_start, v_end+1))

                if len(text_indices) > 0:
                    # 提取text→vision的attention: attn[text, vision]
                    # attn_weights[batch, heads, from, to]
                    # 我们需要: text_tokens → vision_tokens
                    text_to_vision = attn_weights[:, :, text_indices, :][:, :, :, vision_indices]
                    # (batch, num_heads, n_text, n_vision)

                    # 平均: 跨heads和text tokens维度
                    text_to_vision_attn = text_to_vision.mean(dim=(1, 2))  # (batch, n_vision)

                    print(f"[DEBUG] text_to_vision_attn computed: shape={text_to_vision_attn.shape}, "
                          f"range=[{text_to_vision_attn.min().item():.6f}, {text_to_vision_attn.max().item():.6f}], "
                          f"mean={text_to_vision_attn.mean().item():.6f}")
                else:
                    print(f"[DEBUG] No text tokens found (all tokens are vision)")

                # 清空storage，避免影响下一层
                attention_storage['attn_weights'] = None
            else:
                print(f"[DEBUG] Attention NOT captured! storage['attn_weights'] is None")

        # === Step 3: 调用pruner生成soft_mask（传入attention） ===
        with torch.enable_grad():  # 确保梯度开启（即使在eval模式）
            soft_mask = pruner(vision_hidden, question_embeddings, text_to_vision_attn=text_to_vision_attn)  # (batch, n_vision)

        # === Step 4: 收集mask（用于sparsity loss计算） ===
        if mask_collector is not None:
            mask_collector.append(soft_mask)

        # === Step 5: 应用mask（逐元素乘法） ===
        # 等价于同时缩放Q, K, V，影响attention scores
        # 确保soft_mask与vision_hidden的dtype一致
        soft_mask = soft_mask.to(vision_hidden.dtype)
        scaled_vision = vision_hidden * soft_mask.unsqueeze(-1)  # (batch, n_vision, d_model)

        # === Step 6: 替换到完整hidden_states中 ===
        new_hidden = hidden_states.clone()
        new_hidden[:, v_start:v_end+1, :] = scaled_vision

        return new_hidden, attention_mask

    # 如果启用attention residual，返回modifier和attention_storage
    # 否则只返回modifier
    if use_attn_residual:
        # 返回modifier和storage的引用，以便外部可以填充attention weights
        return modifier, attention_storage
    else:
        return modifier, None


def register_multi_layer_hooks(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False
) -> List[Any]:
    """在多个LLM层注册剪枝hooks（旧版本，建议使用 register_multi_layer_hooks_v2）

    参数:
        backbone: LLaVA backbone实例
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (start, end) - vision tokens位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        mask_collector: 可选的列表，用于收集soft_mask（用于计算sparsity loss）
        use_attn_residual: 是否启用attention residual（此版本不支持，将被忽略）

    返回:
        handles: hook handle列表（用于后续清理）
    """
    # 如果启用attention residual，使用新版本
    if use_attn_residual:
        return register_multi_layer_hooks_v2(
            backbone, layer_pruners, vision_positions, question_embeddings,
            mask_collector, use_attn_residual
        )

    handles = []

    for layer_idx in layer_pruners.get_all_layers():
        # 1. 获取该层的pruner
        pruner = layer_pruners.get_pruner(layer_idx)

        # 2. 创建modifier函数
        modifier, _ = create_layer_pruning_modifier(
            pruner, vision_positions, question_embeddings, mask_collector, use_attn_residual=False
        )

        # 3. 获取target layer
        target_layer = backbone.model.model.language_model.layers[layer_idx]

        # 4. 在整个Layer上注册pre-hook来应用pruning
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


def register_multi_layer_hooks_v2(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False
) -> List[Any]:
    """在多个LLM层注册剪枝hooks（V2版本 - 更稳健）

    新方案：使用 Pre-hook + Post-hook 组合
    - Pre-hook: 保存Layer输入的hidden_states
    - Post-hook: 在Layer执行完后，手动计算attention，调用pruner，应用mask

    这样可以确保：
    1. attention在Layer真正执行后计算（基于layernorm后的输入）
    2. mask应用在Layer输出上（已经完成了attention计算）
    3. 时序清晰，不会出现attention未计算就读取的问题

    参数:
        backbone: LLaVA backbone实例
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (start, end) - vision tokens位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        mask_collector: 可选的列表，用于收集soft_mask（用于计算sparsity loss）
        use_attn_residual: 是否启用attention residual

    返回:
        handles: hook handle列表（用于后续清理）
    """
    handles = []
    v_start, v_end = vision_positions

    for layer_idx in layer_pruners.get_all_layers():
        # 1. 获取该层的pruner和target layer
        pruner = layer_pruners.get_pruner(layer_idx)
        target_layer = backbone.model.model.language_model.layers[layer_idx]
        self_attn = target_layer.self_attn

        # 2. 创建该层的context（存储输入hidden_states）
        layer_context = {'input_hidden_states': None}

        # 3. Pre-hook: 保存输入
        def create_pre_hook(ctx):
            def pre_hook(module, args, kwargs):
                if len(args) > 0:
                    hidden_states = args[0]
                else:
                    hidden_states = kwargs.get('hidden_states')
                ctx['input_hidden_states'] = hidden_states
                return args, kwargs
            return pre_hook

        # 4. Post-hook: 计算attention + 剪枝 + 应用mask
        def create_post_hook(ctx, pruner_ref, layer_ref, attn_ref, layer_idx_ref, collector_ref, use_attn_ref):
            def post_hook(module, args, kwargs, output):
                hidden_states_out = output  # Layer输出
                hidden_states_in = ctx['input_hidden_states']  # Layer输入

                # 提取vision hidden states
                vision_hidden = hidden_states_out[:, v_start:v_end+1, :]

                text_to_vision_attn = None
                if use_attn_ref:
                    # 手动计算attention weights
                    # 使用Layer输入（经过input_layernorm后的）
                    with torch.no_grad():
                        normed_input = layer_ref.input_layernorm(hidden_states_in)

                        batch, seq_len, d_model = normed_input.shape
                        num_heads = attn_ref.config.num_attention_heads
                        head_dim = attn_ref.head_dim

                        # 计算Q, K
                        Q = attn_ref.q_proj(normed_input)
                        K = attn_ref.k_proj(normed_input)

                        # Reshape to (batch, num_heads, seq_len, head_dim)
                        Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
                        K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

                        # 计算attention scores
                        scaling = 1.0 / (head_dim ** 0.5)
                        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scaling
                        attn_weights = torch.softmax(attn_weights, dim=-1)
                        # (batch, num_heads, seq_len, seq_len)

                        # 提取question→vision的attention
                        # question位置: [v_end+1, seq_len)，但我们用传入的question_embeddings来确定范围
                        # 这里简化：使用vision之后的所有token作为question
                        q_start = v_end + 1
                        q_end = seq_len  # 包括answer部分

                        if q_start < seq_len:
                            question_indices = list(range(q_start, q_end))
                            vision_indices = list(range(v_start, v_end + 1))

                            # attn_weights[batch, heads, from, to]
                            # 提取 question → vision
                            q_to_v = attn_weights[:, :, question_indices, :][:, :, :, vision_indices]
                            # (batch, num_heads, n_question, n_vision)

                            # 平均跨heads和question tokens
                            text_to_vision_attn = q_to_v.mean(dim=(1, 2))  # (batch, n_vision)

                # 调用pruner生成soft_mask
                with torch.enable_grad():
                    soft_mask = pruner_ref(
                        vision_hidden,
                        question_embeddings,
                        text_to_vision_attn=text_to_vision_attn
                    )  # (batch, n_vision)

                # 收集mask
                if collector_ref is not None:
                    collector_ref.append(soft_mask)

                # 应用mask到vision部分
                soft_mask = soft_mask.to(hidden_states_out.dtype)
                new_hidden = hidden_states_out.clone()
                new_hidden[:, v_start:v_end+1, :] = vision_hidden * soft_mask.unsqueeze(-1)

                return new_hidden

            return post_hook

        # 5. 注册hooks
        pre_handle = target_layer.register_forward_pre_hook(
            create_pre_hook(layer_context),
            with_kwargs=True
        )
        post_handle = target_layer.register_forward_hook(
            create_post_hook(
                layer_context, pruner, target_layer, self_attn,
                layer_idx, mask_collector, use_attn_residual
            ),
            with_kwargs=True
        )

        handles.append(pre_handle)
        handles.append(post_handle)

    return handles


def remove_hooks(handles: List[Any]):
    """移除所有注册的hooks

    参数:
        handles: hook handle列表
    """
    for handle in handles:
        handle.remove()


def replace_vision_tokens_in_embeddings(
    full_embeddings: torch.Tensor,
    original_vision_pos: Tuple[int, int],
    merged_vision_features: torch.Tensor,
    original_attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[int, int], torch.Tensor]:
    """将合并后的vision features替换回完整embeddings

    用于Token Merge阶段：将576个vision tokens合并为~288个后，需要更新序列

    参数:
        full_embeddings: (1, seq_len, d) - 原始完整序列（如676 tokens）
        original_vision_pos: (start, end) - 原始vision位置（如(100, 675)）
        merged_vision_features: (1, M, d) - 合并后的vision features（如M=288）

    返回:
        new_embeddings: (1, new_seq_len, d) - 更新后的序列（如400 tokens）
        new_vision_pos: (new_start, new_end) - 更新后的vision位置（如(100, 387)）
        new_attention_mask: (1, new_seq_len) - 更新后的mask
    """
    v_start, v_end = original_vision_pos

    # 拼接: text_before + merged_vision + text_after
    new_embeddings = torch.cat([
        full_embeddings[:, :v_start, :],           # 图像前的文本
        merged_vision_features,                     # 合并后的vision tokens
        full_embeddings[:, v_end+1:, :]            # 图像后的文本
    ], dim=1)

    # 更新vision位置
    new_v_start = v_start
    new_v_end = v_start + merged_vision_features.shape[1] - 1

    # 更新attention_mask
    new_seq_len = new_embeddings.shape[1]
    # 基于原mask拼接，保持文本部分的padding掩码
    before_mask = original_attention_mask[:, :v_start]
    after_mask = original_attention_mask[:, v_end+1:]
    vision_mask = torch.ones(
        merged_vision_features.shape[:2],  # (batch, M)
        device=merged_vision_features.device,
        dtype=original_attention_mask.dtype
    )
    new_attention_mask = torch.cat([before_mask, vision_mask, after_mask], dim=1)

    return new_embeddings, (new_v_start, new_v_end), new_attention_mask


def update_temperature_for_all(
    token_merger,
    layer_pruners,
    config: Dict,
    current_step: int,
    total_steps: int
):
    """更新Token Merger和所有Layer Pruners的temperature

    Temperature annealing: 训练初期软（高temp），后期硬（低temp）

    参数:
        token_merger: LearnableTokenMerger实例（可选，可为None）
        layer_pruners: LayerSpecificPruner实例（可选，可为None）
        config: 配置字典
        current_step: 当前步数
        total_steps: 总步数
    """
    # 读取配置
    initial_temp = config['method_settings'].get('temperature', 1.0)
    final_temp = config['method_settings'].get('temperature_min', 0.1)
    anneal_rate = config['method_settings'].get('temperature_anneal_rate', 0.5)

    if total_steps == 0:
        current_temp = initial_temp
    else:
        progress = current_step / total_steps

        if progress < anneal_rate:
            # 线性annealing
            current_temp = initial_temp - (progress / anneal_rate) * (initial_temp - final_temp)
        else:
            # Annealing完成
            current_temp = final_temp

    # 更新token_merger
    if token_merger is not None:
        token_merger.set_temperature(current_temp)

    # 更新所有layer_pruners
    if layer_pruners is not None:
        layer_pruners.set_temperature(current_temp)


# ==================== Hard Pruning Hook工具函数 ====================

class HardPruningContext:
    """Hard Pruning的上下文对象，用于在多层之间传递动态变化的vision位置信息

    在多层hard pruning中，每层剪枝后vision token数量会减少，
    需要动态更新vision_positions供下一层使用。

    **重要**：Hard pruning只在prefill阶段执行，decode阶段不执行。
    """

    def __init__(self, initial_vision_positions: Tuple[int, int]):
        """初始化上下文

        参数:
            initial_vision_positions: (start, end) - 初始vision token位置
        """
        self.vision_positions = initial_vision_positions
        self.pruning_stats = []  # 记录每层的剪枝统计信息
        self._is_decode_mode = False  # 标记是否已进入decode阶段
        self.layer_pruners = None  # 存储layer_pruners引用
        self.question_embeddings = None  # 存储question_embeddings引用
        self.threshold = 0.5  # 剪枝阈值

    def update_positions(self, new_positions: Tuple[int, int], layer_idx: int,
                         original_count: int, kept_count: int):
        """更新vision位置并记录统计信息

        参数:
            new_positions: (start, end) - 新的vision位置
            layer_idx: 当前层索引
            original_count: 剪枝前的token数量
            kept_count: 剪枝后保留的token数量
        """
        self.vision_positions = new_positions
        self.pruning_stats.append({
            'layer_idx': layer_idx,
            'original_count': original_count,
            'kept_count': kept_count,
            'pruned_count': original_count - kept_count,
            'keep_ratio': kept_count / original_count if original_count > 0 else 0.0
        })

    def get_positions(self) -> Tuple[int, int]:
        """获取当前vision位置"""
        return self.vision_positions

    def get_stats(self) -> List[Dict]:
        """获取所有层的剪枝统计信息"""
        return self.pruning_stats

    def set_decode_mode(self, is_decode: bool):
        """设置是否进入decode模式

        参数:
            is_decode: True表示进入decode阶段，False表示prefill阶段
        """
        self._is_decode_mode = is_decode

    def is_decode_mode(self) -> bool:
        """检查是否已进入decode模式"""
        return self._is_decode_mode

    def should_prune_layer(self, layer_idx: int) -> bool:
        """检查是否应该对指定层进行剪枝

        参数:
            layer_idx: 层索引

        返回:
            True如果该层需要剪枝
        """
        if self.layer_pruners is None:
            return False
        return layer_idx in self.layer_pruners.get_all_layers()


def create_layer_hard_pruning_modifier(
    pruner,
    context: HardPruningContext,
    question_embeddings: torch.Tensor,
    layer_idx: int,
    threshold: float = 0.5,
    is_first_layer: bool = False,
    min_seq_len_for_pruning: int = 50
) -> Callable:
    """创建层hard剪枝的modifier函数（用于hook）

    与soft版本不同，hard剪枝会真正移除tokens，改变序列长度。

    **重要**：Hard pruning只在prefill阶段（第一次forward）执行，
    decode阶段（seq_len=1）不执行剪枝，直接返回。

    参数:
        pruner: VisionPrunerHead实例（该层的剪枝器）
        context: HardPruningContext实例，用于追踪动态变化的vision位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        layer_idx: 当前层索引（用于统计）
        threshold: hard mask阈值，默认0.5
        is_first_layer: 是否是第一个剪枝层（用于检测prefill/decode）
        min_seq_len_for_pruning: 最小序列长度阈值，低于此值不执行剪枝（避免decode阶段触发）

    返回:
        modifier函数，签名为 (hidden_states, attention_mask) -> (new_hidden, new_mask)
    """

    def modifier(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Hook函数，在layer执行前调用

        参数:
            hidden_states: (batch, seq_len, d_model) - 输入到当前层的hidden states
            attention_mask: (batch, seq_len) - attention mask

        返回:
            (new_hidden_states, new_attention_mask) - 序列长度可能改变
        """
        seq_len = hidden_states.shape[1]
        attn_mask_len = attention_mask.shape[1] if attention_mask is not None else None

        # === 关键检测：是否是decode阶段 ===
        # Decode阶段特征：seq_len很小（通常为1），且已经完成过prefill
        if is_first_layer:
            # 第一个剪枝层负责检测阶段
            print(f"\n[Hard Pruning Layer {layer_idx}] FIRST LAYER DETECTION:")
            print(f"  hidden_states.shape: {hidden_states.shape}")
            print(f"  attention_mask.shape: {attention_mask.shape if attention_mask is not None else None}")
            print(f"  seq_len={seq_len}, threshold={min_seq_len_for_pruning}")

            if seq_len < min_seq_len_for_pruning:
                # Decode阶段：序列太短，不执行剪枝
                print(f"  → DECODE MODE detected (seq_len < {min_seq_len_for_pruning})")
                context.set_decode_mode(True)
                return hidden_states, attention_mask
            else:
                # Prefill阶段：序列足够长
                print(f"  → PREFILL MODE detected (seq_len >= {min_seq_len_for_pruning})")
                context.set_decode_mode(False)

        # 如果已经进入decode模式，直接返回
        if context.is_decode_mode():
            print(f"[Hard Pruning Layer {layer_idx}] SKIP (decode mode), seq_len={seq_len}")
            return hidden_states, attention_mask

        print(f"\n[Hard Pruning Layer {layer_idx}] EXECUTING PRUNING:")
        print(f"  Input: seq_len={seq_len}, attn_mask_len={attn_mask_len}")

        # === Step 1: 获取当前vision位置（可能已被前面的层更新） ===
        v_start, v_end = context.get_positions()
        print(f"  Vision positions: [{v_start}, {v_end}]")

        # 边界检查
        if v_start >= seq_len or v_end >= seq_len:
            # vision tokens已经被完全剪除，直接返回
            print(f"  → SKIP: vision positions out of bounds (v_end={v_end} >= seq_len={seq_len})")
            return hidden_states, attention_mask

        vision_hidden = hidden_states[:, v_start:v_end+1, :]  # (batch, n_vision, d_model)
        n_vision = vision_hidden.shape[1]
        print(f"  Vision tokens: {n_vision}")

        if n_vision == 0:
            # 没有vision tokens了，直接返回
            print(f"  → SKIP: no vision tokens")
            return hidden_states, attention_mask

        # === Step 2: 调用pruner生成soft_mask ===
        with torch.no_grad():  # 评估时不需要梯度
            soft_mask = pruner(vision_hidden, question_embeddings, use_gumbel=False)  # (batch, n_vision)

        # === Step 3: 转换为hard mask ===
        hard_mask = (soft_mask > threshold).float()  # (batch, n_vision)

        # === Step 4: 找出要保留的token索引 ===
        # 假设batch_size=1（评估通常是单样本）
        kept_indices = torch.nonzero(hard_mask[0] > 0.5).squeeze(-1)  # (n_kept,)

        # 处理边界情况：如果所有tokens都被剪除，保留至少1个（得分最高的）
        if len(kept_indices) == 0:
            # 找到soft_mask中得分最高的token
            max_idx = soft_mask[0].argmax()
            kept_indices = max_idx.unsqueeze(0)

        n_kept = len(kept_indices)
        print(f"  Pruning: {n_vision} → {n_kept} (keep_ratio={n_kept/n_vision:.2%})")

        # === Step 5: 提取保留的vision tokens ===
        kept_vision = vision_hidden[:, kept_indices, :]  # (batch, n_kept, d_model)

        # === Step 6: 重新拼接hidden_states ===
        new_hidden = torch.cat([
            hidden_states[:, :v_start, :],      # vision前的文本
            kept_vision,                         # 保留的vision tokens
            hidden_states[:, v_end+1:, :]       # vision后的文本
        ], dim=1)

        # === Step 7: 更新attention_mask ===
        if attention_mask is not None:
            new_attention_mask = torch.cat([
                attention_mask[:, :v_start],
                torch.ones(attention_mask.shape[0], n_kept,
                          device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask[:, v_end+1:]
            ], dim=1)
        else:
            new_attention_mask = None

        new_seq_len = new_hidden.shape[1]
        new_mask_len = new_attention_mask.shape[1] if new_attention_mask is not None else None
        print(f"  Output: new_seq_len={new_seq_len}, new_mask_len={new_mask_len}")

        # === Step 8: 更新context中的vision位置 ===
        new_v_start = v_start
        new_v_end = v_start + n_kept - 1
        context.update_positions((new_v_start, new_v_end), layer_idx, n_vision, n_kept)
        print(f"  Updated vision positions: [{new_v_start}, {new_v_end}]")

        return new_hidden, new_attention_mask

    return modifier


def register_multi_layer_hard_hooks(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[List[Any], HardPruningContext]:
    """在多个LLM层注册hard剪枝hooks

    与soft版本不同，hard剪枝会真正移除tokens。
    使用HardPruningContext在各层之间传递动态变化的vision位置。

    **重要**：Hard pruning只在prefill阶段执行，decode阶段自动跳过。

    参数:
        backbone: LLaVA backbone实例
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (start, end) - 初始vision tokens位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        threshold: hard mask阈值，默认0.5

    返回:
        handles: hook handle列表（用于后续清理）
        context: HardPruningContext实例（包含剪枝统计信息）
    """
    handles = []
    context = HardPruningContext(vision_positions)

    # 按层索引排序，确保按顺序注册hooks
    layer_indices = sorted(layer_pruners.get_all_layers())

    for idx, layer_idx in enumerate(layer_indices):
        # 1. 获取该层的pruner
        pruner = layer_pruners.get_pruner(layer_idx)

        # 2. 创建hard pruning modifier函数
        # 第一个剪枝层负责检测prefill/decode阶段
        is_first = (idx == 0)
        modifier = create_layer_hard_pruning_modifier(
            pruner, context, question_embeddings, layer_idx, threshold,
            is_first_layer=is_first
        )

        # 3. 注册hook到LLaMA的对应层
        target_layer = backbone.model.model.language_model.layers[layer_idx]

        # 注册forward pre-hook
        def hook_fn(module, args, mod=modifier):
            # args: (hidden_states, attention_mask, *rest)
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

    return handles, context


# ==================== Hard Pruning v2: Intercept at LlamaModel level ====================

def register_hard_pruning_at_model_level(
    backbone,
    layer_pruners,
    vision_positions: Tuple[int, int],
    question_embeddings: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[Callable, HardPruningContext]:
    """在LlamaModel层面注册hard pruning，正确处理position_ids等参数

    这个方法通过monkey-patch LlamaModel.forward来实现hard pruning，
    确保每次序列长度改变后，position相关参数都被正确更新。

    参数:
        backbone: LLaVA backbone实例
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (start, end) - 初始vision tokens位置
        question_embeddings: (batch, n_text, d_text) - question embeddings
        threshold: hard mask阈值，默认0.5

    返回:
        restore_fn: 恢复原始forward的函数
        context: HardPruningContext实例（包含剪枝统计信息）
    """
    context = HardPruningContext(vision_positions)
    context.layer_pruners = layer_pruners
    context.question_embeddings = question_embeddings
    context.threshold = threshold

    # 获取LlamaModel
    llama_model = backbone.model.model.language_model

    # 保存原始forward方法
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
        """Wrapped forward that handles hard pruning and updates position parameters"""
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        # === 初始化部分（与原始forward相同） ===
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

        # === 检测prefill/decode阶段 ===
        seq_len = inputs_embeds.shape[1]
        if seq_len < 50:  # Decode阶段
            context.set_decode_mode(True)
        else:  # Prefill阶段
            context.set_decode_mode(False)

        # 创建causal mask
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

        # === Layer循环 - 在这里进行hard pruning ===
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # 调用decoder layer
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # === Hard Pruning逻辑 ===
            if not context.is_decode_mode() and context.should_prune_layer(layer_idx):
                old_seq_len = hidden_states.shape[1]

                # 执行剪枝，返回kept_position_indices
                hidden_states, kept_position_indices = apply_hard_pruning_to_hidden_states(
                    hidden_states,
                    context,
                    layer_idx
                )

                new_seq_len = hidden_states.shape[1]

                # === 关键：使用原始positions重新计算position相关参数 ===
                if new_seq_len != old_seq_len and kept_position_indices is not None:

                    # === 保持原始position_ids ===
                    # 不重新编号[0, 1, 2, ...], 而是使用原始positions
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

                    # 从原始position_ids中选取保留的positions
                    original_position_ids = position_ids[0]  # (seq_len,)

                    # 确保设备一致性（处理多GPU情况）
                    if kept_position_indices.device != original_position_ids.device:
                        kept_position_indices = kept_position_indices.to(original_position_ids.device)

                    kept_position_ids = original_position_ids[kept_position_indices]  # (new_seq_len,)

                    position_ids = kept_position_ids.unsqueeze(0)  # (1, new_seq_len)

                    # 更新cache_position - 也使用原始positions
                    cache_position = kept_position_ids

                    # 重新计算position_embeddings (RoPE)
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    # 重新创建causal_mask (基于新的序列长度)
                    # 注意：这里attention_mask需要更新
                    if attention_mask is not None:
                        # 根据kept_position_indices选择attention_mask
                        new_attention_mask = attention_mask[:, kept_position_indices]
                        attention_mask = new_attention_mask

                    # 创建临时inputs_embeds用于causal_mask计算
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

    # Monkey-patch
    import types
    llama_model.forward = types.MethodType(wrapped_forward, llama_model)

    # 返回恢复函数
    def restore_fn():
        llama_model.forward = original_forward

    return restore_fn, context


def apply_hard_pruning_to_hidden_states(
    hidden_states: torch.Tensor,
    context: HardPruningContext,
    layer_idx: int
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """对hidden_states应用hard pruning

    参数:
        hidden_states: (batch, seq_len, d_model)
        context: HardPruningContext
        layer_idx: 当前层索引

    返回:
        pruned_hidden_states: (batch, new_seq_len, d_model)
        kept_position_indices: (new_seq_len,) - 保留的tokens的原始位置索引，用于更新position_ids
    """
    v_start, v_end = context.get_positions()
    seq_len = hidden_states.shape[1]

    if v_start >= seq_len or v_end >= seq_len:
        return hidden_states, None

    # 提取vision tokens
    vision_hidden = hidden_states[:, v_start:v_end+1, :]
    n_vision = vision_hidden.shape[1]

    if n_vision == 0:
        return hidden_states, None

    # 获取pruner
    pruner = context.layer_pruners.get_pruner(layer_idx)

    # 生成soft_mask
    with torch.no_grad():
        soft_mask = pruner(vision_hidden, context.question_embeddings, use_gumbel=False)

    # 转换为hard mask
    hard_mask = (soft_mask > context.threshold).float()

    # 找出保留的token索引（相对于vision tokens的索引）
    kept_indices = torch.nonzero(hard_mask[0] > 0.5).squeeze(-1)

    if len(kept_indices) == 0:
        # 至少保留1个
        max_idx = soft_mask[0].argmax()
        kept_indices = max_idx.unsqueeze(0)

    # 提取保留的vision tokens
    kept_vision = vision_hidden[:, kept_indices, :]

    # 重新拼接
    new_hidden = torch.cat([
        hidden_states[:, :v_start, :],
        kept_vision,
        hidden_states[:, v_end+1:, :]
    ], dim=1)

    # 计算保留的tokens在原始序列中的绝对位置索引
    # 用于保持原始position_ids
    device = hidden_states.device

    # 确保kept_indices在正确的设备上
    if kept_indices.device != device:
        kept_indices = kept_indices.to(device)

    text_before_indices = torch.arange(0, v_start, device=device)
    vision_kept_indices = v_start + kept_indices  # 转换为绝对索引（现在已在正确设备）
    text_after_indices = torch.arange(v_end+1, seq_len, device=device)

    kept_position_indices = torch.cat([
        text_before_indices,
        vision_kept_indices,
        text_after_indices
    ])

    # 计算保留的vision tokens数量
    n_kept = len(kept_indices)

    # 更新context中的vision位置
    new_v_start = v_start
    new_v_end = v_start + n_kept - 1
    context.update_positions((new_v_start, new_v_end), layer_idx, n_vision, n_kept)

    return new_hidden, kept_position_indices

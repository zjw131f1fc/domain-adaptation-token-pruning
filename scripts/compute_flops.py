#!/usr/bin/env python
"""计算剪枝方法相比原始模型节省的 FLOPs

使用指定的剪枝率计算：
1. 原始模型的 FLOPs
2. 剪枝方法的 FLOPs（包括 Pruner 和 Adapter 的开销）
3. 比较两者的差异

用法:
    python scripts/compute_flops.py
    python scripts/compute_flops.py --l4 0.2283 --l14 0.1004 --l24 0.0736
    python scripts/compute_flops.py --text-len 50
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List


# ============================================================
# 配置类
# ============================================================

@dataclass
class ModelConfig:
    """LLaVA 模型配置"""
    hidden_size: int = 4096
    intermediate_size: int = 11008  # MLP intermediate size
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # GQA, LLaVA 1.5-7B 没有用 GQA
    head_dim: int = 128
    n_vision: int = 576  # vision tokens 数量


@dataclass
class PrunerConfig:
    """Pruner 配置"""
    d_internal: int = 512
    n_heads: int = 4
    n_queries: int = 4


@dataclass
class AdapterConfig:
    """Adapter 配置"""
    bottleneck_dim: int = 512


# ============================================================
# FLOPs 计算公式
# ============================================================

def compute_attention_flops(
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int
) -> int:
    """计算 Attention 层的 FLOPs
    注意: 1 MAC (乘加) = 2 FLOPs
    """
    # Q 投影: seq_len * hidden_size * (num_heads * head_dim)
    q_proj = 2 * seq_len * hidden_size * (num_heads * head_dim)
    # K/V 投影
    kv_proj = 2 * 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    # QK^T
    qk = 2 * num_heads * seq_len * seq_len * head_dim
    # Softmax(QK^T) * V
    attn_v = 2 * num_heads * seq_len * seq_len * head_dim
    # O 投影
    o_proj = 2 * seq_len * (num_heads * head_dim) * hidden_size

    return q_proj + kv_proj + qk + attn_v + o_proj


def compute_mlp_flops(seq_len: int, hidden_size: int, intermediate_size: int) -> int:
    """计算 MLP 层的 FLOPs (SwiGLU)"""
    gate = 2 * seq_len * hidden_size * intermediate_size
    up = 2 * seq_len * hidden_size * intermediate_size
    down = 2 * seq_len * intermediate_size * hidden_size
    return gate + up + down


def compute_transformer_layer_flops(seq_len: int, config: ModelConfig) -> int:
    """计算单个 Transformer 层的 FLOPs"""
    attn = compute_attention_flops(
        seq_len, config.hidden_size, config.num_heads,
        config.num_kv_heads, config.head_dim
    )
    mlp = compute_mlp_flops(seq_len, config.hidden_size, config.intermediate_size)
    return attn + mlp


def compute_pruner_flops(
    n_vision: int, config: ModelConfig, pruner_config: PrunerConfig
) -> int:
    """计算 Pruner (CrossAttentionPruner) 的 FLOPs"""
    d_model = config.hidden_size
    d_internal = pruner_config.d_internal
    n_heads = pruner_config.n_heads
    n_queries = pruner_config.n_queries
    head_dim_pruner = d_internal // n_heads

    # vision_proj
    vision_proj = 2 * n_vision * d_model * d_internal
    # cross_attn
    cross_attn_q = 2 * n_queries * d_internal * d_internal
    cross_attn_kv = 2 * 2 * n_vision * d_internal * d_internal
    cross_attn_qk = 2 * n_heads * n_queries * n_vision * head_dim_pruner
    cross_attn_av = 2 * n_heads * n_queries * n_vision * head_dim_pruner
    cross_attn_o = 2 * n_queries * d_internal * d_internal
    cross_attn = cross_attn_q + cross_attn_kv + cross_attn_qk + cross_attn_av + cross_attn_o
    # token_scorer
    token_scorer = 2 * n_vision * d_internal * d_internal + 2 * n_vision * d_internal
    # query_aggregator
    query_agg = 2 * n_vision * n_queries

    return vision_proj + cross_attn + token_scorer + query_agg


def compute_adapter_flops(
    seq_len: int, n_vision: int, config: ModelConfig, adapter_config: AdapterConfig
) -> int:
    """计算 Adapter (LightweightAdapter) 的 FLOPs"""
    hidden_size = config.hidden_size
    bottleneck = adapter_config.bottleneck_dim

    mask_encoder = 2 * n_vision * bottleneck + 2 * bottleneck * bottleneck
    query_proj = 2 * seq_len * hidden_size * bottleneck
    film = 2 * 2 * seq_len * bottleneck * bottleneck
    down = 2 * seq_len * hidden_size * bottleneck
    up = 2 * seq_len * bottleneck * hidden_size

    return mask_encoder + query_proj + film + down + up


# ============================================================
# 总 FLOPs 计算
# ============================================================

def compute_origin_flops(text_len: int, config: ModelConfig) -> Dict[str, int]:
    """计算原始模型的 FLOPs"""
    seq_len = text_len + config.n_vision
    layer_flops = compute_transformer_layer_flops(seq_len, config)
    total_llm = layer_flops * config.num_layers

    return {
        'seq_len': seq_len,
        'per_layer': layer_flops,
        'total_llm': total_llm,
        'total': total_llm,
    }


def compute_pruned_flops(
    text_len: int,
    config: ModelConfig,
    pruning_layers: List[int],
    kept_ratios: Dict[int, float],
    pruner_config: PrunerConfig,
    adapter_config: AdapterConfig
) -> Dict:
    """计算剪枝方法的 FLOPs"""
    n_vision = config.n_vision
    pruning_layers = sorted(pruning_layers)

    layer_flops_list = []
    current_n_vision = n_vision
    total_pruner = 0
    total_adapter = 0

    for layer_idx in range(config.num_layers):
        seq_len = text_len + current_n_vision
        layer_flops = compute_transformer_layer_flops(seq_len, config)
        layer_flops_list.append({
            'layer_idx': layer_idx,
            'seq_len': seq_len,
            'n_vision': current_n_vision,
            'flops': layer_flops,
        })

        if layer_idx in pruning_layers:
            pruner_flops = compute_pruner_flops(current_n_vision, config, pruner_config)
            total_pruner += pruner_flops
            adapter_flops = compute_adapter_flops(seq_len, n_vision, config, adapter_config)
            total_adapter += adapter_flops
            # 更新 n_vision（剪枝后）
            kept_ratio = kept_ratios.get(layer_idx, 1.0)
            current_n_vision = int(current_n_vision * kept_ratio)

    total_llm = sum(item['flops'] for item in layer_flops_list)

    return {
        'layer_details': layer_flops_list,
        'total_llm': total_llm,
        'total_pruner': total_pruner,
        'total_adapter': total_adapter,
        'total': total_llm + total_pruner + total_adapter,
    }


# ============================================================
# 辅助函数
# ============================================================

def format_flops(flops: int) -> str:
    """格式化 FLOPs 显示"""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compute FLOPs comparison")
    parser.add_argument('--l4', type=float, default=0.2283, help='Layer 4 kept ratio')
    parser.add_argument('--l14', type=float, default=0.1004, help='Layer 14 kept ratio')
    parser.add_argument('--l24', type=float, default=0.0736, help='Layer 24 kept ratio')
    parser.add_argument('--text-len', type=int, default=29, help='Text token length')
    parser.add_argument('--gen-len', type=int, default=32, help='Generated token length (for decode phase)')
    args = parser.parse_args()

    # 配置
    model_config = ModelConfig()
    pruner_config = PrunerConfig()
    adapter_config = AdapterConfig()
    pruning_layers = [4, 14, 24]

    # 累积保留率（相对于原始 n_vision）
    kept_ratios_cumulative = {4: args.l4, 14: args.l14, 24: args.l24}
    text_len = args.text_len
    gen_len = args.gen_len
    n_vision = model_config.n_vision

    print("\n" + "=" * 70)
    print("FLOPs Analysis")
    print("=" * 70)

    print("\n[Model Configuration]")
    print(f"  hidden_size: {model_config.hidden_size}")
    print(f"  intermediate_size: {model_config.intermediate_size}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  num_heads: {model_config.num_heads}")
    print(f"  n_vision: {model_config.n_vision}")

    print("\n[Sample Information]")
    print(f"  Text tokens: {text_len}")
    print(f"  Vision tokens: {n_vision}")
    print(f"  Generated tokens: {gen_len}")
    print(f"  Total sequence length (origin): {text_len + n_vision}")

    print("\n[Pruning Configuration]")
    print(f"  Pruning layers: {pruning_layers}")
    print(f"  Cumulative kept ratios (relative to original n_vision):")
    for layer_idx in pruning_layers:
        ratio = kept_ratios_cumulative[layer_idx]
        n_kept = int(n_vision * ratio)
        print(f"    Layer {layer_idx}: {ratio:.2%} ({n_kept} tokens)")

    # ========== 原始模型 FLOPs ==========
    print("\n" + "-" * 70)
    print("[Origin Model FLOPs]")

    # Prefill 阶段
    origin_prefill = compute_origin_flops(text_len, model_config)

    # Decode 阶段：每个生成的 token
    # 序列长度逐渐增加，但主要计算量来自 KV cache 的 attention
    # 简化计算：使用平均序列长度
    origin_decode_per_token = compute_origin_flops(1, model_config)  # seq_len=1 for new token
    # 但 attention 需要 attend to 所有之前的 tokens
    # 更准确的计算：attention 部分需要考虑 KV cache 长度
    avg_kv_len = text_len + n_vision + gen_len // 2
    origin_decode_attn_correction = 2 * model_config.num_heads * avg_kv_len * model_config.head_dim * 2  # QK^T + Attn*V
    origin_decode_per_token_corrected = origin_decode_per_token['per_layer'] + origin_decode_attn_correction * model_config.num_layers
    origin_decode_total = origin_decode_per_token_corrected * gen_len

    origin_total = origin_prefill['total'] + origin_decode_total

    print(f"  Prefill:")
    print(f"    Sequence length: {origin_prefill['seq_len']}")
    print(f"    Per layer: {format_flops(origin_prefill['per_layer'])}")
    print(f"    Total: {format_flops(origin_prefill['total'])}")
    print(f"  Decode ({gen_len} tokens):")
    print(f"    Per token (avg): {format_flops(origin_decode_per_token_corrected)}")
    print(f"    Total: {format_flops(origin_decode_total)}")
    print(f"  Total: {format_flops(origin_total)}")

    # ========== 剪枝模型 FLOPs ==========
    # 将累积保留率转换为相对保留率
    relative_kept_ratios = {}
    prev_cumulative = 1.0
    for layer_idx in pruning_layers:
        current_cumulative = kept_ratios_cumulative[layer_idx]
        relative_ratio = current_cumulative / prev_cumulative
        relative_kept_ratios[layer_idx] = relative_ratio
        prev_cumulative = current_cumulative

    print("\n" + "-" * 70)
    print("[Pruned Model FLOPs]")

    # Prefill 阶段
    pruned_prefill = compute_pruned_flops(
        text_len, model_config, pruning_layers,
        relative_kept_ratios, pruner_config, adapter_config
    )

    print(f"  Prefill - LLM layers breakdown:")
    for i, layer_info in enumerate(pruned_prefill['layer_details']):
        if i < 3 or i >= model_config.num_layers - 2 or layer_info['layer_idx'] in pruning_layers:
            marker = " *" if layer_info['layer_idx'] in pruning_layers else ""
            print(f"    Layer {layer_info['layer_idx']:2d}: seq={layer_info['seq_len']:4d}, "
                  f"n_vision={layer_info['n_vision']:3d}, "
                  f"flops={format_flops(layer_info['flops'])}{marker}")
        elif i == 3:
            print(f"    ...")

    # Decode 阶段
    # 剪枝后的 KV cache 长度更短
    final_n_vision = int(n_vision * kept_ratios_cumulative[24])
    avg_kv_len_pruned = text_len + final_n_vision + gen_len // 2

    # Decode 阶段的 LLM FLOPs（每个 token）
    pruned_decode_per_token_llm = compute_transformer_layer_flops(1, model_config) * model_config.num_layers
    pruned_decode_attn_correction = 2 * model_config.num_heads * avg_kv_len_pruned * model_config.head_dim * 2 * model_config.num_layers
    pruned_decode_per_token_llm_corrected = pruned_decode_per_token_llm + pruned_decode_attn_correction

    # Decode 阶段的 Adapter FLOPs（每个 token，seq_len=1）
    pruned_decode_adapter_per_token = sum(
        compute_adapter_flops(1, n_vision, model_config, adapter_config)
        for _ in pruning_layers
    )

    pruned_decode_total_llm = pruned_decode_per_token_llm_corrected * gen_len
    pruned_decode_total_adapter = pruned_decode_adapter_per_token * gen_len

    print(f"\n  Prefill totals:")
    print(f"    LLM: {format_flops(pruned_prefill['total_llm'])}")
    print(f"    Pruner: {format_flops(pruned_prefill['total_pruner'])}")
    print(f"    Adapter: {format_flops(pruned_prefill['total_adapter'])}")

    print(f"\n  Decode ({gen_len} tokens):")
    print(f"    LLM per token (avg): {format_flops(pruned_decode_per_token_llm_corrected)}")
    print(f"    Adapter per token: {format_flops(pruned_decode_adapter_per_token)}")
    print(f"    LLM total: {format_flops(pruned_decode_total_llm)}")
    print(f"    Adapter total: {format_flops(pruned_decode_total_adapter)}")

    pruned_total = (pruned_prefill['total'] +
                    pruned_decode_total_llm +
                    pruned_decode_total_adapter)

    print(f"\n  Grand Total:")
    print(f"    Prefill: {format_flops(pruned_prefill['total'])}")
    print(f"    Decode: {format_flops(pruned_decode_total_llm + pruned_decode_total_adapter)}")
    print(f"    Total: {format_flops(pruned_total)}")

    # ========== 比较 ==========
    print("\n" + "-" * 70)
    print("[Comparison]")

    saved = origin_total - pruned_total
    reduction = saved / origin_total * 100

    print(f"  Origin Model: {format_flops(origin_total)}")
    print(f"  Pruned Model: {format_flops(pruned_total)}")
    print(f"  Saved: {format_flops(saved)}")
    print(f"  Reduction: {reduction:.2f}%")
    print(f"  Speedup: {origin_total / pruned_total:.2f}x")

    # 分解节省来源
    print("\n[Savings Breakdown]")
    prefill_saved = origin_prefill['total'] - pruned_prefill['total']
    decode_saved = origin_decode_total - (pruned_decode_total_llm + pruned_decode_total_adapter)

    print(f"  Prefill saved: {format_flops(prefill_saved)} ({prefill_saved/origin_total*100:.2f}%)")
    print(f"  Decode saved: {format_flops(decode_saved)} ({decode_saved/origin_total*100:.2f}%)")
    print(f"  Total Pruner overhead: {format_flops(pruned_prefill['total_pruner'])} ({pruned_prefill['total_pruner']/origin_total*100:.2f}%)")
    print(f"  Total Adapter overhead: {format_flops(pruned_prefill['total_adapter'] + pruned_decode_total_adapter)} ({(pruned_prefill['total_adapter'] + pruned_decode_total_adapter)/origin_total*100:.2f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

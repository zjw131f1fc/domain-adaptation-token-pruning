"""计算 LLaVA 1.5 7B 渐进式视觉 token 剪枝的 FLOPs 节省

剪枝配置: [L4=35.96%, L14=5.30%, L24=1.43%]
- 这是累积保留率，即相对于原始 576 tokens 的比例
- L4 剪枝后保留 35.96%，L14 后保留 5.30%，L24 后保留 1.43%
"""

# LLaVA 1.5 7B 参数
d_model = 4096
num_heads = 32
head_dim = 128
intermediate_size = 11008
num_layers = 32

# Token 数量
n_vision_original = 576
n_text = 20

# 原始保留率 [L4=35.96%, L14=5.30%, L24=1.43%]
# 计算原始配置的平均每层 vision tokens，然后按比例调整到目标 128

# 原始配置
original_ratios = {4: 0.3596, 14: 0.0530, 24: 0.0143}

# 计算原始平均每层 vision tokens
def calc_avg_tokens(ratios, n_vision, n_layers=32):
    n_L4 = int(n_vision * ratios[4])
    n_L14 = int(n_vision * ratios[14])
    n_L24 = int(n_vision * ratios[24])
    total = n_vision * 4 + n_L4 * 10 + n_L14 * 10 + n_L24 * 8
    return total / n_layers

original_avg = calc_avg_tokens(original_ratios, n_vision_original)
target_avg = 128

# 按比例缩放：新比例 = 原比例 * (target_avg / original_avg)
scale = target_avg / original_avg
keep_ratios = {
    4: original_ratios[4] * scale,
    14: original_ratios[14] * scale,
    24: original_ratios[24] * scale,
}

print(f"原始平均每层 vision tokens: {original_avg:.1f}")
print(f"目标平均每层 vision tokens: {target_avg}")
print(f"缩放系数: {scale:.3f}")

# 计算各阶段的 vision token 数量
n_vision_after_L4 = int(n_vision_original * keep_ratios[4])    # ≈ 207
n_vision_after_L14 = int(n_vision_original * keep_ratios[14])  # ≈ 31
n_vision_after_L24 = int(n_vision_original * keep_ratios[24])  # ≈ 8

print("=" * 70)
print("LLaVA 1.5 7B 渐进式剪枝 FLOPs 分析")
print("=" * 70)
print(f"\n剪枝配置: [L4={keep_ratios[4]*100:.2f}%, L14={keep_ratios[14]*100:.2f}%, L24={keep_ratios[24]*100:.2f}%]")
print(f"\nVision tokens 变化:")
print(f"  L0-L3:   {n_vision_original} tokens (原始)")
print(f"  L4-L13:  {n_vision_after_L4} tokens ({keep_ratios[4]*100:.1f}%)")
print(f"  L14-L23: {n_vision_after_L14} tokens ({keep_ratios[14]*100:.1f}%)")
print(f"  L24-L31: {n_vision_after_L24} tokens ({keep_ratios[24]*100:.1f}%)")

# 各阶段序列长度
seq_stage0 = n_vision_original + n_text      # 596 (L0-L3)
seq_stage1 = n_vision_after_L4 + n_text      # 227 (L4-L13)
seq_stage2 = n_vision_after_L14 + n_text     # 51  (L14-L23)
seq_stage3 = n_vision_after_L24 + n_text     # 28  (L24-L31)

print(f"\n总序列长度 (vision + text):")
print(f"  L0-L3:   {seq_stage0}")
print(f"  L4-L13:  {seq_stage1}")
print(f"  L14-L23: {seq_stage2}")
print(f"  L24-L31: {seq_stage3}")

def calc_layer_flops(seq_len, d_model, intermediate_size):
    """计算单层 Transformer 的 FLOPs"""
    # Attention: QKV proj + scores + output + out proj
    qkv_proj = 3 * seq_len * d_model * d_model * 2
    attn_scores = seq_len * seq_len * d_model * 2
    attn_output = seq_len * seq_len * d_model * 2
    out_proj = seq_len * d_model * d_model * 2
    attention = qkv_proj + attn_scores + attn_output + out_proj

    # FFN (SwiGLU): gate + up + down
    ffn = 3 * seq_len * d_model * intermediate_size * 2

    return {'attention': attention, 'ffn': ffn, 'total': attention + ffn}

def calc_pruner_flops(n_vision, d_model, d_internal=128, n_queries=4, n_heads=4):
    """计算 CrossAttentionPruner 的 FLOPs"""
    # vision_proj: n_vision * d_model * d_internal * 2
    vision_proj = n_vision * d_model * d_internal * 2

    # cross_attn: Q @ K^T + softmax @ V
    # Q: (n_queries, d_internal), K: (n_vision, d_internal)
    # scores: n_queries * n_vision * d_internal * 2
    # output: n_queries * n_vision * d_internal * 2
    cross_attn = 2 * n_queries * n_vision * d_internal * 2

    # token_scorer MLP: n_vision * d_internal * d_internal * 2 (两层)
    token_scorer = 2 * n_vision * d_internal * d_internal * 2

    # query_aggregator: n_vision * n_queries * 1 * 2
    query_agg = n_vision * n_queries * 2

    return vision_proj + cross_attn + token_scorer + query_agg

def calc_adapter_flops(seq_len, d_model, bottleneck=512, n_vision=576):
    """计算 LightweightAdapter 的 FLOPs"""
    # down: seq * d_model * bottleneck * 2
    down = seq_len * d_model * bottleneck * 2

    # mask_encoder (attention pooling): ~n_vision * 64 * 2 + 64 * bottleneck * 2
    mask_enc = n_vision * 64 * 2 + 64 * bottleneck * 2

    # query_proj: seq * d_model * bottleneck * 2
    query_proj = seq_len * d_model * bottleneck * 2

    # gamma_net + beta_net: 2 * seq * bottleneck * bottleneck * 2
    film = 2 * seq_len * bottleneck * bottleneck * 2

    # up: seq * bottleneck * d_model * 2
    up = seq_len * bottleneck * d_model * 2

    return down + mask_enc + query_proj + film + up

# 计算各阶段 FLOPs
flops_stage0 = calc_layer_flops(seq_stage0, d_model, intermediate_size)
flops_stage1 = calc_layer_flops(seq_stage1, d_model, intermediate_size)
flops_stage2 = calc_layer_flops(seq_stage2, d_model, intermediate_size)
flops_stage3 = calc_layer_flops(seq_stage3, d_model, intermediate_size)

print(f"\n单层 FLOPs (GFLOPs):")
print(f"  L0-L3 (seq={seq_stage0}):   {flops_stage0['total']/1e9:.2f}G")
print(f"  L4-L13 (seq={seq_stage1}):  {flops_stage1['total']/1e9:.2f}G")
print(f"  L14-L23 (seq={seq_stage2}): {flops_stage2['total']/1e9:.2f}G")
print(f"  L24-L31 (seq={seq_stage3}): {flops_stage3['total']/1e9:.2f}G")

# Pruner 和 Adapter 的 FLOPs
pruner_L4 = calc_pruner_flops(n_vision_original, d_model)
pruner_L14 = calc_pruner_flops(n_vision_after_L4, d_model)
pruner_L24 = calc_pruner_flops(n_vision_after_L14, d_model)

adapter_L4 = calc_adapter_flops(seq_stage1, d_model, n_vision=n_vision_original)
adapter_L14 = calc_adapter_flops(seq_stage2, d_model, n_vision=n_vision_after_L4)
adapter_L24 = calc_adapter_flops(seq_stage3, d_model, n_vision=n_vision_after_L14)

print(f"\nPruner FLOPs (MFLOPs):")
print(f"  L4 Pruner (input={n_vision_original}):  {pruner_L4/1e6:.2f}M")
print(f"  L14 Pruner (input={n_vision_after_L4}): {pruner_L14/1e6:.2f}M")
print(f"  L24 Pruner (input={n_vision_after_L14}): {pruner_L24/1e6:.2f}M")

print(f"\nAdapter FLOPs (GFLOPs):")
print(f"  L4 Adapter (seq={seq_stage1}):  {adapter_L4/1e9:.3f}G")
print(f"  L14 Adapter (seq={seq_stage2}): {adapter_L14/1e9:.3f}G")
print(f"  L24 Adapter (seq={seq_stage3}): {adapter_L24/1e9:.3f}G")

# 总 FLOPs 计算
# 原始 (无剪枝)
original_total = flops_stage0['total'] * num_layers

# 剪枝后
# L0-L3: 4 层全序列
# L4: 剪枝层 (全序列 + Pruner + Adapter)
# L5-L13: 9 层短序列 stage1
# L14: 剪枝层 (stage1 序列 + Pruner + Adapter)
# L15-L23: 9 层短序列 stage2
# L24: 剪枝层 (stage2 序列 + Pruner + Adapter)
# L25-L31: 7 层短序列 stage3

pruned_total = (
    flops_stage0['total'] * 4 +           # L0-L3
    flops_stage0['total'] + pruner_L4 + adapter_L4 +   # L4 (剪枝发生在 attention 后)
    flops_stage1['total'] * 9 +           # L5-L13
    flops_stage1['total'] + pruner_L14 + adapter_L14 + # L14
    flops_stage2['total'] * 9 +           # L15-L23
    flops_stage2['total'] + pruner_L24 + adapter_L24 + # L24
    flops_stage3['total'] * 7             # L25-L31
)

saved = original_total - pruned_total
ratio = saved / original_total * 100

print(f"\n" + "=" * 70)
print("总 FLOPs 对比")
print("=" * 70)
print(f"\n原始 (无剪枝): {original_total/1e12:.3f} TFLOPs")
print(f"剪枝后:        {pruned_total/1e12:.3f} TFLOPs")
print(f"节省:          {saved/1e12:.3f} TFLOPs ({ratio:.1f}%)")

# 详细分解
print(f"\n剪枝后 FLOPs 分解:")
print(f"  L0-L3 (4层):   {flops_stage0['total'] * 4 / 1e12:.3f} TFLOPs")
print(f"  L4 (剪枝层):   {(flops_stage0['total'] + pruner_L4 + adapter_L4) / 1e12:.3f} TFLOPs")
print(f"  L5-L13 (9层):  {flops_stage1['total'] * 9 / 1e12:.3f} TFLOPs")
print(f"  L14 (剪枝层):  {(flops_stage1['total'] + pruner_L14 + adapter_L14) / 1e12:.3f} TFLOPs")
print(f"  L15-L23 (9层): {flops_stage2['total'] * 9 / 1e12:.3f} TFLOPs")
print(f"  L24 (剪枝层):  {(flops_stage2['total'] + pruner_L24 + adapter_L24) / 1e12:.3f} TFLOPs")
print(f"  L25-L31 (7层): {flops_stage3['total'] * 7 / 1e12:.3f} TFLOPs")

# Pruner + Adapter 总开销
extra_overhead = pruner_L4 + pruner_L14 + pruner_L24 + adapter_L4 + adapter_L14 + adapter_L24
print(f"\nPruner + Adapter 总开销: {extra_overhead/1e9:.2f} GFLOPs ({extra_overhead/pruned_total*100:.2f}%)")

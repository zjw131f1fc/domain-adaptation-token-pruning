"""
Generator: Mini Transformer for query-aware vision token mask generation.

Architecture:
    Each layer: Self-Attention (on V) → Cross-Attention (V queries, Q as KV) → FFN
    Output: sigmoid mask for each vision token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.d_k)
        
        self._init_weights()
    
    def _init_weights(self):
        """Scaled initialization for attention projections."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: optional attention mask
        
        Returns:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Project and reshape to (batch, num_heads, seq_len, d_k)
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention: (batch, num_heads, seq_len_q, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project: (batch, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        return self.linear2(F.gelu(self.linear1(x)))


class GeneratorLayer(nn.Module):
    """Single Generator transformer layer (self-attention only)."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch, N, d_model) - input features
        
        Returns:
            x_out: (batch, N, d_model)
        """
        # Self-attention
        x1 = self.norm1(x)
        x1 = x + self.self_attn(x1, x1, x1)
        
        # Feed-forward
        x2 = self.norm2(x1)
        x_out = x1 + self.ffn(x2)
        
        return x_out


class Generator(nn.Module):
    """
    Mini Transformer Generator for query-aware vision token masking.
    
    Architecture:
    1. Bottleneck: project d_v -> d_internal for efficient processing
    2. Cross-attention: vision tokens attend to text query -> fused features
    3. Transformer layers on fused features
    4. Output: threshold for each token
    
    Args:
        d_v: dimension of vision tokens (input/output)
        d_t: dimension of text tokens
        d_internal: internal bottleneck dimension for efficient processing
        num_layers: number of transformer layers (applied on fused features)
        num_heads: number of attention heads
        d_ff: dimension of feed-forward network
        use_pos_encoding: whether to use positional encoding for V
    """
    
    def __init__(
        self,
        d_v=768,
        d_t=768,
        d_internal=512,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        use_pos_encoding=False,
        use_gumbel=False,
        temperature=1.0
    ):
        super().__init__()
        
        self.d_v = d_v
        self.d_t = d_t
        self.d_internal = d_internal
        self.num_layers = num_layers
        self.use_pos_encoding = use_pos_encoding
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        
        # Bottleneck: d_v -> d_internal
        self.v_down_proj = nn.Linear(d_v, d_internal)
        nn.init.xavier_uniform_(self.v_down_proj.weight)
        nn.init.zeros_(self.v_down_proj.bias)
        
        # Text to internal dimension projection
        self.text_proj = nn.Linear(d_t, d_internal)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        
        # Optional positional encoding for vision tokens (in internal dim)
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, d_internal))
            nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Cross-attention: vision tokens attend to text query (in internal dim)
        self.cross_attn = MultiHeadAttention(d_internal, num_heads)
        self.cross_attn_norm = nn.LayerNorm(d_internal)
        
        # Transformer layers on fused features (in internal dim)
        self.layers = nn.ModuleList([
            GeneratorLayer(d_internal, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output projection to mask logits
        self.output_norm = nn.LayerNorm(d_internal)
        self.output_proj = nn.Linear(d_internal, 1)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def set_temperature(self, temperature: float):
        """动态设置temperature（用于temperature annealing）"""
        self.temperature = temperature
    
    def forward(self, V, Q, padding_mask=None):
        """
        Generate query-aware mask for vision tokens.
        
        Args:
            V: (batch, N, d_v) - vision tokens (may contain padding)
            Q: (batch, T_q, d_t) - question embeddings (only question part, NOT answer)
            padding_mask: (batch, N) - True for valid tokens, False for padding (optional)
        
        Returns:
            m: (batch, N) - sigmoid/gumbel mask probabilities for each vision token
            V_internal: (batch, N, d_internal) - internal features (for analysis)
        """
        batch_size, N, _ = V.shape
        
        # Bottleneck: d_v -> d_internal
        V_internal = self.v_down_proj(V)  # (batch, N, d_internal)
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            V_internal = V_internal + self.pos_encoding[:, :N, :]
        
        # Project text to internal dimension
        Q_proj = self.text_proj(Q)  # (batch, T_q, d_internal)
        
        # Cross-attention: vision tokens attend to ALL question tokens
        V_normed = self.cross_attn_norm(V_internal)
        fused = V_internal + self.cross_attn(V_normed, Q_proj, Q_proj)  # (batch, N, d_internal)
        
        # Apply transformer layers on fused features
        V_out = fused
        for layer in self.layers:
            V_out = layer(V_out)
        
        # Generate mask
        V_normalized = self.output_norm(V_out)  # (batch, N, d_internal)
        logits = self.output_proj(V_normalized).squeeze(-1)  # (batch, N)

        if self.use_gumbel:
            # Gumbel-Softmax: 对每个token的keep/drop进行采样
            stacked_logits = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # (batch, N, 2)
            if self.training:
                # 训练模式：添加Gumbel噪声实现可微分采样
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(stacked_logits) + 1e-8) + 1e-8)
                gumbel_logits = (stacked_logits + gumbel_noise) / self.temperature
                m_soft = F.softmax(gumbel_logits, dim=-1)
            else:
                # 推理模式：不添加噪声，但使用相同的softmax公式保持一致性
                m_soft = F.softmax(stacked_logits / self.temperature, dim=-1)
            m = m_soft[..., 1]  # 取P(keep)
        else:
            # 不使用Gumbel时，直接用sigmoid
            m = torch.sigmoid(logits / self.temperature)

        # Apply padding mask
        if padding_mask is not None:
            m = m * padding_mask.float()

        return m, V_out


# if __name__ == "__main__":
#     # Test the Generator
#     batch_size = 2
#     N = 576  # number of vision tokens (e.g., 24x24)
#     T_q = 32  # number of text tokens
#     d_v = 768
#     d_t = 768
    
#     generator = Generator(
#         d_v=d_v,
#         d_t=d_t,
#         num_layers=4,
#         num_heads=8,
#         d_ff=2048,
#         use_pos_encoding=True
#     )
    
#     # Random inputs
#     V = torch.randn(batch_size, N, d_v)
#     Q = torch.randn(batch_size, T_q, d_t)
    
#     # Forward pass
#     mask, V_out = generator(V, Q)
    
#     print(f"Input V shape: {V.shape}")
#     print(f"Input Q shape: {Q.shape}")
#     print(f"Output mask shape: {mask.shape}")
#     print(f"Output V_out shape: {V_out.shape}")
#     print(f"Mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
#     print(f"Mask mean: {mask.mean().item():.4f}")
#     print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")

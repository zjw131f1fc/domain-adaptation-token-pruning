"""
Discriminator: 2-layer MLP for token-level classification.

Task: Judge whether a token's hidden state comes from unpruned or pruned LLM.
Architecture: Linear(d_model → d_d) → GELU → Linear(d_d → 1) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    2-layer MLP Discriminator for token-level real/fake classification.
    
    Args:
        d_model: dimension of input hidden states from LLM
        num_layers: number of LLM layers to aggregate (concatenate their hidden states)
        d_d: dimension of hidden layer (default 1024 or 2048)
        dropout: dropout rate for regularization
        use_layer_norm: whether to apply LayerNorm to input
        use_spectral_norm: whether to apply spectral normalization to linear layers (for GAN stability)
    """
    
    def __init__(
        self,
        d_model=4096,
        num_layers=1,
        d_d=2048,
        dropout=0.1,
        use_layer_norm=True,
        use_spectral_norm=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_d = d_d
        self.use_layer_norm = use_layer_norm
        self.use_spectral_norm = use_spectral_norm
        
        # 谱归一化包装函数
        def maybe_spectral_norm(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer
        
        # Input dimension is d_model * num_layers (concatenated)
        input_dim = d_model

        self.first_linears = [maybe_spectral_norm(nn.Linear(input_dim, d_d)) for _ in range(num_layers)]
        self.second_linears = [maybe_spectral_norm(nn.Linear(d_d, d_d)) for _ in range(num_layers)]
        self.fusion_linear1 = maybe_spectral_norm(nn.Linear(d_d * num_layers, d_d))
        self.fusion_linear2 = maybe_spectral_norm(nn.Linear(d_d, d_d))
        self.fusion_linear3 = maybe_spectral_norm(nn.Linear(d_d, 1))
        
        # Optional input normalization - 使用 BatchNorm1d 替代 LayerNorm
        if use_layer_norm:
            # BatchNorm1d 对特征维度进行归一化，更适合处理大数值的 LLM hidden states
            self.input_norm = nn.BatchNorm1d(d_model, momentum=0.1, eps=1e-5)
        
        # Register linear layers as ModuleList
        self.first_linears = nn.ModuleList(self.first_linears)
        self.second_linears = nn.ModuleList(self.second_linears)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for linear in self.first_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        for linear in self.second_linears:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.fusion_linear1.weight)
        nn.init.zeros_(self.fusion_linear1.bias)
        nn.init.xavier_uniform_(self.fusion_linear2.weight)
        nn.init.zeros_(self.fusion_linear2.bias)
        nn.init.xavier_uniform_(self.fusion_linear3.weight)
        nn.init.zeros_(self.fusion_linear3.bias)
    
    def forward(self, hidden_states, padding_mask=None):
        """
        Classify token hidden state as real (from unpruned) or fake (from pruned).
        
        Args:
            hidden_states: list of tensors, each (batch, seq_len, d_model) or (batch * seq_len, d_model)
                          Hidden states from multiple LLM layers
                          Length of list should be num_layers
            padding_mask: (batch, seq_len) - True for valid tokens, False for padding (optional)
        
        Returns:
            prob: (batch, seq_len) or (batch * seq_len,)
                  Sigmoid probability in (0, 1)
                  Close to 1 = real (unpruned), close to 0 = fake (pruned)
        """
        # Handle single tensor input (backward compatibility)
        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]
        
        # Check number of layers
        assert len(hidden_states) == self.num_layers, \
            f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
        
        # Get shape info from first tensor
        first_shape = hidden_states[0].shape
        if len(first_shape) == 3:
            batch_size, seq_len, _ = first_shape
            reshape_needed = True
        else:
            reshape_needed = False

        # 目标设备：优先使用输入所在设备，否则使用模块设备
        target_device = hidden_states[0].device if isinstance(hidden_states[0], torch.Tensor) else next(self.parameters()).device

        # Process each layer with residual connection
        layer_outputs = []
        for i, h in enumerate(hidden_states):
            if len(h.shape) == 3:
                h = h.view(-1, self.d_model)  # (batch * seq_len, d_model)
            h = h.to(device=target_device, dtype=torch.float32)
            # 【简化方案】直接用 tanh 压缩到 [-1, 1]，避免 BatchNorm 在小 batch 下不稳定
            # 先除以一个大的常数（如 100）缩小数值范围，再用 tanh
            h = torch.tanh(h / 100.0)  # 将 [-300, 300] 映射到约 [-1, 1]
            
            # First linear layer with GELU
            h = self.first_linears[i](h)       # (N, d_d)
            h = F.gelu(h)                      # (N, d_d)
            h = self.dropout(h)                # (N, d_d)
            
            # Second linear layer with residual connection
            h_res = self.second_linears[i](h)  # (N, d_d)
            h = h + h_res                      # residual connection
            
            layer_outputs.append(h)
        
        # Concatenate all layer outputs
        h_fused = torch.cat(layer_outputs, dim=-1)  # (N, d_d * num_layers)
        
        # Fusion layer
        h_fused = self.fusion_linear1(h_fused)  # (N, d_d)
        h_fused = F.gelu(h_fused)
        h_fused = self.fusion_linear2(h_fused)  # (N, d_d)
        h_fused = F.gelu(h_fused)
        logits = self.fusion_linear3(h_fused)  # (N, 1)
        prob = torch.sigmoid(logits.squeeze(-1))  # (N,)
        
        # Reshape back if input was 3D
        if reshape_needed:
            prob = prob.view(batch_size, seq_len)
        
        # Apply padding mask: zero out padding positions
        if padding_mask is not None:
            prob = prob * padding_mask.float()
        
        return prob
    
    def get_logits(self, hidden_states):
        """
        Get raw logits before sigmoid (useful for loss computation).
        
        Args:
            hidden_states: list of tensors, each (batch, seq_len, d_model) or (batch * seq_len, d_model)
        
        Returns:
            logits: (batch, seq_len) or (batch * seq_len,)
        """
        # Handle single tensor input
        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]
        
        assert len(hidden_states) == self.num_layers
        
        # Get shape info
        first_shape = hidden_states[0].shape
        if len(first_shape) == 3:
            batch_size, seq_len, _ = first_shape
            reshape_needed = True
        else:
            reshape_needed = False
        
        # Process each layer with residual connection
        layer_outputs = []
        for i, h in enumerate(hidden_states):
            if len(h.shape) == 3:
                h = h.view(-1, self.d_model)
            
            # 【简化方案】直接用 tanh 压缩到 [-1, 1]
            h = torch.tanh(h / 100.0)
            
            # First linear layer with GELU
            h = self.first_linears[i](h)
            h = F.gelu(h)
            h = self.dropout(h)
            
            # Second linear layer with residual connection
            h_res = self.second_linears[i](h)
            h = h + h_res
            h = F.gelu(h)
            
            layer_outputs.append(h)
        
        # Concatenate all layer outputs
        h_fused = torch.cat(layer_outputs, dim=-1)
        
        # Fusion layer
        logits = self.fusion_linear(h_fused)  # (N, 1)
        logits = logits.squeeze(-1)  # (N,)
        
        if reshape_needed:
            logits = logits.view(batch_size, seq_len)
        
        return logits





# if __name__ == "__main__":
#     # Test the Discriminator
#     print("=" * 60)
#     print("Testing Single-layer Discriminator")
#     print("=" * 60)
    
#     batch_size = 4
#     seq_len = 128
#     d_model = 4096
    
#     discriminator = Discriminator(d_model=d_model, d_d=2048, dropout=0.1)
    
#     # Test with 3D input (batch, seq_len, d_model)
#     h_3d = torch.randn(batch_size, seq_len, d_model)
#     prob_3d = discriminator(h_3d)
#     print(f"Input shape (3D): {h_3d.shape}")
#     print(f"Output shape: {prob_3d.shape}")
#     print(f"Output range: [{prob_3d.min().item():.4f}, {prob_3d.max().item():.4f}]")
#     print(f"Output mean: {prob_3d.mean().item():.4f}")
    
#     # Test with 2D input (batch * seq_len, d_model)
#     h_2d = torch.randn(batch_size * seq_len, d_model)
#     prob_2d = discriminator(h_2d)
#     print(f"\nInput shape (2D): {h_2d.shape}")
#     print(f"Output shape: {prob_2d.shape}")
    
#     # Test get_logits
#     logits = discriminator.get_logits(h_3d)
#     print(f"\nLogits shape: {logits.shape}")
#     print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    
#     print(f"\nDiscriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
#     # Test Multi-layer input Discriminator
#     print("\n" + "=" * 60)
#     print("Testing Discriminator with multiple layer inputs")
#     print("=" * 60)
    
#     # Discriminator that takes 3 layers as input
#     multi_layer_disc = Discriminator(d_model=d_model, num_layers=3, d_d=2048, dropout=0.1)
    
#     # Hidden states from 3 different layers
#     hidden_states = [
#         torch.randn(batch_size, seq_len, d_model),
#         torch.randn(batch_size, seq_len, d_model),
#         torch.randn(batch_size, seq_len, d_model)
#     ]
    
#     prob_multi = multi_layer_disc(hidden_states)
#     print(f"Input: {len(hidden_states)} layers, each shape {hidden_states[0].shape}")
#     print(f"Output shape: {prob_multi.shape}")
#     print(f"Output range: [{prob_multi.min().item():.4f}, {prob_multi.max().item():.4f}]")
#     print(f"Output mean: {prob_multi.mean().item():.4f}")
    
#     # Test get_logits with multiple layers
#     logits_multi = multi_layer_disc.get_logits(hidden_states)
#     print(f"\nLogits shape: {logits_multi.shape}")
#     print(f"Logits range: [{logits_multi.min().item():.4f}, {logits_multi.max().item():.4f}]")
    
#     print(f"\nMulti-layer Discriminator parameters: {sum(p.numel() for p in multi_layer_disc.parameters()):,}")
#     print(f"Note: Input dim = {d_model} * {3} = {d_model * 3}, hidden dim = {2048}")

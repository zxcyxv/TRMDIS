"""
Transformer backbone with RMSNorm, SwiGLU, and RoPE
Based on modern LLM architectures (Llama-style)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    x / sqrt(mean(x²) + eps) * scale
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation: SiLU(Wx) ⊙ Vx
    Used in feedforward networks
    """

    def __init__(self, dim: int, hidden_dim: int, use_spectral_norm: bool = False):
        super().__init__()
        if use_spectral_norm:
            self.w = nn.utils.spectral_norm(nn.Linear(dim, hidden_dim, bias=False))
            self.v = nn.utils.spectral_norm(nn.Linear(dim, hidden_dim, bias=False))
            self.out_proj = nn.utils.spectral_norm(nn.Linear(hidden_dim, dim, bias=False))
        else:
            self.w = nn.Linear(dim, hidden_dim, bias=False)
            self.v = nn.Linear(dim, hidden_dim, bias=False)
            self.out_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.w(x)) * self.v(x))


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Applies rotations to query and key vectors based on position
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies for dim//2 (since we split and rotate pairs)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        # Don't duplicate - keep as dim//2
        self.register_buffer('cos_cached', freqs.cos())  # [max_seq_len, dim//2]
        self.register_buffer('sin_cached', freqs.sin())  # [max_seq_len, dim//2]

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) for applying rotation

        Args:
            x: Input tensor (not used, just for reference)
            seq_len: Sequence length

        Returns:
            (cos, sin): [seq_len, dim]
        """
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor

    Args:
        x: [B, n_heads, seq_len, head_dim]
        cos: [seq_len, head_dim//2]
        sin: [seq_len, head_dim//2]

    Returns:
        Rotated x
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split x into two halves
    x1, x2 = x.chunk(2, dim=-1)  # Each [B, n_heads, seq_len, head_dim//2]

    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_spectral_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if use_spectral_norm:
            self.q_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model, bias=False))
            self.k_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model, bias=False))
            self.v_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model, bias=False))
            self.out_proj = nn.utils.spectral_norm(nn.Linear(d_model, d_model, bias=False))
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
            mask: [B, seq_len] - attention mask (1=valid, 0=pad)
            rope_cos: [seq_len, head_dim]
            rope_sin: [seq_len, head_dim]

        Returns:
            output: [B, seq_len, d_model]
        """
        B, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, n_heads, seq_len, head_dim]

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [B, n_heads, seq_len, seq_len]

        # Apply attention mask
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
        attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, head_dim]

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class TransformerLayer(nn.Module):
    """
    Single Transformer layer with RMSNorm and SwiGLU
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_spectral_norm: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_spectral_norm)

        self.ff_norm = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, use_spectral_norm)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
            mask: [B, seq_len]
            rope_cos: [seq_len, head_dim]
            rope_sin: [seq_len, head_dim]

        Returns:
            output: [B, seq_len, d_model]
        """
        # Pre-norm attention (no residual)
        x = self.dropout(self.attn(self.attn_norm(x), mask, rope_cos, rope_sin))

        # Pre-norm feedforward (no residual)
        x = self.dropout(self.ff(self.ff_norm(x)))

        return x


class TransformerBackbone(nn.Module):
    """
    2-layer Transformer encoder with RMSNorm, SwiGLU, and RoPE
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 16,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Rotary position embedding
        head_dim = d_model // n_heads
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, use_spectral_norm)
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model] - embedded sequences
            mask: [B, seq_len] - attention mask

        Returns:
            output: [B, seq_len, d_model]
        """
        B, seq_len, _ = x.shape

        # Get RoPE embeddings
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Pass through layers
        for layer in self.layers:
            x = layer(x, mask, rope_cos, rope_sin)

        # Final normalization
        x = self.final_norm(x)

        return x

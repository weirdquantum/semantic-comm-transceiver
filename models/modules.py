"""
Core neural-network building blocks for the semantic transceiver.

Key innovation:  **SNR-Adaptive Multi-Head Attention**
  -- embeds the current channel SNR into the Transformer's Query vectors,
     allowing the encoder / decoder to dynamically adjust which semantic
     features to prioritise under varying channel conditions.
  Reference: SNR-EQ-JSCC (arXiv 2501.04732, 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Positional Encoding (standard sinusoidal)
# ====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ====================================================================
# SNR Embedding  –  maps scalar SNR (dB) → d_model vector
# ====================================================================
class SNREmbedding(nn.Module):
    """Two-layer MLP that lifts a scalar SNR value to a d_model-dim vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, snr_db: torch.Tensor):
        """
        Args:
            snr_db: (B,) or (B,1) — SNR in dB.
        Returns:
            (B, d_model)
        """
        if snr_db.dim() == 1:
            snr_db = snr_db.unsqueeze(-1)
        return self.net(snr_db)


# ====================================================================
# SNR-Adaptive Multi-Head Attention
# ====================================================================
class SNRAdaptiveMultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product MHA augmented with a
    Channel-Adaptive Query (CAQ) path.

    Q_adaptive = W_q(x) + W_caq(SNR_emb)

    This lets attention scores shift according to channel quality —
    e.g. under low SNR the model can learn to put more weight on
    high-energy / robust semantic features.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.caq = nn.Linear(d_model, d_model)       # channel-adaptive query

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, snr_emb, mask=None):
        """
        Args:
            q, k, v: (B, L, d_model)
            snr_emb: (B, d_model) from SNREmbedding
            mask:    (B, 1, 1, L) or broadcastable
        Returns:
            (B, L, d_model)
        """
        B = q.size(0)

        Q = self.W_q(q) + self.caq(snr_emb).unsqueeze(1)   # CAQ injection
        K = self.W_k(k)
        V = self.W_v(v)

        # reshape → (B, n_heads, L, d_k)
        Q = Q.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)

        ctx = ctx.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(ctx)


# ====================================================================
# Standard Multi-Head Attention (no SNR awareness — used in baseline)
# ====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.W_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(ctx)


# ====================================================================
# Feed-Forward Network
# ====================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ====================================================================
# SNR-Adaptive Encoder Layer
# ====================================================================
class SNRAdaptiveEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SNRAdaptiveMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, snr_emb, mask=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, snr_emb, mask))
        _x = self.norm2(x)
        x = x + self.drop2(self.ff(_x))
        return x


# ====================================================================
# Baseline Encoder Layer (no SNR awareness)
# ====================================================================
class BaselineEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, mask))
        _x = self.norm2(x)
        x = x + self.drop2(self.ff(_x))
        return x


# ====================================================================
# Decoder Layer  –  cross-attention to encoder memory + causal self-attn
# ====================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, mem_mask=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, tgt_mask))
        _x = self.norm2(x)
        x = x + self.drop2(self.cross_attn(_x, memory, memory, mem_mask))
        _x = self.norm3(x)
        x = x + self.drop3(self.ff(_x))
        return x


# ====================================================================
# SNR-Adaptive Decoder Layer  –  cross-attention is channel-aware
# ====================================================================
class SNRAdaptiveDecoderLayer(nn.Module):
    """Decoder layer whose cross-attention uses SNRAdaptiveMultiHeadAttention.

    Self-attention remains standard (the target sequence has no direct
    relationship to channel quality), but cross-attention benefits from
    knowing the SNR so the decoder can weight memory features according
    to channel reliability.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = SNRAdaptiveMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, snr_emb, tgt_mask=None, mem_mask=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, tgt_mask))
        _x = self.norm2(x)
        x = x + self.drop2(self.cross_attn(_x, memory, memory, snr_emb, mem_mask))
        _x = self.norm3(x)
        x = x + self.drop3(self.ff(_x))
        return x


# ====================================================================
# FiLM  –  SNR-conditioned feature-wise affine modulation
# ====================================================================
class FiLM(nn.Module):
    """h -> h * (1 + gamma(c)) + beta(c).

    Zero-initialised so that at the start of training the layer is the
    identity and the model behaves exactly like its un-conditioned version.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, cond):
        """h: (B, L, F),  cond: (B, cond_dim)."""
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


# ====================================================================
# Channel encoder / decoder  (dense projection + power normalisation)
# ====================================================================
class ChannelEncoder(nn.Module):
    """Maps d_model → channel_dim per time-step, then power-normalises.

    Args:
        film_dim:      if set, the hidden layer is FiLM-modulated by an
                       SNR embedding of this size.
        masked_power:  if True, padded positions transmit zero energy and
                       the per-sentence energy budget (L * channel_dim) is
                       spent on the valid tokens only.  Otherwise every
                       position, padding included, gets unit average power.
    """

    def __init__(self, d_model: int, channel_dim: int,
                 film_dim: int = None, masked_power: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, channel_dim),
        )
        self.film = FiLM(film_dim, 256) if film_dim else None
        self.masked_power = masked_power

    def forward(self, x, snr_emb=None, valid_mask=None):
        h = self.net[1](self.net[0](x))
        if self.film is not None:
            h = self.film(h, snr_emb)
        out = self.net[2](h).float()                # (B, L, channel_dim)

        if self.masked_power and valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
            budget = out.size(-2) * out.size(-1)    # same energy as unit power on all slots
            energy = torch.sum(out ** 2, dim=(-2, -1), keepdim=True) + 1e-12
            return out * torch.sqrt(budget / energy)

        # power normalisation: unit average power per sample
        power = torch.mean(out ** 2, dim=(-2, -1), keepdim=True) + 1e-12
        return out / torch.sqrt(power)


class ChannelDecoder(nn.Module):
    """Maps channel_dim → d_model per time-step (optionally SNR-conditioned)."""

    def __init__(self, channel_dim: int, d_model: int, film_dim: int = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel_dim, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
        )
        self.film = FiLM(film_dim, 256) if film_dim else None

    def forward(self, x, snr_emb=None):
        h = self.net[1](self.net[0](x))
        if self.film is not None:
            h = self.film(h, snr_emb)
        return self.net[2](h)

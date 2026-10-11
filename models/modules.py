"""
Neural-network building blocks for the semantic transceiver.

SNR-adaptive attention (inspired by SNR-EQ-JSCC, arXiv 2501.04732) adds an
embedding of the channel SNR to the attention queries,
    Q = W_q(x) + W_caq(snr_emb),
so that attention can shift with channel quality.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding followed by dropout."""

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


class SNREmbedding(nn.Module):
    """Two-layer MLP that maps a scalar SNR (dB) to a d_model vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, snr_db: torch.Tensor):
        """snr_db: (B,) -> (B, d_model)."""
        return self.net(snr_db.unsqueeze(-1))


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    With ``snr_adaptive=True`` a channel-adaptive query term ``caq(snr_emb)``
    is added to the queries.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 snr_adaptive: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.caq = nn.Linear(d_model, d_model) if snr_adaptive else None
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, snr_emb=None):
        """q, k, v: (B, L, d_model); mask broadcastable to (B, H, Lq, Lk)."""
        B = q.size(0)
        Q = self.W_q(q)
        if self.caq is not None:
            Q = Q + self.caq(snr_emb).unsqueeze(1)
        Q = Q.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(ctx)


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


class EncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer (self-attention optionally SNR-adaptive)."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, snr_adaptive=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, snr_adaptive)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, snr_emb=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, mask, snr_emb))
        _x = self.norm2(x)
        return x + self.drop2(self.ff(_x))


class DecoderLayer(nn.Module):
    """Pre-LN Transformer decoder layer.

    Causal self-attention is always standard; with ``snr_adaptive=True`` the
    cross-attention over the received features is SNR-adaptive.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, snr_adaptive=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, snr_adaptive)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, mem_mask=None, snr_emb=None):
        _x = self.norm1(x)
        x = x + self.drop1(self.self_attn(_x, _x, _x, tgt_mask))
        _x = self.norm2(x)
        x = x + self.drop2(self.cross_attn(_x, memory, memory, mem_mask, snr_emb))
        _x = self.norm3(x)
        return x + self.drop3(self.ff(_x))


class FiLM(nn.Module):
    """h -> h * (1 + gamma(c)) + beta(c).

    Zero-initialised, so the layer starts as the identity and the model
    initially behaves exactly like its unconditioned version.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, cond):
        """h: (B, L, F), cond: (B, cond_dim)."""
        gamma, beta = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class ChannelEncoder(nn.Module):
    """Maps d_model -> channel_dim per token, then power-normalises.

    Args:
        film_dim:     if set, the hidden layer is FiLM-modulated by an SNR
                      embedding of this size.
        masked_power: if True, padding transmits zero energy and the
                      per-sentence budget (L * channel_dim) goes to the valid
                      tokens; otherwise every position gets unit average power.
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

        if self.masked_power:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
            budget = out.size(-2) * out.size(-1)
            energy = torch.sum(out ** 2, dim=(-2, -1), keepdim=True) + 1e-12
            return out * torch.sqrt(budget / energy)

        power = torch.mean(out ** 2, dim=(-2, -1), keepdim=True) + 1e-12
        return out / torch.sqrt(power)


class ChannelDecoder(nn.Module):
    """Maps channel_dim -> d_model per token (optionally SNR-conditioned)."""

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

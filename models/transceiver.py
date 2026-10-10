"""
End-to-end semantic communication transceivers.

* BaselineDeepSC  — reproduction of the original DeepSC (Xie et al., 2021)
* ImprovedDeepSC  — enhanced with:
      1. SNR-Adaptive Attention in the semantic encoder / decoder
      2. SNR-conditioned (FiLM) channel encoder / decoder
      3. Length-aware power normalisation (no energy spent on padding)
      4. CSI-aware receiver: decoder conditions on the effective SNR
      5. Feature-denoising composite loss (handled externally in losses.py)
      6. LLM fusion refinement at inference time (handled in llm_refiner.py)

Both models share the decoding utilities in ``_DeepSCBase`` (greedy,
batched beam search, per-position candidate extraction).
"""

import torch
import torch.nn as nn

from .modules import (
    PositionalEncoding,
    SNREmbedding,
    SNRAdaptiveEncoderLayer,
    BaselineEncoderLayer,
    DecoderLayer,
    SNRAdaptiveDecoderLayer,
    ChannelEncoder,
    ChannelDecoder,
)
from .channel import ChannelModel


# ====================================================================
# helpers
# ====================================================================
def _causal_mask(seq_len: int, device: torch.device):
    """Lower-triangular causal mask: (1, 1, L, L)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def _pad_mask(ids: torch.Tensor, pad_idx: int = 0):
    """(B, 1, 1, L) — True where NOT padded."""
    return (ids != pad_idx).unsqueeze(1).unsqueeze(2)


def _repeat_ctx(ctx, k):
    return None if ctx is None else ctx.repeat_interleave(k, dim=0)


# ====================================================================
# Shared decoding utilities
# ====================================================================
class _DeepSCBase(nn.Module):
    """Subclasses implement ``_transmit`` and ``_decode``.

    _transmit(src, snr_db, channel_type) -> (enc_out, memory, src_mask, ctx)
    _decode(trg, memory, src_mask, ctx)  -> logits (B, L_trg, V)
    """

    def forward(self, src, trg, snr_db, channel_type="awgn"):
        """
        Args:
            src: (B, L_src) source token ids
            trg: (B, L_trg) target token ids (teacher forcing)
            snr_db: (B,) SNR in dB
            channel_type: 'awgn' | 'rayleigh'
        Returns:
            logits: (B, L_trg, V)
            enc_out: (B, L_src, d_model)
            ch_dec_out: (B, L_src, d_model)
        """
        enc_out, memory, src_mask, ctx = self._transmit(src, snr_db, channel_type)
        return self._decode(trg, memory, src_mask, ctx), enc_out, memory

    # ------------------------------------------------------------------
    def _greedy(self, memory, src_mask, ctx, max_len, start_idx, end_idx):
        B = memory.size(0)
        ys = torch.full((B, 1), start_idx, dtype=torch.long, device=memory.device)
        for _ in range(max_len - 1):
            logits = self._decode(ys, memory, src_mask, ctx)[:, -1:, :]
            next_tok = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == end_idx).all():
                break
        return ys

    def _beam(self, memory, src_mask, ctx, max_len, start_idx, end_idx, beam_width):
        """Batched beam search: all B*K hypotheses are decoded together."""
        B, K = memory.size(0), beam_width
        device = memory.device
        memory = memory.repeat_interleave(K, dim=0)
        src_mask = src_mask.repeat_interleave(K, dim=0)
        ctx = _repeat_ctx(ctx, K)

        ys = torch.full((B * K, 1), start_idx, dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0                      # only one live hypothesis at start
        finished = torch.zeros(B * K, dtype=torch.bool, device=device)
        offset = (torch.arange(B, device=device) * K).unsqueeze(1)
        V = self.vocab_size
        # finished hypotheses can only be extended with PAD, at no cost
        pad_only = torch.full((1, V), float("-inf"), device=device)
        pad_only[0, self.pad_idx] = 0.0

        for _ in range(max_len - 1):
            logits = self._decode(ys, memory, src_mask, ctx)[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)              # (BK, V)
            logp = torch.where(finished.unsqueeze(1), pad_only, logp)

            cand = (scores.view(-1, 1) + logp).view(B, K * V)
            scores, top_idx = cand.topk(K, dim=-1)                # (B, K)
            sel = (offset + top_idx // V).view(-1)
            tok = (top_idx % V).view(-1)

            ys = torch.cat([ys[sel], tok.unsqueeze(1)], dim=1)
            finished = finished[sel] | (tok == end_idx)
            if finished.all():
                break

        best = offset.view(-1) + scores.argmax(dim=-1)
        return ys[best]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, src, snr_db, channel_type="awgn", max_len=32,
               start_idx=1, end_idx=2, beam_width=1, topk=0):
        """Transmit ``src`` once and decode it.

        Args:
            beam_width: 1 = greedy, >1 = batched beam search.
            topk:       if >0, also return decoder candidates for every
                        output position (used by the LLM refiner).
        Returns:
            ys:   (B, L) token ids starting with START
            info: None, or dict with
                  conf      (B, L-1) probability of the chosen token ys[:, 1:]
                  cand_ids  (B, L-1, topk) top-k token ids at each position
                  cand_logp (B, L-1, topk) their log-probabilities
        """
        self.eval()
        _, memory, src_mask, ctx = self._transmit(src, snr_db, channel_type)
        if beam_width > 1:
            ys = self._beam(memory, src_mask, ctx, max_len, start_idx, end_idx,
                            beam_width)
        else:
            ys = self._greedy(memory, src_mask, ctx, max_len, start_idx, end_idx)
        if topk <= 0:
            return ys, None

        # one teacher-forced pass over the decoded sequence, same channel output
        logits = self._decode(ys[:, :-1], memory, src_mask, ctx).float()
        logp = torch.log_softmax(logits, dim=-1)
        conf = logp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).exp()
        cand_logp, cand_ids = logp.topk(topk, dim=-1)
        return ys, {"conf": conf, "cand_ids": cand_ids, "cand_logp": cand_logp}

    @torch.no_grad()
    def greedy_decode(self, src, snr_db, channel_type="awgn",
                      max_len=32, start_idx=1, end_idx=2):
        """Auto-regressive greedy decoding for inference."""
        return self.decode(src, snr_db, channel_type, max_len,
                           start_idx, end_idx)[0]

    @torch.no_grad()
    def beam_search_decode(self, src, snr_db, channel_type="awgn",
                           max_len=32, start_idx=1, end_idx=2, beam_width=5):
        return self.decode(src, snr_db, channel_type, max_len,
                           start_idx, end_idx, beam_width=beam_width)[0]


# ====================================================================
# Baseline DeepSC
# ====================================================================
class BaselineDeepSC(_DeepSCBase):
    """Faithful re-implementation of the original DeepSC architecture."""

    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512,
                 n_enc_layers=4, n_dec_layers=4, channel_dim=16,
                 max_len=32, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.enc_layers = nn.ModuleList([
            BaselineEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        self.ch_enc = ChannelEncoder(d_model, channel_dim)
        self.channel = ChannelModel()
        self.ch_dec = ChannelDecoder(channel_dim, d_model)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def _transmit(self, src, snr_db, channel_type):
        src_mask = _pad_mask(src, self.pad_idx)
        enc_out = self.encode(src, src_mask)
        tx = self.ch_enc(enc_out)
        rx, _ = self.channel(tx, snr_db, channel_type)
        return enc_out, self.ch_dec(rx), src_mask, None

    def _decode(self, trg, memory, src_mask, ctx):
        x = self.pos_enc(self.embedding(trg))
        trg_mask = _causal_mask(trg.size(1), trg.device)
        for layer in self.dec_layers:
            x = layer(x, memory, trg_mask, src_mask)
        return self.fc_out(self.dec_norm(x))


# ====================================================================
# Improved DeepSC
# ====================================================================
class ImprovedDeepSC(_DeepSCBase):
    """
    Enhanced DeepSC.  Each contribution can be switched off for ablation:

      use_film      SNR-conditioned FiLM in the channel encoder / decoder
      masked_power  power normalisation over valid tokens only
      csi_snr       receiver conditions on the effective (post-fading) SNR
                    instead of the average SNR (no effect on AWGN)
    """

    SNR_EFF_RANGE = (-30.0, 40.0)   # clamp for deep fades before embedding

    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512,
                 n_enc_layers=4, n_dec_layers=4, channel_dim=16,
                 max_len=32, dropout=0.1, pad_idx=0,
                 use_film=True, masked_power=True, csi_snr=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_film = use_film
        self.csi_snr = csi_snr

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # SNR embedding shared across encoder, channel coder and decoder
        self.snr_emb = SNREmbedding(d_model)

        # SNR-adaptive encoder
        self.enc_layers = nn.ModuleList([
            SNRAdaptiveEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        film_dim = d_model if use_film else None
        self.ch_enc = ChannelEncoder(d_model, channel_dim, film_dim, masked_power)
        self.channel = ChannelModel()
        self.ch_dec = ChannelDecoder(channel_dim, d_model, film_dim)

        # SNR-adaptive decoder (cross-attention is channel-aware)
        self.dec_layers = nn.ModuleList([
            SNRAdaptiveDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, snr_e, src_mask=None):
        x = self.pos_enc(self.embedding(src))
        for layer in self.enc_layers:
            x = layer(x, snr_e, src_mask)
        return self.enc_norm(x)

    def _transmit(self, src, snr_db, channel_type):
        src_mask = _pad_mask(src, self.pad_idx)
        # transmitter only knows the average SNR
        snr_tx = self.snr_emb(snr_db)
        enc_out = self.encode(src, snr_tx, src_mask)

        tx = self.ch_enc(enc_out, snr_tx, src != self.pad_idx)
        rx, snr_eff = self.channel(tx, snr_db, channel_type)

        # receiver has perfect CSI, so it may condition on the effective SNR
        if self.csi_snr:
            snr_rx = self.snr_emb(snr_eff.clamp(*self.SNR_EFF_RANGE))
        else:
            snr_rx = snr_tx
        return enc_out, self.ch_dec(rx, snr_rx), src_mask, snr_rx

    def _decode(self, trg, memory, src_mask, snr_rx):
        x = self.pos_enc(self.embedding(trg))
        trg_mask = _causal_mask(trg.size(1), trg.device)
        for layer in self.dec_layers:
            x = layer(x, memory, snr_rx, trg_mask, src_mask)
        return self.fc_out(self.dec_norm(x))

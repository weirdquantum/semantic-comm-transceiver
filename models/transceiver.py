"""
End-to-end semantic communication transceivers.

* BaselineDeepSC — reproduction of DeepSC (Xie et al., 2021).
* ImprovedDeepSC — adds, each switchable for ablation:
      - SNR-adaptive attention in the semantic encoder and decoder
      - SNR-conditioned (FiLM) channel encoder / decoder      (use_film)
      - length-aware power normalisation                       (masked_power)
      - CSI-aware receiver conditioned on the effective SNR    (csi_snr)

Both share the transmit / decode path and the decoding utilities
(greedy, batched beam search, per-position candidate extraction).
"""

import torch
import torch.nn as nn

from .channel import apply_channel
from .modules import (
    PositionalEncoding,
    SNREmbedding,
    EncoderLayer,
    DecoderLayer,
    ChannelEncoder,
    ChannelDecoder,
)


def _causal_mask(seq_len: int, device: torch.device):
    """Lower-triangular causal mask: (1, 1, L, L)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)


def _pad_mask(ids: torch.Tensor, pad_idx: int = 0):
    """(B, 1, 1, L) — True where NOT padded."""
    return (ids != pad_idx).unsqueeze(1).unsqueeze(2)


class DeepSC(nn.Module):
    """Transformer semantic codec + dense channel codec + differentiable channel.

    With ``snr_adaptive=False`` this is the original DeepSC; the remaining
    flags only apply to the SNR-adaptive (improved) variant.
    """

    SNR_EFF_RANGE = (-30.0, 40.0)   # clamp deep fades before embedding

    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512,
                 n_enc_layers=4, n_dec_layers=4, channel_dim=16,
                 max_len=32, dropout=0.1, pad_idx=0, snr_adaptive=False,
                 use_film=False, masked_power=False, csi_snr=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.snr_adaptive = snr_adaptive
        self.csi_snr = snr_adaptive and csi_snr
        film_dim = d_model if snr_adaptive and use_film else None

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        if snr_adaptive:
            self.snr_emb = SNREmbedding(d_model)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, snr_adaptive)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        self.ch_enc = ChannelEncoder(d_model, channel_dim, film_dim,
                                     snr_adaptive and masked_power)
        self.ch_dec = ChannelDecoder(channel_dim, d_model, film_dim)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, snr_adaptive)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    def _transmit(self, src, snr_db, channel_type):
        """Encode, send through the channel and decode the channel symbols.

        Returns (enc_out, memory, src_mask, snr_rx) where snr_rx is the SNR
        embedding available at the receiver (None for the baseline).
        """
        src_mask = _pad_mask(src, self.pad_idx)
        # the transmitter only knows the average SNR
        snr_tx = self.snr_emb(snr_db) if self.snr_adaptive else None

        x = self.pos_enc(self.embedding(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask, snr_tx)
        enc_out = self.enc_norm(x)

        tx = self.ch_enc(enc_out, snr_tx, src != self.pad_idx)
        rx, snr_eff = apply_channel(tx, snr_db, channel_type)

        # with perfect CSI the receiver may condition on the effective SNR
        snr_rx = snr_tx
        if self.csi_snr:
            snr_rx = self.snr_emb(snr_eff.clamp(*self.SNR_EFF_RANGE))
        return enc_out, self.ch_dec(rx, snr_rx), src_mask, snr_rx

    def _decode(self, trg, memory, src_mask, snr_rx):
        x = self.pos_enc(self.embedding(trg))
        trg_mask = _causal_mask(trg.size(1), trg.device)
        for layer in self.dec_layers:
            x = layer(x, memory, trg_mask, src_mask, snr_rx)
        return self.fc_out(self.dec_norm(x))

    def forward(self, src, trg, snr_db, channel_type="awgn"):
        """Teacher-forced pass.

        Returns logits (B, L_trg, V), the semantic encoder output and the
        channel decoder output (both (B, L_src, d_model)).
        """
        enc_out, memory, src_mask, snr_rx = self._transmit(src, snr_db, channel_type)
        return self._decode(trg, memory, src_mask, snr_rx), enc_out, memory

    # ------------------------------------------------------------------
    def _greedy(self, memory, src_mask, snr_rx, max_len, start_idx, end_idx):
        B = memory.size(0)
        ys = torch.full((B, 1), start_idx, dtype=torch.long, device=memory.device)
        for _ in range(max_len - 1):
            next_tok = self._decode(ys, memory, src_mask, snr_rx)[:, -1:, :].argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == end_idx).all():
                break
        return ys

    def _beam(self, memory, src_mask, snr_rx, max_len, start_idx, end_idx, beam_width):
        """Batched beam search: all B*K hypotheses are decoded together."""
        B, K, V = memory.size(0), beam_width, self.vocab_size
        device = memory.device
        memory = memory.repeat_interleave(K, dim=0)
        src_mask = src_mask.repeat_interleave(K, dim=0)
        if snr_rx is not None:
            snr_rx = snr_rx.repeat_interleave(K, dim=0)

        ys = torch.full((B * K, 1), start_idx, dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0                      # one live hypothesis at the start
        finished = torch.zeros(B * K, dtype=torch.bool, device=device)
        offset = (torch.arange(B, device=device) * K).unsqueeze(1)
        # finished hypotheses can only be extended with PAD, at no cost
        pad_only = torch.full((1, V), float("-inf"), device=device)
        pad_only[0, self.pad_idx] = 0.0

        for _ in range(max_len - 1):
            logits = self._decode(ys, memory, src_mask, snr_rx)[:, -1, :].float()
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

        return ys[offset.view(-1) + scores.argmax(dim=-1)]

    @torch.no_grad()
    def decode(self, src, snr_db, channel_type="awgn", max_len=32,
               start_idx=1, end_idx=2, beam_width=1, topk=0):
        """Transmit ``src`` once and decode it.

        Args:
            beam_width: 1 = greedy, >1 = batched beam search.
            topk:       if >0, also return the decoder's candidates for every
                        output position (used by the LLM refiner).
        Returns:
            ys:   (B, L) token ids starting with START
            info: None, or dict with
                  conf      (B, L-1) probability of the chosen token ys[:, 1:]
                  cand_ids  (B, L-1, topk) top-k token ids at each position
                  cand_logp (B, L-1, topk) their log-probabilities
        """
        self.eval()
        _, memory, src_mask, snr_rx = self._transmit(src, snr_db, channel_type)
        if beam_width > 1:
            ys = self._beam(memory, src_mask, snr_rx, max_len, start_idx, end_idx,
                            beam_width)
        else:
            ys = self._greedy(memory, src_mask, snr_rx, max_len, start_idx, end_idx)
        if topk <= 0:
            return ys, None

        # one teacher-forced pass over the decoded sequence, same channel output
        logits = self._decode(ys[:, :-1], memory, src_mask, snr_rx).float()
        logp = torch.log_softmax(logits, dim=-1)
        conf = logp.gather(-1, ys[:, 1:].unsqueeze(-1)).squeeze(-1).exp()
        cand_logp, cand_ids = logp.topk(topk, dim=-1)
        return ys, {"conf": conf, "cand_ids": cand_ids, "cand_logp": cand_logp}


class BaselineDeepSC(DeepSC):
    """Reproduction of the original DeepSC architecture."""

    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, snr_adaptive=False, **kwargs)


class ImprovedDeepSC(DeepSC):
    """DeepSC with SNR-adaptive attention and the switchable improvements."""

    def __init__(self, vocab_size, use_film=True, masked_power=True,
                 csi_snr=True, **kwargs):
        super().__init__(vocab_size, snr_adaptive=True, use_film=use_film,
                         masked_power=masked_power, csi_snr=csi_snr, **kwargs)

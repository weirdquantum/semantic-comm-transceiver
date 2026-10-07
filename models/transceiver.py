"""
End-to-end semantic communication transceivers.

* BaselineDeepSC  — reproduction of the original DeepSC (Xie et al., 2021)
* ImprovedDeepSC  — enhanced with:
      1. SNR-Adaptive Attention in the semantic encoder
      2. Semantic-aware composite loss (handled externally in losses.py)
      3. LLM refinement at inference time (handled in llm_refiner.py)
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


# ====================================================================
# Baseline DeepSC
# ====================================================================
class BaselineDeepSC(nn.Module):
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
        self.ch_enc = ChannelEncoder(d_model, channel_dim)
        self.channel = ChannelModel()
        self.ch_dec = ChannelDecoder(channel_dim, d_model)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.embedding(src))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

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
        src_mask = _pad_mask(src, self.pad_idx)

        # ---- Encoder ----
        enc_out = self.encode(src, src_mask)

        # ---- Channel ----
        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        ch_dec_out = self.ch_dec(rx)

        # ---- Decoder (teacher forcing) ----
        trg_emb = self.pos_enc(self.embedding(trg))
        trg_mask = _causal_mask(trg.size(1), trg.device)
        x = trg_emb
        for layer in self.dec_layers:
            x = layer(x, ch_dec_out, trg_mask, src_mask)
        logits = self.fc_out(x)
        return logits, enc_out, ch_dec_out

    @torch.no_grad()
    def greedy_decode(self, src, snr_db, channel_type="awgn",
                      max_len=32, start_idx=1, end_idx=2):
        """Auto-regressive greedy decoding for inference."""
        self.eval()
        B = src.size(0)
        device = src.device
        src_mask = _pad_mask(src, self.pad_idx)

        enc_out = self.encode(src, src_mask)
        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        memory = self.ch_dec(rx)

        ys = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            trg_emb = self.pos_enc(self.embedding(ys))
            trg_mask = _causal_mask(ys.size(1), device)
            x = trg_emb
            for layer in self.dec_layers:
                x = layer(x, memory, trg_mask, src_mask)
            logits = self.fc_out(x[:, -1:, :])
            next_tok = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == end_idx).all():
                break
        return ys


# ====================================================================
# Improved DeepSC  (with SNR-Adaptive Attention)
# ====================================================================
class ImprovedDeepSC(nn.Module):
    """
    Enhanced DeepSC with:
      * SNR-Adaptive Attention in the semantic encoder
      * SNR-Adaptive cross-attention in the semantic decoder
      * Composite loss integration via extra return values
    """

    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512,
                 n_enc_layers=4, n_dec_layers=4, channel_dim=16,
                 max_len=32, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # SNR embedding shared across encoder and decoder layers
        self.snr_emb = SNREmbedding(d_model)

        # SNR-adaptive encoder
        self.enc_layers = nn.ModuleList([
            SNRAdaptiveEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])

        self.ch_enc = ChannelEncoder(d_model, channel_dim)
        self.channel = ChannelModel()
        self.ch_dec = ChannelDecoder(channel_dim, d_model)

        # SNR-adaptive decoder (cross-attention is channel-aware)
        self.dec_layers = nn.ModuleList([
            SNRAdaptiveDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, snr_db, src_mask=None):
        x = self.pos_enc(self.embedding(src))
        snr_e = self.snr_emb(snr_db)
        for layer in self.enc_layers:
            x = layer(x, snr_e, src_mask)
        return x

    def forward(self, src, trg, snr_db, channel_type="awgn"):
        src_mask = _pad_mask(src, self.pad_idx)
        snr_e = self.snr_emb(snr_db)

        enc_out = self.encode(src, snr_db, src_mask)

        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        ch_dec_out = self.ch_dec(rx)

        trg_emb = self.pos_enc(self.embedding(trg))
        trg_mask = _causal_mask(trg.size(1), trg.device)
        x = trg_emb
        for layer in self.dec_layers:
            x = layer(x, ch_dec_out, snr_e, trg_mask, src_mask)
        logits = self.fc_out(x)
        return logits, enc_out, ch_dec_out

    @torch.no_grad()
    def greedy_decode(self, src, snr_db, channel_type="awgn",
                      max_len=32, start_idx=1, end_idx=2):
        self.eval()
        B = src.size(0)
        device = src.device
        src_mask = _pad_mask(src, self.pad_idx)
        snr_e = self.snr_emb(snr_db)

        enc_out = self.encode(src, snr_db, src_mask)
        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        memory = self.ch_dec(rx)

        ys = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            trg_emb = self.pos_enc(self.embedding(ys))
            trg_mask = _causal_mask(ys.size(1), device)
            x = trg_emb
            for layer in self.dec_layers:
                x = layer(x, memory, snr_e, trg_mask, src_mask)
            logits = self.fc_out(x[:, -1:, :])
            next_tok = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == end_idx).all():
                break
        return ys

    @torch.no_grad()
    def beam_search_decode(self, src, snr_db, channel_type="awgn",
                           max_len=32, start_idx=1, end_idx=2,
                           beam_width=5):
        """Beam search decoding for higher-quality output.

        Runs beam search independently per sample in the batch.
        Returns (B, L) token ids and (B, L) per-position confidence.
        """
        self.eval()
        B = src.size(0)
        device = src.device
        src_mask = _pad_mask(src, self.pad_idx)
        snr_e = self.snr_emb(snr_db)

        enc_out = self.encode(src, snr_db, src_mask)
        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        memory = self.ch_dec(rx)

        all_seqs = []
        all_confs = []
        for b in range(B):
            mem_b = memory[b:b+1]
            src_mask_b = src_mask[b:b+1]
            snr_e_b = snr_e[b:b+1]

            # Each beam: (log_prob, token_list, conf_list)
            beams = [(0.0, [start_idx], [1.0])]
            finished = []

            for _ in range(max_len - 1):
                candidates = []
                for log_p, seq, confs in beams:
                    if seq[-1] == end_idx:
                        finished.append((log_p, seq, confs))
                        continue
                    ys = torch.tensor([seq], dtype=torch.long, device=device)
                    trg_emb = self.pos_enc(self.embedding(ys))
                    trg_mask = _causal_mask(ys.size(1), device)
                    x = trg_emb
                    for layer in self.dec_layers:
                        x = layer(x, mem_b, snr_e_b, trg_mask, src_mask_b)
                    logits = self.fc_out(x[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)
                    topk_lp, topk_ids = log_probs.topk(beam_width, dim=-1)

                    for k in range(beam_width):
                        tid = topk_ids[0, k].item()
                        lp = topk_lp[0, k].item()
                        conf = probs[0, tid].item()
                        candidates.append(
                            (log_p + lp, seq + [tid], confs + [conf])
                        )

                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]

                if all(s[-1] == end_idx for _, s, _ in beams):
                    finished.extend(beams)
                    break

            finished.extend(beams)
            best = max(finished, key=lambda x: x[0])
            all_seqs.append(best[1])
            all_confs.append(best[2])

        # Pad to same length
        max_out_len = max(len(s) for s in all_seqs)
        padded_seqs = torch.full((B, max_out_len), 0, dtype=torch.long, device=device)
        padded_confs = torch.ones((B, max_out_len), device=device)
        for i, (seq, conf) in enumerate(zip(all_seqs, all_confs)):
            padded_seqs[i, :len(seq)] = torch.tensor(seq, device=device)
            padded_confs[i, :len(conf)] = torch.tensor(conf, device=device)

        return padded_seqs, padded_confs

    @torch.no_grad()
    def greedy_decode_with_confidence(self, src, snr_db, channel_type="awgn",
                                      max_len=32, start_idx=1, end_idx=2):
        """Greedy decode returning both token ids and per-position confidence.

        Returns:
            ys:    (B, L) predicted token ids
            confs: (B, L) max softmax probability at each decoded position
        """
        self.eval()
        B = src.size(0)
        device = src.device
        src_mask = _pad_mask(src, self.pad_idx)
        snr_e = self.snr_emb(snr_db)

        enc_out = self.encode(src, snr_db, src_mask)
        tx = self.ch_enc(enc_out)
        rx = self.channel(tx, snr_db, channel_type)
        memory = self.ch_dec(rx)

        ys = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        all_confs = [torch.ones(B, 1, device=device)]
        for _ in range(max_len - 1):
            trg_emb = self.pos_enc(self.embedding(ys))
            trg_mask = _causal_mask(ys.size(1), device)
            x = trg_emb
            for layer in self.dec_layers:
                x = layer(x, memory, snr_e, trg_mask, src_mask)
            logits = self.fc_out(x[:, -1:, :])
            probs = torch.softmax(logits, dim=-1)
            max_prob, _ = probs.max(dim=-1)
            next_tok = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_tok], dim=1)
            all_confs.append(max_prob)
            if (next_tok == end_idx).all():
                break
        confs = torch.cat(all_confs, dim=1)
        return ys, confs

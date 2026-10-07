"""
Composite loss for semantic communication training.

L_total = alpha * L_CE  +  beta * L_semantic

* L_CE        — standard cross-entropy on per-token predictions.
* L_semantic  — 1 - cosine_similarity between the encoder's semantic
                representation and the post-channel semantic representation.
                This term encourages the channel encoder / decoder to
                maximally preserve semantic content through the physical
                channel, going beyond token-level accuracy.

For *evaluation* we additionally use external metrics (BLEU, BERTScore)
computed in evaluate.py, but those are not used during training to
keep the training loop efficient and fully differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticConsistencyLoss(nn.Module):
    """
    Cosine-distance between the mean-pooled encoder output (before channel)
    and the mean-pooled channel-decoder output (after channel).

    A small value means the channel preserved semantics well.
    """

    def forward(self, enc_out: torch.Tensor, ch_dec_out: torch.Tensor,
                pad_mask: torch.Tensor = None):
        """
        Args:
            enc_out:    (B, L, d_model) — semantic encoder output
            ch_dec_out: (B, L, d_model) — channel decoder output
            pad_mask:   (B, L) — True where NOT padded
        """
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()            # (B, L, 1)
            enc_mean = (enc_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            dec_mean = (ch_dec_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            enc_mean = enc_out.mean(dim=1)
            dec_mean = ch_dec_out.mean(dim=1)

        cos_sim = F.cosine_similarity(enc_mean, dec_mean, dim=-1)   # (B,)
        return (1.0 - cos_sim).mean()


class SoftEmbeddingSemanticLoss(nn.Module):
    """
    Uses the decoder's softmax probability distribution to compute
    a differentiable 'soft' embedding for each output position, then
    compares it to the source embedding via cosine distance.

    This provides a fully differentiable, BERT-free semantic loss that
    directly measures whether the reconstructed meaning matches the source.
    """

    def forward(self, logits: torch.Tensor, src_ids: torch.Tensor,
                embedding_weight: torch.Tensor, pad_idx: int = 0):
        """
        Args:
            logits:           (B, L_trg, V) — decoder output logits
            src_ids:          (B, L_src) — source token ids
            embedding_weight: (V, d_model) — shared embedding matrix
            pad_idx:          padding token id
        """
        probs = F.softmax(logits, dim=-1)                              # (B, L, V)
        pred_emb = torch.matmul(probs, embedding_weight)              # (B, L, d)

        src_emb = F.embedding(src_ids, embedding_weight)              # (B, L, d)

        src_mask = (src_ids != pad_idx).unsqueeze(-1).float()
        pred_mask = torch.ones(pred_emb.shape[:2], device=pred_emb.device).unsqueeze(-1)

        src_mean = (src_emb * src_mask).sum(1) / src_mask.sum(1).clamp(min=1)
        pred_mean = (pred_emb * pred_mask).sum(1) / pred_mask.sum(1).clamp(min=1)

        cos_sim = F.cosine_similarity(src_mean, pred_mean, dim=-1)
        return (1.0 - cos_sim).mean()


class CompositeLoss(nn.Module):
    """
    L = alpha * CE  +  beta * Semantic

    The semantic term can come from either:
      - `SemanticConsistencyLoss` (channel-preservation, needs enc_out & ch_dec_out)
      - `SoftEmbeddingSemanticLoss` (output-level, needs logits & embedding)
    Both are fully differentiable.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0,
                 alpha: float = 1.0, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.sem_channel = SemanticConsistencyLoss()
        self.sem_embed = SoftEmbeddingSemanticLoss()

    def forward(self, logits, targets, enc_out=None, ch_dec_out=None,
                src_ids=None, embedding_weight=None, pad_mask=None,
                pad_idx=0, snr_db=None):
        """
        Returns (total_loss, ce_loss, sem_loss).

        When *snr_db* is provided the effective beta is scaled by
        sigmoid((mean_snr - 0) / 5) so that semantic loss contributes
        less at low SNR (where CE-driven reconstruction matters most)
        and more at high SNR (where semantic refinement helps).
        """
        ce_loss = self.ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        sem_loss = torch.tensor(0.0, device=logits.device)
        if self.beta > 0:
            if enc_out is not None and ch_dec_out is not None:
                sem_loss = self.sem_channel(enc_out, ch_dec_out, pad_mask)
            elif src_ids is not None and embedding_weight is not None:
                sem_loss = self.sem_embed(logits, src_ids, embedding_weight, pad_idx)

        if self.beta > 0 and snr_db is not None:
            beta_eff = self.beta * torch.sigmoid(snr_db.mean() / 5.0)
        else:
            beta_eff = self.beta

        total = self.alpha * ce_loss + beta_eff * sem_loss
        return total, ce_loss, sem_loss

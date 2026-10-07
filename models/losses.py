"""
Composite loss for semantic communication training.

L_total = alpha * L_CE  +  beta * L_feat

* L_CE    — standard cross-entropy on per-token predictions.
* L_feat  — per-token feature-denoising loss: for every valid position,
            1 - cos(channel-decoder output, stop_grad(semantic-encoder output)).
            It trains the channel decoder to recover the clean semantic
            features of each token from the noisy received signal.  The
            target is detached so the encoder cannot trivially satisfy the
            term by collapsing its features.

For *evaluation* we additionally use external metrics (BLEU, BERTScore)
computed in evaluate.py, but those are not used during training to
keep the training loop efficient and fully differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDenoisingLoss(nn.Module):
    """Mean over valid tokens of 1 - cos(ch_dec_out_i, sg(enc_out_i))."""

    def forward(self, enc_out: torch.Tensor, ch_dec_out: torch.Tensor,
                valid_mask: torch.Tensor):
        """
        Args:
            enc_out:    (B, L, d_model) — semantic encoder output
            ch_dec_out: (B, L, d_model) — channel decoder output
            valid_mask: (B, L) — True where NOT padded
        """
        cos = F.cosine_similarity(ch_dec_out.float(), enc_out.detach().float(),
                                  dim=-1)                              # (B, L)
        m = valid_mask.float()
        return ((1.0 - cos) * m).sum() / m.sum().clamp(min=1)


class CompositeLoss(nn.Module):
    """L = alpha * CE  +  beta * FeatureDenoising."""

    def __init__(self, vocab_size: int, pad_idx: int = 0,
                 alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.feat = FeatureDenoisingLoss()

    def forward(self, logits, targets, enc_out=None, ch_dec_out=None,
                pad_mask=None):
        """Returns (total_loss, ce_loss, feat_loss)."""
        ce_loss = self.ce(logits.float().reshape(-1, logits.size(-1)),
                          targets.reshape(-1))

        feat_loss = torch.zeros((), device=logits.device)
        if self.beta > 0 and enc_out is not None and ch_dec_out is not None:
            feat_loss = self.feat(enc_out, ch_dec_out, pad_mask)

        total = self.alpha * ce_loss + self.beta * feat_loss
        return total, ce_loss, feat_loss

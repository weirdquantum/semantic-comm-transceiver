"""
Differentiable channel models.

Both channels take a power-normalised signal x (B, L, C) and a per-sample
SNR in dB (B,), and return the received signal together with the
*effective* SNR seen by the receiver: the nominal SNR for AWGN and
SNR * |h|^2 for Rayleigh fading.
"""

import math

import torch


def _noise_std(snr_db):
    """Noise standard deviation for unit signal power, shaped (B, 1, 1)."""
    return (1.0 / torch.sqrt(10.0 ** (snr_db / 10.0))).view(-1, 1, 1)


def awgn(x: torch.Tensor, snr_db: torch.Tensor):
    """y = x + n,  n ~ N(0, 1/SNR)."""
    sigma = _noise_std(snr_db)
    return x + sigma * torch.randn_like(x), snr_db


def rayleigh(x: torch.Tensor, snr_db: torch.Tensor):
    """Block fading y = h x + n with h = |g|, g ~ CN(0, 1) (so E[|h|^2] = 1),
    one coefficient per sentence, followed by perfect-CSI equalisation y / h."""
    sigma = _noise_std(snr_db)
    h_real = torch.randn(x.size(0), 1, 1, device=x.device) * math.sqrt(0.5)
    h_imag = torch.randn(x.size(0), 1, 1, device=x.device) * math.sqrt(0.5)
    h = torch.sqrt(h_real ** 2 + h_imag ** 2)                    # (B, 1, 1)

    y = h * x + sigma * torch.randn_like(x)
    snr_eff_db = snr_db + 10.0 * torch.log10(h.view(-1) ** 2 + 1e-10)
    return y / (h + 1e-10), snr_eff_db


def apply_channel(x: torch.Tensor, snr_db: torch.Tensor, channel_type: str = "awgn"):
    """Returns (y, snr_eff_db)."""
    if channel_type == "rayleigh":
        return rayleigh(x, snr_db)
    return awgn(x, snr_db)

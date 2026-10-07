"""
Differentiable channel models for end-to-end semantic communication training.

Supported:
  * AWGN  — additive white Gaussian noise
  * Rayleigh  — flat block-fading with perfect CSI equalisation

Both channels are parameterised by SNR (dB) and produce noisy / faded
versions of the transmitted signal while allowing gradient flow.
"""

import math
import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """y = x + n,   n ~ N(0, sigma^2 I),   sigma = 1/sqrt(SNR_linear)."""

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, L, C) power-normalised transmitted signal
            snr_db: (B,) per-sample SNR in dB
        Returns:
            y: same shape as x
        """
        snr_linear = 10.0 ** (snr_db / 10.0)                         # (B,)
        sigma = (1.0 / torch.sqrt(snr_linear)).view(-1, 1, 1)        # (B,1,1)
        noise = sigma * torch.randn_like(x)
        return x + noise


class RayleighChannel(nn.Module):
    """
    Block Rayleigh fading:  y = h * x + n
      h ~ Rayleigh (magnitude of CN(0,1))  — one coefficient per sample
      n ~ N(0, sigma^2)

    With perfect CSI at receiver:  x_hat = y / h
    """

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma = (1.0 / torch.sqrt(snr_linear)).view(-1, 1, 1)

        # Rayleigh fading coefficient (one per sample, constant across time & dim)
        h_real = torch.randn(x.size(0), 1, 1, device=x.device)
        h_imag = torch.randn(x.size(0), 1, 1, device=x.device)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2)                    # (B,1,1)

        noise = sigma * torch.randn_like(x)
        y = h * x + noise

        # Perfect-CSI equalisation
        y_eq = y / (h + 1e-10)
        return y_eq


class ChannelModel(nn.Module):
    """Unified wrapper that selects AWGN or Rayleigh at runtime."""

    def __init__(self):
        super().__init__()
        self.awgn = AWGNChannel()
        self.rayleigh = RayleighChannel()

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor,
                channel_type: str = "awgn") -> torch.Tensor:
        if channel_type == "rayleigh":
            return self.rayleigh(x, snr_db)
        return self.awgn(x, snr_db)

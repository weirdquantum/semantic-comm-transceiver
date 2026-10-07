import math
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snr_db_to_noise_std(snr_db: float) -> float:
    """Convert SNR in dB to noise standard deviation (assuming unit signal power)."""
    snr_linear = 10 ** (snr_db / 10.0)
    return math.sqrt(1.0 / snr_linear)


def ids_to_text(ids, vocab_inv, end_idx=2):
    """Convert a 1-D tensor of token ids back to a string."""
    tokens = []
    for idx in ids:
        idx = idx.item() if hasattr(idx, "item") else int(idx)
        if idx == end_idx:
            break
        word = vocab_inv.get(idx, "")
        if word and word not in ("<PAD>", "<START>"):
            tokens.append(word)
    return " ".join(tokens)


def batch_ids_to_texts(batch_ids, vocab_inv, end_idx=2):
    """Convert a batch of token-id sequences to a list of strings."""
    return [ids_to_text(seq, vocab_inv, end_idx) for seq in batch_ids]


class AverageMeter:
    """Track running averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def amp_dtype(device):
    """Mixed-precision dtype for CUDA: bf16 on Ampere+, fp16 otherwise."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16

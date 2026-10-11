import contextlib
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def preserve_rng(device: torch.device):
    """Run a block (e.g. seeded validation) without disturbing the global RNG
    streams of the CPU and of the given accelerator."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)


def amp_dtype(device: torch.device):
    """Mixed-precision dtype for CUDA: bf16 on Ampere+, fp16 otherwise."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def ids_to_text(ids, vocab_inv, end_idx=2):
    """Convert a 1-D sequence of token ids to a string, stopping at END."""
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == end_idx:
            break
        word = vocab_inv.get(idx, "")
        if word and word not in ("<PAD>", "<START>"):
            tokens.append(word)
    return " ".join(tokens)


def batch_ids_to_texts(batch_ids, vocab_inv, end_idx=2):
    return [ids_to_text(seq, vocab_inv, end_idx) for seq in batch_ids]


class AverageMeter:
    """Running (weighted) average."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

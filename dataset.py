import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


def _load(name, data_dir=Config.data_dir):
    with open(os.path.join(data_dir, name), "rb") as f:
        return pickle.load(f)


def load_vocab(data_dir: str = Config.data_dir):
    v = _load("vocab.pkl", data_dir)
    return v["word2idx"], v["idx2word"]


def load_sentences(data_dir: str = Config.data_dir):
    """{'train': [...], 'test': [...]} raw sentences, aligned with the id files."""
    return _load("sentences.pkl", data_dir)


def train_val_indices(n: int, val_fraction: float = Config.val_fraction, seed: int = 0):
    """Deterministic split of the training file into (train, val) indices."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_fraction)
    return sorted(perm[n_val:]), sorted(perm[:n_val])


class EuroparlDataset(Dataset):
    """Pre-processed Europarl token-id sequences (all padded to max_seq_len)."""

    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def get_dataloader(split: str = "train", batch_size: int = Config.batch_size,
                   data_dir: str = Config.data_dir, shuffle: bool = None,
                   limit: int = None):
    """split: 'train' | 'val' (held out from the training file) | 'test'."""
    if split == "test":
        data = _load("test_data.pkl", data_dir)
    else:
        data = _load("train_data.pkl", data_dir)
        train_idx, val_idx = train_val_indices(len(data))
        data = [data[i] for i in (train_idx if split == "train" else val_idx)]
    if limit is not None:
        data = data[:limit]
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(EuroparlDataset(data), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      drop_last=(split == "train"))

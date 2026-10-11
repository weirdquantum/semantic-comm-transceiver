import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class EuroparlDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed Europarl token-id sequences."""

    def __init__(self, data_path: str, indices=None, limit: int = None):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if indices is not None:
            data = [data[i] for i in indices]
        if limit is not None:
            data = data[:limit]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def train_val_indices(n: int, val_fraction: float = Config.val_fraction,
                      seed: int = 0):
    """Deterministic split of the training file into (train, val) indices."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_fraction)
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def load_vocab(data_dir: str = Config.data_dir):
    path = os.path.join(data_dir, "vocab.pkl")
    with open(path, "rb") as f:
        v = pickle.load(f)
    return v["word2idx"], v["idx2word"]


def load_sentences(data_dir: str = Config.data_dir):
    path = os.path.join(data_dir, "sentences.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_dataloader(split: str = "train", batch_size: int = Config.batch_size,
                   data_dir: str = Config.data_dir, shuffle: bool = None,
                   limit: int = None):
    """split: 'train' | 'val' (held out from the training file) | 'test'."""
    if shuffle is None:
        shuffle = split == "train"
    if split == "test":
        ds = EuroparlDataset(os.path.join(data_dir, "test_data.pkl"), limit=limit)
    else:
        path = os.path.join(data_dir, "train_data.pkl")
        with open(path, "rb") as f:
            n = len(pickle.load(f))
        train_idx, val_idx = train_val_indices(n)
        ds = EuroparlDataset(path, train_idx if split == "train" else val_idx,
                             limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      drop_last=(split == "train"))

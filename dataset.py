import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class EuroparlDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed Europarl token-id sequences."""

    def __init__(self, data_path: str):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


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
                   data_dir: str = Config.data_dir, shuffle: bool = True):
    fname = "train_data.pkl" if split == "train" else "test_data.pkl"
    ds = EuroparlDataset(os.path.join(data_dir, fname))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      drop_last=(split == "train"))

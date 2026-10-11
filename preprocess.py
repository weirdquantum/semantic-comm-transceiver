"""
Preprocess the Europarl corpus for the DeepSC semantic communication system.

Usage:
    1. Download Europarl v7:
       wget http://www.statmt.org/europarl/v7/europarl.tgz
       tar xzf europarl.tgz
    2. Run this script:
       python preprocess.py --raw_dir europarl/txt/en --out_dir data
"""

import argparse
import os
import pickle
import re
from collections import Counter

from config import Config


def clean_sentence(sent: str) -> str:
    sent = sent.lower().strip()
    sent = re.sub(r"[^a-z0-9\s']", " ", sent)
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


def build_vocab(sentences, max_vocab):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    special = [Config.PAD_TOKEN, Config.START_TOKEN, Config.END_TOKEN, Config.UNK_TOKEN]
    most_common = [w for w, _ in counter.most_common(max_vocab - len(special))]
    word2idx = {tok: i for i, tok in enumerate(special + most_common)}
    idx2word = {i: tok for tok, i in word2idx.items()}
    return word2idx, idx2word


def sentence_to_ids(sent, word2idx, max_len):
    tokens = sent.split()[:max_len - 2]
    ids = [Config.START_IDX]
    for t in tokens:
        ids.append(word2idx.get(t, Config.UNK_IDX))
    ids.append(Config.END_IDX)
    while len(ids) < max_len:
        ids.append(Config.PAD_IDX)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="europarl/txt/en",
                        help="Path to raw Europarl English text files")
    parser.add_argument("--out_dir", type=str, default=Config.data_dir)
    parser.add_argument("--max_vocab", type=int, default=Config.vocab_size)
    parser.add_argument("--max_len", type=int, default=Config.max_seq_len)
    parser.add_argument("--min_len", type=int, default=Config.min_seq_len)
    parser.add_argument("--max_sentences", type=int, default=500_000,
                        help="Cap on total sentences to keep")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- collect raw sentences ---
    sentences = []
    if os.path.isdir(args.raw_dir):
        files = sorted(f for f in os.listdir(args.raw_dir) if f.endswith(".txt"))
        for fname in files:
            with open(os.path.join(args.raw_dir, fname), "r", encoding="utf-8",
                      errors="ignore") as f:
                for line in f:
                    s = clean_sentence(line)
                    n_words = len(s.split())
                    if args.min_len <= n_words <= args.max_len - 2:
                        sentences.append(s)
            if len(sentences) >= args.max_sentences:
                break
    elif os.path.isfile(args.raw_dir):
        with open(args.raw_dir, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = clean_sentence(line)
                n_words = len(s.split())
                if args.min_len <= n_words <= args.max_len - 2:
                    sentences.append(s)
                if len(sentences) >= args.max_sentences:
                    break

    sentences = sentences[:args.max_sentences]
    print(f"Collected {len(sentences)} sentences")

    # --- build vocabulary ---
    word2idx, idx2word = build_vocab(sentences, args.max_vocab)
    print(f"Vocabulary size: {len(word2idx)}")

    # --- convert to id sequences ---
    all_ids = [sentence_to_ids(s, word2idx, args.max_len) for s in sentences]

    # --- train / test split (90 / 10) ---
    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    test_ids = all_ids[split:]
    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

    # --- save ---
    with open(os.path.join(args.out_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_ids, f)
    with open(os.path.join(args.out_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(test_ids, f)
    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump({"word2idx": word2idx, "idx2word": idx2word}, f)
    with open(os.path.join(args.out_dir, "sentences.pkl"), "wb") as f:
        pickle.dump({"train": sentences[:split], "test": sentences[split:]}, f)
    print("Preprocessing done.")


if __name__ == "__main__":
    main()

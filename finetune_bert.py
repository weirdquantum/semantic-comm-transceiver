"""
Fine-tune bert-base-uncased on Europarl data for Masked Language Modeling.

This produces a domain-adapted BERT MLM that the LLMRefiner can use
to correct low-confidence decoder outputs more accurately.

Usage:
    python finetune_bert.py                        # defaults
    python finetune_bert.py --epochs 5 --lr 2e-5   # custom
"""

import argparse
import os
import pickle
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
)

from config import Config


class EuroparlTextDataset(Dataset):
    """Simple dataset returning raw sentences for BERT tokenizer."""

    def __init__(self, sentences, tokenizer, max_length=128):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--max_sentences", type=int, default=100_000,
                        help="Max training sentences (for speed)")
    parser.add_argument("--save_dir", default=os.path.join(Config.checkpoint_dir, "bert_europarl"))
    args = parser.parse_args()

    device = Config.device
    os.makedirs(args.save_dir, exist_ok=True)

    sent_path = os.path.join(Config.data_dir, "sentences.pkl")
    with open(sent_path, "rb") as f:
        all_sents = pickle.load(f)
    train_sents = all_sents["train"][:args.max_sentences]
    print(f"Fine-tuning on {len(train_sents)} Europarl sentences")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    dataset = EuroparlTextDataset(train_sents, tokenizer, max_length=64)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"Device: {device} | Epochs: {args.epochs} | Steps/epoch: {len(loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 200 == 0:
                avg = total_loss / step
                ppl = math.exp(min(avg, 10))
                print(f"  Epoch {epoch} step {step}/{len(loader)} | "
                      f"loss={avg:.4f} ppl={ppl:.2f}")

        avg_loss = total_loss / len(loader)
        ppl = math.exp(min(avg_loss, 10))
        print(f"Epoch {epoch}/{args.epochs} done | loss={avg_loss:.4f} ppl={ppl:.2f}")

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"\nFine-tuned model saved to {args.save_dir}")


if __name__ == "__main__":
    main()

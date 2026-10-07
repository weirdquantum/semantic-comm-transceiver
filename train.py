"""
Training script for Baseline DeepSC and Improved DeepSC.

Usage examples:

  # Train baseline DeepSC on AWGN channel
  python train.py --model baseline --channel awgn --epochs 80

  # Train improved DeepSC on AWGN channel
  python train.py --model improved --channel awgn --epochs 80

  # Ablation: improved without semantic loss (CE only)
  python train.py --model improved --channel awgn --beta 0.0 --tag no_sem

  # Train on Rayleigh channel
  python train.py --model improved --channel rayleigh --epochs 80 --tag rayleigh
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from dataset import get_dataloader, load_vocab
from models.transceiver import BaselineDeepSC, ImprovedDeepSC
from models.losses import CompositeLoss
from utils import set_seed, AverageMeter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "improved"], default="improved")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--alpha", type=float, default=Config.alpha_ce)
    p.add_argument("--beta", type=float, default=Config.beta_semantic)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="",
                   help="Extra tag appended to checkpoint dir name")
    p.add_argument("--resume", type=str, default="",
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def build_model(model_type, vocab_size, device):
    kwargs = dict(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        n_heads=Config.n_heads,
        d_ff=Config.d_ff,
        n_enc_layers=Config.n_encoder_layers,
        n_dec_layers=Config.n_decoder_layers,
        channel_dim=Config.channel_dim,
        max_len=Config.max_seq_len,
        dropout=Config.dropout,
        pad_idx=Config.PAD_IDX,
    )
    if model_type == "baseline":
        return BaselineDeepSC(**kwargs).to(device)
    return ImprovedDeepSC(**kwargs).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    channel_type, epoch, total_epochs):
    model.train()
    loss_m = AverageMeter()
    ce_m = AverageMeter()
    sem_m = AverageMeter()
    acc_m = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        src = batch
        trg_in = batch[:, :-1]
        trg_out = batch[:, 1:]

        # Random SNR per sample
        snr_db = (torch.rand(batch.size(0), device=device)
                  * (Config.snr_train_high - Config.snr_train_low)
                  + Config.snr_train_low)

        logits, enc_out, ch_dec_out = model(src, trg_in, snr_db, channel_type)

        pad_mask = (src != Config.PAD_IDX)
        total, ce, sem = criterion(
            logits, trg_out,
            enc_out=enc_out, ch_dec_out=ch_dec_out,
            src_ids=src,
            embedding_weight=model.embedding.weight,
            pad_mask=pad_mask,
            pad_idx=Config.PAD_IDX,
            snr_db=snr_db,
        )

        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        optimizer.step()

        # word-level accuracy (ignoring padding)
        preds = logits.argmax(dim=-1)
        mask = trg_out != Config.PAD_IDX
        correct = ((preds == trg_out) & mask).sum().item()
        n_tokens = mask.sum().item()
        acc = correct / max(n_tokens, 1)

        loss_m.update(total.item(), batch.size(0))
        ce_m.update(ce.item(), batch.size(0))
        sem_m.update(sem.item(), batch.size(0))
        acc_m.update(acc, batch.size(0))

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}",
                         ce=f"{ce_m.avg:.4f}",
                         sem=f"{sem_m.avg:.4f}",
                         acc=f"{acc_m.avg:.4f}")

    return loss_m.avg, ce_m.avg, sem_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, criterion, device, channel_type, snr_db_val=10.0):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for batch in loader:
        batch = batch.to(device)
        src = batch
        trg_in = batch[:, :-1]
        trg_out = batch[:, 1:]
        snr_db = torch.full((batch.size(0),), snr_db_val, device=device)

        logits, enc_out, ch_dec_out = model(src, trg_in, snr_db, channel_type)
        pad_mask = (src != Config.PAD_IDX)
        total, ce, sem = criterion(
            logits, trg_out,
            enc_out=enc_out, ch_dec_out=ch_dec_out,
            pad_mask=pad_mask, pad_idx=Config.PAD_IDX,
            snr_db=snr_db,
        )
        preds = logits.argmax(dim=-1)
        mask = trg_out != Config.PAD_IDX
        acc = ((preds == trg_out) & mask).sum().item() / max(mask.sum().item(), 1)
        loss_m.update(total.item(), batch.size(0))
        acc_m.update(acc, batch.size(0))

    return loss_m.avg, acc_m.avg


def main():
    args = parse_args()
    set_seed(args.seed)
    Config.ensure_dirs()
    device = Config.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # ---- data ----
    word2idx, idx2word = load_vocab()
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    train_loader = get_dataloader("train", args.batch_size)
    val_loader = get_dataloader("test", args.batch_size, shuffle=False)

    # ---- model ----
    model = build_model(args.model, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}")

    # ---- loss & optimiser ----
    criterion = CompositeLoss(vocab_size, Config.PAD_IDX,
                              alpha=args.alpha, beta=args.beta)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=Config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=Config.lr_min)

    # ---- checkpoint dir ----
    tag = f"{args.model}_{args.channel}"
    if args.tag:
        tag += f"_{args.tag}"
    ckpt_dir = os.path.join(Config.checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0
    history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, ce_loss, sem_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            args.channel, epoch, args.epochs)
        val_loss, val_acc = validate(model, val_loader, criterion,
                                     device, args.channel)
        scheduler.step()
        elapsed = time.time() - t0

        record = dict(epoch=epoch, train_loss=train_loss, ce_loss=ce_loss,
                      sem_loss=sem_loss, train_acc=train_acc,
                      val_loss=val_loss, val_acc=val_acc, time=elapsed)
        history.append(record)

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  ce={ce_loss:.4f}  sem={sem_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        state = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            best_val_acc=best_val_acc,
            args=vars(args),
        )
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))

    # ---- save history ----
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete.  Best val acc = {best_val_acc:.4f}")
    print(f"Checkpoints saved in {ckpt_dir}")


if __name__ == "__main__":
    main()

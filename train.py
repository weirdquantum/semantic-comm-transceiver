"""
Training script for Baseline DeepSC and Improved DeepSC.

Usage examples:

  # Train baseline DeepSC on AWGN channel
  python train.py --model baseline --channel awgn --epochs 80

  # Train improved DeepSC on AWGN channel
  python train.py --model improved --channel awgn --epochs 80

  # Ablations of the improved model
  python train.py --model improved --channel awgn --no_film --tag no_film
  python train.py --model improved --channel awgn --no_masked_power --tag no_maskpow
  python train.py --model improved --channel awgn --beta 0.0 --tag no_feat
  python train.py --model improved --channel rayleigh --no_csi_snr --tag no_csi

  # Resume automatically from <ckpt_dir>/last.pt if it exists
  python train.py --model improved --channel awgn --resume auto
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import Config
from dataset import get_dataloader, load_vocab
from models.transceiver import BaselineDeepSC, ImprovedDeepSC
from models.losses import CompositeLoss
from utils import AverageMeter, amp_dtype, preserve_rng, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "improved"], default="improved")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--alpha", type=float, default=Config.alpha_ce)
    p.add_argument("--beta", type=float, default=None,
                   help="Feature-denoising loss weight "
                        "(default: 0 for baseline, Config.beta_feat for improved)")
    p.add_argument("--no_film", action="store_true",
                   help="Improved: disable SNR-conditioned channel coder")
    p.add_argument("--no_masked_power", action="store_true",
                   help="Improved: spend power on padding like the baseline")
    p.add_argument("--no_csi_snr", action="store_true",
                   help="Improved: receiver uses average instead of effective SNR")
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision on CUDA (bf16 if supported, else fp16)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="",
                   help="Extra tag appended to checkpoint dir name")
    p.add_argument("--resume", type=str, default="",
                   help="Checkpoint path to resume from, or 'auto' for <ckpt_dir>/last.pt")
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="Smoke testing: stop each epoch after N batches")
    p.add_argument("--limit_val_samples", type=int, default=None,
                   help="Smoke testing: use only N validation sentences")
    args = p.parse_args()
    if args.beta is None:
        args.beta = 0.0 if args.model == "baseline" else Config.beta_feat
    return args


def build_model(model_type, vocab_size, device, use_film=True,
                masked_power=True, csi_snr=True):
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
    return ImprovedDeepSC(use_film=use_film, masked_power=masked_power,
                          csi_snr=csi_snr, **kwargs).to(device)


def build_model_from_ckpt(ckpt, vocab_size, device):
    """Rebuild the exact architecture a checkpoint was trained with."""
    a = ckpt.get("args", {})
    model = build_model(a.get("model", "improved"), vocab_size, device,
                        use_film=not a.get("no_film", False),
                        masked_power=not a.get("no_masked_power", False),
                        csi_snr=not a.get("no_csi_snr", False))
    model.load_state_dict(ckpt["model"])
    return model


def make_scheduler(optimizer, total_steps, warmup_steps, lr, lr_min):
    """Linear warm-up followed by cosine decay to lr_min (stepped per batch)."""
    floor = lr_min / lr

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, channel_type, epoch, total_epochs, dtype,
                    limit_batches=0):
    model.train()
    loss_m = AverageMeter()
    ce_m = AverageMeter()
    feat_m = AverageMeter()
    acc_m = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for i, batch in enumerate(pbar):
        if limit_batches and i >= limit_batches:
            break
        batch = batch.to(device)
        src = batch
        trg_in = batch[:, :-1]
        trg_out = batch[:, 1:]

        # Random SNR per sample
        snr_db = (torch.rand(batch.size(0), device=device)
                  * (Config.snr_train_high - Config.snr_train_low)
                  + Config.snr_train_low)

        with torch.autocast(device_type=device.type, dtype=dtype,
                            enabled=dtype is not None):
            logits, enc_out, ch_dec_out = model(src, trg_in, snr_db, channel_type)
            total, ce, feat = criterion(
                logits, trg_out,
                enc_out=enc_out, ch_dec_out=ch_dec_out,
                pad_mask=(src != Config.PAD_IDX),
            )

        optimizer.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # word-level accuracy (ignoring padding)
        preds = logits.argmax(dim=-1)
        mask = trg_out != Config.PAD_IDX
        correct = ((preds == trg_out) & mask).sum().item()
        n_tokens = mask.sum().item()
        acc = correct / max(n_tokens, 1)

        loss_m.update(total.item(), batch.size(0))
        ce_m.update(ce.item(), batch.size(0))
        feat_m.update(feat.item(), batch.size(0))
        acc_m.update(acc, batch.size(0))

        pbar.set_postfix(loss=f"{loss_m.avg:.4f}",
                         ce=f"{ce_m.avg:.4f}",
                         feat=f"{feat_m.avg:.4f}",
                         acc=f"{acc_m.avg:.4f}")

    return loss_m.avg, ce_m.avg, feat_m.avg, acc_m.avg


@torch.no_grad()
def validate(model, loader, device, channel_type, dtype,
             snr_list=Config.val_snr_list):
    """Teacher-forced token accuracy on the validation split at each SNR.

    Uses a fixed noise seed (without disturbing the training RNG) so that
    epochs are compared on identical channel realisations.
    """
    model.eval()
    accs = {}
    with preserve_rng(device):
        torch.manual_seed(1234)
        for snr_val in snr_list:
            correct = total = 0
            for batch in loader:
                batch = batch.to(device)
                trg_in = batch[:, :-1]
                trg_out = batch[:, 1:]
                snr_db = torch.full((batch.size(0),), snr_val, device=device)
                with torch.autocast(device_type=device.type, dtype=dtype,
                                    enabled=dtype is not None):
                    logits, _, _ = model(batch, trg_in, snr_db, channel_type)
                mask = trg_out != Config.PAD_IDX
                correct += ((logits.argmax(-1) == trg_out) & mask).sum().item()
                total += mask.sum().item()
            accs[float(snr_val)] = correct / max(total, 1)
    return sum(accs.values()) / len(accs), accs


def main():
    args = parse_args()
    set_seed(args.seed)
    device = Config.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    dtype = amp_dtype(device) if args.amp else None
    print(f"Device: {device}  |  AMP: {dtype}")

    # ---- data ----
    word2idx, idx2word = load_vocab()
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    train_loader = get_dataloader("train", args.batch_size)
    val_loader = get_dataloader("val", args.batch_size,
                                limit=args.limit_val_samples)
    print(f"Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    # ---- model ----
    model = build_model(args.model, vocab_size, device,
                        use_film=not args.no_film,
                        masked_power=not args.no_masked_power,
                        csi_snr=not args.no_csi_snr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}  |  beta={args.beta}")

    # ---- loss & optimiser ----
    criterion = CompositeLoss(vocab_size, Config.PAD_IDX,
                              alpha=args.alpha, beta=args.beta)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=Config.weight_decay)
    steps_per_epoch = len(train_loader)
    if args.limit_train_batches:
        steps_per_epoch = min(steps_per_epoch, args.limit_train_batches)
    scheduler = make_scheduler(optimizer, args.epochs * steps_per_epoch,
                               Config.warmup_epochs * steps_per_epoch,
                               args.lr, Config.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    # ---- checkpoint dir ----
    tag = f"{args.model}_{args.channel}"
    if args.tag:
        tag += f"_{args.tag}"
    ckpt_dir = os.path.join(Config.checkpoint_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0
    history = []

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = os.path.join(ckpt_dir, "last.pt")
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history = ckpt.get("history", [])
        print(f"Resumed from {resume_path}, epoch {start_epoch}")

    # ---- training loop ----
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, ce_loss, feat_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, args.channel, epoch, args.epochs, dtype,
            args.limit_train_batches)
        val_acc, val_accs = validate(model, val_loader, device, args.channel, dtype)
        elapsed = time.time() - t0

        record = dict(epoch=epoch, train_loss=train_loss, ce_loss=ce_loss,
                      feat_loss=feat_loss, train_acc=train_acc,
                      val_acc=val_acc, val_acc_per_snr=val_accs,
                      lr=scheduler.get_last_lr()[0], time=elapsed)
        history.append(record)

        per_snr = "  ".join(f"{s:+.0f}dB={a:.4f}" for s, a in val_accs.items())
        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  ce={ce_loss:.4f}  feat={feat_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f} ({per_snr})  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  time={elapsed:.1f}s")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        state = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_val_acc=best_val_acc,
            history=history,
            args=vars(args),
        )
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
        with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ---- mark finished (run_all.py skips finished variants) ----
    with open(os.path.join(ckpt_dir, "DONE"), "w") as f:
        f.write(f"{args.epochs}\n")
    print(f"Training complete.  Best val acc = {best_val_acc:.4f}")
    print(f"Checkpoints saved in {ckpt_dir}")


if __name__ == "__main__":
    main()

"""
Lightweight model optimisation via knowledge distillation and quantisation.

This script:
  1. Loads a trained *teacher* model (full Improved DeepSC).
  2. Builds a smaller *student* model (fewer layers & smaller d_model).
  3. Trains the student with a KD loss:
       L = (1-alpha)*CE_hard + alpha*KL(teacher_soft, student_soft)
  4. Optionally applies dynamic quantisation to the student.
  5. Reports parameter count, inference speed, and accuracy comparison.

Usage:
  python lightweight.py --teacher checkpoints/improved_awgn/best.pt \\
                        --epochs 40 --channel awgn
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from dataset import get_dataloader, load_vocab
from train import build_model_from_ckpt
from utils import set_seed, AverageMeter


class StudentDeepSC(nn.Module):
    """
    Compact variant: half the layers, half the d_model.
    Still uses SNR-Adaptive Attention for a fair comparison.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256,
                 n_enc_layers=2, n_dec_layers=2, channel_dim=16,
                 max_len=32, dropout=0.1, pad_idx=0, **flags):
        super().__init__()
        from models.transceiver import ImprovedDeepSC
        self.inner = ImprovedDeepSC(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
            channel_dim=channel_dim, max_len=max_len, dropout=dropout,
            pad_idx=pad_idx, **flags)

    @property
    def embedding(self):
        return self.inner.embedding

    def forward(self, src, trg, snr_db, channel_type="awgn"):
        return self.inner(src, trg, snr_db, channel_type)

    def greedy_decode(self, *a, **kw):
        return self.inner.greedy_decode(*a, **kw)


def distillation_loss(student_logits, teacher_logits, targets,
                      temperature=4.0, alpha=0.7, pad_idx=0):
    """
    KD loss = alpha * KL(soft_teacher, soft_student) + (1-alpha) * CE_hard.
    """
    ce_hard = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        targets.reshape(-1), ignore_index=pad_idx)

    soft_s = F.log_softmax(student_logits / temperature, dim=-1)
    soft_t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(soft_s.reshape(-1, soft_s.size(-1)),
                  soft_t.reshape(-1, soft_t.size(-1)),
                  reduction="batchmean") * (temperature ** 2)

    return alpha * kl + (1 - alpha) * ce_hard


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_speed(model, sample_input, snr, device, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model.greedy_decode(sample_input, snr, "awgn",
                                max_len=Config.max_seq_len)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        for _ in range(n_runs):
            model.greedy_decode(sample_input, snr, "awgn",
                                max_len=Config.max_seq_len)
        torch.cuda.synchronize() if device.type == "cuda" else None
        return (time.time() - t0) / n_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", type=str, required=True,
                   help="Path to trained teacher checkpoint")
    p.add_argument("--channel", type=str, default="awgn")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha_kd", type=float, default=0.7)
    p.add_argument("--quantize", action="store_true",
                   help="Apply dynamic quantisation to the student")
    args = p.parse_args()

    set_seed(42)
    Config.ensure_dirs()
    device = Config.device
    word2idx, _ = load_vocab()
    vocab_size = len(word2idx)

    # ---- Teacher ----
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher = build_model_from_ckpt(ckpt, vocab_size, device)
    teacher.eval()
    for p_t in teacher.parameters():
        p_t.requires_grad = False
    print(f"Teacher params: {count_params(teacher):,}")

    # ---- Student (half-size) ----
    t_args = ckpt.get("args", {})
    student = StudentDeepSC(
        vocab_size,
        use_film=not t_args.get("no_film", False),
        masked_power=not t_args.get("no_masked_power", False),
        csi_snr=not t_args.get("no_csi_snr", False)).inner.to(device)
    print(f"Student params: {count_params(student):,}")
    print(f"Compression:    {count_params(teacher)/count_params(student):.1f}x")

    # ---- Data ----
    train_loader = get_dataloader("train", Config.batch_size)
    val_loader = get_dataloader("val", Config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)

    # ---- KD Training ----
    for epoch in range(1, args.epochs + 1):
        student.train()
        loss_m = AverageMeter()
        acc_m = AverageMeter()
        for batch in tqdm(train_loader, desc=f"KD Epoch {epoch}", leave=False):
            batch = batch.to(device)
            src = batch
            trg_in = batch[:, :-1]
            trg_out = batch[:, 1:]
            snr_db = (torch.rand(batch.size(0), device=device)
                      * (Config.snr_train_high - Config.snr_train_low)
                      + Config.snr_train_low)

            with torch.no_grad():
                t_logits, _, _ = teacher(src, trg_in, snr_db, args.channel)
            s_logits, _, _ = student(src, trg_in, snr_db, args.channel)

            loss = distillation_loss(s_logits, t_logits, trg_out,
                                     args.temperature, args.alpha_kd,
                                     Config.PAD_IDX)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            preds = s_logits.argmax(-1)
            mask = trg_out != Config.PAD_IDX
            acc = ((preds == trg_out) & mask).sum().item() / mask.sum().item()
            loss_m.update(loss.item(), batch.size(0))
            acc_m.update(acc, batch.size(0))

        print(f"  Epoch {epoch}: loss={loss_m.avg:.4f}  acc={acc_m.avg:.4f}")

    # ---- Save student ----
    out_dir = os.path.join(Config.checkpoint_dir, "student_awgn")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": student.state_dict()},
               os.path.join(out_dir, "student.pt"))

    # ---- Optional quantisation ----
    if args.quantize:
        student_q = torch.quantization.quantize_dynamic(
            student.cpu(), {nn.Linear}, dtype=torch.qint8)
        torch.save(student_q.state_dict(),
                   os.path.join(out_dir, "student_quant.pt"))
        print(f"Quantised student saved. Size reduction: "
              f"~{count_params(student) * 4 / 1024 / 1024:.1f} MB → "
              f"~{count_params(student) * 1 / 1024 / 1024:.1f} MB (int8)")

    # ---- Inference speed comparison ----
    sample = next(iter(val_loader))[:4].to(device)
    snr_s = torch.tensor([10.0] * 4, device=device)
    t_speed = measure_inference_speed(teacher, sample, snr_s, device)
    s_speed = measure_inference_speed(student.to(device), sample, snr_s, device)
    print(f"\nInference latency (batch=4):")
    print(f"  Teacher:  {t_speed*1000:.1f} ms")
    print(f"  Student:  {s_speed*1000:.1f} ms")
    print(f"  Speedup:  {t_speed/s_speed:.2f}x")


if __name__ == "__main__":
    main()

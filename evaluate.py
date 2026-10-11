"""
Comprehensive evaluation of semantic communication models.

Metrics computed per SNR point:
  - Word Accuracy   (token-level match)
  - BLEU-1 … BLEU-4 (n-gram precision; BLEU-4 with a bootstrap 95% CI)
  - BERTScore       (contextual embedding similarity)
  - Sentence Similarity (cosine sim of DistilBERT [CLS] embeddings)

The architecture is rebuilt from the arguments stored in the checkpoint.
The channel noise is re-seeded at every SNR point, so different models
(and greedy vs. beam decoding) see the same noise realisations.

Usage:
  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt

  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt \\
                     --beam_width 5 --eval_tag beam5

  # tune the LLM fusion weights on the validation split, then apply them
  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt \\
                     --beam_width 5 --tune_llm
  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt \\
                     --beam_width 5 --use_llm --eval_tag beam5_llm
"""

import argparse
import json
import math
import os

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import Config
from dataset import get_dataloader, load_vocab
from train import build_model_from_ckpt
from utils import set_seed, batch_ids_to_texts


# ====================================================================
# Metric helpers
# ====================================================================
def compute_word_accuracy(pred_ids, ref_ids, pad_idx=0, end_idx=2):
    """Token-level accuracy ignoring PAD.  Returns (correct, total)."""
    correct = 0
    total = 0
    for p, r in zip(pred_ids.tolist(), ref_ids.tolist()):
        for pi, ri in zip(p, r):
            if ri == pad_idx:
                break
            total += 1
            if pi == ri:
                correct += 1
    return correct, total


def compute_bleu(pred_texts, ref_texts, orders=(1, 2, 3, 4)):
    """Average sentence-level BLEU-n, plus the per-sentence BLEU-4 scores."""
    sf = SmoothingFunction().method1
    bleu_scores = {f"BLEU-{n}": [] for n in orders}
    for pred, ref in zip(pred_texts, ref_texts):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if len(pred_tokens) == 0:
            pred_tokens = [""]
        for n in orders:
            w = [1.0 / n] * n + [0.0] * (4 - n)
            score = sentence_bleu([ref_tokens], pred_tokens,
                                  weights=w, smoothing_function=sf)
            bleu_scores[f"BLEU-{n}"].append(score)
    means = {k: float(np.mean(v)) for k, v in bleu_scores.items()}
    return means, bleu_scores.get("BLEU-4")


def bootstrap_ci(values, n_boot=1000, seed=0):
    """95% percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bleu_with_ci(pred_texts, ref_texts):
    means, b4 = compute_bleu(pred_texts, ref_texts)
    lo, hi = bootstrap_ci(b4)
    return {**means, "BLEU-4_CI_low": lo, "BLEU-4_CI_high": hi}


_SCORERS = {}


def compute_bert_score(pred_texts, ref_texts, device="cpu"):
    """BERTScore (Precision, Recall, F1) using the bert-score library.

    Empty predictions score 0, as bert-score intends; they are scored here
    because bert-score's own empty-sentence path fails with transformers 5.
    """
    try:
        if "bertscore" not in _SCORERS:
            from bert_score import BERTScorer
            _SCORERS["bertscore"] = BERTScorer(lang="en", device=str(device), batch_size=128)
        keep = [i for i, p in enumerate(pred_texts) if p.strip()]
        scores = np.zeros((3, len(pred_texts)))
        if keep:
            P, R, F1 = _SCORERS["bertscore"].score([pred_texts[i] for i in keep],
                                                   [ref_texts[i] for i in keep],
                                                   verbose=False)
            scores[:, keep] = torch.stack([P, R, F1]).cpu().numpy()
        P, R, F1 = scores.mean(axis=1)
        return {"BERTScore_P": float(P), "BERTScore_R": float(R), "BERTScore_F1": float(F1)}
    except Exception as e:
        print(f"  [warn] BERTScore failed: {e}")
        return {}


def compute_sentence_similarity(pred_texts, ref_texts, device="cpu"):
    """Mean cosine similarity of sentence-level DistilBERT [CLS] embeddings."""
    try:
        if "distilbert" not in _SCORERS:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            mdl = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
            mdl.eval()
            _SCORERS["distilbert"] = (tok, mdl)
        tok, mdl = _SCORERS["distilbert"]
        sims = []
        bs = 128
        for i in range(0, len(pred_texts), bs):
            p_batch = pred_texts[i:i + bs]
            r_batch = ref_texts[i:i + bs]
            with torch.no_grad():
                pe = tok(p_batch, padding=True, truncation=True,
                         max_length=64, return_tensors="pt").to(device)
                re_ = tok(r_batch, padding=True, truncation=True,
                          max_length=64, return_tensors="pt").to(device)
                p_emb = mdl(**pe).last_hidden_state[:, 0]
                r_emb = mdl(**re_).last_hidden_state[:, 0]
            cos = torch.nn.functional.cosine_similarity(p_emb, r_emb)
            sims.extend(cos.cpu().tolist())
        return {"SentSim": float(np.mean(sims))}
    except Exception as e:
        print(f"  [warn] SentSim failed: {e}")
        return {}


def text_metrics(pred_texts, ref_texts, device, with_embeddings=True):
    m = bleu_with_ci(pred_texts, ref_texts)
    if with_embeddings:
        m.update(compute_bert_score(pred_texts, ref_texts, device))
        m.update(compute_sentence_similarity(pred_texts, ref_texts, device))
    return m


# ====================================================================
# Decoding
# ====================================================================
def snr_seed(snr_val):
    """Same seed for the same SNR, whatever the model or decoding strategy."""
    return 42 + int(round((snr_val + 100) * 10))


def build_refine_items(ys, info, idx2word, end_idx=Config.END_IDX):
    """Per-sentence (words, confidences, candidates) for the LLM refiner.

    Words follow exactly the same rules as ``utils.ids_to_text`` so that
    " ".join(words) equals the unrefined decoded text.
    """
    ys = ys.tolist()
    conf = info["conf"].tolist()
    cand_ids = info["cand_ids"].tolist()
    cand_logp = info["cand_logp"].tolist()
    items = []
    for i, seq in enumerate(ys):
        words, confs, cands = [], [], []
        for j in range(1, len(seq)):
            if seq[j] == end_idx:
                break
            w = idx2word.get(seq[j], "")
            if not w or w in ("<PAD>", "<START>"):
                continue
            c = conf[i][j - 1]
            cs = [(idx2word.get(t, ""), lp)
                  for t, lp in zip(cand_ids[i][j - 1], cand_logp[i][j - 1])]
            if w not in {cw for cw, _ in cs}:
                cs.append((w, math.log(max(c, 1e-12))))
            words.append(w)
            confs.append(c)
            cands.append(cs)
        items.append((words, confs, cands))
    return items


@torch.no_grad()
def decode_at_snr(model, loader, idx2word, snr_val, channel_type, device,
                  max_samples, beam_width=1, topk=0):
    """Decode the split at a fixed SNR.

    Returns pred_texts, ref_texts, (correct, total) word counts and,
    if topk > 0, the refiner items.
    """
    set_seed(snr_seed(snr_val))
    model.eval()
    pred_texts, ref_texts, items = [], [], []
    correct = total = n = 0

    for batch in loader:
        batch = batch[: max_samples - n].to(device)
        B = batch.size(0)
        snr_db = torch.full((B,), snr_val, device=device, dtype=torch.float)
        ys, info = model.decode(batch, snr_db, channel_type,
                                max_len=Config.max_seq_len,
                                start_idx=Config.START_IDX,
                                end_idx=Config.END_IDX,
                                beam_width=beam_width, topk=topk)

        pred = torch.full((B, Config.max_seq_len), Config.PAD_IDX,
                          dtype=torch.long, device=ys.device)
        pred[:, :ys.size(1)] = ys[:, :Config.max_seq_len]
        c, t = compute_word_accuracy(pred.cpu(), batch.cpu(),
                                     Config.PAD_IDX, Config.END_IDX)
        correct += c
        total += t

        pred_texts += batch_ids_to_texts(pred.cpu(), idx2word, Config.END_IDX)
        ref_texts += batch_ids_to_texts(batch.cpu(), idx2word, Config.END_IDX)
        if topk > 0:
            items += build_refine_items(ys.cpu(), {k: v.cpu() for k, v in info.items()},
                                        idx2word)
        n += B
        if n >= max_samples:
            break
    return pred_texts, ref_texts, (correct, total), items


# ====================================================================
# Entry points
# ====================================================================
def load_model(args, device):
    word2idx, idx2word = load_vocab()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, len(word2idx), device)
    model.eval()
    ckpt_args = ckpt.get("args", {})
    channel = args.channel or ckpt_args.get("channel", "awgn")
    print(f"Loaded {ckpt_args.get('model', '?')} from {args.checkpoint} "
          f"(epoch {ckpt.get('epoch')}, channel={channel})")
    return model, idx2word, channel


def default_llm_model():
    has_weights = any(os.path.isfile(os.path.join(Config.bert_dir, f))
                      for f in ("model.safetensors", "pytorch_model.bin"))
    return Config.bert_dir if has_weights else "bert-base-uncased"


def fusion_path(args):
    return os.path.join(os.path.dirname(args.checkpoint),
                        f"llm_fusion_beam{args.beam_width}.json")


def make_refiner(args, device):
    from models.llm_refiner import LLMRefiner
    llm_model = args.llm_model or default_llm_model()
    refiner = LLMRefiner(model_name=llm_model, device=device)
    print(f"LLM refiner loaded ({llm_model})")
    return refiner


def run_tuning(args):
    """Grid-search (lambda, tau) on the validation split; save the best."""
    device = Config.device
    model, idx2word, channel = load_model(args, device)
    refiner = make_refiner(args, device)
    loader = get_dataloader("val", batch_size=args.batch_size, shuffle=False)
    max_samples = args.max_samples or Config.llm_tune_samples
    snrs = args.snr_list or Config.llm_tune_snrs

    grid = {}   # "lam,tau" -> list of BLEU-4 per SNR
    for snr_val in snrs:
        _, refs, _, items = decode_at_snr(model, loader, idx2word, snr_val,
                                          channel, device, max_samples,
                                          args.beam_width, Config.llm_topk)
        for tau in Config.llm_tau_grid:
            outs = refiner.refine_grid(items, Config.llm_lambda_grid, tau)
            for lam, texts in outs.items():
                b4 = compute_bleu(texts, refs, orders=(4,))[0]["BLEU-4"]
                grid.setdefault(f"{lam},{tau}", []).append(b4)
        print(f"  tuned at {snr_val} dB")

    mean = {k: float(np.mean(v)) for k, v in grid.items()}
    # best mean BLEU-4; ties go to the smaller lambda (lambda=0 is no-op)
    best = max(mean, key=lambda k: (round(mean[k], 6), -float(k.split(",")[0])))
    lam, tau = (float(x) for x in best.split(","))
    out = {"lam": lam, "tau": tau, "val_bleu4": mean[best],
           "val_bleu4_no_refine": mean[f"0.0,{Config.llm_tau_grid[0]}"],
           "grid": mean, "snrs": snrs, "beam_width": args.beam_width}
    with open(fusion_path(args), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Best lambda={lam} tau={tau}  val BLEU-4 {mean[best]:.4f} "
          f"(no refinement {out['val_bleu4_no_refine']:.4f})")
    print(f"Saved to {fusion_path(args)}")


def run_evaluation(args):
    device = Config.device
    model, idx2word, channel = load_model(args, device)
    loader = get_dataloader("test", batch_size=args.batch_size, shuffle=False)

    if args.beam_width > 1:
        print(f"Beam search enabled (width={args.beam_width})")

    # optional LLM refiner
    refiner = None
    if args.use_llm:
        refiner = make_refiner(args, device)
        if os.path.isfile(fusion_path(args)):
            with open(fusion_path(args)) as f:
                tuned = json.load(f)
            refiner.lam, refiner.tau = tuned["lam"], tuned["tau"]
            print(f"Using tuned fusion weights lambda={refiner.lam} tau={refiner.tau}")
        else:
            print(f"[warn] {fusion_path(args)} not found; using defaults "
                  f"lambda={refiner.lam} tau={refiner.tau}")

    results = {}
    snr_list = args.snr_list or Config.eval_snr_list
    with_emb = not args.no_bertscore

    for snr_val in snr_list:
        print(f"\n===== SNR = {snr_val} dB  |  channel = {channel} =====")
        pred_texts, ref_texts, (c, t), items = decode_at_snr(
            model, loader, idx2word, snr_val, channel, device,
            args.max_samples, args.beam_width,
            Config.llm_topk if refiner is not None else 0)

        metrics = {"WordAcc": c / max(t, 1),
                   **text_metrics(pred_texts, ref_texts, device, with_emb)}

        for j in range(min(3, len(pred_texts))):
            print(f"  REF : {ref_texts[j]}")
            print(f"  PRED: {pred_texts[j]}")
            print()

        if refiner is not None:
            refined = refiner.refine(items)
            n_changed = sum(a != b for a, b in zip(pred_texts, refined))
            r = text_metrics(refined, ref_texts, device, with_emb)
            metrics.update({f"LLM_{k}": v for k, v in r.items()})
            metrics["LLM_changed_frac"] = n_changed / max(len(refined), 1)
            for a, b, ref in zip(pred_texts, refined, ref_texts):
                if a != b:
                    print(f"  REF : {ref}\n  PRED: {a}\n  LLM : {b}\n")
                    break

        results[int(snr_val)] = metrics
        print("  Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

    # ---- save results ----
    out_dir = os.path.dirname(args.checkpoint)
    suffix = f"_{args.eval_tag}" if args.eval_tag else ""
    out_path = os.path.join(out_dir, f"eval_{channel}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default=None,
                   help="Default: the channel the checkpoint was trained on")
    p.add_argument("--use_llm", action="store_true",
                   help="Apply LLM fusion refinement and report before/after metrics")
    p.add_argument("--tune_llm", action="store_true",
                   help="Tune LLM fusion (lambda, tau) on the validation split")
    p.add_argument("--llm_model", type=str, default=None,
                   help="HuggingFace model name or local path for the refiner "
                        "(default: fine-tuned Europarl BERT if present)")
    p.add_argument("--beam_width", type=int, default=1,
                   help="Beam width for decoding (1=greedy, >1=beam search)")
    p.add_argument("--eval_tag", type=str, default="",
                   help="Tag appended to output filename, e.g. 'beam5'")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--snr_list", type=float, nargs="+", default=None)
    p.add_argument("--no_bertscore", action="store_true",
                   help="Skip BERTScore / SentSim (fast smoke runs)")
    args = p.parse_args()
    if args.tune_llm:
        run_tuning(args)
        return
    if args.max_samples is None:
        args.max_samples = Config.eval_max_samples
    run_evaluation(args)


if __name__ == "__main__":
    main()

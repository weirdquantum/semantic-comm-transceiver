"""
Comprehensive evaluation of semantic communication models.

Metrics computed per SNR point:
  - Word Accuracy   (token-level match)
  - BLEU-1 … BLEU-4 (n-gram precision)
  - BERTScore       (contextual embedding similarity)
  - Sentence Similarity (cosine sim of mean BERT embeddings)

Supports beam search decoding (--beam_width) and optional LLM refinement
(--use_llm) with configurable BERT model path (--llm_model).

Usage:
  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt \\
                     --model improved --channel awgn --beam_width 5 \\
                     --eval_tag beam5

  python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt \\
                     --model improved --channel awgn --use_llm \\
                     --llm_model checkpoints/bert_europarl --eval_tag ftbert
"""

import argparse
import json
import os
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import Config
from dataset import get_dataloader, load_vocab
from train import build_model
from utils import set_seed, ids_to_text, batch_ids_to_texts


# ====================================================================
# Metric helpers
# ====================================================================
def compute_word_accuracy(pred_ids, ref_ids, pad_idx=0, end_idx=2):
    """Token-level accuracy ignoring PAD."""
    correct = 0
    total = 0
    for p, r in zip(pred_ids, ref_ids):
        for pi, ri in zip(p, r):
            pi = pi.item() if hasattr(pi, "item") else int(pi)
            ri = ri.item() if hasattr(ri, "item") else int(ri)
            if ri == pad_idx:
                break
            total += 1
            if pi == ri:
                correct += 1
    return correct / max(total, 1)


def compute_bleu(pred_texts, ref_texts):
    """Average sentence-level BLEU-1..4."""
    sf = SmoothingFunction().method1
    bleu_scores = {f"BLEU-{n}": [] for n in range(1, 5)}
    for pred, ref in zip(pred_texts, ref_texts):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if len(pred_tokens) == 0:
            pred_tokens = [""]
        for n in range(1, 5):
            w = [1.0 / n] * n + [0.0] * (4 - n)
            score = sentence_bleu([ref_tokens], pred_tokens,
                                  weights=w, smoothing_function=sf)
            bleu_scores[f"BLEU-{n}"].append(score)
    return {k: float(np.mean(v)) for k, v in bleu_scores.items()}


def compute_bert_score(pred_texts, ref_texts, device="cpu"):
    """BERTScore (Precision, Recall, F1) using the bert-score library."""
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(pred_texts, ref_texts, lang="en",
                                  device=str(device), verbose=False)
        return {
            "BERTScore_P": float(P.mean()),
            "BERTScore_R": float(R.mean()),
            "BERTScore_F1": float(F1.mean()),
        }
    except Exception:
        return {"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F1": 0.0}


def compute_sentence_similarity(pred_texts, ref_texts, device="cpu"):
    """Mean cosine similarity of sentence-level DistilBERT [CLS] embeddings."""
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        mdl = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        mdl.eval()
        sims = []
        bs = 64
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
    except Exception:
        return {"SentSim": 0.0}


# ====================================================================
# Main evaluation loop
# ====================================================================
def evaluate_at_snr(model, loader, vocab_inv, snr_val, channel_type,
                    device, max_samples=2000, collect_confidence=False,
                    beam_width=1):
    """Decode on the test set at a fixed SNR and collect predictions.

    Args:
        beam_width: 1 = greedy decode, >1 = beam search.
        collect_confidence: also return per-position softmax confidence.
    """
    model.eval()
    all_pred_ids = []
    all_ref_ids = []
    all_confs = [] if collect_confidence else None
    n = 0
    use_beam = beam_width > 1 and hasattr(model, "beam_search_decode")

    for batch in loader:
        batch = batch.to(device)
        B = batch.size(0)
        snr_db = torch.full((B,), snr_val, device=device, dtype=torch.float)

        if use_beam:
            pred_ids, confs = model.beam_search_decode(
                batch, snr_db, channel_type,
                max_len=Config.max_seq_len,
                start_idx=Config.START_IDX,
                end_idx=Config.END_IDX,
                beam_width=beam_width,
            )
            if collect_confidence:
                all_confs.append(confs.cpu())
        elif collect_confidence and hasattr(model, "greedy_decode_with_confidence"):
            pred_ids, confs = model.greedy_decode_with_confidence(
                batch, snr_db, channel_type,
                max_len=Config.max_seq_len,
                start_idx=Config.START_IDX,
                end_idx=Config.END_IDX,
            )
            all_confs.append(confs.cpu())
        else:
            pred_ids = model.greedy_decode(
                batch, snr_db, channel_type,
                max_len=Config.max_seq_len,
                start_idx=Config.START_IDX,
                end_idx=Config.END_IDX,
            )

        max_len = Config.max_seq_len
        if pred_ids.size(1) < max_len:
            pad = torch.full((pred_ids.size(0), max_len - pred_ids.size(1)),
                             Config.PAD_IDX, dtype=pred_ids.dtype,
                             device=pred_ids.device)
            pred_ids = torch.cat([pred_ids, pad], dim=1)
        if collect_confidence and all_confs and all_confs[-1].size(1) < max_len:
            cpad = torch.ones(all_confs[-1].size(0),
                              max_len - all_confs[-1].size(1))
            all_confs[-1] = torch.cat([all_confs[-1], cpad], dim=1)

        all_pred_ids.append(pred_ids.cpu())
        all_ref_ids.append(batch.cpu())
        n += B
        if n >= max_samples:
            break

    all_pred_ids = torch.cat(all_pred_ids, dim=0)[:max_samples]
    all_ref_ids = torch.cat(all_ref_ids, dim=0)[:max_samples]
    if collect_confidence and all_confs:
        all_confs = torch.cat(all_confs, dim=0)[:max_samples]
    else:
        all_confs = None
    return all_pred_ids, all_ref_ids, all_confs


def _build_word_confidences(pred_ids, conf_tensor, idx2word, end_idx):
    """Convert token-level confidence tensor to per-word confidence lists.

    Each sentence gets a list of floats whose length matches the number
    of words produced by ``batch_ids_to_texts`` (tokens up to END,
    skipping START at position 0).
    """
    if conf_tensor is None:
        return None
    result = []
    for i in range(pred_ids.size(0)):
        word_confs = []
        for j in range(1, pred_ids.size(1)):
            tid = pred_ids[i, j].item()
            if tid == end_idx:
                break
            if tid == 0:
                break
            c = conf_tensor[i, j].item() if j < conf_tensor.size(1) else 1.0
            word_confs.append(c)
        result.append(word_confs)
    return result


def run_evaluation(args):
    set_seed(42)
    device = Config.device
    word2idx, idx2word = load_vocab()
    vocab_size = len(word2idx)

    model = build_model(args.model, vocab_size, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.model} from {args.checkpoint}")

    loader = get_dataloader("test", batch_size=128, shuffle=False)

    beam_width = getattr(args, "beam_width", 1)
    if beam_width > 1:
        print(f"Beam search enabled (width={beam_width})")

    # optional LLM refiner
    refiner = None
    if args.use_llm:
        from models.llm_refiner import LLMRefiner
        llm_model = getattr(args, "llm_model", "bert-base-uncased")
        refiner = LLMRefiner(model_name=llm_model, device=device)
        print(f"LLM refiner loaded ({llm_model})")

    results = {}
    snr_list = args.snr_list or Config.eval_snr_list

    for snr_val in snr_list:
        print(f"\n===== SNR = {snr_val} dB  |  channel = {args.channel} =====")
        pred_ids, ref_ids, conf_tensor = evaluate_at_snr(
            model, loader, idx2word, snr_val, args.channel, device,
            max_samples=args.max_samples,
            collect_confidence=(refiner is not None),
            beam_width=beam_width)

        word_acc = compute_word_accuracy(pred_ids, ref_ids,
                                         Config.PAD_IDX, Config.END_IDX)

        pred_texts = batch_ids_to_texts(pred_ids, idx2word, Config.END_IDX)
        ref_texts = batch_ids_to_texts(ref_ids, idx2word, Config.END_IDX)

        bleu = compute_bleu(pred_texts, ref_texts)
        bert_s = compute_bert_score(pred_texts, ref_texts, device)
        sent_sim = compute_sentence_similarity(pred_texts, ref_texts, device)

        metrics = {"WordAcc": word_acc, **bleu, **bert_s, **sent_sim}

        for j in range(min(3, len(pred_texts))):
            print(f"  REF : {ref_texts[j]}")
            print(f"  PRED: {pred_texts[j]}")
            print()

        # LLM refinement (if enabled)
        if refiner is not None:
            decoder_confidences = _build_word_confidences(
                pred_ids, conf_tensor, idx2word, Config.END_IDX
            )
            refined_texts = refiner.refine(pred_texts, decoder_confidences)
            r_bleu = compute_bleu(refined_texts, ref_texts)
            r_bert = compute_bert_score(refined_texts, ref_texts, device)
            r_sim = compute_sentence_similarity(refined_texts, ref_texts, device)
            metrics.update({f"LLM_{k}": v for k, v in r_bleu.items()})
            metrics.update({f"LLM_{k}": v for k, v in r_bert.items()})
            metrics.update({f"LLM_{k}": v for k, v in r_sim.items()})
            for j in range(min(3, len(refined_texts))):
                print(f"  LLM : {refined_texts[j]}")
                print()

        results[int(snr_val)] = metrics
        print("  Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})

    # ---- save results ----
    out_dir = os.path.dirname(args.checkpoint)
    eval_tag = getattr(args, "eval_tag", "")
    suffix = f"_{eval_tag}" if eval_tag else ""
    out_path = os.path.join(out_dir, f"eval_{args.channel}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["baseline", "improved"], default="improved")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--use_llm", action="store_true",
                   help="Apply LLM refinement and report before/after metrics")
    p.add_argument("--llm_model", type=str, default="bert-base-uncased",
                   help="HuggingFace model name or local path for LLM refiner")
    p.add_argument("--beam_width", type=int, default=1,
                   help="Beam width for decoding (1=greedy, >1=beam search)")
    p.add_argument("--eval_tag", type=str, default="",
                   help="Tag appended to output filename, e.g. 'beam5'")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--snr_list", type=float, nargs="+", default=None)
    args = p.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

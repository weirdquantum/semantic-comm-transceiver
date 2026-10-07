"""
Ablation study runner and result visualization.

Runs (or collects) evaluation results for multiple model configurations
and produces comparison plots suitable for the final report.

Configurations evaluated:
  A) Baseline DeepSC  (no innovations)
  B) + SNR-Adaptive Attention only
  C) + Composite Loss only
  D) + SNR-Adaptive + Composite Loss (full Improved, no LLM)
  E) + Full Improved + LLM refinement

Usage:
  # After training all variants, collect & plot
  python ablation.py --results_dir checkpoints --out_dir results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = {
    "Baseline":           "baseline_awgn",
    "Improved (full)":    "improved_awgn",
    "Improved+no_sem":    "improved_awgn_no_sem",
    "Improved+rayleigh":  "improved_rayleigh",
    "Baseline+rayleigh":  "baseline_rayleigh",
}

METRICS = ["WordAcc", "BLEU-1", "BLEU-4", "BERTScore_F1", "SentSim"]
METRIC_LABELS = {
    "WordAcc":       "Word Accuracy",
    "BLEU-1":        "BLEU-1",
    "BLEU-4":        "BLEU-4",
    "BERTScore_F1":  "BERTScore F1",
    "SentSim":       "Sentence Similarity",
}


def load_results(results_dir):
    """Load eval_awgn.json / eval_rayleigh.json from each config dir."""
    data = {}
    for label, dirname in CONFIGS.items():
        for ch in ["awgn", "rayleigh"]:
            fpath = os.path.join(results_dir, dirname, f"eval_{ch}.json")
            if os.path.isfile(fpath):
                with open(fpath) as f:
                    raw = json.load(f)
                data[label] = {int(float(k)): v for k, v in raw.items()}
                break
    return data


def plot_metric_vs_snr(data, metric, out_dir):
    """Plot one metric across SNR for all available configurations."""
    plt.figure(figsize=(8, 5))
    for label, snr_dict in data.items():
        snrs = sorted(snr_dict.keys())
        vals = [snr_dict[s].get(metric, 0) for s in snrs]
        plt.plot(snrs, vals, marker="o", linewidth=2, label=label)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} vs SNR", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{metric}_vs_snr.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_llm_comparison(data_before, data_after, metric, out_dir):
    """Compare a metric before and after LLM refinement."""
    if not data_before or not data_after:
        return
    plt.figure(figsize=(8, 5))
    snrs = sorted(data_before.keys())
    before = [data_before[s].get(metric, 0) for s in snrs]
    after = [data_after[s].get(f"LLM_{metric}", 0) for s in snrs]
    plt.plot(snrs, before, "o--", linewidth=2, label="Before LLM")
    plt.plot(snrs, after, "s-", linewidth=2, label="After LLM Refine")
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    plt.title(f"LLM Refinement Effect on {METRIC_LABELS.get(metric, metric)}",
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"llm_{metric}_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_training_curves(results_dir, out_dir):
    """Plot training loss and accuracy curves from history.json files."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, dirname in CONFIGS.items():
        hpath = os.path.join(results_dir, dirname, "history.json")
        if not os.path.isfile(hpath):
            continue
        with open(hpath) as f:
            hist = json.load(f)
        epochs = [r["epoch"] for r in hist]
        axes[0].plot(epochs, [r["train_loss"] for r in hist], label=label)
        axes[1].plot(epochs, [r["val_acc"] for r in hist], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Training Loss",
                title="Training Loss Curves")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Validation Accuracy",
                title="Validation Accuracy Curves")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def print_summary_table(data):
    """Print a summary table of all metrics at selected SNR points."""
    highlight_snrs = [-5, 0, 10]
    print("\n" + "=" * 90)
    print(f"{'Configuration':<25}", end="")
    for snr in highlight_snrs:
        print(f"{'SNR=' + str(snr) + 'dB':>20}", end="")
    print()
    print("-" * 90)
    for metric in METRICS:
        print(f"\n  {METRIC_LABELS.get(metric, metric)}:")
        for label, snr_dict in data.items():
            print(f"    {label:<23}", end="")
            for snr in highlight_snrs:
                val = snr_dict.get(snr, {}).get(metric, float("nan"))
                print(f"{val:>20.4f}", end="")
            print()
    print("=" * 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="checkpoints")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_results(args.results_dir)
    if not data:
        print("No evaluation result files found.  Run evaluate.py first.")
        return

    print(f"Loaded results for: {list(data.keys())}")

    # metric-vs-SNR plots
    for metric in METRICS:
        plot_metric_vs_snr(data, metric, args.out_dir)

    # LLM comparison (if LLM metrics available in 'Improved (full)')
    if "Improved (full)" in data:
        llm_data = data["Improved (full)"]
        sample_snr = list(llm_data.keys())[0]
        has_llm = any(k.startswith("LLM_") for k in llm_data[sample_snr])
        if has_llm:
            for metric in ["BLEU-1", "BLEU-4", "BERTScore_F1", "SentSim"]:
                plot_llm_comparison(llm_data, llm_data, metric, args.out_dir)

    # training curves
    plot_training_curves(args.results_dir, args.out_dir)

    # summary table
    print_summary_table(data)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

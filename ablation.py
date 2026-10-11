"""
Ablation study result collection and visualization.

Reads the evaluation JSON files written by evaluate.py and produces
comparison plots and a summary table suitable for the final report.

Configurations (see run_all.py):
  AWGN:      Baseline, Improved (full), -FiLM, -masked power, -feature loss,
             Baseline/Improved with beam search, Improved + beam + LLM fusion
  Rayleigh:  Baseline, Improved (full), -CSI-aware SNR, beam / LLM variants

Usage:
  # After training and evaluating all variants, collect & plot
  python ablation.py --results_dir checkpoints --out_dir results
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# label -> (checkpoint dir, eval file, metric prefix)
CONFIGS = {
    "awgn": {
        "Baseline":                 ("baseline_awgn", "eval_awgn.json", ""),
        "Improved (full)":          ("improved_awgn", "eval_awgn.json", ""),
        "Improved - FiLM":          ("improved_awgn_no_film", "eval_awgn.json", ""),
        "Improved - masked power":  ("improved_awgn_no_maskpow", "eval_awgn.json", ""),
        "Improved - feat. loss":    ("improved_awgn_no_feat", "eval_awgn.json", ""),
        "Baseline + beam5":         ("baseline_awgn", "eval_awgn_beam5.json", ""),
        "Improved + beam5":         ("improved_awgn", "eval_awgn_beam5.json", ""),
        "Improved + beam5 + LLM":   ("improved_awgn", "eval_awgn_beam5_llm.json", "LLM_"),
    },
    "rayleigh": {
        "Baseline":                 ("baseline_rayleigh", "eval_rayleigh.json", ""),
        "Improved (full)":          ("improved_rayleigh", "eval_rayleigh.json", ""),
        "Improved - CSI SNR":       ("improved_rayleigh_no_csi", "eval_rayleigh.json", ""),
        "Baseline + beam5":         ("baseline_rayleigh", "eval_rayleigh_beam5.json", ""),
        "Improved + beam5":         ("improved_rayleigh", "eval_rayleigh_beam5.json", ""),
        "Improved + beam5 + LLM":   ("improved_rayleigh", "eval_rayleigh_beam5_llm.json", "LLM_"),
    },
}

METRICS = ["WordAcc", "BLEU-1", "BLEU-4", "BERTScore_F1", "SentSim"]
METRIC_LABELS = {
    "WordAcc":       "Word Accuracy",
    "BLEU-1":        "BLEU-1",
    "BLEU-4":        "BLEU-4",
    "BERTScore_F1":  "BERTScore F1",
    "SentSim":       "Sentence Similarity",
}


def load_results(results_dir, channel):
    """{label: {snr: {metric: value}}} for every evaluation file found."""
    data = {}
    for label, (dirname, fname, prefix) in CONFIGS[channel].items():
        fpath = os.path.join(results_dir, dirname, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            raw = json.load(f)
        data[label] = {
            int(float(snr)): {k[len(prefix):]: v for k, v in m.items()
                              if k.startswith(prefix) and (prefix or not k.startswith("LLM_"))}
            for snr, m in raw.items()
        }
    return data


def plot_metric_vs_snr(data, metric, channel, out_dir):
    """Plot one metric across SNR for all available configurations."""
    plt.figure(figsize=(8, 5))
    for label, snr_dict in data.items():
        snrs = sorted(snr_dict.keys())
        if not all(metric in snr_dict[s] for s in snrs):
            continue
        vals = [snr_dict[s][metric] for s in snrs]
        style = "--" if "beam" in label else "-"
        plt.plot(snrs, vals, style, marker="o", linewidth=2, label=label)
        if metric == "BLEU-4" and "BLEU-4_CI_low" in snr_dict[snrs[0]]:
            plt.fill_between(snrs,
                             [snr_dict[s]["BLEU-4_CI_low"] for s in snrs],
                             [snr_dict[s]["BLEU-4_CI_high"] for s in snrs],
                             alpha=0.15)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
    plt.title(f"{METRIC_LABELS.get(metric, metric)} vs SNR ({channel.upper()})",
              fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{channel}_{metric}_vs_snr.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_gain_vs_baseline(data, metric, channel, out_dir):
    """Difference to the greedy Baseline, which makes small gaps visible."""
    if "Baseline" not in data:
        return
    base = data["Baseline"]
    plt.figure(figsize=(8, 5))
    for label, snr_dict in data.items():
        if label == "Baseline":
            continue
        snrs = [s for s in sorted(snr_dict) if metric in snr_dict[s] and metric in base.get(s, {})]
        if not snrs:
            continue
        diff = [100 * (snr_dict[s][metric] - base[s][metric]) for s in snrs]
        style = "--" if "beam" in label else "-"
        plt.plot(snrs, diff, style, marker="o", linewidth=2, label=label)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("SNR (dB)", fontsize=13)
    plt.ylabel(f"Δ {METRIC_LABELS.get(metric, metric)} vs Baseline (points)", fontsize=12)
    plt.title(f"Gain over greedy Baseline ({channel.upper()})", fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{channel}_{metric}_gain_vs_baseline.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_training_curves(results_dir, out_dir):
    """Plot training loss and validation accuracy from history.json files."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dirs = sorted({d for cfg in CONFIGS.values() for d, _, _ in cfg.values()})
    for dirname in dirs:
        hpath = os.path.join(results_dir, dirname, "history.json")
        if not os.path.isfile(hpath):
            continue
        with open(hpath) as f:
            hist = json.load(f)
        epochs = [r["epoch"] for r in hist]
        axes[0].plot(epochs, [r["train_loss"] for r in hist], label=dirname)
        axes[1].plot(epochs, [r["val_acc"] for r in hist], label=dirname)
    axes[0].set(xlabel="Epoch", ylabel="Training Loss",
                title="Training Loss Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Validation Accuracy (mean of -5/0/5 dB)",
                title="Validation Accuracy Curves")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def summary_table(data, channel, snrs=(-10, -5, 0, 5, 10)):
    """Markdown table of BLEU-4 (with 95% CI) and WordAcc."""
    lines = [f"### {channel.upper()}", ""]
    for metric in ["BLEU-4", "WordAcc", "BERTScore_F1"]:
        lines.append(f"**{METRIC_LABELS[metric]}**\n")
        lines.append("| Configuration | " + " | ".join(f"{s} dB" for s in snrs) + " |")
        lines.append("|---|" + "---|" * len(snrs))
        for label, snr_dict in data.items():
            cells = []
            for s in snrs:
                m = snr_dict.get(s, {})
                if metric not in m:
                    cells.append("–")
                elif metric == "BLEU-4" and "BLEU-4_CI_low" in m:
                    half = (m["BLEU-4_CI_high"] - m["BLEU-4_CI_low"]) / 2
                    cells.append(f"{m[metric]:.4f} ± {half:.4f}")
                else:
                    cells.append(f"{m[metric]:.4f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="checkpoints")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables = []
    for channel in CONFIGS:
        data = load_results(args.results_dir, channel)
        if not data:
            print(f"No {channel} evaluation files found.")
            continue
        print(f"Loaded {channel} results for: {list(data.keys())}")
        for metric in METRICS:
            plot_metric_vs_snr(data, metric, channel, args.out_dir)
        for metric in ["BLEU-4", "WordAcc"]:
            plot_gain_vs_baseline(data, metric, channel, args.out_dir)
        tables.append(summary_table(data, channel))

    if any(os.path.isfile(os.path.join(args.results_dir, d, "history.json"))
           for cfg in CONFIGS.values() for d, _, _ in cfg.values()):
        plot_training_curves(args.results_dir, args.out_dir)

    if tables:
        text = "\n\n".join(tables)
        with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
            f.write(text + "\n")
        print("\n" + text)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()

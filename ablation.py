"""
Collect evaluation results, write a summary table and plot the comparisons.

Reads the eval_*.json files written by evaluate.py from <results_dir>/<model>/
and writes <out_dir>/summary.md and <out_dir>/figures/*.png.

Usage:
  python ablation.py --results_dir checkpoints --out_dir results
  python ablation.py --results_dir experiments/v2/runs --out_dir experiments/v2
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# label -> (model dir, eval file, metric prefix)
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

METRIC_LABELS = {
    "WordAcc": "Word Accuracy",
    "BLEU-1": "BLEU-1",
    "BLEU-4": "BLEU-4",
    "BERTScore_F1": "BERTScore F1",
    "SentSim": "Sentence Similarity",
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


def _finish(path, xlabel, ylabel, title):
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metric_vs_snr(data, metric, channel, fig_dir):
    """One metric across SNR for every configuration (BLEU-4 with its CI)."""
    plt.figure(figsize=(8, 5))
    for label, snr_dict in data.items():
        snrs = sorted(snr_dict)
        if not all(metric in snr_dict[s] for s in snrs):
            continue
        style = "--" if "beam" in label else "-"
        plt.plot(snrs, [snr_dict[s][metric] for s in snrs], style,
                 marker="o", linewidth=2, label=label)
        if metric == "BLEU-4" and "BLEU-4_CI_low" in snr_dict[snrs[0]]:
            plt.fill_between(snrs, [snr_dict[s]["BLEU-4_CI_low"] for s in snrs],
                             [snr_dict[s]["BLEU-4_CI_high"] for s in snrs], alpha=0.15)
    name = METRIC_LABELS[metric]
    _finish(os.path.join(fig_dir, f"{channel}_{metric}_vs_snr.png"),
            "SNR (dB)", name, f"{name} vs SNR ({channel.upper()})")


def plot_gain_vs_baseline(data, metric, channel, fig_dir):
    """Difference to the greedy Baseline, which makes small gaps visible."""
    base = data.get("Baseline")
    if base is None:
        return
    plt.figure(figsize=(8, 5))
    for label, snr_dict in data.items():
        snrs = [s for s in sorted(snr_dict) if metric in snr_dict[s] and metric in base.get(s, {})]
        if label == "Baseline" or not snrs:
            continue
        style = "--" if "beam" in label else "-"
        plt.plot(snrs, [100 * (snr_dict[s][metric] - base[s][metric]) for s in snrs],
                 style, marker="o", linewidth=2, label=label)
    plt.axhline(0, color="black", linewidth=1)
    _finish(os.path.join(fig_dir, f"{channel}_{metric}_gain_vs_baseline.png"),
            "SNR (dB)", f"Δ {METRIC_LABELS[metric]} vs Baseline (points)",
            f"Gain over greedy Baseline ({channel.upper()})")


def plot_training_curves(results_dir, fig_dir):
    """Validation accuracy per epoch from the history.json files, if present."""
    dirs = sorted({d for cfg in CONFIGS.values() for d, _, _ in cfg.values()})
    histories = {}
    for d in dirs:
        path = os.path.join(results_dir, d, "history.json")
        if os.path.isfile(path):
            with open(path) as f:
                histories[d] = json.load(f)
    if not histories:
        return
    plt.figure(figsize=(8, 5))
    for d, hist in histories.items():
        plt.plot([r["epoch"] for r in hist], [r["val_acc"] for r in hist], label=d)
    _finish(os.path.join(fig_dir, "training_curves.png"), "Epoch",
            "Validation accuracy (mean of -5/0/5 dB)", "Training progress")


def summary_table(data, channel, snrs=(-10, -5, 0, 5, 10)):
    """Markdown tables of BLEU-4 (± 95% CI half-width), Word Acc and BERTScore."""
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
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    tables = []
    for channel in CONFIGS:
        data = load_results(args.results_dir, channel)
        if not data:
            print(f"No {channel} evaluation files found in {args.results_dir}.")
            continue
        for metric in METRIC_LABELS:
            plot_metric_vs_snr(data, metric, channel, fig_dir)
        for metric in ["BLEU-4", "WordAcc"]:
            plot_gain_vs_baseline(data, metric, channel, fig_dir)
        tables.append(summary_table(data, channel))
    plot_training_curves(args.results_dir, fig_dir)

    if tables:
        text = "\n\n".join(tables)
        with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
            f.write(text + "\n")
        print(text)
    print(f"\nSummary and figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()

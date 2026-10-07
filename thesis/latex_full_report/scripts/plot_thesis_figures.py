"""
Generate data-driven figures for the LaTeX thesis from checkpoints JSON.
Run from repository root:  python thesis/latex_full_report/scripts/plot_thesis_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # FinalProject (…/thesis/latex_full_report/scripts)
CKPT = ROOT / "checkpoints_v1"  # results of the first (v1) experiments
OUT = Path(__file__).resolve().parents[1] / "figures"

SNR_ORDER = [-10, -5, 0, 5, 10, 15, 20]


def load_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def series_bleu4(data: dict, use_llm: bool = False) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for s in SNR_ORDER:
        k = str(s)
        if k not in data:
            continue
        row = data[k]
        if use_llm and "LLM_BLEU-4" in row:
            ys.append(float(row["LLM_BLEU-4"]))
        else:
            ys.append(float(row["BLEU-4"]))
        xs.append(s)
    return xs, ys


def series_wordacc(data: dict) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for s in SNR_ORDER:
        k = str(s)
        if k not in data:
            continue
        xs.append(s)
        ys.append(float(data[k]["WordAcc"]))
    return xs, ys


def plot_bleu_awgn_strategies() -> None:
    base = load_metrics(CKPT / "baseline_awgn" / "eval_awgn.json")
    imp = load_metrics(CKPT / "improved_awgn" / "eval_awgn.json")
    beam = load_metrics(CKPT / "improved_awgn" / "eval_awgn_beam5.json")
    ft = load_metrics(CKPT / "improved_awgn" / "eval_awgn_ftbert.json")
    bft = load_metrics(CKPT / "improved_awgn" / "eval_awgn_beam5_ftbert.json")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    styles = [
        ("Baseline (AWGN)", base, False, "C0", "-", "o"),
        ("Improved + greedy", imp, False, "C1", "-", "s"),
        ("Improved + Beam5", beam, False, "C2", "-", "^"),
        ("Improved + FT-BERT (post)", ft, True, "C3", "--", "D"),
        ("Improved + Beam5 + FT-BERT (post)", bft, True, "C4", ":", "v"),
    ]
    for label, data, llm, color, ls, marker in styles:
        xs, ys = series_bleu4(data, use_llm=llm)
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker, ms=4, lw=1.6, label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLEU-4")
    ax.set_title("AWGN: BLEU-4 vs SNR (from evaluation JSON)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xticks(SNR_ORDER)
    ax.set_ylim(0.35, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bleu_awgn_strategies.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_bleu_awgn_strategies.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bleu_rayleigh() -> None:
    base = load_metrics(CKPT / "baseline_rayleigh" / "eval_rayleigh.json")
    imp = load_metrics(CKPT / "improved_rayleigh" / "eval_rayleigh.json")
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=150)
    for label, data, color, marker in [
        ("Baseline (Rayleigh)", base, "C0", "o"),
        ("Improved (Rayleigh)", imp, "C1", "s"),
    ]:
        xs, ys = series_bleu4(data, use_llm=False)
        ax.plot(xs, ys, color=color, marker=marker, ms=4, lw=1.6, label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLEU-4")
    ax.set_title("Rayleigh: BLEU-4 vs SNR")
    ax.legend(loc="lower right")
    ax.set_xticks(SNR_ORDER)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bleu_rayleigh.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_bleu_rayleigh.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_wordacc_awgn() -> None:
    base = load_metrics(CKPT / "baseline_awgn" / "eval_awgn.json")
    imp = load_metrics(CKPT / "improved_awgn" / "eval_awgn.json")
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=150)
    xs_b, ys_b = series_wordacc(base)
    xs_i, ys_i = series_wordacc(imp)
    ax.plot(xs_b, ys_b, "o-", color="C0", lw=1.6, label="Baseline")
    ax.plot(xs_i, ys_i, "s-", color="C1", lw=1.6, label="Improved (greedy)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Word accuracy")
    ax.set_title("AWGN: Word accuracy vs SNR")
    ax.legend()
    ax.set_xticks(SNR_ORDER)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_wordacc_awgn.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_wordacc_awgn.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_channel_schematic() -> None:
    """Simple schematic for AWGN vs Rayleigh (not real constellation data)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.6), dpi=150)
    rng = np.random.default_rng(0)
    tx = np.array([1.0, 0.0])
    noise = rng.normal(scale=0.22, size=(120, 2))
    rx = tx + noise
    ax1.scatter(rx[:, 0], rx[:, 1], s=8, alpha=0.45, c="C0")
    ax1.scatter([tx[0]], [tx[1]], s=80, c="red", zorder=5, label="Transmitted")
    ax1.set_title("AWGN (conceptual)")
    ax1.set_xlabel("I")
    ax1.set_ylabel("Q")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper left", fontsize=7)

    t = np.linspace(0, 2 * np.pi, 200)
    env = 0.35 * (1.0 + 0.55 * np.abs(np.sin(2.3 * t)))
    ax2.plot(t, env, color="C1", lw=1.8, label="|h(t)| envelope")
    ax2.plot(t, -env, color="C1", lw=1.8)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_title("Rayleigh fading (conceptual)")
    ax2.set_xlabel("Time / symbol index")
    ax2.set_ylabel("Gain")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_channel_models.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_channel_models.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_bleu_awgn_strategies()
    plot_bleu_rayleigh()
    plot_wordacc_awgn()
    plot_channel_schematic()
    print("Wrote:", "\n  ".join(str(p.name) for p in sorted(OUT.glob("fig_*.pdf"))))


if __name__ == "__main__":
    main()

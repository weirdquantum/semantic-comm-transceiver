"""
Convenience script that trains all ablation variants sequentially
and then runs evaluation + plotting.

Usage:
  python run_all.py                     # full pipeline
  python run_all.py --epochs 20         # quick test run
  python run_all.py --skip_train        # skip training, only evaluate + plot
"""

import argparse
import os
import subprocess
import sys

from config import Config


TRAIN_CONFIGS = [
    # (tag-used-for-dir, model, channel, extra-args)
    ("baseline_awgn",        "baseline",  "awgn",     []),
    ("improved_awgn",        "improved",  "awgn",     []),
    ("improved_awgn_no_sem", "improved",  "awgn",     ["--beta", "0.0", "--tag", "no_sem"]),
    ("baseline_rayleigh",    "baseline",  "rayleigh", []),
    ("improved_rayleigh",    "improved",  "rayleigh", []),
]


def run_cmd(cmd):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    py = sys.executable

    # ---- Training ----
    if not args.skip_train:
        for tag, model, channel, extras in TRAIN_CONFIGS:
            cmd = [py, "train.py",
                   "--model", model,
                   "--channel", channel,
                   "--epochs", str(args.epochs)] + extras
            run_cmd(cmd)

    # ---- Evaluation ----
    if not args.skip_eval:
        for tag, model, channel, _ in TRAIN_CONFIGS:
            ckpt = os.path.join(Config.checkpoint_dir, tag, "best.pt")
            if not os.path.isfile(ckpt):
                ckpt = os.path.join(Config.checkpoint_dir, tag, "last.pt")
            if not os.path.isfile(ckpt):
                print(f"  Checkpoint not found for {tag}, skipping eval.")
                continue
            cmd = [py, "evaluate.py",
                   "--checkpoint", ckpt,
                   "--model", model,
                   "--channel", channel]
            run_cmd(cmd)

        # Evaluate improved with LLM refinement
        ckpt_llm = os.path.join(Config.checkpoint_dir, "improved_awgn", "best.pt")
        if os.path.isfile(ckpt_llm):
            cmd = [py, "evaluate.py",
                   "--checkpoint", ckpt_llm,
                   "--model", "improved",
                   "--channel", "awgn",
                   "--use_llm"]
            run_cmd(cmd)

    # ---- Ablation plots ----
    run_cmd([py, "ablation.py",
             "--results_dir", Config.checkpoint_dir,
             "--out_dir", Config.result_dir])

    print("\nAll done!  Check the 'results/' directory for plots and tables.")


if __name__ == "__main__":
    main()

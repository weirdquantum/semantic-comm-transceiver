"""
Convenience script that trains all model variants sequentially, then
fine-tunes the refiner BERT, evaluates everything and plots the ablation.

The script is idempotent: finished variants (DONE marker) are skipped,
interrupted ones resume from last.pt, and existing evaluation files are
not recomputed.  Re-running the same command after a disconnect continues
where it stopped.

Usage:
  python run_all.py --amp               # full pipeline on a CUDA GPU
  python run_all.py --epochs 20         # shorter run
  python run_all.py --skip_train        # only evaluate + plot
  python run_all.py --smoke             # tiny end-to-end check (CPU ok)
"""

import argparse
import os
import subprocess
import sys

from config import Config


TRAIN_CONFIGS = [
    # (dir name, model, channel, extra train args)
    ("baseline_awgn",            "baseline", "awgn",     []),
    ("improved_awgn",            "improved", "awgn",     []),
    ("improved_awgn_no_film",    "improved", "awgn",     ["--no_film", "--tag", "no_film"]),
    ("improved_awgn_no_maskpow", "improved", "awgn",     ["--no_masked_power", "--tag", "no_maskpow"]),
    ("improved_awgn_no_feat",    "improved", "awgn",     ["--beta", "0.0", "--tag", "no_feat"]),
    ("baseline_rayleigh",        "baseline", "rayleigh", []),
    ("improved_rayleigh",        "improved", "rayleigh", []),
    ("improved_rayleigh_no_csi", "improved", "rayleigh", ["--no_csi_snr", "--tag", "no_csi"]),
]

# Main models get the full metric set (BERTScore, SentSim) and beam search.
# Ablation variants only need BLEU / WordAcc with greedy decoding to show
# each component's contribution, which keeps evaluation time manageable.
MAIN_CONFIGS = ["baseline_awgn", "improved_awgn", "baseline_rayleigh", "improved_rayleigh"]
# models that additionally get the LLM fusion refiner
LLM_CONFIGS = ["improved_awgn", "improved_rayleigh"]
BEAM = 5


ROOT = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, env):
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--amp", action="store_true", help="Mixed precision training")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--no_llm", action="store_true", help="Skip BERT fine-tuning and refinement")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run into smoke_checkpoints/ to check the pipeline")
    args = p.parse_args()

    py = sys.executable
    env = dict(os.environ)
    train_extra, eval_extra, bert_extra = [], [], []
    if args.smoke:
        # always isolated, so a smoke run can never mark real runs as DONE
        env["SC_CKPT_DIR"] = os.path.join(ROOT, "smoke_checkpoints")
        env["SC_RESULT_DIR"] = os.path.join(ROOT, "smoke_checkpoints", "results")
        args.epochs = 1
        train_extra = ["--limit_train_batches", "20", "--limit_val_samples", "256"]
        eval_extra = ["--max_samples", "64", "--snr_list", "0", "--no_bertscore"]
        bert_extra = ["--epochs", "1", "--max_sentences", "64"]
    if args.amp:
        train_extra.append("--amp")
    ckpt_root = env.get("SC_CKPT_DIR", Config.checkpoint_dir)
    result_dir = env.get("SC_RESULT_DIR", Config.result_dir)

    def ckpt_of(tag):
        path = os.path.join(ckpt_root, tag, "best.pt")
        return path if os.path.isfile(path) else os.path.join(ckpt_root, tag, "last.pt")

    def evaluate(tag, channel, extra, eval_tag):
        out = os.path.join(ckpt_root, tag, f"eval_{channel}{'_' + eval_tag if eval_tag else ''}.json")
        if os.path.isfile(out) and not args.smoke:
            print(f"  {out} exists, skipping.")
            return
        cmd = [py, "evaluate.py", "--checkpoint", ckpt_of(tag)] + extra
        if eval_tag:
            cmd += ["--eval_tag", eval_tag]
        run_cmd(cmd + eval_extra, env)

    # ---- Training ----
    if not args.skip_train:
        for tag, model, channel, extras in TRAIN_CONFIGS:
            if os.path.isfile(os.path.join(ckpt_root, tag, "DONE")) and not args.smoke:
                print(f"  {tag} already trained, skipping.")
                continue
            cmd = [py, "train.py",
                   "--model", model,
                   "--channel", channel,
                   "--epochs", str(args.epochs),
                   "--resume", "auto"] + extras + train_extra
            run_cmd(cmd, env)

    # ---- Evaluation ----
    if not args.skip_eval:
        for tag, _, channel, _ in TRAIN_CONFIGS:
            if not os.path.isfile(ckpt_of(tag)):
                print(f"  Checkpoint not found for {tag}, skipping eval.")
                continue
            if tag in MAIN_CONFIGS:
                evaluate(tag, channel, [], "")
                # beam vs greedy is compared on BLEU; BERTScore is the slowest metric
                evaluate(tag, channel, ["--beam_width", str(BEAM), "--no_bertscore"], f"beam{BEAM}")
            else:
                evaluate(tag, channel, ["--no_bertscore"], "")

        if not args.no_llm:
            bert_dir = os.path.join(ckpt_root, "bert_europarl")
            if not os.path.isfile(os.path.join(bert_dir, "config.json")) or args.smoke:
                run_cmd([py, "finetune_bert.py", "--save_dir", bert_dir] + bert_extra, env)
            for tag in LLM_CONFIGS:
                channel = "rayleigh" if "rayleigh" in tag else "awgn"
                if not os.path.isfile(ckpt_of(tag)):
                    continue
                beam = ["--beam_width", str(BEAM)]
                fusion = os.path.join(ckpt_root, tag, f"llm_fusion_beam{BEAM}.json")
                if not os.path.isfile(fusion) or args.smoke:
                    tune_extra = ["--max_samples", "64", "--snr_list", "0"] if args.smoke else []
                    run_cmd([py, "evaluate.py", "--checkpoint", ckpt_of(tag),
                             "--tune_llm", "--llm_model", bert_dir] + beam + tune_extra, env)
                evaluate(tag, channel, beam + ["--use_llm", "--llm_model", bert_dir],
                         f"beam{BEAM}_llm")

    # ---- Ablation plots ----
    run_cmd([py, "ablation.py",
             "--results_dir", ckpt_root,
             "--out_dir", result_dir], env)

    print(f"\nAll done!  Check {result_dir} for plots and tables.")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# One-command pipeline for a cloud GPU (AutoDL terminal, or Colab via `!bash`).
#
#   bash scripts/run_cloud.sh                 # full run (8 models x 80 epochs)
#   bash scripts/run_cloud.sh --epochs 2      # quick check before the real run
#
# Re-running the same command resumes where it stopped.
# Set SC_CKPT_DIR (and optionally SC_RESULT_DIR) to a persistent location,
# e.g. a mounted Google Drive folder on Colab or /root/autodl-tmp on AutoDL.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q -r requirements.txt
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# HuggingFace models (BERT, DistilBERT, BERTScore) — use a mirror if the hub is unreachable
if [ -z "${HF_ENDPOINT:-}" ] && ! curl -s -m 5 -o /dev/null https://huggingface.co; then
  export HF_ENDPOINT=https://hf-mirror.com
  echo "huggingface.co unreachable, using HF_ENDPOINT=$HF_ENDPOINT"
fi

# Data: upload your existing data/*.pkl to keep the exact same split.
# Otherwise download Europarl v7 and preprocess it here.
if [ ! -f data/train_data.pkl ]; then
  echo "data/train_data.pkl not found — downloading Europarl v7 (~1.5 GB)"
  wget -q -c http://www.statmt.org/europarl/v7/europarl.tgz
  tar xzf europarl.tgz --wildcards '*txt/en/*'
  RAW_DIR=$(dirname "$(find . -path '*txt/en/*.txt' | head -1)")
  python preprocess.py --raw_dir "$RAW_DIR"
fi

CKPT_DIR="${SC_CKPT_DIR:-checkpoints}"
mkdir -p "$CKPT_DIR"
python run_all.py --amp "$@" 2>&1 | tee -a "$CKPT_DIR/run_all.log"

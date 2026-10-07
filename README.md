# A Novel Transceiver Design for Semantic Communication

**Final Year Project — Glasgow College, UESTC (2025-26)**

## Overview

An end-to-end deep semantic communication system for text, built on top of
DeepSC (Xie et al., 2021). A Transformer semantic encoder maps a sentence to
features, a dense channel encoder compresses them into a power-normalised
signal, the signal passes through a differentiable AWGN or Rayleigh channel,
and the receiver reconstructs the sentence.

On top of the baseline the improved transceiver adds (each one can be
switched off for ablation):

1. **SNR-Adaptive Attention** — a learned SNR embedding is injected into the
   attention queries (`Q = W_q·x + W_caq·SNR_emb`) of every encoder
   self-attention and decoder cross-attention layer (inspired by
   SNR-EQ-JSCC, 2025).
2. **SNR-conditioned channel coder (FiLM)** — the channel encoder/decoder
   hidden layers are modulated by the SNR embedding (`--no_film`).
3. **Length-aware power normalisation** — padding transmits zero energy, so
   the same per-sentence energy budget is spent on real tokens only
   (`--no_masked_power`).
4. **CSI-aware receiver** — under Rayleigh fading the receiver conditions on
   the effective SNR `SNR·|h|²` it already knows from perfect CSI, while the
   transmitter only knows the average SNR (`--no_csi_snr`).
5. **Feature-denoising loss** — cross-entropy plus a per-token cosine loss
   between the channel-decoder output and the (detached) clean encoder
   features (`--beta 0` disables it).
6. **LLM fusion refinement** — at low-confidence positions a frozen,
   Europarl-fine-tuned BERT MLM re-ranks the decoder's own top-5 candidates
   (`log p_dec + λ·log p_BERT`), with λ and the confidence threshold tuned
   on a validation split.
7. **Batched beam search** decoding, available to both models.

All comparisons use the same channel uses (32 × 16 real symbols) and the
same transmit energy per sentence.

## Pipeline

```
sentence ─► Embedding + PosEnc ─► Semantic Encoder (SNR-adaptive) ─► Channel Encoder (power norm)
                                                                          │
                                                               AWGN / Rayleigh channel
                                                                          │
reconstruction ◄─ [BERT MLM refiner] ◄─ Semantic Decoder (SNR-adaptive) ◄─ Channel Decoder
```

## Project Structure

```
.
├── config.py              # Hyperparameters, special tokens, paths
├── preprocess.py          # Europarl v7 cleaning, vocab, 90/10 split → data/*.pkl
├── dataset.py             # Dataset / DataLoader; 5% of train held out as validation
├── utils.py               # Seeding, id→text, AverageMeter
├── models/
│   ├── modules.py         # PosEnc, SNR embedding, SNR-adaptive MHA, enc/dec layers,
│   │                      # channel encoder/decoder
│   ├── channel.py         # AWGN & Rayleigh (perfect-CSI) channels, effective SNR
│   ├── losses.py          # Composite loss (CE + feature denoising)
│   ├── transceiver.py     # BaselineDeepSC, ImprovedDeepSC, greedy / batched beam search
│   └── llm_refiner.py     # Decoder–BERT fusion refinement
├── train.py               # Train one variant (warm-up + cosine LR, AMP, auto-resume)
├── evaluate.py            # WordAcc / BLEU-1..4 (+95% CI) / BERTScore / SentSim per SNR,
│                          # LLM fusion tuning (--tune_llm)
├── finetune_bert.py       # Domain-adapt BERT MLM on Europarl for the refiner
├── ablation.py            # Comparison plots & summary tables
├── lightweight.py         # Knowledge distillation + quantisation
├── run_all.py             # Train all 8 variants → evaluate → refine → plot (resumable)
├── scripts/run_cloud.sh   # One-command run on a cloud GPU
├── checkpoints_v1/        # Results of the first experiment round (eval_*.json, history.json)
└── thesis/
    ├── latex_full_report/ # Final report (LaTeX: chapters/, tables/, figures/, refs.bib)
    │   └── scripts/plot_thesis_figures.py   # Builds result figures from checkpoints/*.json
    ├── reports.tex
    ├── thesis_draft_v1.md
    └── thesis_draft_v2_english.md
```

Model weights (`*.pt`) and preprocessed data (`data/`) are not included in the
repository; regenerate them with the steps below. New runs write to
`checkpoints/` (or `$SC_CKPT_DIR`).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Prepare data

```bash
wget http://www.statmt.org/europarl/v7/europarl.tgz
tar xzf europarl.tgz
python preprocess.py --raw_dir europarl/txt/en
```

### 3. Train

```bash
python train.py --model baseline --channel awgn --epochs 80
python train.py --model improved --channel awgn --epochs 80
python train.py --model improved --channel awgn --no_film --tag no_film
python train.py --model improved --channel awgn --no_masked_power --tag no_maskpow
python train.py --model improved --channel awgn --beta 0.0 --tag no_feat
python train.py --model baseline --channel rayleigh --epochs 80
python train.py --model improved --channel rayleigh --epochs 80
python train.py --model improved --channel rayleigh --no_csi_snr --tag no_csi
```

Add `--amp` on a CUDA GPU for mixed precision. Or run the whole pipeline
(training, evaluation, BERT fine-tuning, fusion tuning, plots) in one go;
re-running it resumes where it stopped:

```bash
python run_all.py --amp
python run_all.py --smoke   # tiny end-to-end check, writes to smoke_checkpoints/
```

### 4. Evaluate

```bash
# greedy / beam search (architecture and channel are read from the checkpoint)
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --beam_width 5 --eval_tag beam5

# LLM fusion: fine-tune BERT, tune (lambda, tau) on validation, then evaluate
python finetune_bert.py
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --beam_width 5 --tune_llm
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --beam_width 5 \
  --use_llm --eval_tag beam5_llm
```

### 5. Lightweight model

```bash
python lightweight.py --teacher checkpoints/improved_awgn/best.pt --epochs 40 --quantize
```

### 6. Thesis figures and PDF

```bash
python thesis/latex_full_report/scripts/plot_thesis_figures.py
cd thesis/latex_full_report && latexmk -pdf main.tex
```

## Running on a cloud GPU

`scripts/run_cloud.sh` installs dependencies, falls back to a HuggingFace
mirror if the hub is unreachable, downloads and preprocesses Europarl if
`data/*.pkl` is missing (uploading your existing `data/` keeps the exact same
split), and runs `run_all.py --amp`. Try `--epochs 2` first.

**AutoDL** (terminal):

```bash
git clone -b v2-improvements https://github.com/weirdquantum/semantic-comm-transceiver.git
cd semantic-comm-transceiver   # upload data/*.pkl into ./data
export SC_CKPT_DIR=/root/autodl-tmp/checkpoints
nohup bash scripts/run_cloud.sh > run.log 2>&1 &
```

**Colab** (GPU runtime):

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -b v2-improvements https://github.com/weirdquantum/semantic-comm-transceiver.git
%cd semantic-comm-transceiver
!mkdir -p data && cp /content/drive/MyDrive/sc_data/*.pkl data/
%env SC_CKPT_DIR=/content/drive/MyDrive/sc_checkpoints
%env SC_RESULT_DIR=/content/drive/MyDrive/sc_results
!bash scripts/run_cloud.sh
```

## Results (first experiment round)

The numbers below come from the first round of experiments
(`checkpoints_v1/`), before the v2 changes listed above. BLEU-4 on 2,000
Europarl test sentences:

| Model                    | Channel  | −10 dB | −5 dB | 0 dB  | 5 dB  | 10 dB | 20 dB |
| ------------------------ | -------- | ------ | ----- | ----- | ----- | ----- | ----- |
| Baseline DeepSC          | AWGN     | 0.419  | 0.734 | 0.837 | 0.963 | 0.991 | 0.993 |
| Improved, no sem. loss   | AWGN     | 0.449  | 0.745 | 0.840 | 0.971 | 0.995 | 0.996 |
| Improved (full)          | AWGN     | 0.433  | 0.739 | 0.837 | 0.969 | 0.993 | 0.994 |
| Improved + beam search 5 | AWGN     | 0.455  | 0.750 | 0.849 | 0.973 | 0.993 | 0.995 |
| Improved + BERT refiner  | AWGN     | 0.405  | 0.733 | 0.834 | 0.965 | 0.991 | 0.994 |
| Baseline DeepSC          | Rayleigh | 0.470  | 0.666 | 0.817 | 0.917 | 0.965 | 0.987 |
| Improved (full)          | Rayleigh | 0.464  | 0.660 | 0.805 | 0.915 | 0.965 | 0.988 |

In that round, SNR-adaptive attention and beam search gave the clearest gains
at low SNR on AWGN, while the semantic loss and the BERT refiner did not
improve BLEU. The thesis (`thesis/latex_full_report`) discusses these
results. The v1 Rayleigh channel also had `E[|h|²] = 2`, which v2 fixes.

## Evaluation Metrics

| Metric              | Level    | Description                                 |
| ------------------- | -------- | ------------------------------------------- |
| Word Accuracy       | Token    | Exact token match rate                      |
| BLEU-1 … BLEU-4     | Token    | n-gram precision                            |
| BERTScore F1        | Semantic | Contextual embedding similarity             |
| Sentence Similarity | Semantic | Cosine similarity of DistilBERT [CLS] vectors |

## Key References

1. Xie et al., "Deep Learning Enabled Semantic Communication Systems,"
   IEEE TSP, 2021.
2. SNR-EQ-JSCC, arXiv:2501.04732, 2025.
3. LLM-SC, Applied Sciences 15(13):7227, 2025.
4. KG-LLM, arXiv:2504.07431, 2025.
5. Yi et al., "Semantic Communication with a Shared Knowledge Base,"
   IEEE TWC, 2023.

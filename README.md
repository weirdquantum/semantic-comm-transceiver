# A Novel Transceiver Design for Semantic Communication

**Final Year Project — Glasgow College, UESTC (2025-26)**

## Overview

An end-to-end deep semantic communication system for text, built on top of
DeepSC (Xie et al., 2021). A Transformer semantic encoder maps a sentence to
features, a dense channel encoder compresses them into a power-normalised
signal, the signal passes through a differentiable AWGN or Rayleigh channel,
and the receiver reconstructs the sentence.

On top of the baseline the project adds:

1. **SNR-Adaptive Attention** — a learned SNR embedding is injected into the
   attention queries (`Q = W_q·x + W_caq·SNR_emb`) of every encoder
   self-attention and decoder cross-attention layer, so the transceiver can
   adapt to channel quality (inspired by SNR-EQ-JSCC, 2025).
2. **Semantic-Aware Composite Loss** — cross-entropy plus a cosine-distance
   term between the semantic features before and after the channel, with the
   semantic weight scaled by `sigmoid(SNR/5)`.
3. **LLM-Aided Refinement** — at inference time, words whose decoder softmax
   confidence is below a threshold are masked and re-predicted by a frozen
   BERT masked-LM (optionally fine-tuned on Europarl).
4. **Beam search decoding** for the improved model.
5. **Lightweight variant** via knowledge distillation and dynamic quantisation.

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
├── dataset.py             # PyTorch Dataset / DataLoader, vocab loading
├── utils.py               # Seeding, id→text, AverageMeter
├── models/
│   ├── modules.py         # PosEnc, SNR embedding, SNR-adaptive MHA, enc/dec layers,
│   │                      # channel encoder/decoder
│   ├── channel.py         # AWGN & Rayleigh (perfect-CSI) channel models
│   ├── losses.py          # Composite loss (CE + semantic consistency)
│   ├── transceiver.py     # BaselineDeepSC, ImprovedDeepSC (greedy / beam search)
│   └── llm_refiner.py     # Confidence-driven BERT MLM refinement
├── train.py               # Train one model variant
├── evaluate.py            # WordAcc / BLEU-1..4 / BERTScore / SentSim per SNR
├── finetune_bert.py       # Domain-adapt BERT MLM on Europarl for the refiner
├── ablation.py            # Comparison plots & summary tables
├── lightweight.py         # Knowledge distillation + quantisation
├── run_all.py             # Train all 5 variants → evaluate → plot
├── _diag_llm.py           # Old debugging script for the refiner (outdated API)
├── checkpoints/           # Per-variant eval_*.json and history.json (weights not tracked)
└── thesis/
    ├── latex_full_report/ # Final report (LaTeX: chapters/, tables/, figures/, refs.bib)
    │   └── scripts/plot_thesis_figures.py   # Builds result figures from checkpoints/*.json
    ├── reports.tex
    ├── thesis_draft_v1.md
    └── thesis_draft_v2_english.md
```

Model weights (`*.pt`) and preprocessed data (`data/`) are not included in the
repository; regenerate them with the steps below.

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
python train.py --model improved --channel awgn --epochs 80 --beta 0.0 --tag no_sem
python train.py --model baseline --channel rayleigh --epochs 80
python train.py --model improved --channel rayleigh --epochs 80
```

Or run everything (training, evaluation, plots) in one go:

```bash
python run_all.py --epochs 80
```

### 4. Evaluate

```bash
# greedy
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn

# beam search
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn \
  --beam_width 5 --eval_tag beam5

# with BERT refinement (optionally fine-tune first: python finetune_bert.py)
python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn \
  --use_llm --llm_model checkpoints/bert_europarl --eval_tag ftbert
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

## Results

BLEU-4 on 2,000 Europarl test sentences (from `checkpoints/*/eval_*.json`):

| Model                    | Channel  | −10 dB | −5 dB | 0 dB  | 5 dB  | 10 dB | 20 dB |
| ------------------------ | -------- | ------ | ----- | ----- | ----- | ----- | ----- |
| Baseline DeepSC          | AWGN     | 0.419  | 0.734 | 0.837 | 0.963 | 0.991 | 0.993 |
| Improved, no sem. loss   | AWGN     | 0.449  | 0.745 | 0.840 | 0.971 | 0.995 | 0.996 |
| Improved (full)          | AWGN     | 0.433  | 0.739 | 0.837 | 0.969 | 0.993 | 0.994 |
| Improved + beam search 5 | AWGN     | 0.455  | 0.750 | 0.849 | 0.973 | 0.993 | 0.995 |
| Improved + BERT refiner  | AWGN     | 0.405  | 0.733 | 0.834 | 0.965 | 0.991 | 0.994 |
| Baseline DeepSC          | Rayleigh | 0.470  | 0.666 | 0.817 | 0.917 | 0.965 | 0.987 |
| Improved (full)          | Rayleigh | 0.464  | 0.660 | 0.805 | 0.915 | 0.965 | 0.988 |

SNR-adaptive attention and beam search give the clearest gains at low SNR on
AWGN; the semantic loss and the BERT refiner do not improve BLEU in these
experiments. The thesis (`thesis/latex_full_report`) discusses these results.

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

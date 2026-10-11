# A Novel Transceiver Design for Semantic Communication

**Final Year Project — Glasgow College, UESTC (2025-26)**

An end-to-end deep semantic communication system for text, built on DeepSC
(Xie et al., 2021). A Transformer semantic encoder maps a sentence to
features, a dense channel encoder turns them into a power-normalised signal,
the signal passes through a differentiable AWGN or Rayleigh-fading channel,
and the receiver reconstructs the sentence. The improved transceiver beats
DeepSC on both channels under the same bandwidth (32 × 16 real symbols per
sentence) and transmit energy.

```
sentence → embedding → semantic encoder → channel encoder → AWGN / Rayleigh channel
         → channel decoder → semantic decoder (greedy / beam) → [BERT fusion] → sentence
```

## Results

BLEU-4 on 10,000 Europarl test sentences per SNR point (95 % bootstrap CI
≈ ±0.8 points at ≤ 0 dB). Full analysis, ablations and caveats:
[`experiments/v2/RESULTS.md`](experiments/v2/RESULTS.md).

| Model | Channel | −10 dB | −5 dB | 0 dB | 5 dB | 10 dB |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline DeepSC | AWGN | 0.429 | 0.736 | 0.838 | 0.963 | 0.991 |
| Improved | AWGN | 0.443 | 0.749 | 0.848 | 0.975 | 0.996 |
| Improved + beam 5 | AWGN | **0.461** | **0.758** | **0.859** | **0.978** | **0.996** |
| Baseline DeepSC | Rayleigh | 0.345 | 0.572 | 0.746 | 0.870 | 0.941 |
| Improved | Rayleigh | 0.349 | 0.569 | 0.750 | 0.878 | 0.951 |
| Improved + beam 5 | Rayleigh | **0.353** | **0.579** | **0.759** | **0.885** | **0.954** |

## What the improved transceiver adds

| Component | Idea | Switch | Effect (ablation) |
| --- | --- | --- | --- |
| SNR-adaptive attention | SNR embedding added to the attention queries of the encoder and of the decoder cross-attention (after SNR-EQ-JSCC) | — | Shared by all Improved variants |
| CSI-aware receiver | Under fading the receiver conditions on the effective SNR·\|h\|² it knows from perfect CSI | `--no_csi_snr` | Helps at every Rayleigh SNR |
| SNR-conditioned channel coder | FiLM modulation of the channel encoder / decoder | `--no_film` | Helps at −5…10 dB |
| Length-aware power normalisation | Padding transmits no energy | `--no_masked_power` | No measurable gain |
| Feature-denoising loss | Per-token cosine loss to the clean encoder features | `--beta 0` | No measurable gain |
| Decoder–BERT fusion | A fine-tuned BERT re-ranks the decoder's top-5 words at low-confidence positions | `--use_llm` (inference) | Roughly neutral |
| Batched beam search | Available to both models | `--beam_width 5` | +0.3…2 BLEU-4 points |

## Project structure

```
.
├── config.py               # hyperparameters and paths
├── preprocess.py           # Europarl v7 → vocabulary, 90/10 split → data/*.pkl
├── dataset.py              # data loading; 5 % of train held out for validation
├── utils.py                # seeding, RNG isolation, id → text
├── models/
│   ├── modules.py          # attention, encoder/decoder layers, FiLM, channel coder
│   ├── channel.py          # AWGN and Rayleigh channels (+ effective SNR)
│   ├── transceiver.py      # DeepSC (BaselineDeepSC / ImprovedDeepSC), greedy and beam decoding
│   ├── losses.py           # cross-entropy + feature-denoising loss
│   └── llm_refiner.py      # decoder–BERT fusion refiner
├── train.py                # train one model (warm-up + cosine LR, AMP, auto-resume)
├── evaluate.py             # metrics per SNR (WordAcc, BLEU + CI, BERTScore, SentSim)
├── finetune_bert.py        # domain-adapt BERT for the refiner
├── ablation.py             # summary table and figures from the eval files
├── run_all.py              # full pipeline for the 8-model experiment (resumable)
├── notebooks/train_colab.ipynb   # self-contained Colab notebook
├── scripts/
│   ├── make_colab_notebook.py    # regenerates the notebook from the sources
│   └── run_cloud.sh              # one-command run on any cloud GPU
├── tests/                  # fast model sanity checks (pytest)
├── experiments/            # archived results: v1/ and v2/ (runs/, figures/, summary.md)
└── thesis/                 # LaTeX report and drafts
```

Data (`data/`), model weights and new runs (`checkpoints/`, `results/`) are
not tracked.

## Quick start

```bash
pip install -r requirements.txt
```

**Data.** Download Europarl v7 and preprocess it:

```bash
wget http://www.statmt.org/europarl/v7/europarl.tgz && tar xzf europarl.tgz
python preprocess.py --raw_dir europarl/txt/en
```

**Full experiment.** Trains the 8 models, evaluates them, fine-tunes BERT,
tunes the fusion refiner and writes `results/summary.md` and
`results/figures/`. Re-running continues where it stopped.

```bash
python run_all.py --smoke   # quick end-to-end check (writes to smoke_checkpoints/)
python run_all.py --amp     # full run on a CUDA GPU
```

**Single model.**

```bash
python train.py --model improved --channel rayleigh --amp
python evaluate.py --checkpoint checkpoints/improved_rayleigh/best.pt --beam_width 5 --eval_tag beam5
```

**Cloud GPU.** On Colab, open
[`notebooks/train_colab.ipynb`](https://colab.research.google.com/github/weirdquantum/semantic-comm-transceiver/blob/main/notebooks/train_colab.ipynb):
it embeds every source file, reads the data parts from
`MyDrive/semantic-comm-transceiver/data/` and saves all results to Drive.
On other machines run `bash scripts/run_cloud.sh` (set `SC_CKPT_DIR` to
persistent storage). After editing any source file, regenerate the notebook
with `python scripts/make_colab_notebook.py`.

**Tests, figures, thesis.**

```bash
pip install pytest && pytest
python ablation.py --results_dir experiments/v2/runs --out_dir experiments/v2
python thesis/latex_full_report/scripts/plot_thesis_figures.py
```

## Evaluation metrics

| Metric | Level | Description |
| --- | --- | --- |
| Word Accuracy | Token | Position-wise exact match |
| BLEU-1 … BLEU-4 | Token | n-gram precision (BLEU-4 with bootstrap CI) |
| BERTScore F1 | Semantic | Contextual embedding similarity (roberta-large) |
| Sentence Similarity | Semantic | Cosine similarity of DistilBERT [CLS] vectors |

## Key references

1. Xie et al., "Deep Learning Enabled Semantic Communication Systems," IEEE TSP, 2021.
2. SNR-EQ-JSCC, arXiv:2501.04732, 2025.
3. LLM-SC, Applied Sciences 15(13):7227, 2025.
4. KG-LLM, arXiv:2504.07431, 2025.
5. Yi et al., "Semantic Communication with a Shared Knowledge Base," IEEE TWC, 2023.

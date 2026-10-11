# Experiments

Archived results of each experiment round. Each round has the same layout:
`runs/<model>/` holds the raw outputs of `evaluate.py` / `train.py`
(model weights are not tracked), `summary.md` and `figures/` are produced by
`ablation.py`, and `RESULTS.md` (if present) discusses the round.

| Round | Contents |
|---|---|
| [`v2/`](v2/RESULTS.md) | Current code. 8 models × 80 epochs, 10k test sentences per SNR, ablations of every v2 component, beam search and decoder–BERT fusion. |
| [`v1/`](v1/runs) | First round with the original code (2k test sentences, test-set model selection, Rayleigh channel with E[\|h\|²] = 2). Kept for reference and for the thesis figures (`thesis/latex_full_report/scripts/plot_thesis_figures.py`). |

Regenerate a round's summary and figures:

```bash
python ablation.py --results_dir experiments/v2/runs --out_dir experiments/v2
```

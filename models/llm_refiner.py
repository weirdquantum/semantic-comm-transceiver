"""
LLM-Aided Semantic Decoder  (decoder / BERT-MLM fusion refinement)

At inference time the transceiver produces a reconstruction together with,
for every output position, its confidence and its top-k candidate tokens.
The refiner then:

  1. Selects positions whose decoder confidence is below ``tau`` (at most
     ``max_corrections`` per sentence, least confident first, never <UNK>).
  2. Masks those positions and runs a frozen BERT masked-LM once.
  3. Re-scores only the decoder's own top-k candidates at each masked
     position with   log p_dec(w) + lambda * log p_BERT(w | context)
     and keeps the best one.

Restricting the choice to the decoder's candidates keeps the output inside
the transceiver vocabulary and anchored to what was actually received;
BERT only arbitrates between channel-plausible words.  lambda = 0 leaves
the sentence unchanged, so tuning (lambda, tau) on the validation split
can never do worse than no refinement there.

The refiner is applied **only at inference** — it does not participate
in gradient-based training of the transceiver.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

# One sentence to refine: (words, per-word confidence, per-word candidates)
# where candidates is a list of (word, decoder log-prob) pairs.
RefineItem = Tuple[List[str], List[float], List[List[Tuple[str, float]]]]

SPECIAL_WORDS = {"<PAD>", "<START>", "<END>", "<UNK>"}


class LLMRefiner:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        lam: float = 0.5,
        tau: float = 0.7,
        max_corrections: int = 3,
        batch_size: int = 64,
        device: torch.device | str = "cpu",
    ):
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.device = torch.device(device)
        self.lam = lam
        self.tau = tau
        self.max_corrections = max_corrections
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id
        self._piece_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _first_piece(self, word: str) -> int:
        """BERT id of the first word-piece of ``word`` (cached)."""
        if word not in self._piece_cache:
            pieces = self.tokenizer.tokenize(word)
            self._piece_cache[word] = (
                self.tokenizer.convert_tokens_to_ids(pieces[0])
                if pieces else self.tokenizer.unk_token_id)
        return self._piece_cache[word]

    def _positions(self, item: RefineItem, tau: float) -> List[int]:
        words, confs, _ = item
        low = [i for i in range(min(len(words), len(confs)))
               if confs[i] < tau and words[i] not in SPECIAL_WORDS]
        low.sort(key=lambda i: confs[i])
        return sorted(low[: self.max_corrections])

    @torch.no_grad()
    def _bert_logprobs(self, items: Sequence[RefineItem],
                       positions: List[List[int]]):
        """For each item, a {word_pos: log-prob vector over BERT vocab}."""
        out: List[Dict[int, torch.Tensor]] = [{} for _ in items]
        todo = [i for i, pos in enumerate(positions)
                if pos and len(items[i][0]) >= 3]
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s:s + self.batch_size]
            texts = []
            for i in chunk:
                masked = list(items[i][0])
                for p in positions[i]:
                    masked[p] = self.mask_token
                texts.append(" ".join(masked))
            enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128).to(self.device)
            logp = torch.log_softmax(self.model(**enc).logits.float(), dim=-1)
            for row, i in enumerate(chunk):
                mask_idx = (enc["input_ids"][row] == self.mask_id).nonzero(
                    as_tuple=True)[0].tolist()
                # each [MASK] is a single word-piece, in word order
                for p, b in zip(positions[i], mask_idx):
                    out[i][p] = logp[row, b].cpu()
        return out

    # ------------------------------------------------------------------
    def refine_grid(self, items: Sequence[RefineItem], lams: Sequence[float],
                    tau: float) -> Dict[float, List[str]]:
        """Refine every item for several lambdas sharing one BERT pass."""
        positions = [self._positions(it, tau) for it in items]
        bert = self._bert_logprobs(items, positions)
        results = {}
        for lam in lams:
            texts = []
            for it, pos_lp in zip(items, bert):
                words = list(it[0])
                if lam > 0:
                    for p, lp_bert in pos_lp.items():
                        best_w, best_s = words[p], float("-inf")
                        for w, lp_dec in it[2][p]:
                            if w in SPECIAL_WORDS:
                                continue
                            score = lp_dec + lam * lp_bert[self._first_piece(w)].item()
                            if score > best_s:
                                best_w, best_s = w, score
                        words[p] = best_w
                texts.append(" ".join(words))
            results[lam] = texts
        return results

    def refine(self, items: Sequence[RefineItem]) -> List[str]:
        return self.refine_grid(items, [self.lam], self.tau)[self.lam]

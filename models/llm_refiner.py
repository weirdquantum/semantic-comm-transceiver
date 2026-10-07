"""
LLM-Aided Semantic Decoder  (BERT MLM refinement module)

At inference time the Transformer decoder produces an initial text
reconstruction together with per-position softmax confidence scores.
The LLM refiner then:

  1. Identifies low-confidence positions using the decoder's own
     softmax probabilities (confidence < threshold).
  2. Masks those positions and feeds the sentence to a frozen
     BERT Masked-LM to obtain bidirectional predictions.
  3. Accepts a replacement only when it differs from the original
     word AND when it is a valid alphabetic token.

The refiner is applied **only at inference** — it does not participate
in gradient-based training of the transceiver, keeping training efficient.

Key difference from the previous DistilGPT-2 approach:
  - Bidirectional context (BERT) instead of left-only (GPT-2)
  - Triggered by the transceiver's OWN uncertainty, not external LM ppl
  - Eliminates domain-mismatch: no threshold depends on the LM's
    familiarity with the target corpus
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


class LLMRefiner:
    """Post-hoc semantic refinement using a frozen BERT Masked-LM.

    Strategy: **decoder-confidence-driven mask-then-predict**.

    For each sentence the transceiver decoded:
      1. Look at the per-position softmax confidence from the decoder.
         Positions with confidence < *confidence_threshold* are candidates.
      2. Replace those words with [MASK] and run the frozen BERT MLM.
      3. Accept the BERT prediction only if it differs from the original
         and is a valid alphabetic word.
      4. At most *max_corrections* replacements per sentence.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        confidence_threshold: float = 0.6,
        max_corrections: int = 3,
        device: torch.device | str = "cpu",
    ):
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.max_corrections = max_corrections

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine(
        self,
        sentences: List[str],
        decoder_confidences: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Refine a batch of draft sentences.

        Args:
            sentences: list of decoded sentences (space-separated words).
            decoder_confidences: per-sentence list of per-word confidence
                scores (max softmax probability). ``None`` → skip all
                refinement (pass-through).

        Returns:
            List of (possibly refined) sentences.
        """
        if decoder_confidences is None:
            return sentences
        return [
            self._refine_single(s, c)
            for s, c in zip(sentences, decoder_confidences)
        ]

    # ------------------------------------------------------------------
    def _refine_single(
        self, sentence: str, word_confs: List[float]
    ) -> str:
        words = sentence.split()
        if len(words) < 3:
            return sentence

        n_words = min(len(words), len(word_confs))

        low_conf = [
            (i, word_confs[i])
            for i in range(n_words)
            if word_confs[i] < self.confidence_threshold
        ]
        if not low_conf:
            return sentence

        low_conf.sort(key=lambda x: x[1])
        candidates = low_conf[: self.max_corrections]

        mask_positions = [idx for idx, _ in candidates]
        masked_words = words.copy()
        for pos in mask_positions:
            masked_words[pos] = self.mask_token
        masked_sentence = " ".join(masked_words)

        enc = self.tokenizer(
            masked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        logits = self.model(**enc).logits  # (1, T_bpe, V)
        input_ids = enc["input_ids"][0]

        mask_bpe_positions = (input_ids == self.mask_id).nonzero(as_tuple=True)[0]

        word_to_bpe_masks = self._map_word_masks_to_bpe(
            words, mask_positions, mask_bpe_positions
        )

        result_words = words.copy()
        for word_pos, bpe_indices in word_to_bpe_masks.items():
            predicted_pieces = []
            for bpe_idx in bpe_indices:
                top_id = logits[0, bpe_idx].argmax().item()
                predicted_pieces.append(top_id)
            replacement = self.tokenizer.decode(predicted_pieces).strip()

            if not replacement:
                continue
            replacement_clean = replacement.split()[0] if replacement.split() else ""
            if not replacement_clean:
                continue
            if not replacement_clean.isalpha():
                continue
            if replacement_clean.lower() == result_words[word_pos].lower():
                continue
            result_words[word_pos] = replacement_clean

        return " ".join(result_words)

    # ------------------------------------------------------------------
    def _map_word_masks_to_bpe(
        self,
        words: List[str],
        mask_word_positions: List[int],
        mask_bpe_positions: torch.Tensor,
    ) -> dict:
        """Map each masked word index to its corresponding BPE [MASK] indices.

        Returns dict  {word_pos: [bpe_idx, ...]}
        """
        mapping: dict = {}
        bpe_iter = iter(mask_bpe_positions.tolist())

        masked_words = words.copy()
        for pos in mask_word_positions:
            masked_words[pos] = self.mask_token
        masked_sentence = " ".join(masked_words)

        tokens = self.tokenizer.tokenize(masked_sentence)

        mask_count = 0
        word_idx = 0
        mask_word_set = set(mask_word_positions)
        mask_word_sorted = sorted(mask_word_positions)
        current_mask_word_ptr = 0

        bpe_positions_list = mask_bpe_positions.tolist()
        bpe_ptr = 0

        for word_pos in mask_word_sorted:
            orig_word = words[word_pos]
            n_bpe_for_word = len(
                self.tokenizer.tokenize(self.mask_token)
            )
            if bpe_ptr < len(bpe_positions_list):
                mapping[word_pos] = [bpe_positions_list[bpe_ptr]]
                bpe_ptr += 1
            else:
                mapping[word_pos] = []

        return mapping

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_sentences(self, sentences: List[str]) -> List[float]:
        """Return per-sentence pseudo-log-likelihood under the MLM."""
        scores = []
        for sent in sentences:
            enc = self.tokenizer(
                sent, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            input_ids = enc["input_ids"]
            if input_ids.size(1) < 2:
                scores.append(0.0)
                continue
            logits = self.model(**enc).logits
            log_probs = F.log_softmax(logits, dim=-1)
            token_lp = log_probs[0, range(input_ids.size(1)), input_ids[0]]
            scores.append(token_lp.mean().item())
        return scores

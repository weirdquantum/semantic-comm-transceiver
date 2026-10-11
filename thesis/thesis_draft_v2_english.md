# Final Year Project Report

Bachelor of Engineering  
UESTC4006P INDIVIDUAL PROJECT 4

**Project Title:** A Novel Transceiver Design for Semantic Communication  
**Student:** [Your Name]  
**GUID:** [Your GUID]  
**UESTC ID:** [Your ID]  
**Supervisor:** [Keep blank for Moodle submission; fill before hardcopy submission]  
**Academic Year:** 2025-26

---

# Coursework Declaration and Feedback Form

Student Name: [Your Name]  
Student GUID: [Your GUID]  
Course Code: UESTC4006P  
Course Name: INDIVIDUAL PROJECT 4  
Name of Supervisor: [Keep blank for Moodle submission; fill before hardcopy submission]  
Title of Project: A Novel Transceiver Design for Semantic Communication

## Declaration of Originality and Submission Information

I affirm that this submission is all my own work in accordance with the University regulations and the School of Engineering requirements.

Signed (Student): ____________________  
Date of Submission: ____________________

## Feedback from Lecturer to Student (to be completed by Lecturer)

Grade Awarded: ____________________  
Feedback: ____________________  
Lecturer/Demonstrator: ____________________  
Date returned to ESAS: ____________________

---

# Abstract

Semantic communication is a promising paradigm for beyond-Shannon communications, where the objective shifts from bit-level fidelity to meaning-level fidelity. Although deep-learning-based semantic communication systems have shown strong potential, they still face key limitations under low signal-to-noise ratio (SNR) and fading channels, especially in robust semantic recovery and stable optimization. This project investigates how to improve a Transformer-based semantic transceiver through channel-aware modeling, adaptive training objectives, and inference-stage refinement.

First, a baseline DeepSC-style text transceiver was implemented and reproduced under both AWGN and Rayleigh channels. Then three innovation paths were studied: (i) SNR-adaptive attention, (ii) adaptive semantic-loss weighting, and (iii) language-model-based semantic refinement. During iterative experiments, initial assumptions were critically re-evaluated: the semantic-loss term did not consistently yield positive gains, and LLM post-refinement could degrade performance through over-correction. Based on this diagnosis, the design was updated by extending SNR-awareness to decoder cross-attention, introducing SNR-conditioned semantic-loss scheduling, and replacing early GPT-style correction with confidence-driven BERT masked refinement. In addition, beam search decoding was introduced as an inference-time enhancement.

Experimental results across SNR points from -10 dB to 20 dB show that channel-aware transceiver improvements produce clear gains in low-SNR AWGN conditions. At -10 dB, BLEU-4 improves from 0.4189 (baseline) to 0.4329 (improved), while Word Accuracy improves from 0.4425 to 0.4773. Ablation experiments indicate weak negative transfer from the semantic-loss component in the current formulation. Under Rayleigh fading, improvements are mixed and generally smaller, showing that AWGN-effective mechanisms do not always transfer to stronger channel dynamics. LLM-based refinement reduces degradation compared with earlier strategies but still fails to deliver robust net positive gains. In contrast, beam search decoding yields stable and substantial low-SNR improvements without retraining the backbone model, reaching BLEU-4 = 0.4552 at -10 dB.

This project delivers a complete research loop from hypothesis to failure diagnosis and iterative redesign. The final recommendation is to prioritize robust channel-aware transceiver design and decoding strategy optimization (especially beam search), while treating external LLM post-refinement as conditional and high-risk in high-quality decoding regimes.

**Keywords:** Semantic Communication; DeepSC; SNR-Adaptive Attention; Beam Search; Rayleigh Fading; Transformer

---

# Acknowledgements

I would like to express my sincere gratitude to my supervisor for continuous guidance throughout this project, including problem formulation, experimental design, and technical writing. I am also grateful to the research group members for valuable discussions on model debugging, evaluation strategy, and result interpretation. Special thanks go to the open-source community and tool developers whose frameworks and libraries made this implementation possible. Finally, I deeply appreciate my family and friends for their support and encouragement during the project period.

---

# Contents

Abstract  
Acknowledgements  
1 Introduction  
1.1 Project Background and Motivation  
1.2 Research Questions and Objectives  
1.3 Contribution Summary  
2 Body of the Report  
2.1 Literature and Technical Background  
2.2 Baseline System Design  
2.3 Initial Innovation Hypotheses  
2.4 Diagnosed Limitations and Root Causes  
2.5 Iterative Redesign and Implementation  
2.6 Experimental Protocol and Evaluation Metrics  
2.7 Result Presentation  
3 Analysis and Discussion  
3.1 Main Findings under AWGN  
3.2 Ablation Analysis: Semantic Loss  
3.3 Cross-Channel Analysis: Rayleigh Robustness  
3.4 LLM Refinement Analysis  
3.5 Decoding Strategy Analysis  
3.6 Integrated Insight and Design Guidelines  
4 Conclusions and Further Work  
4.1 Conclusions  
4.2 Suggestions for Further Work  
References  
A Appendices

---

# 1 Introduction

## 1.1 Project Background and Motivation

Conventional communication systems are built around Shannon’s framework, which optimizes reliable bit transport through noisy channels. This framework has been successful for decades; however, many modern applications do not require exact bit-level recovery as long as the delivered meaning remains useful for downstream tasks. Semantic communication addresses this shift by transmitting task-relevant meaning rather than all syntactic details.

Recent deep-learning-based semantic communication systems, especially Transformer-based designs, have demonstrated promising performance for text transmission. A representative line is DeepSC, which learns semantic encoding and decoding in an end-to-end fashion. Despite these advances, practical limitations remain in low-SNR and fading environments: semantic representations become less stable, optimization objectives can conflict, and post-processing by external language models may introduce additional errors instead of corrections.

This project is motivated by an engineering question with direct practical value: can a semantic transceiver be improved through targeted, testable modifications that are effective under realistic channel conditions? More specifically, the project aims to produce a reproducible design path that does not stop at proposing ideas, but also includes rigorous failure analysis and iterative refinement.

## 1.2 Research Questions and Objectives

This report addresses the following research questions:

1. Can SNR-aware attention mechanisms improve semantic reconstruction quality, especially at low SNR?
2. Does adding semantic regularization in training consistently improve performance, or can it cause optimization conflict?
3. Can LLM-based post-refinement improve decoded text quality in this communication setting?
4. If LLM post-refinement is unstable, is there a lower-risk inference strategy that delivers robust gains?

To answer these questions, the project objectives were:

- Build and verify a baseline semantic transceiver for text transmission.
- Implement three innovation modules and evaluate each one with controlled comparisons.
- Run AWGN and Rayleigh channel experiments under unified settings.
- Perform ablation and combination experiments to identify effective and ineffective components.
- Derive practical design guidance based on measured evidence.

## 1.3 Contribution Summary

The main contributions are:

- A complete reproducible semantic communication pipeline (data preprocessing, training, evaluation, and comparison).
- A channel-aware redesign that extends SNR-adaptive attention to decoder cross-attention.
- An adaptive semantic-loss weighting mechanism conditioned on SNR.
- A confidence-driven BERT masked refinement pipeline for post-decoding correction.
- A decoding-strategy upgrade via beam search showing consistent low-SNR gains.
- A full hypothesis-to-diagnosis research narrative including negative findings and failure mechanisms.

---

# 2 Body of the Report

## 2.1 Literature and Technical Background

Shannon’s communication theory defines reliable transmission limits through channel capacity, but does not explicitly model semantic usefulness. Semantic communication extends this view by emphasizing meaning-preserving transfer. DeepSC demonstrated that end-to-end neural transceivers can preserve sentence-level semantics under noisy channels and may outperform traditional bit-centric pipelines for specific language tasks.

Transformer architectures are central to modern text representation and generation due to attention-based global context modeling. In communication scenarios, this capability allows semantic compression and reconstruction through learned latent representations. However, semantic communication models remain sensitive to channel variations; therefore, integrating channel-quality awareness into model internals is a natural direction.

For evaluation, token-level and semantic-level metrics are both required. BLEU measures n-gram overlap, while BERTScore captures contextual semantic similarity. Word Accuracy provides fine-grained correctness, and sentence-level cosine similarity gives a global semantic estimate.

## 2.2 Baseline System Design

The baseline transceiver follows an Encoder-Channel-Decoder paradigm:

- Source text is tokenized, indexed, and embedded.
- A multi-layer Transformer encoder produces semantic latent features.
- A channel encoder maps features into channel symbols.
- Channel effects are simulated by AWGN or Rayleigh modules.
- A channel decoder reconstructs latent features.
- A Transformer decoder autoregressively reconstructs target text.

The baseline is trained with cross-entropy at token level. This setting serves as the control group for all innovation experiments.

## 2.3 Initial Innovation Hypotheses

Three initial hypotheses were proposed:

### (A) SNR-Adaptive Attention

Inject SNR embeddings into attention computation so that representation extraction can adapt to channel quality.

### (B) Composite Semantic Loss

Combine cross-entropy and semantic consistency loss to improve meaning preservation:

L_total = alpha * L_CE + beta * L_sem.

### (C) LLM Post-Refinement

Apply language-model-based post-correction to decoded sentences to repair semantically plausible errors.

## 2.4 Diagnosed Limitations and Root Causes

After running the initial experiment cycle, the following limitations were observed:

1. **Partial effectiveness of (A):** gains were visible in AWGN low-SNR settings but weak in fading scenarios.  
2. **Instability of (B):** semantic loss did not consistently improve final metrics and showed signs of negative transfer.  
3. **Risk of (C):** LLM post-refinement often over-corrected outputs, reducing BLEU and related metrics.

Root-cause analysis suggested:

- Channel-awareness only at one side (encoder) was insufficient for robust reconstruction under uncertainty.
- Fixed semantic-loss weight did not match the dynamic reliability of supervision across SNR values.
- External LLM objectives were misaligned with communication reconstruction objectives, especially when decoded text was already high quality.

## 2.5 Iterative Redesign and Implementation

Based on diagnosis, the system was redesigned:

### (A') Decoder-side SNR-aware Cross-Attention

SNR embedding was introduced into decoder cross-attention, enabling channel-conditioned memory retrieval at reconstruction time.

### (B') SNR-Adaptive Semantic-Loss Weight

The semantic-loss coefficient was changed from fixed beta to:

beta_eff = beta * sigmoid(mean(SNR) / 5.0),

reducing semantic-loss impact in low-SNR regimes.

### (C') Confidence-Driven BERT Masked Refinement

Refinement was changed to confidence-triggered masking:

- Identify low-confidence decoded positions.
- Replace them with [MASK].
- Use BERT MLM for constrained replacement.

The intention was to avoid global rewriting and reduce over-correction.

### (D) Beam Search Decoding

A beam search decoder was added for inference-time optimization without retraining. This was designed as a practical fallback when post-refinement was unstable.

## 2.6 Experimental Protocol and Evaluation Metrics

Training and evaluation were conducted under consistent settings:

- Channel types: AWGN and Rayleigh.
- Evaluation SNR points: -10, -5, 0, 5, 10, 15, 20 dB.
- Main metrics: WordAcc, BLEU-1/2/3/4, BERTScore (P/R/F1), Sentence Similarity.

Model groups included:

1. Baseline-AWGN  
2. Improved-AWGN  
3. Improved-AWGN-NoSem (ablation)  
4. Baseline-Rayleigh  
5. Improved-Rayleigh  
6. Improved-AWGN + Beam search  
7. Improved-AWGN + FT-BERT refinement  
8. Improved-AWGN + Beam + FT-BERT

## 2.7 Result Presentation

### 2.7.1 AWGN Main Comparison

At -10 dB:

- Baseline BLEU-4 = 0.4189
- Improved BLEU-4 = 0.4329
- Absolute gain = +0.0140

At -10 dB WordAcc:

- Baseline = 0.4425
- Improved = 0.4773
- Absolute gain = +0.0348

These gains indicate that channel-aware transceiver redesign improves robustness in the most challenging noise conditions.

### 2.7.2 Semantic-Loss Ablation

Comparing Improved vs Improved-NoSem:

- At -10 dB BLEU-4: 0.4329 vs 0.4488
- At -5 dB BLEU-4: 0.7386 vs 0.7448
- At 5 dB BLEU-4: 0.9685 vs 0.9709

NoSem performs slightly better across several points, indicating weak negative transfer from the semantic-loss term in its current formulation.

### 2.7.3 Rayleigh Comparison

Rayleigh BLEU-4 (Baseline vs Improved):

- -10 dB: 0.4697 vs 0.4640
- -5 dB: 0.6662 vs 0.6603
- 0 dB: 0.8170 vs 0.8055

Improvements are limited and mixed, revealing that AWGN-positive mechanisms do not automatically generalize to stronger fading dynamics.

### 2.7.4 LLM and Decoding Strategy Results

Using FT-BERT refinement on Improved-AWGN generally reduced metrics:

- -10 dB BLEU-4: 0.4329 -> 0.4191
- 5 dB BLEU-4: 0.9685 -> 0.9650

Beam search, however, produced stable gains:

- -10 dB BLEU-4: 0.4329 -> 0.4552
- -5 dB BLEU-4: 0.7386 -> 0.7496
- 0 dB BLEU-4: 0.8375 -> 0.8494

Beam + FT-BERT combination underperformed beam-only, confirming that aggressive post-refinement can damage already improved beam outputs.

---

# 3 Analysis and Discussion

## 3.1 Main Findings under AWGN

The improved transceiver consistently outperforms the baseline in low-SNR AWGN settings. This supports the core claim that explicit channel-quality conditioning inside attention improves semantic feature robustness. As SNR increases, both models approach saturation, reducing relative gains. This pattern is expected in communication systems where noise becomes less dominant.

## 3.2 Ablation Analysis: Semantic Loss

The semantic-loss ablation reveals a critical insight: adding a semantic term is not guaranteed to improve communication reconstruction. Even with adaptive weighting, the semantic objective may conflict with token-level optimization under noise. This indicates that future semantic-loss design should consider gradient compatibility, not only conceptual alignment.

## 3.3 Cross-Channel Analysis: Rayleigh Robustness

Rayleigh results show that stochastic fading introduces a harder distribution shift than AWGN. The current SNR-conditioned mechanism does not fully capture instantaneous fading behavior. This suggests that richer channel-state features or temporal fading-aware modules may be necessary for robust gains in realistic wireless environments.

## 3.4 LLM Refinement Analysis

The LLM refinement module was designed to correct uncertain words, but experiments indicate persistent over-correction. A likely reason is objective mismatch: communication decoding aims to recover the transmitted sentence, while generic language models favor linguistic plausibility over transmission fidelity. When the decoded sentence is already close to correct, the expected benefit from external correction is small and the risk of harmful edits becomes dominant.

## 3.5 Decoding Strategy Analysis

Beam search provides the strongest practical improvement among all post-training modifications. It improves low-SNR performance without retraining the transceiver, making it computationally and operationally attractive. The evidence suggests that better sequence search can outperform external rewriting in this task.

## 3.6 Integrated Insight and Design Guidelines

From a system design perspective, three guidelines emerge:

1. Prioritize in-model channel awareness before external post-processing.
2. Treat semantic regularization as conditional, not universally beneficial.
3. Prefer conservative, model-consistent inference enhancements (beam search) over unconstrained external correction.

These guidelines are actionable for future semantic communication development and also justify the final system recommendation in this report.

---

# 4 Conclusions and Further Work

## 4.1 Conclusions

This project delivered a complete engineering and research cycle for semantic communication transceiver optimization. The following conclusions are supported by experimental evidence:

1. SNR-adaptive transceiver redesign improves low-SNR AWGN performance and increases semantic reconstruction robustness.
2. The current semantic-loss formulation shows weak negative transfer despite adaptive weighting, and should not be treated as universally beneficial.
3. Confidence-driven BERT refinement reduces some severe failure modes compared with earlier post-processing ideas, but still does not achieve reliable net gains.
4. Beam search decoding is the most effective and practical enhancement in this project, providing consistent low-SNR improvements without retraining.
5. Cross-channel generalization remains challenging: methods successful in AWGN may not transfer directly to Rayleigh fading.

Overall, the most reliable configuration in this project is the improved transceiver with beam search decoding.

## 4.2 Suggestions for Further Work

Future work can focus on:

1. **Rayleigh-specific modeling:** integrate richer channel-state representations beyond scalar SNR.
2. **Gradient-conflict-aware loss balancing:** apply modern multi-objective optimization techniques to reduce CE-semantic loss interference.
3. **Constrained refinement:** replace free-form LLM correction with minimal-edit or n-best reranking strategies.
4. **Joint training of confidence and refinement:** calibrate uncertainty estimates so that low-confidence triggering aligns better with true error positions.
5. **Multi-domain validation:** evaluate on additional corpora and tasks to test generalization beyond Europarl.

---

# References

[1] C. E. Shannon, "A Mathematical Theory of Communication," Bell System Technical Journal, vol. 27, no. 3, pp. 379-423, 1948.  
[2] R. Carnap and Y. Bar-Hillel, "An Outline of a Theory of Semantic Information," MIT Technical Report, 1954.  
[3] J. Bao, P. Basu, M. Dean, et al., "Towards a Theory of Semantic Communication," Proc. IEEE Network Science Workshop, 2011.  
[4] Z. Lu, R. Li, K. Lu, et al., "Semantics-Empowered Communication: A Tutorial-Cum-Survey," IEEE Communications Surveys & Tutorials, vol. 26, no. 1, pp. 534-572, 2024.  
[5] H. Xie, Z. Qin, G. Y. Li, and B. F. Juang, "Deep Learning Enabled Semantic Communication Systems," IEEE Transactions on Signal Processing, vol. 69, pp. 2663-2675, 2021.  
[6] A. Goldsmith, Wireless Communications, Cambridge University Press, 2005.  
[7] T. S. Rappaport, Wireless Communications: Principles and Practice, 2nd ed., Prentice Hall, 2002.  
[8] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention Is All You Need," Advances in Neural Information Processing Systems, vol. 30, pp. 5998-6008, 2017.  
[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," Proc. NAACL-HLT, pp. 4171-4186, 2019.  
[10] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A Method for Automatic Evaluation of Machine Translation," Proc. ACL, pp. 311-318, 2002.  
[11] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating Text Generation with BERT," Proc. ICLR, 2020.  
[12] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter," arXiv:1910.01108, 2019.  
[13] A. Radford, J. Wu, R. Child, et al., "Language Models are Unsupervised Multitask Learners," OpenAI Technical Report, 2019.  
[14] A. Kendall, Y. Gal, and R. Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics," Proc. CVPR, pp. 7482-7491, 2018.  
[15] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks," Proc. ICML, pp. 794-803, 2018.

---

# A Appendices

## Appendix A.1 Reproducible Experimental Commands

Environment setup:

- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`

Training:

- `python train.py --model baseline --channel awgn --epochs 80`
- `python train.py --model improved --channel awgn --epochs 80`
- `python train.py --model improved --channel awgn --epochs 80 --beta 0.0 --tag no_sem`
- `python train.py --model baseline --channel rayleigh --epochs 80`
- `python train.py --model improved --channel rayleigh --epochs 80`

Evaluation:

- `python evaluate.py --checkpoint checkpoints/baseline_awgn/best.pt --model baseline --channel awgn`
- `python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn`
- `python evaluate.py --checkpoint checkpoints/improved_awgn_no_sem/best.pt --model improved --channel awgn`
- `python evaluate.py --checkpoint checkpoints/baseline_rayleigh/best.pt --model baseline --channel rayleigh`
- `python evaluate.py --checkpoint checkpoints/improved_rayleigh/best.pt --model improved --channel rayleigh`
- `python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --beam_width 5 --eval_tag beam5`
- `python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --use_llm --llm_model checkpoints/bert_europarl --eval_tag ftbert`
- `python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --beam_width 5 --use_llm --llm_model checkpoints/bert_europarl --eval_tag beam5_ftbert`

## Appendix A.2 Data Sources Used in Quantitative Comparison

- `checkpoints/baseline_awgn/eval_awgn.json`
- `checkpoints/improved_awgn/eval_awgn.json`
- `checkpoints/improved_awgn_no_sem/eval_awgn.json`
- `checkpoints/baseline_rayleigh/eval_rayleigh.json`
- `checkpoints/improved_rayleigh/eval_rayleigh.json`
- `checkpoints/improved_awgn/eval_awgn_beam5.json`
- `checkpoints/improved_awgn/eval_awgn_ftbert.json`
- `checkpoints/improved_awgn/eval_awgn_beam5_ftbert.json`


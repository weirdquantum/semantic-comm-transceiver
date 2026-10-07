# Cover Page

**Project Title:** A Novel Transceiver Design for Semantic Communication  
**Candidate:** [Your Name]  
**Programme:** BEng Final Year Project  
**School:** [Your School / Department]  
**Supervisor:** [Supervisor Name]  
**Submission Date:** [Date]

---

# Abstract

语义通信被认为是后香农时代通信系统的重要发展方向，其核心目标是由“可靠传输比特”转向“有效传输语义”。与传统通信中以误码率为中心的优化目标不同，语义通信更关注任务语义的保真度和下游可用性。现有 DeepSC 类方法虽然在文本传输任务中表现出潜力，但在低信噪比（SNR）和复杂衰落信道下仍面临重构鲁棒性不足、损失函数约束不稳定以及外部语言模型后处理可能反向降级等问题。针对上述问题，本文围绕“信道感知建模 + 语义约束优化 + 推理后处理”提出并验证一套可复现实验框架。

本文首先以 Transformer 为主干搭建基线语义通信收发机，并在 AWGN 与 Rayleigh 两类信道下进行统一训练与评估。随后提出三类改进：1）在收发机注意力机制中引入 SNR 自适应建模，并进一步扩展到解码器 cross-attention；2）将复合损失中的语义损失权重由固定值改为随 SNR 动态调节的自适应系数；3）构建置信度驱动的 BERT MLM 语义精炼器，对低置信度词位进行掩码修复。为检验创新点的独立贡献与耦合效应，本文设计了主实验、消融实验、跨信道对比实验及推理策略组合实验（greedy / beam / LLM refinement）。

实验结果表明：在 AWGN 信道下，改进模型相较基线在低 SNR 场景获得稳定增益，-10 dB 下 BLEU-4 由 0.4189 提升至 0.4329，WordAcc 由 0.4425 提升至 0.4773；在中高 SNR 下两者均接近饱和。消融结果显示，语义损失项在当前实现下仍存在弱负迁移现象（去除语义损失后 BLEU-4 在多个 SNR 点略高）。Rayleigh 对比结果显示双端 SNR 感知机制对该复杂信道的提升有限且呈混合效应。针对创新点 3，BERT 精炼器相比早期 GPT 风格后处理显著降低了性能降级幅度，但在整体指标上仍以轻微负增益为主。进一步引入 Beam Search 后，低 SNR 场景显著改善（-10 dB BLEU-4 达 0.4552），证明推理策略优化在本任务中具有更高工程性价比。

本文的主要贡献在于：完成了一个面向文本语义通信的端到端可复现系统；形成了“初始假设—缺陷暴露—针对性改进—再验证”的完整研究闭环；给出创新模块在不同信道和不同 SNR 下的定量有效性边界，为后续语义通信收发机设计提供可复用的实验范式和工程参考。

**Keywords:** Semantic Communication; DeepSC; SNR-Adaptive Attention; Beam Search; Rayleigh Fading

---

# Abstract (English)

Semantic communication is regarded as a key paradigm for post-Shannon communication systems, where the objective shifts from reliable bit delivery to effective semantic delivery. Compared with conventional communication systems optimized mainly by bit error metrics, semantic communication focuses on semantic fidelity and downstream usability. Existing DeepSC-style approaches have shown promising results for text transmission, but still suffer from limited robustness under low signal-to-noise ratio (SNR) and fading channels, unstable behavior of semantic regularization terms, and potential performance degradation caused by external language-model-based post-processing.

This thesis develops a reproducible experimental framework from three perspectives: channel-aware modeling, semantic-loss optimization, and inference-stage refinement. A Transformer-based baseline semantic transceiver is first implemented and evaluated under both AWGN and Rayleigh channels. Then three categories of improvements are introduced: (1) SNR-adaptive attention modeling, further extended to decoder-side cross-attention; (2) adaptive semantic-loss weighting, where a fixed semantic coefficient is replaced by an SNR-dependent schedule; and (3) a confidence-driven BERT masked language model (MLM) refiner, which masks low-confidence token positions and predicts replacements. To assess individual and coupled effects, this work includes main experiments, ablation studies, cross-channel comparisons, and inference strategy combinations (greedy / beam / LLM refinement).

Results show that in AWGN, the improved model consistently outperforms the baseline in low-SNR settings: at -10 dB, BLEU-4 increases from 0.4189 to 0.4329, and Word Accuracy increases from 0.4425 to 0.4773. At medium-to-high SNR, both models approach saturation. Ablation indicates that the semantic loss term still causes weak negative transfer in the current implementation, as removing it yields slightly better BLEU-4 at several SNR points. Under Rayleigh fading, the dual-side SNR-aware design brings limited and mixed gains. For innovation point 3, the BERT refiner significantly mitigates degradation compared with earlier GPT-style post-processing, yet still tends to produce slight negative gains overall. Further integrating Beam Search yields clear improvements in low-SNR conditions (BLEU-4 reaches 0.4552 at -10 dB), suggesting that inference strategy optimization is more cost-effective in this task.

The key contributions are: (i) building an end-to-end reproducible semantic communication system for text transmission; (ii) establishing a complete research loop from hypothesis to diagnosis and iterative improvement; and (iii) providing quantitative effectiveness boundaries of each innovation across channel types and SNR regimes, which offers practical guidance for future semantic transceiver design.

---

# Acknowledgements

本课题在导师的指导下完成。感谢导师在选题设计、实验方案、论文结构与学术规范方面的持续指导。感谢课题组同学在环境配置、实验复现和结果讨论中的支持。感谢实验平台与开源社区提供的工具与数据资源。最后感谢家人与朋友在毕业设计期间给予的理解和鼓励。

---

# Contents

1. Introduction  
2. Literature Survey  
3. Materials and Methods  
4. Results  
5. Discussion  
6. Conclusions  
References  
Appendices

---

# List of Figures

Figure 1-1 Semantic communication versus conventional communication framework  
Figure 3-1 Overall architecture of the proposed semantic transceiver  
Figure 3-2 SNR-adaptive attention mechanism in encoder and decoder  
Figure 4-1 WordAcc vs SNR under AWGN (Baseline vs Improved vs NoSem)  
Figure 4-2 BLEU-4 vs SNR under AWGN (Baseline vs Improved vs NoSem)  
Figure 4-3 BLEU-4 vs SNR under Rayleigh (Baseline vs Improved)  
Figure 4-4 LLM refinement impact under AWGN (before vs after)  
Figure 4-5 Beam search and combined strategies comparison

---

# List of Tables

Table 3-1 Core hyper-parameters and training setup  
Table 4-1 AWGN main comparison metrics  
Table 4-2 Ablation study of semantic loss term  
Table 4-3 Rayleigh channel comparison  
Table 4-4 Beam / FT-BERT / Combined strategy comparison  
Table 6-1 Final conclusion matrix for three innovation points

---

# 1 Introduction

## 1.1 Background and Motivation

经典通信理论以香农范式为核心，其目标是在给定信道容量下尽可能可靠地传输比特序列[1]。在该范式中，系统优化对象通常是符号误码率、比特误码率或分组误码率，而不是信息本身的语义效用。随着移动互联网、物联网、智能体协同和人机交互场景的普及，通信系统面临大量“语义冗余高、任务相关性强”的数据类型。对于这类场景，仅追求比特级无差错并不总是最优，通信系统需要转向语义层面的有效传输。

语义通信的核心思想是：在发端提取与任务相关的语义表征，在收端重构与决策所需的语义信息，而非逐比特恢复原始载荷[2]-[4]。近年来，基于深度学习的端到端语义通信架构逐步成熟，尤其是以 Transformer 为核心编码器-解码器骨干的 DeepSC 系列方法，在文本任务上显示出可行性[5]。但是，已有工作仍面临三个关键挑战：其一，低 SNR 与复杂衰落下语义表示脆弱；其二，多目标损失中的语义约束存在不稳定耦合；其三，外部大语言模型后处理在某些条件下可能“过修正”并降低最终指标。

本科毕业设计既要求技术可实现，也要求研究过程完整可验证。基于这一目标，本文选择文本语义通信作为问题载体，围绕“结构侧信道感知、训练侧损失调度、推理侧后处理策略”构建一条可落地的创新路径，并通过系统实验回答“哪些改进真正有效、在哪些条件下有效、代价和边界是什么”。

## 1.2 Research Questions

围绕上述背景，本文聚焦以下研究问题：

1. 在文本语义通信任务中，SNR 感知机制是否能够稳定提升低信噪比下的重构质量？
2. 语义损失项在端到端训练中的作用是正向增强还是潜在负迁移？是否可通过自适应权重缓解？
3. 基于预训练语言模型的后处理精炼是否能够带来净增益？若不能，其失效机理是什么？
4. 在不重新训练主模型的条件下，推理策略（如 Beam Search）能否提供更高性价比的性能提升？

## 1.3 Main Contributions

本文主要贡献如下：

- 搭建并复现了完整的文本语义通信实验链路，包括数据预处理、模型训练、统一评估和可视化分析；
- 在原有 SNR 自适应注意力思路基础上，将信道感知扩展到解码器 cross-attention，形成双端信道感知结构；
- 提出 SNR 自适应语义损失权重机制，探索固定语义权重在低 SNR 下的失配问题；
- 设计置信度驱动的 BERT MLM 精炼流程，替代早期 GPT 风格后处理；
- 基于大量对比与消融实验，识别出创新点的有效边界：Beam Search 在本任务中比外置 LLM 精炼器更稳定有效；
- 给出“初始假设—问题暴露—改进迭代—再验证”的完整研究叙事，为工程型毕业论文提供可复用范式。

## 1.4 Thesis Organization

第 2 章介绍语义通信、信道感知建模、语义损失与 LLM 后处理等相关研究；第 3 章给出本文系统设计、创新点实现与实验配置；第 4 章展示 AWGN、Rayleigh、消融和组合实验结果；第 5 章分析结果背后的机制与局限；第 6 章给出结论与后续工作方向。

---

# 2 Literature Survey

## 2.1 From Shannon Communication to Semantic Communication

香农信息论强调语法层可靠传输，其理论上限由信道容量决定[1]。该框架极大推动了现代通信发展，但并不直接刻画“消息对任务的意义”。语义信息思想可以追溯到 Carnap 与 Bar-Hillel 对语义信息形式化的讨论[2]，近年在通信领域得到系统重访[3][4]。随着深度神经网络在表征学习中的成功，研究者开始尝试直接学习“任务相关语义压缩—信道传输—语义重建”的端到端模型。

Xie 等提出的 DeepSC 是代表性工作之一[5]。其使用 Transformer 提取句子语义，在噪声信道中传输低维语义表示，并在收端重建文本。与传统信道编码相比，该类方法在低 SNR 下能维持更高的语义相似度，为语义通信可行性提供了实证基础。后续研究逐步拓展到多用户、语音、多模态和任务导向场景[6]-[9]。

## 2.2 Channel-Aware Modeling in Deep Learning Systems

传统无线系统中，信道状态信息（CSI）用于自适应调制编码、功率分配与资源调度。对于深度学习语义通信模型，如何显式注入信道质量信息同样关键。已有研究通常采用 SNR 条件输入、条件归一化或门控机制，以提高模型对信道分布变化的泛化能力[6][10]。然而，很多实现只在编码侧使用信道感知，解码侧并未充分利用“接收语义特征的不确定性”信息。

Rayleigh 衰落信道相比 AWGN 具有更高的随机性与时变性[11][12]。在该信道中，语义表示可能经历幅度起伏和深衰落，导致注意力聚焦被扰动。因此，针对 Rayleigh 的鲁棒设计通常需要跨模块协同，而非单点改动。

## 2.3 Multi-Objective Learning and Semantic Loss

在文本生成和语义重构任务中，交叉熵损失可稳定优化词级预测，但并不直接保证句子语义一致性。为此，一些工作引入额外语义一致性约束，如嵌入空间相似度、对比学习损失或句级判别损失[13]-[15]。多目标训练的难点在于损失之间的梯度冲突：在噪声较大时，语义项可能提供不稳定监督，反而拖累主任务收敛。

自适应损失加权是缓解冲突的常见策略，包括基于不确定性估计、梯度范数平衡或条件门控的动态权重[16][17]。在语义通信场景中，SNR 可作为天然条件变量，用于调节语义项强度，这构成了本文改进 B 的理论动机。

## 2.4 LLM-based Post-Refinement for Text Reconstruction

预训练语言模型（PLM）在纠错、补全与重写任务中广泛应用。BERT 提供双向上下文建模能力，适合掩码恢复任务[18]；GPT 系列偏向自回归生成，擅长续写但在某些纠错场景容易受左上下文偏置影响[19]。在语义通信中，外置 LLM 作为后处理器可用于“二次修复”低置信度词位。

但相关研究也指出，当主模型输出已较高质量时，后处理模型可能因域偏差和目标错配引发过修正[20]-[22]。因此，后处理是否有效取决于触发策略、置信度估计和领域匹配程度。本文采用“解码器置信度驱动掩码”的方式尝试降低不必要修改，并通过实验验证其收益边界。

## 2.5 Evaluation Metrics for Semantic Reconstruction

文本重构质量评价通常结合词级和句级指标。BLEU 从 n-gram 匹配角度评价生成文本与参考文本一致性[23]；BERTScore 利用预训练上下文嵌入衡量语义相似性[24]；句向量余弦相似度可作为全局语义一致性的补充。语义通信研究常采用多指标联合评估，以减少单指标偏差[5][8]。本文同时报告 Word Accuracy、BLEU-1~4、BERTScore 与 Sentence Similarity，以全面反映模型行为。

---

# 3 Materials and Methods

## 3.1 Dataset and Preprocessing

本文实验数据来自 Europarl v7 英文语料。预处理流程包括：文本小写化、字符清洗、长度过滤、词表构建、序列索引化与固定长度填充。依据项目配置，词表规模为 10,000，序列最大长度为 32，最小长度为 4。样本按 9:1 划分为训练集与测试集。

为保证可复现性，预处理脚本固定了符号定义：`<PAD>`、`<START>`、`<END>`、`<UNK>`，对应索引分别为 0/1/2/3。训练与评估统一使用同一词表与句长约束，避免由数据处理不一致导致的偏差。

## 3.2 Baseline Transceiver Architecture

基线模型采用 Encoder-Channel-Decoder 架构：源句子先经词嵌入与位置编码进入 4 层 Transformer Encoder；编码输出经信道编码层降维到信道空间，输入可微分 AWGN 或 Rayleigh 信道；接收端通过信道解码层恢复特征，并由 4 层 Transformer Decoder 逐步生成目标序列。训练目标为词级交叉熵损失。

该设计与 DeepSC 思路一致[5]，其优势是端到端可训练、可直接面向语义重构优化；局限是对信道变化敏感，在低 SNR 与衰落条件下易出现语义漂移。本文所有改进均建立在该基线上，保证比较公平。

## 3.3 Initial Innovation Hypotheses

### 3.3.1 Innovation A (Initial): Encoder-side SNR-Adaptive Attention

初始版本将 SNR 嵌入引入编码器多头注意力，使注意力权重可随信道质量变化。直觉上，高 SNR 时模型可保留更细粒度语义，低 SNR 时偏向鲁棒性更强的表示。

### 3.3.2 Innovation B (Initial): Composite Semantic Loss

训练损失设为 `L = alpha * L_ce + beta * L_sem`。其中 `L_sem` 可由编码-解码特征相似性或嵌入一致性构成。初始设置下 `beta` 为固定常数 0.2。

### 3.3.3 Innovation C (Initial): LLM Post-Refinement

在推理阶段加入外部语言模型对初始解码句进行纠错，早期方案采用 GPT 风格模型并基于外部困惑度触发替换。

## 3.4 Iterative Improvements after Problem Diagnosis

### 3.4.1 Improved A: Decoder-side SNR-Adaptive Cross-Attention

针对“仅编码器感知不足”的问题，本文在解码器 cross-attention 也注入 SNR 嵌入。这样，解码端在访问接收记忆特征时可动态调整权重，更符合“信道不确定性应在重构端持续可见”的原则。

### 3.4.2 Improved B: Adaptive Semantic Loss Weight

针对固定 `beta` 在低 SNR 可能误导训练的问题，将其改为

`beta_eff = beta * sigmoid(mean(SNR)/5)`。

该机制在低 SNR 自动降低语义项影响，在高 SNR 逐步恢复语义约束强度，目标是缓解梯度冲突。

### 3.4.3 Improved C: Confidence-Driven BERT MLM Refiner

将后处理从 GPT 风格模型替换为 BERT MLM。流程为：先获取解码器每个词位的最大 softmax 置信度；对低于阈值的位置掩码；由 BERT 预测替换。该策略强调“仅修复不确定位置”，避免全句盲目改写。

## 3.5 Training and Evaluation Setup

训练核心参数：batch size 256，epochs 80，学习率 1e-3，优化器 AdamW，训练 SNR 范围 -10~20 dB。评估在固定 SNR 点 `[-10, -5, 0, 5, 10, 15, 20]` 上进行。

评估指标包括：

- WordAcc（词级准确率）
- BLEU-1/2/3/4
- BERTScore（P/R/F1）
- Sentence Similarity（句向量余弦相似）

实验分四类：

1. 主实验：Baseline vs Improved（AWGN）
2. 消融实验：Improved vs Improved-NoSem（AWGN）
3. 跨信道实验：Baseline-Rayleigh vs Improved-Rayleigh
4. 推理策略实验：Beam only / FT-BERT only / Beam + FT-BERT

---

# 4 Results

## 4.1 Main Results under AWGN

### 4.1.1 Baseline vs Improved (Greedy)

在 AWGN 下，改进模型整体优于基线，尤其在低 SNR 更明显。关键结果如下（BLEU-4）：

- -10 dB：0.4189 → 0.4329（+1.40%）
- -5 dB：0.7340 → 0.7386（+0.46%）
- 0 dB：0.8371 → 0.8375（+0.04%）
- 5 dB：0.9633 → 0.9685（+0.52%）

对应 WordAcc 在 -10 dB 由 0.4425 提升到 0.4773，增幅更明显，说明 SNR 感知机制确实增强了低信噪比下的词级恢复能力。10 dB 以上两者差距迅速缩小，表明系统在高 SNR 已接近性能饱和区间。

### 4.1.2 Observation

结果符合语义通信常见规律：在中高 SNR，重构任务本身难度下降，结构改进边际收益有限；在低 SNR，结构鲁棒性决定了性能上限。该现象支持了创新点 A 的基本有效性。

## 4.2 Ablation Study: Impact of Semantic Loss

将 Improved 与 `Improved-NoSem` 对比发现，去除语义损失后多个 SNR 点 BLEU-4 更高：

- -10 dB：0.4329（Improved） vs 0.4488（NoSem）
- -5 dB：0.7386 vs 0.7448
- 0 dB：0.8375 vs 0.8398
- 5 dB：0.9685 vs 0.9709

这说明即便采用自适应 `beta_eff`，语义损失项仍可能存在弱负迁移。可解释为：在噪声较强时，语义项监督信号不稳定，与 CE 梯度方向不一致，导致优化目标冲突。该结论并不意味着语义损失无意义，而是提示其形式或权重调度仍需进一步改进。

## 4.3 Cross-Channel Comparison: Rayleigh

Rayleigh 信道下，Baseline 与 Improved 对比如下（BLEU-4）：

- -10 dB：0.4697（Baseline） vs 0.4640（Improved）
- -5 dB：0.6662 vs 0.6603
- 0 dB：0.8170 vs 0.8055
- 5 dB：0.9167 vs 0.9148

结果显示改进模型在 Rayleigh 下提升有限甚至略降，呈现“混合效应”。这提示 AWGN 有效机制不一定可直接迁移到强衰落场景。可能原因包括：Rayleigh 分布引入更复杂幅度随机性，现有 SNR 条件嵌入不足以刻画瞬时衰落状态。

## 4.4 LLM Refinement Results

在 Improved（greedy）基础上使用微调 BERT 精炼后（`eval_awgn_ftbert.json` 中 LLM_*），BLEU-4 在各 SNR 下均有轻微下降：

- -10 dB：0.4329 → 0.4191
- -5 dB：0.7386 → 0.7348
- 0 dB：0.8375 → 0.8339
- 10 dB：0.9934 → 0.9905

尽管相比早期 GPT 风格方案，降级幅度已明显缩小，但总体仍为负增益。该结果说明：当基础解码质量已较高时，外置 LLM 容易过修正，将局部正确词替换为不合任务语境的词。

## 4.5 Beam Search and Combined Strategies

### 4.5.1 Beam Search only

引入 Beam Search（width=5）后，低 SNR 提升显著：

- -10 dB BLEU-4：0.4329 → 0.4552（+2.23%）
- -5 dB：0.7386 → 0.7496（+1.10%）
- 0 dB：0.8375 → 0.8494（+1.19%）

该改进不依赖重新训练主模型，工程成本低且收益明确。

### 4.5.2 FT-BERT only and Combined

FT-BERT only（greedy+LLM）整体为轻微负增益；Beam+FT-BERT 组合在低 SNR 下降更明显：

- -10 dB：Beam only 0.4552 vs Beam+FT-BERT 0.4046
- -5 dB：0.7496 vs 0.7326

结论：在本任务与当前阈值机制下，Beam Search 是更稳定、更高收益的推理策略；LLM 精炼不宜与高质量 Beam 输出直接串联。

---

# 5 Discussion

## 5.1 Why SNR-Adaptive Attention Works in AWGN

AWGN 主要表现为加性高斯扰动，其统计性质稳定。将 SNR 显式注入注意力层，相当于为模型提供“噪声强度条件”，帮助其在语义特征选择时实现条件化权衡：低 SNR 下更偏向鲁棒语义骨架，高 SNR 下保留更多细粒度词级信息。因此在低 SNR 场景出现稳定增益。

## 5.2 Why Semantic Loss Still Shows Weak Negative Transfer

语义损失并非天然正向。当前实现中，语义约束和 CE 目标在低 SNR 下可能存在梯度冲突；即使采用 SNR 调度，冲突仍未彻底消除。此外，语义损失对句级一致性的偏好与词级准确率指标并不总是一致，可能导致 BLEU-4 和 WordAcc 不同步变化。

## 5.3 Why LLM Refinement Fails to Bring Net Gains

LLM 精炼失效的核心在于“目标错配”与“收益空间错估”。当主模型输出已较高质量（中高 SNR）时，可改进空间很小，任意替换都可能破坏原有正确词。即使只处理低置信度位置，置信度与真实错误并非一一对应。再加上 Europarl 语域与通用预训练语料仍有差异，导致修复动作不稳定。

## 5.4 Engineering Implications

对于毕业设计和工程部署，优先级建议为：

1. 优先做结构侧与解码策略侧优化（如双端信道感知 + beam）；
2. 对后处理 LLM 采取保守策略，仅在极低置信度和特定错误类型触发；
3. 若继续研究 LLM 路线，应考虑联合训练或最小编辑约束，而非自由替换。

## 5.5 Limitations and Future Work

本文局限包括：数据域单一（Europarl），任务为英文文本重构；Rayleigh 信道建模仍较简化；LLM 精炼器未进行联合优化。后续可考虑：

- 引入 CSI 估计特征与时变信道序列建模；
- 使用冲突感知多任务优化（如梯度投影）改善语义损失稳定性；
- 将后处理改为“最小编辑校正器”或“受限解码重排序器”；
- 扩展到多语种与任务导向语义通信。

---

# 6 Conclusions

本文围绕文本语义通信收发机开展了完整的系统研究，形成以下结论：

1. **创新点 A（双端 SNR 自适应注意力）在 AWGN 低 SNR 条件下有效。** 与基线相比，-10 dB 下 BLEU-4 与 WordAcc 均显著提升，验证了信道感知结构设计的价值。  
2. **创新点 B（自适应语义损失）相比固定权重已有改进，但当前实现仍呈弱负迁移。** 消融实验表明去除语义项后部分指标更优，说明该模块仍需进一步优化。  
3. **创新点 C（BERT MLM 精炼）相比早期方案降低了性能退化，但总体尚未实现正增益。** 其主要问题为过修正与域偏差。  
4. **Beam Search 是当前最具工程收益的改进。** 在不重训主模型的前提下，低 SNR 获得显著增益，且稳定性优于 LLM 后处理。  
5. 本课题完成了从“假设提出—问题发现—改进迭代—再验证”的闭环，满足本科毕业设计对研究性与工程性的双重要求。

综合来看，本文推荐的实用配置为：**Improved transceiver + Beam Search decoding**。该方案在复杂噪声条件下具有更优的性能/成本比，也为后续语义通信系统优化提供了可复用的基线。

---

# References

[1] Shannon C E. A Mathematical Theory of Communication[J]. Bell System Technical Journal, 1948, 27(3): 379-423; 27(4): 623-656.  
[2] Carnap R, Bar-Hillel Y. An Outline of a Theory of Semantic Information[R]. MIT, 1952/1954.  
[3] Bao J, Basu P, Dean M, et al. Towards a Theory of Semantic Communication[C]//IEEE Network Science Workshop. 2011.  
[4] Weaver W. Recent Contributions to the Mathematical Theory of Communication[M]//Shannon C E, Weaver W. The Mathematical Theory of Communication. University of Illinois Press, 1949.  
[5] Xie H, Qin Z, Li G Y, Juang B H. Deep Learning Enabled Semantic Communication Systems[J]. IEEE Transactions on Signal Processing, 2021, 69: 2663-2675.  
[6] Guo J, Wen C K, Jin S, et al. Constellation Design for Semantic Communications with Large Language Models[C]//IEEE ICC Workshops, 2024.  
[7] Shi Y, Zhang S, Yang K B, et al. Semantic Communications for Speech Recognition and Synthesis[J]. IEEE Journal on Selected Areas in Communications, 2023, 41(8): 2530-2546.  
[8] Lu Z, Li R, Lu K, et al. Semantics-Empowered Communication: A Tutorial-Cum-Survey[J]. IEEE Communications Surveys & Tutorials, 2024, 26(1): 534-572.  
[9] Getu T M, Kaddoum G, Bennis M. Tutorial-Cum-Survey on Semantic and Goal-Oriented Communication: Research Landscape, Challenges, and Future Directions[J]. IEEE Access, 2024, 12: 68021-68058.  
[10] Peng H, Niyato D, Wang P, et al. Artificial Intelligence for Wireless Communications in 5G and Beyond: A Survey[J]. IEEE Open Journal of the Communications Society, 2021, 2: 264-294.  
[11] Goldsmith A. Wireless Communications[M]. Cambridge University Press, 2005.  
[12] Rappaport T S. Wireless Communications: Principles and Practice[M]. 2nd ed. Prentice Hall, 2002.  
[13] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks[C]//EMNLP-IJCNLP. 2019: 3982-3992.  
[14] Gao T, Yao X, Chen D. SimCSE: Simple Contrastive Learning of Sentence Embeddings[C]//EMNLP. 2021: 6894-6910.  
[15] Cer D, Yang Y, Kong S Y, et al. Universal Sentence Encoder for English[J]. arXiv:1803.11175, 2018.  
[16] Kendall A, Gal Y, Cipolla R. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics[C]//CVPR. 2018: 7482-7491.  
[17] Chen Z, Badrinarayanan V, Lee C Y, Rabinovich A. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks[C]//ICML. 2018: 794-803.  
[18] Devlin J, Chang M W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]//NAACL-HLT. 2019: 4171-4186.  
[19] Radford A, Wu J, Child R, et al. Language Models are Unsupervised Multitask Learners[R]. OpenAI Technical Report, 2019.  
[20] Brown T, Mann B, Ryder N, et al. Language Models are Few-Shot Learners[C]//NeurIPS. 2020, 33: 1877-1901.  
[21] Zhang T, Kishore V, Wu F, Weinberger K Q, Artzi Y. BERTScore: Evaluating Text Generation with BERT[C]//ICLR. 2020.  
[22] Sanh V, Debut L, Chaumond J, Wolf T. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter[J]. arXiv:1910.01108, 2019.  
[23] Papineni K, Roukos S, Ward T, Zhu W J. BLEU: a Method for Automatic Evaluation of Machine Translation[C]//ACL. 2002: 311-318.  
[24] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]//NeurIPS. 2017, 30: 5998-6008.  
[25] Kingma D P, Ba J. Adam: A Method for Stochastic Optimization[C]//ICLR. 2015.  
[26] Loshchilov I, Hutter F. Decoupled Weight Decay Regularization[C]//ICLR. 2019.  
[27] Hochreiter S, Schmidhuber J. Long Short-Term Memory[J]. Neural Computation, 1997, 9(8): 1735-1780.  
[28] Bahdanau D, Cho K, Bengio Y. Neural Machine Translation by Jointly Learning to Align and Translate[C]//ICLR. 2015.  
[29] Graves A. Sequence Transduction with Recurrent Neural Networks[J]. arXiv:1211.3711, 2012.  
[30] Sutskever I, Vinyals O, Le Q V. Sequence to Sequence Learning with Neural Networks[C]//NeurIPS. 2014, 27: 3104-3112.

---

# Appendices

## Appendix A: Reproducible Command Flow

本项目关键命令流程如下（节选）：

1. 环境准备与 GPU 检查  
`python -m venv .venv`  
`.\.venv\Scripts\Activate.ps1`  
`python -c "import torch; print(torch.cuda.is_available())"`

2. 训练各模型  
`python train.py --model baseline --channel awgn --epochs 80`  
`python train.py --model improved --channel awgn --epochs 80`  
`python train.py --model improved --channel awgn --epochs 80 --beta 0.0 --tag no_sem`  
`python train.py --model baseline --channel rayleigh --epochs 80`  
`python train.py --model improved --channel rayleigh --epochs 80`

3. 评估与组合实验  
`python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn`  
`python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --beam_width 5 --eval_tag beam5`  
`python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --use_llm --llm_model checkpoints/bert_europarl --eval_tag ftbert`  
`python evaluate.py --checkpoint checkpoints/improved_awgn/best.pt --model improved --channel awgn --beam_width 5 --use_llm --llm_model checkpoints/bert_europarl --eval_tag beam5_ftbert`

## Appendix B: Suggested Figure/Table Captions

- Figure 4-2 BLEU-4 comparison under AWGN across seven SNR points, showing that Improved outperforms Baseline mainly in low SNR regions and Beam Search further improves low-SNR robustness.  
- Table 4-4 Quantitative comparison of inference strategies (greedy, Beam5, FT-BERT, Beam5+FT-BERT), indicating that Beam Search yields the most stable positive gains while LLM refinement introduces slight to moderate degradation.


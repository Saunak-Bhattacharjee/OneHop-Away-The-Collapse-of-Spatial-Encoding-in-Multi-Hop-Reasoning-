# OneHop-Away-The-Collapse-of-Spatial-Encoding-in-Multi-Hop-Reasoning-
A probing study of spatial relation encoding across transformer layers in Pythia-2.8B, using the StepGame and SpaRTUN benchmarks.

## Abstract
 
We investigate how spatial directional relations are encoded in the internal representations of a large language model (Pythia-2.8B) across all 32 transformer layers. Using linear probing classifiers trained on entity-targeted hidden state extractions, we show that direct (single-hop) spatial relations are encoded with high linear separability in the earliest layers (Layer 1 accuracy: 0.715, chance: 0.250), and that this signal degrades monotonically through the network. More strikingly, adding a single intermediate reasoning hop causes probe accuracy to collapse from 0.715 to 0.344 at Layer 1 (a drop of 0.371), and all multi-hop conditions (k ≥ 2) remain near chance across all layers. We further show that the extraction strategy matters more than dataset choice: mean-pooling over the full sequence yields flat, near-chance curves regardless of layer, while entity-targeted extraction (concatenating hidden states at the subject and object entity positions) recovers a strong spatial signal for k=1. These findings suggest that Pythia encodes direct spatial relations linearly and early, but compositional spatial reasoning does not produce linearly-readable traces in individual entity representations at any network depth.


 ## 1. Introduction
 
A central question in large language model (LLM) interpretability is whether these models form genuine internal representations of the world, including spatial structure, or whether they rely on surface-level pattern matching. Spatial reasoning is a particularly clean test case: sentences like "X is to the left of K" have unambiguous relational semantics, and multi-hop chains like "X is left of K, K is above Y" require explicit compositional inference to determine the relation between X and Y.
 
Prior work has explored spatial reasoning *behaviorally*, i.e., measuring whether LLMs answer spatial questions correctly, but comparatively little work has examined *where* and *how* spatial relations are encoded in model internals across layers. The logit lens and probing classifier traditions offer complementary tools: the former asks what the model would predict at each layer; the latter asks whether specific properties are linearly decodable from hidden states.
 
In this work, we apply entity-targeted linear probing to Pythia-2.8B across all 32 layers, using two spatial reasoning benchmarks: **StepGame** (directional relations, k=1–5 hops) and **SpaRTUN** (richer relation taxonomy including topological relations). The primary contributions of this project are:
 
1. Entity-targeted hidden state extraction (concatenating representations at subject and object token positions) dramatically outperforms mean-pooling over the full sequence for spatial probing, recovering probe accuracies more than double chance at Layer 1.
2. For direct (k=1) spatial relations, linear probe accuracy peaks at Layer 1 (0.715) and declines monotonically to Layer 32 (0.636), suggesting that spatial direction is encoded early and transformed rather than built up through subsequent layers.
3. Probe accuracy collapses sharply from k=1 to k=2 (0.715 → 0.344) and remains near chance for all k ≥ 2 across all layers, indicating that compositional spatial relations are not linearly readable from entity-level representations regardless of network depth.

## 2. Datasets
 
### 2.1 [StepGame] (https://arxiv.org/pdf/2204.08292)
StepGame (Shi et al., AAAI 2022) is a multi-hop spatial reasoning benchmark in which entities are arranged in directional chains. Each example consists of a context describing pairwise relations (e.g. "X is to the left of K and is on the same horizontal plane"), a question asking for the relation between a non-adjacent pair, and a ground-truth directional label from {left, right, above, below}.
 
The benchmark parameterises difficulty by `k`, the number of hops required to answer the question. We use the HuggingFace release (`UKPLab/sparp`, config `SpaRP-PS2 (StepGame)`), which augments the original with symbolic reasoning paths and hop annotations.
 
For our experiments, we restrict to four-class directional labels (left, right, above, below), yielding a balanced dataset (chance = 0.25) with 14,347 examples at k=1, 5,389 at k=2, 1,927 at k=3, 788 at k=4, and 213 at k=5.
 
### 2.2 [SpaRTUN] (https://arxiv.org/pdf/2210.16952)
 
SpaRTUN (Mirzaee et al., NAACL 2021; updated) is a spatial QA benchmark with a richer relation taxonomy including directional (left, right, above, below, in front, behind), proximity (near, far), and topological (inside, outside, contains, partially overlapping, and touching variants) relations. We use the HuggingFace release (`UKPLab/sparp`, config `SpaRP-PS1 (SpaRTUN)`).
 
We collapse the 15 fine-grained relation types into 5 coarse classes, i.e., lateral, vertical, depth, proximity, and topological, to mitigate severe class imbalance (the topological class accounts for 54% of single-label examples).

## 3. Method
 
### 3.1 Model
 
We use **Pythia-2.8B** (EleutherAI), a 32-layer decoder-only transformer with hidden dimension 2,560, accessed via HuggingFace Transformers with weights frozen throughout. Pythia was chosen for its full openness, native TransformerLens support, and the availability of pre-trained Tuned Lens translators.
 
### 3.2 Hidden State Extraction
 
For each example we pass the context string through the frozen model with `output_hidden_states=True`, obtaining 32 hidden state tensors of shape `(seq_len, 2560)`, one per layer, skipping the embedding layer.
 
We experiment with two extraction strategies:
 
**Mean-pool**: average hidden states over all non-padding token positions, producing a single `(2560,)` vector per layer. This is the standard approach in many probing studies.
 
**Entity-targeted**: locate the token spans of the subject entity (e.g., "X") and object entity (e.g., "K") in the tokenized context using a space-prefixed token search. Extract and mean-pool over each entity's span separately, then concatenate the two vectors, producing a `(5120,)` vector per layer. If an entity is not found, fall back to the last token.
 
### 3.3 Probe Training
 
For each layer L (1–32), we train an independent **logistic regression** probe on the extracted representations. We use `sklearn.linear_model.LogisticRegression` with L2 regularization (C=1.0), standardised inputs (`StandardScaler` fitted on the training split only), stratified 80/20 train/test split, and report accuracy and macro-F1 on the held-out test set.
 
Linear probes are chosen deliberately: a complex probe might learn the task independently rather than reading out information already present in the representation. High linear probe accuracy is therefore evidence that the property is geometrically clean in the hidden state, not merely present in some nonlinear sense.
 
### 3.4 Hop-Depth Analysis
 
For the StepGame entity-targeted setting, we additionally stratify examples by `num_hop` (k=1–5) and train separate probe curves per hop depth, enabling direct comparison of how spatial encoding varies with reasoning complexity.

## 4. Results
 
### 4.1 Extraction Strategy Comparison
 
| Method | Dataset | Classes | L1 acc | Peak acc | L32 acc |
|---|---|---|---|---|---|
| Mean-pool | SpaRTUN | 5 (coarse) | 0.537 | 0.537 | 0.458 |
| Mean-pool | StepGame | 4 (directional) | 0.354 | 0.354 | 0.351 |
| Entity-targeted | StepGame | 4 (directional) | **0.564** | **0.564** | **0.449** |
| Entity-targeted (k=1 only) | StepGame | 4 (directional) | **0.715** | **0.715** | **0.636** |
| Chance | SpaRTUN | 5 | 0.200 | — | — |
| Chance | StepGame | 4 | 0.250 | — | — |
 
Mean-pooling over the full sequence yields flat curves hovering near the majority-class baseline for SpaRTUN and near chance for StepGame. Entity-targeted extraction recovers a strong and structured signal, particularly for k=1 examples.

 [Spatial Probe Accuracy in SpaRTUN](Figures/SpatialProbeAcconSpartrun.png)
 [Spatial Probe Accuracy in StepGame](Figures/SpatialProbeAccuracyStepGame.png)
### 4.2 Layer-wise Probe Curve (k=1)
 
For single-hop StepGame examples, Layer 1 achieves 0.715 accuracy. The curve declines monotonically and smoothly to 0.636 at Layer 32. There is no mid-network peak; the model does not appear to build up spatial encoding progressively, the signal is strongest at the bottom, and eroded through subsequent computation.

 [Spatial Probe Accuracy by layer and hop depth in StepGame](Figures/HoppingbyLayer.png)
 
### 4.3 Hop-Depth Analysis
 
| k | n | L1 acc | Peak layer | Peak acc | L32 acc |
|---|---|---|---|---|---|
| 1 | 14,347 | **0.715** | 1 | 0.715 | 0.636 |
| 2 | 5,389 | 0.344 | 1 | 0.344 | 0.276 |
| 3 | 1,927 | 0.342 | 4 | 0.360 | 0.345 |
| 4 | 788 | 0.361 | 1 | 0.361 | 0.253 |
| 5 | 213 | 0.233 | 28 | 0.326 | 0.302 |
| — | — | chance: 0.250 | — | — | — |
 
The k=1 curve is entirely separated from k≥2. Adding one intermediate hop causes a 0.371 drop in L1 accuracy, and all multi-hop conditions remain near or below chance across all 32 layers. The k=5 condition exhibits a tentative late peak (L28: 0.326), indicating that very long chains may accumulate a slight signal in deeper layers; however, the sample size (n=213) limits confidence in this observation.
 [Hop-Depth Analysis](Figures/Hop-Depth Exp.png)
---

## 5. Discussion
 
**Direct spatial relations are encoded early and linearly.**: The 0.715 L1 accuracy for k=1 examples is a strong positive result: the concatenated entity representations carry enough directional geometry immediately after the first transformer block for a linear classifier to recover the answer at well above chance. This finding is consistent with those in other probing studies, which suggest that surface-level semantic features are established in early layers.
 
**Deeper layers transform rather than preserve this signal.**:  The monotonic decline from L1 to L32 (0.715 → 0.636 for k=1) suggests that later layers are transforming the early encoding into a format suited to next-token prediction, which is geometrically less clean for a direction-classification probe.
 
**Compositional spatial reasoning is not linearly readable from entity tokens.**: The collapse at k=2 is the central finding. The model cannot compose even a two-hop spatial chain into a linearly-decodable relation at the individual entity token positions. This does not mean the model fails at multi-hop spatial reasoning. It means the answer is not stored *there*. 
 
**Extraction strategy is the critical methodological decision.** The jump from 0.354 (mean-pool) to 0.715 (entity-targeted, k=1) on the same dataset and model shows that *where* you extract from matters far more than which dataset or model you use. Probing studies that rely on sentence-level representations may be systematically underestimating the spatial encoding capacity of LLMs.

## References
 
- Shi, Z., Zhang, Q., & Lipani, A. (2022). StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts. *AAAI 2022*.
- Mirzaee, R., Faghihi, H. R., Ning, Q., & Kordjamshidi, P. (2021). SPARTQA: A Textual Question Answering Benchmark for Spatial Reasoning. *NAACL 2021*.
- Rizvi, M. I., Zhu, X., & Gurevych, I. (2024). SpaRC and SpaRP: Spatial Reasoning Characterization and Path Generation for Understanding Spatial Reasoning Capability of Large Language Models. *ACL 2024*.
- Biderman, S. et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML 2023*.
- Alain, G., & Bengio, Y. (2016). Understanding Intermediate Layers Using Linear Classifier Probes. *ICLR Workshop 2017*.
- Belrose, N. et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. *arXiv:2303.08112*.


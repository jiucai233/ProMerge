# Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation

**Paper Link**: [arXiv:2602.06575](https://arxiv.org/abs/2602.06575)  
**Project Page**: [https://nicehiro.github.io/ThinkProprio](https://nicehiro.github.io/ThinkProprio)

---

## Abstract

Vision-language-action (VLA) models typically inject proprioception only as a late conditioning signal, which prevents robot state from shaping instruction understanding and from influencing which visual tokens are attended throughout the policy. We introduce **ThinkProprio**, which converts proprioception into a sequence of text tokens in the VLM embedding space and fuses them with the task instruction at the input. This early fusion lets embodied state participate in subsequent visual reasoning and token selection, biasing computation toward action-critical evidence while suppressing redundant visual tokens. 

In a systematic ablation over proprioception encoding, state entry point, and action-head conditioning, we find that text tokenization is more effective than learned projectors, and that retaining roughly 15% of visual tokens can match the performance of using the full token set. Across CALVIN, LIBERO, and real-world manipulation, ThinkProprio matches or improves over strong baselines while reducing end-to-end inference latency over 50%. 

---

## 1. Introduction

Vision-language-action (VLA) models have emerged as a powerful paradigm for generalizable robot control because they translate visual observations and natural-language instructions into executable actions through large-scale pretraining (Zitkovich et al., 2023; Kim et al., 2025b; Black et al., 2024a). Yet most current VLA pipelines introduce proprioceptive signals late in the processing stack, couple them only weakly to perception, or ignore them entirely. Contact-rich manipulation depends not only on what is visible and what action is requested, but also on the robot’s embodiment, including its joint configuration and motion. This raises a basic design question: when a robot interprets what it sees and what to do, how can it better leverage physically grounded state to understand the task and environment?

A striking example is the evolution from $\pi_0$ (Black et al., 2024a), where proprioceptive state is encoded by a simple projector and fed to the action head, to $\pi_{0.5}$ (Intelligence et al., 2025), where proprioceptive state is tokenized as text and fed through the VLM encoder. $\pi_{0.5}$ achieves an 18% improvement on the CALVIN (Mees et al., 2022) $ABC \to D$ benchmark, with average completion length increasing from 2.01 to 2.38. Although other changes are present, the change in proprioceptive encoding and entry point is distinctive and plausibly contributes to the gain.

However, in existing systems, such as $\pi_{0.5}$, the proprioception design is typically chosen jointly with vision-language aggregation and the token conditioning mechanism at the action head, which makes the functional role of proprioception difficult to isolate. Figure 1(a) and Table 1 show that modern VLA models vary along several coupled axes, and these choices trade expressivity against efficiency and information retention. Cross-attention conditioning (Reuss et al., 2025) is flexible but expensive, while pooled tokens (Huang et al., 2025) or compressed tokens (Li et al., 2024) reduce compute at the cost of spatial detail. For proprioception, learned projections preserve continuous values but must map into an embedding space that is not pretrained, whereas text tokenization (Intelligence et al., 2025) leverages the VLM text interface but discretizes continuous state.

We conduct a systematic study to uncouple design choices by varying vision-language aggregation, the conditioning mechanism, and proprioception encoding and entry. To our knowledge, this provides the first systematic comparison. Building on the view that proprioception should be explicitly provided and actively involved during task understanding, we introduce ThinkProprio. Unlike LightVLA, which uses instruction-guided token selection primarily to reduce computation (Jiang et al., 2025), ThinkProprio uses joint instruction-and-proprioception guidance to study how embodied state shapes which visual evidence is retained during reasoning. In this framing, task-relevant evidence consists of object-centric visual cues together with configuration-specific proprioceptive context, which is sufficient to drive action generation without relying on heavier conditioning pathways. On CALVIN $ABC \to D$, ThinkProprio improves average completion length from 4.44 to 4.55, and on LIBERO it increases average success from 96.9% to 97.3%. Moreover, it reduces end-to-end inference latency by 58%, from 52 ms to 22 ms on CALVIN $ABC \to D$.

---

## 2. Related Works

VLA models integrate perception, language understanding, and action generation to execute language instructions (Din et al., 2025). Single-system approaches such as OpenVLA (Kim et al., 2025b) learn an end-to-end mapping in a unified token space, whereas dual-system architectures separate semantic processing from motor execution with a dedicated action head. Diffusion-based policies are a common choice for action generation (He et al., 2025a, b).

### 2.1 Task Understanding in VLA
* **Vision-Language Feature Extraction**: VLA systems differ in how vision-language backbone outputs are presented to the action head. Some methods pass dense token sets directly, as in $\pi_0$ (Black et al., 2024a), $\pi_{0.5}$ (Intelligence et al., 2025), and FLOWER (Reuss et al., 2025). This choice preserves fine-grained information but increases computation at the action head. Other methods apply token aggregation before control. CogACT (Li et al., 2024) reduces the token set through compression, and DiT-Block (Dasari et al., 2024), OTTER (Huang et al., 2025), and LightVLA (Jiang et al., 2025) use pooling-style reductions that connect to a broader literature on token reduction (Rao et al., 2021; Ryoo et al., 2021). Across these designs, the robot’s proprioceptive state is typically not part of the representation that drives which vision-language tokens are retained, even though robot configuration can change which scene elements are relevant for spatially grounded actions.
* **Proprioceptive Encoding and Entry**: Prior work integrates proprioception with different encodings and entry points. $\pi_{0.5}$ encodes proprioception as text and prepends it to the VLM input sequence, which aligns it with pretrained token embeddings. GR00T-N1 (Bjorck et al., 2025) and SmolVLA (Shukor et al., 2025) instead use multilayer perceptrons to project proprioception into the VLM feature space, which increases representational flexibility but can introduce mismatch relative to pretrained token features. FLOWER (Reuss et al., 2025) conditions the action head directly on proprioceptive inputs, bypassing the backbone.

### 2.2 Conditioning Mechanisms
Following the taxonomy introduced by Diffusion Transformer (DiT) (Peebles & Xie, 2023), VLAs commonly use three conditioning mechanisms that differ in how vision-language tokens and proprioception enter the action head:
- **In-context Conditioning**: Dita (Hou et al., 2025) concatenates vision-language features and proprioceptive embeddings with the action sequence and processes them in self-attention. In single-system VLA models (Kim et al., 2025b), vision and language features are embedded as tokens and provided as prefix context to a decoder-only Transformer, enabling action generation via in-context conditioning.
- **Adaptive Layer Normalization (AdaLN)**: DiT-Block (Dasari et al., 2024) applies global modulation by predicting layer-wise scale and shift parameters from pooled vision-language features and using them to modulate normalization during action generation. FLOWER (Reuss et al., 2025) and MDT (Reuss et al., 2024) use action-space Global AdaLN-Zero conditioning with proprioception.
- **Cross-Attention**: $\pi_0$ (Black et al., 2024a) and $\pi_{0.5}$ (Intelligence et al., 2025) condition action computation via block-wise cross-attention with causal masking used to preserve the action-generation ordering. FLOWER (Reuss et al., 2025) integrates multimodal features into its Flow Transformer through cross-attention. Cross-attention retains token-level access to visual and language context, at the cost of higher computation.

Our approach targets this interface explicitly by representing robot configuration in the VLM token space and using it together with the instruction to prioritize a compact set of task-relevant visual evidence. This yields state-aware, token-level conditioning via cross-attention while improving efficiency by reducing the conditioning tokens.

---

## 3. Method

ThinkProprio is built on the principle of *thinking proprioceptively*. We first tokenize proprioception into the VLM embedding space. We then use instruction and proprioceptive tokens jointly to guide the selection of task-relevant visual patches. The resulting compact token sequence is processed by the VLM, and the action head conditions on the VLM features through cross-attention to generate actions.

```
+------------------+     +------------------------+
| RGB Observations |     | Proprioceptive State q |
+--------+---------+     +-----------+------------+
         |                           | (Uniform Binning)
         v                           v
+--------+---------+     +-----------+------------+
| Vision Tokens H_v|     |   Proprio Tokens H_p   |
+--------+---------+     +-----------+------------+
         |                           |
         |     +---------------------+
         |     | (Guidance Tokens H_q = [H_l; H_p])
         v     v
+--------+-----+---+     +------------------------+
| Physically       |     |  Language Instruction  |
| Grounded Selector|     +-----------+------------+
+--------+---------+                 |
         | (15% kept + H^ctx)        v
         v                       +---+--------------------+
+--------+---------+             |  Language Tokens H_l   |
| Compact Visual   |<------------+                        |
| Tokens H_v^cond  |             +---+--------------------+
+--------+---------+                 |
         |                           |
         +-------------+-------------+
                       |
                       v
            +----------+----------+
            |     VLM Backbone    |
            +----------+----------+
                       |
                       v
            +----------+----------+
            |     Action Head     |
            | (Flow Matching DiT) |
            +----------+----------+
                       |
                       v
            +----------+----------+
            |     Action Chunk    |
            +---------------------+
```

### 3.1 Problem Setup
We consider a dual-system VLA policy $\pi_\theta$ with a vision-language backbone and a separate action head. At timestep $t$, the policy receives an observation $o_t$ comprising $n$ RGB images $(I_t^1, \dots, I_t^n)$, a language instruction $\ell$, and a proprioceptive state $\bm{q}_t \in \mathbb{R}^p$ that encodes the robot’s current configuration, such as joint angles and end-effector pose.

We represent each modality with tokens. A vision encoder maps the images to patch embeddings, yielding vision tokens $H_v \in \mathbb{R}^{N_v \times D}$, where $N_v$ is the total number of patches across the $n$ views and $D$ is the embedding dimension. The instruction $\ell$ is tokenized into language tokens $H_l \in \mathbb{R}^{N_l \times D}$, where $N_l$ is the number of text tokens. The proprioceptive state is encoded into proprio tokens $H_p \in \mathbb{R}^{N_p \times D}$ via an encoding function $f_q$.

The architecture comprises a vision-language backbone $f_{\text{VLM}}$, initialized from a pretrained VLM, and an action head $f_{\text{ACT}}$. The backbone maps the tokenized observation to conditioning features $C$, and the action head predicts a continuous action chunk $\bm{a}_{t:t+\mathcal{H}}$ conditioned on $C$.

### 3.2 Proprioceptive State Encoding
We implement the proprioceptive encoding function $f_q$ by discretizing continuous state values and reusing the VLM token embedding table. At time $t$, the proprioceptive state $\bm{q}_t \in \mathbb{R}^p$ contains $p$ scalar values. We discretize each value using uniform binning over a clipped range, where $q_{\min}$ and $q_{\max}$ denote scalar clipping bounds applied elementwise and $B$ is the number of bins. For each element $q_{t,k}$ with $k \in \{1, \dots, p\}$, we compute the bin index:

$$b_{t,k} = \text{clip}\left(\left\lfloor \frac{q_{t,k} - q_{\min}}{q_{\max} - q_{\min}} \times B \right\rfloor, 0, B-1\right)$$

Let $V$ denote the VLM vocabulary size. We map each bin index to one of the last $B$ vocabulary token IDs using a reverse mapping:
$$\tau_{t,k} = V - 1 - b_{t,k}$$

We then obtain the corresponding embeddings from the VLM token embedding table:
$$H_p = f_q(\bm{q}_t) = \operatorname{Embed}(\bm{\tau}_t) \in \mathbb{R}^{N_p \times D}$$

where $\bm{\tau}_t = [\tau_{t,1}, \dots, \tau_{t,p}]$ and $N_p = p$. This tokenization places proprioception in the same embedding space as language, enabling state-aware visual conditioning via cross-modal attention without introducing an additional projection.

### 3.3 Physically Grounded Token Selection
We use the task instruction and the robot’s current proprioceptive state to guide VLM visual token selection, so that subsequent attention layers focus on embodiment-relevant evidence while operating on a shorter effective sequence.

#### Query generation and scoring
Let the guidance tokens be $H_q = [H_l; H_p] \in \mathbb{R}^{N_q \times D}$, which concatenate instruction and proprio tokens. We apply RMSNorm to obtain $\tilde{H}_v = \text{RMSNorm}(H_v)$ and $\tilde{H}_q = \text{RMSNorm}(H_q)$. Each visual token attends to the guidance tokens to form a physically grounded query:

$$Q = \text{Softmax}\left( \frac{\tilde{H}_v \tilde{H}_q^T}{\sqrt{D}} \right) H_q \in \mathbb{R}^{N_v \times D}$$

We then compute a score matrix by matching normalized queries to normalized visual tokens:

$$S = \tilde{Q} \tilde{H}_v^T \in \mathbb{R}^{N_v \times N_v}$$

where $\tilde{Q} = \text{RMSNorm}(Q)$. Each row of $S$ scores all visual tokens for potential retention using a query grounded in instruction and proprioception.

#### Vote-based selection with annealed Gumbel noise
We use a vote-based discrete selection with a straight-through relaxation as one practical instantiation. We perturb the scores with i.i.d. Gumbel noise $G$ and temperature $\alpha$:

$$\hat{S} = S + \alpha G$$

where $G_{i,j} = -\log(-\log(U_{i,j}))$ and $U \sim \text{Unif}(0, 1)^{N_v \times N_v}$. As $\alpha \to 0$, the perturbation vanishes and selection becomes deterministic. We anneal $\alpha$ from $\alpha_{\text{start}}$ to $\alpha_{\text{end}}$ with a cosine schedule, which encourages exploration early in training and sharpens toward deterministic selection at convergence. Each row $i$ of $\hat{S}$ casts one vote by selecting the column index $j = \arg\max_k \hat{S}_{i,k}$. We retain all visual tokens that receive at least one vote, yielding a binary mask $\bm{m} = (m_1, \dots, m_{N_v})$ where $m_j = 1$ if token $j$ receives at least one vote and $m_j = 0$ otherwise.

#### Straight-Through Estimation (STE)
The mask $\bm{m}$ gives the compact token set we need, but discrete selection has zero gradient with respect to the scores, which prevents the scoring computation from learning. We therefore use the straight-through estimator (STE), which uses hard selection in the forward pass while routing gradients through a differentiable surrogate in the backward pass. We form soft selection probabilities using the Gumbel-softmax relaxation:

$$P_{i,j} = \frac{\exp(\hat{S}_{i,j} / \alpha)}{\sum_k \exp(\hat{S}_{i,k} / \alpha)}$$

where each row of $P$ is a distribution over visual tokens, i.e., $P_{i,j}$ is the relaxed probability that query $i$ would cast its vote for token $j$. Averaging across rows gives a per-token expected selection probability:

$$\bar{p}_j = \frac{1}{N_v} \sum_{i=1}^{N_v} P_{i,j}$$

We combine hard and soft signals with an STE vector:

$$\bm{w} = \bm{m} + \bar{\bm{p}} - \text{sg}(\bar{\bm{p}})$$

where $\text{sg}(\cdot)$ denotes stop-gradient. In the forward pass, $\bm{w}$ evaluates to $\bm{m}$, while in the backward pass the gradient satisfies $\nabla_\theta \bm{w} = \nabla_\theta \bar{\bm{p}}$ because $\bm{m}$ and $\text{sg}(\bar{\bm{p}})$ do not contribute gradients. The conditioned visual tokens are $H_v^{\text{cond}} = \text{diag}(\bm{w}) H_v$, which scales each token by its STE weight before retaining the selected subset in the forward computation. The dominant selector computation scales as $\mathcal{O}(N_v^2 D)$.

#### Global context token
We append a global context token that preserves coarse scene information:

$$H^{\text{ctx}} = \frac{1}{N_v} \sum_{i=1}^{N_v} H_{v,i}$$

where $H_{v,i} \in \mathbb{R}^D$ denotes the $i$-th token of $H_v$. The final sequence $[H_v^{\text{cond}}; H^{\text{ctx}}]$ is forwarded to the VLM alongside proprio and language tokens.

### 3.4 Training Objective
The VLM processes the compact token sequence to produce fused conditioning features:

$$C = f_{\text{VLM}}\left( [H_v^{\text{cond}}; H^{\text{ctx}}; H_l; H_p] \right)$$

We train the action head with flow matching (Lipman et al., 2023). We sample a continuous flow time $T \sim \text{Unif}(0, 1)$ and construct a noised action chunk:

$$\bm{a}_{t:t+\mathcal{H}}^T = (1-T)\bm{a}_{t:t+\mathcal{H}} + T \epsilon$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. The action head conditions on $C$ through cross-attention and also conditions on $T$ as a global diffusion-time signal. Following FLOWER, we embed $T$ and inject it into the action Transformer via global AdaLN-style modulation while cross-attending to $C$.

Under the linear interpolation above, the target velocity field is $\epsilon - \bm{a}_{t:t+\mathcal{H}}$. The action head predicts $v_\theta = f_{\text{ACT}}(\bm{a}_{t:t+\mathcal{H}}^T, T, C)$, and we optimize:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{T, \epsilon, \bm{a}} \left[ \| v_\theta(\bm{a}_{t:t+\mathcal{H}}^T, T, C) - (\epsilon - \bm{a}_{t:t+\mathcal{H}}) \|^2 \right]$$

The action head receives proprioception through the cross-attention over $C$ since proprioception is tokenized and integrated into the VLM input.

---

## 4. Main Results

We evaluate ThinkProprio on two questions:
1. Does ThinkProprio improve long-horizon task performance over strong baselines on CALVIN and LIBERO?
2. Can ThinkProprio reduce inference cost (latency and VRAM) without degrading performance?

### 4.1 Experimental Setup
* **Benchmarks**: We evaluate on two simulation benchmarks. CALVIN (Mees et al., 2022) requires completing chains of five language-conditioned tasks. We use the $ABC \to D$ split by training on environments A, B, and C and testing on the unseen environment D. We report success rate at chain lengths from 1 to 5, noted as LH-1 to LH-5, and the average completed chain length (Avg. Len.). LIBERO (Liu et al., 2023) comprises four suites that probe complementary generalization axes: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. We report the average success rate across the 10 tasks in each suite.
* **Proprioceptive state details**: The proprioceptive state follows each benchmark’s robot interface. For CALVIN, we use a 15-dimensional state consisting of tool-center-point position and orientation in world coordinates, seven arm joint angles, gripper opening width, and a binary gripper action indicator. For LIBERO, we use a 9-dimensional state consisting of seven joint angles and a two-dimensional gripper position. We text-tokenize proprioception by discretizing each dimension into 256 bins over the range $[-3, 3]$.
* **Implementation and measurement**: We implement ThinkProprio on top of FLOWER (Reuss et al., 2025), which uses Florence-2-Large as the vision-language model. The action head is a DiT with hidden dimension 1024, 18 layers, and 16 attention heads, with dropout 0.1 throughout the transformer. We train for 50k steps with AdamW using learning rate $2 \times 10^{-5}$, $\beta = (0.9, 0.95)$, and weight decay 0.05. The noise scale $\alpha_t$ is cosine-annealed from 1.0 to 0.01. We measure per-timestep inference latency and peak VRAM during evaluation on a single RTX 4090 GPU in bfloat16. We report results averaged over 5 independent evaluation runs.

### 4.2 Baselines
We compare against:
1. **Single-system VLAs**: OpenVLA (Kim et al., 2025b), which decodes actions autoregressively as discrete tokens.
2. **Dual-system VLAs**: GR-1 (Wu et al., 2024), RoboFlamingo (Li et al., 2023), $\pi_0$ (Black et al., 2024a), $\pi_{0.5}$ (Intelligence et al., 2025), and FLOWER (Reuss et al., 2025).
3. **Visual planning / predicting baselines**: SuSIE (Black et al., 2024b), VPP (Hu et al., 2025), and Seer (Tian et al., 2024).

For LIBERO, we also report OpenVLA-OFT (Kim et al., 2025a), COA-VLA (Li et al., 2025), and LightVLA (Jiang et al., 2025).

### 4.3 Task Performance
On CALVIN $ABC \to D$ (Table 2), success decreases as the chain length increases, consistent with compounding errors. ThinkProprio achieves the best Avg. Len. (4.55), narrowly improving over FLOWER† (4.53) which uses external pretraining. The gain is most pronounced at the longest horizon: at LH-5, ThinkProprio reaches 82.1% versus 77.8% for FLOWER† (an ~19% relative reduction in failure rate). This supports the intuition that proprio-guided token selection helps retain interaction-relevant visual evidence as the robot configuration evolves.

Table 3 shows that ThinkProprio is competitive with the strongest LIBERO baselines. LightVLA achieves the best overall average (97.4%), while ThinkProprio is close (97.3%) and achieves the best performance on LIBERO-Long (95.2%). Relative to FLOWER, ThinkProprio improves LIBERO-Long from 94.9% to 95.2%, matching the CALVIN observation that proprioceptive guidance is most helpful in extended environments.

### 4.4 Computational Efficiency
Table 4 compares inference cost on CALVIN $ABC \to D$. ThinkProprio keeps only 15 of the 100 available visual tokens per step on average, which shortens the effective sequence. Despite the additional selector, ThinkProprio achieves lower end-to-end latency (22 ms vs. 52 ms for FLOWER). Peak VRAM increases slightly relative to FLOWER (1899 MB vs. 1848 MB) due to selector parameters, but remains far below OpenVLA (14,574 MB). 

Table 5 decomposes total latency into vision encoding, selector, VLM inference, and the diffusion action head, reporting token counts as kept/available (15/100 on CALVIN and 6/34 on LIBERO). The selector remains fast in this regime because it scores short token sets (100 or 34 tokens), while the diffusion action head dominates runtime.

---

## 5. Ablation Studies

We ablate two key design choices: the proprioception encoding and entry point, and the token-retention query signal.

### 5.1 Component Ablation
Table 6 reports a controlled incremental ablation. Starting from FLOWER, we first add text-tokenized proprioception to the VLM input, then introduce physically grounded token selection to select visual patches before the VLM, and finally append the global context token $H^{\text{ctx}}$ that summarizes the full visual input.

Adding proprio tokens slightly improves Avg. Len. from 4.44 to 4.48 while modestly increasing latency from 52 to 55 ms. Introducing physically grounded selection sharply reduces latency to 20 ms by retaining a smaller visual token set, but it also reduces Avg. Len. to 4.35, which suggests that selection without an additional global summary can drop useful scene context. Appending $H^{\text{ctx}}$ restores and improves performance to 4.55 with 22 ms latency, indicating that the global context token complements the selected patches by preserving coarse layout information.

### 5.2 Proprioceptive Encoding and Entry Point
Table 7 isolates proprioceptive integration by varying the encoding scheme and where state enters the model, while keeping the backbone and action head fixed:
1. **MLP-to-ACT**: Maps the continuous state vector directly into the action head and uses AdaLN to modulate generation.
2. **MLP-to-VLM**: Projects the state into a token that is concatenated with vision and language tokens at the VLM input.
3. **Text-to-VLM**: Discretizes the continuous state into bins and retrieves corresponding token embeddings from the VLM vocabulary, reaching the action head through cross-attention over VLM outputs.

The baseline taking vision and language as input already achieves 4.44, consistent with images encoding many implicit state cues. MLP-to-ACT matches the no-proprio baseline at 4.44, suggesting limited benefit from late global modulation. Injecting an MLP-projected state token at the VLM input degrades performance to 4.15, consistent with a distribution mismatch relative to pretrained token embeddings. By contrast, text tokenization with VLM entry achieves the best result at 4.48, indicating that using the VLM’s native token interface is a more effective way to make state available for downstream action generation. Although discretization sacrifices some numerical precision, it offers two advantages: the resulting embeddings lie in a space the VLM was pretrained to process, and the proprioceptive tokens participate directly in the VLM’s reasoning.

### 5.3 Vision Token Retention
Table 8 compares pooling-based compression against query-guided retention. Pooling methods reduce tokens without instruction or proprioception dependent queries. Mean pooling averages all tokens into one, max pooling keeps the top-$K$ tokens by a fixed score, and random sampling uniformly draws $K$ tokens. Query-guided retention scores visual tokens against guidance tokens $H_q$ and retains those selected by the voting mechanism (with $H^{\text{ctx}}$ appended in all query-guided variants).

Among pooling baselines, max pooling achieves 4.33 with 25% tokens. In query-guided retention, using only instruction guidance ($H_q = H_l$) or only proprioceptive guidance ($H_q = H_p$) performs poorly, which is consistent with selecting primarily object-related or robot-related vision patches. In contrast, combining both signals ($H_q = [H_l; H_p]$) achieves 4.55 with only 15% tokens. These results suggest that physically grounded guidance from instruction and proprioception can help sustain long-horizon performance under aggressive token selection.

---

## 6. Analysis

### 6.1 Visualizing Physically Grounded Token Retention
To examine what visual evidence the physically grounded conditioning mechanism preserves, we project token-level retention scores onto the input images as heatmaps. Figure 3 shows two representative tasks over four timesteps each, with paired views from a static camera and a wrist-mounted gripper camera. The overlay in each frame (e.g., Sel: 17/100) reports how many of the 100 available visual tokens are retained at that step. For each view, we label whether the retained tokens are predominantly object-centric, proprioception-centric, or reflect both foci. Across both tasks, the mechanism typically retains only 10% to 20% of tokens while still covering the interaction-relevant region, and the attended evidence transitions smoothly between object-centric and proprioception-centric focus as manipulation progresses.

* **When does proprioceptive guidance matter?** We find that the balance between object-centric and proprioception-centric tokens depends on task phase. During approach, the selector emphasizes object tokens such as the drawer handle or the target block to support localizing the interaction target. During contact and manipulation, the focus shifts toward the gripper and arm, consistent with greater reliance on proprioceptive grounding under contact dynamics. Overall, this phase-dependent pattern indicates that joint instruction-proprioception guidance can modulate which visual evidence is prioritized based on task context, whereas instruction-only selection lacks this pathway.

### 6.2 Recovery in Long-Horizon Tasks
We inspect challenging episodes to characterize policy behavior when task success requires extended correction. Figure 4 shows a stacking sequence that succeeds only after more than 300 steps. In the early frames, the policy picks up the pink block and places it roughly above the red support. Because the red block is smaller than the pink block, the placement induces slipping and tilting that prevents immediate success. In later frames, the selector heatmap remains concentrated on the pink block, gripper, and arm as the policy executes repeated micro-adjustments to stabilize the stack. This example suggests that even when the selector attends to the correct interaction region, completion can depend on contact dynamics that require many corrective steps or may exceed typical evaluation horizons.

---

## 7. Conclusion

ThinkProprio treats proprioception as a first-class modality throughout the pipeline and improves task performance on both CALVIN and LIBERO, with the largest gains in long-horizon evaluation. The strong vision-language baseline in our ablations suggests that images already provide many implicit state cues, while proprioception can provide additional grounded information when integrated properly.

Additionally, we studied how proprioception should enter, be encoded in, and condition vision-language-action pipelines. In controlled ablations, discretizing proprioception into text tokens and mapping it into VLM token embeddings consistently outperformed MLP-based projectors, and physically grounded token selection guided by the instruction and proprioception enabled aggressive visual token reduction. Our real-world experiments in Appendix D provide preliminary validation that the approach transfers to physical hardware, though comprehensive real-world evaluation remains future work.

---

## References

1. Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
2. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L. X., Tanner, J., Vuong, Q., Walling, A., Wang, H., and Zhilinsky, U. $\pi_0$: A vision-language-action flow model for general robot control, 2024a.
3. Black, K., Nakamoto, M., Atreya, P., Walke, H. R., Finn, C., Kumar, A., and Levine, S. Zero-shot robotic manipulation with pre-trained image-editing diffusion models. In The Twelfth International Conference on Learning Representations, 2024b.
4. Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.
5. Dasari, S., Mees, O., Zhao, S., Srirama, M. K., and Levine, S. The ingredients for robotic diffusion transformers. arXiv preprint arXiv:2410.10088, 2024.
6. Din, M. U., Akram, W., Saoud, L. S., Rosell, J., and Hussain, I. Vision language action models in robotic manipulation: A systematic review. arXiv preprint arXiv:2507.10672, 2025.
7. He, C., Camps, G. S., Liu, X., Schwager, M., and Sartoretti, G. Latent theory of mind: A decentralized diffusion architecture for cooperative manipulation. arXiv preprint arXiv:2505.09144, 2025a.
8. He, C., Liu, X., Camps, G. S., Sartoretti, G., and Schwager, M. Demystifying diffusion policies: Action memorization and simple lookup table alternatives, 2025b.
9. Hou, Z., Zhang, T., Xiong, Y., Duan, H., Pu, H., Tong, R., Zhao, C., Zhu, X., Qiao, Y., Dai, J., and Chen, Y. Dita: Scaling diffusion transformer for generalist vision-language-action policy. arXiv preprint arXiv:2503.19757, 2025.
10. Hu, Y., Guo, Y., Wang, P., Chen, X., Wang, Y.-J., Zhang, J., Sreenath, K., Lu, C., and Chen, J. Video prediction policy: A generalist robot policy with predictive visual representations. In Forty-second International Conference on Machine Learning, 2025.
11. Huang, H., Liu, F., Fu, L., Wu, T., Mukadam, M., Malik, J., Goldberg, K., and Abbeel, P. Otter: A vision-language-action model with text-aware visual feature extraction. arXiv preprint arXiv:2503.03734, 2025.
12. Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Galliker, M. Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., and Zhilinsky, U. $\pi_{0.5}$: a vision-language-action model with open-world generalization, 2025.
13. Jiang, T., Jiang, X., Ma, Y., Wen, X., Li, B., Zhan, K., Jia, P., Liu, Y., Sun, S., and Lang, X. The better you learn, the smarter you prune: Towards efficient vision-language-action models via differentiable token pruning. arXiv preprint arXiv:2509.12594, 2025.
14. Kim, M. J., Finn, C., and Liang, P. Fine-tuning vision-language-action models: Optimizing speed and success, 2025a.
15. Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E. P., Sanketi, P. R., Vuong, Q., et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning, pp. 2679–2713. PMLR, 2025b.
16. Li, J., Zhu, Y., Tang, Z., Wen, J., Zhu, M., Liu, X., Li, C., Cheng, R., Peng, Y., Peng, Y., and Feng, F. Coa-vla: Improving vision-language-action models via visual-textual chain-of-affordance, 2025.
17. Li, Q., Liang, Y., Wang, Z., Luo, L., Chen, X., Liao, M., Wei, F., Deng, Y., Xu, S., Zhang, Y., et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.
18. Li, X., Liu, M., Zhang, H., Yu, C., Xu, J., Wu, H., Cheang, C., Jing, Y., Zhang, W., Liu, H., Li, H., and Kong, T. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.
19. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling, 2023.
20. Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.
21. Mees, O., Hermann, L., Rosete-Beas, E., and Burgard, W. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.
22. Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.
23. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., and Hsieh, C.-J. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021.
24. Reuss, M., Yağmurlu, Ö. E., Wenzel, F., and Lioutikov, R. Multimodal diffusion transformer: Learning versatile behavior from multimodal goals. arXiv preprint arXiv:2407.05996, 2024.
25. Reuss, M., Zhou, H., Rühle, M., Yağmurlu, O. E., Otto, F., and Lioutikov, R. Flower: Democratizing generalist robot policies with efficient vision-language-flow models. In Lim, J., Song, S., and Park, H.-W. (eds.), Proceedings of The 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 3736–3761. PMLR, 2025.
26. Ryoo, M. S., Piergiovanni, A., Arnab, A., Dehghani, M., and Angelova, A. Tokenlearner: What can 8 learned tokens do for images and videos? arXiv preprint arXiv:2106.11297, 2021.
27. Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.
28. Tian, Y., Yang, S., Zeng, J., Wang, P., Lin, D., Dong, H., and Pang, J. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2024.
29. Wu, H., Jing, Y., Cheang, C., Chen, G., Xu, J., Li, X., Liu, M., Li, H., and Kong, T. Unleashing large-scale video generative pre-training for visual robot manipulation. In The Twelfth International Conference on Learning Representations, 2024.
30. Yue, Y., Wang, Y., Kang, B., Han, Y., Wang, S., Song, S., Feng, J., and Huang, G. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems, 37:56619–56643, 2024.
31. Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.

---

## Appendix A: Training and Inference Details

### A.1 Observations and Proprioceptive State
Image preprocessing follows the dataset transform configs. For CALVIN, both views are resized to $224 \times 224$. During training, we apply `RandomShiftsAug` with padding 10 for `rgb_static` and padding 4 for `rgb_gripper`, then scale images to $[0, 1]$ and normalize with CLIP statistics. Validation disables `RandomShiftsAug` but keeps resize and normalization. For LIBERO, we use the same augmentation/normalization structure, resizing images to $112 \times 112$.

For CALVIN, the proprioceptive state is `robot_obs` (15D), normalized using dataset statistics and with additional normalization of orientation entries. For LIBERO, we form a 9D proprio state by concatenating 7 joint values with the 2D gripper state. In both benchmarks, actions are 7D relative actions (`rel_actions`) scaled to $[-1, 1]$.

When ThinkProprio tokenizes proprioception for selection, each scalar is clipped to $[-3, 3]$ and discretized into 256 uniform bins. Bin indices are mapped to the last 256 token ids in the VLM vocabulary, and the corresponding embeddings are retrieved from the VLM input embedding table.

### A.2 Pre-VLM Visual Token Selection Implementation
Our pre-VLM selector takes vision tokens $H_v \in \mathbb{R}^{B \times V \times D}$ and query embeddings (instruction and/or tokenized proprioception) and produces a shorter vision sequence for the VLM encoder. It generates per-vision queries and computes a score tensor of shape $[B, V, V]$, where each of the $V$ queries casts a vote over the same $V$ candidate vision tokens. During training, it adds Gumbel noise with an annealed scale, applies a softmax to obtain a differentiable voting distribution, and takes an argmax over the noisy scores to form hard votes. A vision token is kept if it receives at least one vote, and we use a straight-through indicator of the form $\bm{w} = \bm{m} + \bm{p} - \text{sg}(\bm{p})$ so that the forward pass uses hard retention while gradients flow through the soft indicator.

Importantly, unkept tokens are removed rather than zeroed. For each batch element, we gather the kept tokens into a variable-length sequence, multiply them by the corresponding indicator weights during training, pad within the batch to the maximum kept length $M$, and return an attention mask that marks real versus padded positions. We also append a learned global context token computed from the mean of the original vision tokens. Compute savings therefore come from running the VLM encoder and the action head on the shorter effective length $M$ (with masking), not from masking dense sequences. The selector’s dominant cost scales as $\mathcal{O}(V^2 D)$, which is small compared with dense transformer computations in VLM and action head.

### A.3 Model Architecture
We build on FLOWER and use Florence-2-Large (`microsoft/Florence-2-large`) as the vision-language backbone. We fine-tune the model rather than freezing Florence parameters. A special token `<Flow>` is embedded and inserted to mark the conditioning boundary. We apply token dropout with probability 0.1 to the VLM encoder outputs during training.

The action generator is a rectified-flow / Diffusion Transformer. It uses hidden size 1024, 18 transformer layers, and 16 attention heads, with dropout 0.1 in attention, residual, and MLP blocks. The policy predicts an action chunk of length 10 (`act_window_size=10`), and executes it with chunked replanning every 10 environment steps (`multistep=10`).

### A.4 Optimization Hyperparameters
We train end-to-end with AdamW using learning rate $2 \times 10^{-5}$, betas $(0.9, 0.95)$, and weight decay 0.05 applied to non-normalization and non-bias parameters (norm/bias parameters use zero weight decay). Training uses bf16-mixed precision. We use a tri-stage learning-rate schedule over 50k steps (matching `max_epochs=50` and `limit_train_batches=1000`):
1. **Warmup**: Linear warmup for 5% of steps from $0.1 \times \text{lr}$ to $\text{lr}$.
2. **Hold**: Constant hold for 10% of steps.
3. **Decay**: Cosine decay for the remaining 85% to a final learning rate of $0.5 \times \text{lr}$.

We additionally maintain an exponential moving average (EMA) of parameters with decay 0.999 and use EMA weights for evaluation.

### A.5 Inference Settings and Measurement Protocol
At inference time, we run rectified-flow sampling with 4 steps (`num_sampling_steps=4`) per action chunk. The model predicts a 10-step chunk and replans every 10 environment steps. Inference is performed in bf16 and always uses both camera views.

CALVIN long-horizon evaluation follows the standard 5-subtask instruction-chain protocol: each subtask is given up to 360 environment steps, and we report success rates for completing 1 through 5 subtasks as well as the average successful sequence length. LIBERO evaluation follows the standard success-rate protocol over fixed rollouts per task with a maximum horizon of 520 environment steps; we report per-task success and suite averages.

For latency and VRAM profiling, we measure per-environment-step end-to-end inference time including vision encoding, the pre-VLM selector, the Florence encoder forward pass, and all diffusion sampling steps. We compute latency with CUDA synchronization to avoid asynchronous kernel overlap artifacts, and report the mean over a fixed number of inference steps (1000 steps in our efficiency tables). Peak VRAM is reported as the maximum allocated GPU memory during inference. All profiling numbers in the paper are collected on a single RTX 4090 GPU under the same bf16 and two-view settings as evaluation.

---

## Appendix B: Additional Results for CALVIN Benchmark

### B.1 Baseline Details
For the extended tables in this appendix, we report a small set of architectural attributes and group methods into three families to reduce sparsity: Single-system VLAs (a single model directly predicts actions), Dual-system VLAs (a VLM conditions a separate action generator), and Non-VLM planner/predictor baselines. For proprioception, we summarize the encoding (Enc.), entry point (Entry; VLM vs. action head), and the conditioning interface into the action head (Cond.; e.g., AdaLN vs. cross-attention). Tokens reports the number of visual tokens per timestep when explicitly specified. We use “–” when not applicable or not reported.

- **FLOWER** (Reuss et al., 2025) uses Florence-2-Large and conditions a flow-based action generator on the backbone’s intermediate vision-language token sequence via cross-attention. This dense token conditioning preserves fine-grained spatial and semantic information, but its compute scales with the number of conditioning tokens processed by the VLM and consumed by the action head. In the simulated settings reported in Reuss et al. (2025), including CALVIN $ABC \to D$, FLOWER does not explicitly feed proprioception into the backbone; therefore, the proprioception fields are marked as “–” for FLOWER in our extended tables. FLOWER† injects a continuous proprio embedding into the action head via AdaLN.
- **Diff-P-CNN** (Diffusion Policy CNN; Chi et al. 2025) is a non-VLM diffusion policy that denoises actions conditioned on visual observations (two RGB views) and the instruction. In our taxonomy, it serves as a CNN-based diffusion baseline without an explicit VLM backbone or a token-level cross-attention interface between a VLM and a separate action head.
- **GR-1** (Wu et al., 2024) is a video-generative-pretrained causal Transformer policy. It encodes language with CLIP and images with an MAE-pretrained ViT, reduces patch tokens with a Perceiver resampler, and injects robot state via an MLP as part of the input token sequence. Actions are generated in-context from the sequence, so proprioception conditions control via self-attention rather than a separate action-head conditioning interface.
- **DeerVLA** (Yue et al., 2024) introduces dynamic early-exit for Flamingo-style multimodal LLM backbones to trade compute for performance. It pools the backbone hidden states and predicts actions with a lightweight temporal head (e.g., LSTM+MLP). The paper does not specify a consistent proprioception for the CALVIN results we cite, so we mark the proprioception columns as “–” when not explicitly stated.
- **$\pi_0$ and $\pi_{0.5}$** (Black et al., 2024a; Intelligence et al., 2025) are dual-system VLAs that combine a PaliGemma VLM backbone with a flow-based action generator. In the full $\pi_{0.5}$ system, hierarchical inference is a key ingredient for long-horizon real-home tasks: it predicts a high-level semantic subtask as text and conditions low-level action generation on the predicted subtask. This hierarchical component is not implemented in the open-sourced $\pi_{0.5}$ variant we use. We therefore focus on the architectural distinction: how proprioception is represented and where it enters the network. We fine-tune $\pi_0$ and $\pi_{0.5}$ ourselves on CALVIN task $ABC \to D$, and we reuse the reported results on LIBERO.
- **OpenVLA** (Kim et al., 2025b) is a 7B-scale open-source VLA that uses a Llama 2 language model backbone and a two-branch vision encoder combining DINOv2 and SigLIP features. OpenVLA is a single-system VLA that decodes actions autoregressively as discrete tokens. OpenVLA-OFT (Kim et al., 2025a) keeps the same OpenVLA backbone but introduces an optimized fine-tuning recipe that changes the action interface to improve speed and success, combining parallel decoding, action chunking, and a continuous action representation with a simple regression objective.
- **RoboFlamingo** (Li et al., 2023) adapts the Flamingo/OpenFlamingo vision-language fusion mechanism for robotic control. For the CALVIN numbers we report, the paper does not provide a consistent description of proprioception integration, so we mark the proprioception columns (Enc./Entry/Cond.) as “–” when not explicitly stated.
- **SuSIE** (Black et al., 2024b) is a planning-style baseline that uses intermediate predictions to structure multi-step behavior. We mark the unreported attributes in our extended tables are marked as “–”.
- **VPP** (Hu et al., 2025) is a visual planning approach built on generative video prediction components. We mark non-applicable or unreported entries as “–”.
- **Seer** (Tian et al., 2024) is a predictive/planning baseline that uses intermediate predictions to guide control. Its CALVIN reporting does not provide a consistent mapping to our VLM-conditioning and proprio-injection fields, so we mark those attributes as “–” when unspecified.

### B.2 CALVIN Long-Horizon Success Metrics
Tables 10–12 report CALVIN long-horizon performance for $ABC \to D$, $ABCD \to D$, and $D \to D$. We report LH-1–LH-5, where LH-$k$ is the success rate (%) of completing $k$ consecutive subtasks in the five-subtask evaluation chain, and Avg. Len. is the mean number of consecutively completed subtasks. Table 10 additionally includes architecture and proprioception-interface details.

In the $ABCD \to D$ setting (more training data than $ABC \to D$), ThinkProprio achieves an Avg. Len. of 4.74, compared to 4.62 for FLOWER and 4.67 for FLOWER†. The advantage is consistent at longer chains, where ThinkProprio reaches 88.5 on LH-5 versus 88.3 for FLOWER†.

In $D \to D$, ThinkProprio achieves an Avg. Len. of 4.23 and 72.7 on LH-5, which is slightly below FLOWER† (4.35 Avg. Len., 74.9 on LH-5). This comparison is not training-matched, since we train ThinkProprio on split D only, whereas FLOWER† uses additional multi-dataset pretraining before CALVIN fine-tuning (Appendix A (Pretraining Details) of Reuss et al. (2025)). The performance gap is therefore consistent with differences in pretraining and data diversity rather than isolating the effect of our method. Under the non-pretrained split-D setting, ThinkProprio still exceeds MDT (3.72 Avg. Len.) and RoboUniView (3.85 Avg. Len.), indicating a benefit over prior non-pretrained baselines in this regime.

Figure 5 breaks down performance by subtask for task $ABC \to D$ and shows that most subtasks achieve success rates close to 1.00. The remaining failures concentrate in a smaller set of subtasks that fall toward the lower end of the plotted range near 0.75. These lower-scoring cases appear predominantly in Push, Rotate, and Slider-related categories, suggesting these interactions are less uniformly reliable than the others.

### B.3 Failed Case Analysis: Push Pink Block Right
We analyze Push Pink Block Right because it is one of the least reliable subtasks in the per-subtask breakdown: its success rate falls below 80% and is the second-lowest among the 34 CALVIN subtasks in Figure 5. Figure 6 visualizes a representative failure using token-retention heatmaps over both camera views, making it possible to inspect how attention over interaction-relevant regions evolves during execution.

A recurring pattern is that the gripper reaches the vicinity of the block but hesitates to commit to firm contact, or produces only a brief, shallow push before drifting away and re-approaching, rather than sustaining a consistent rightward push over multiple steps. This suggests that even visually simple pushing subtasks can be long-horizon and contact-sensitive: progress depends on maintaining contact while continuously correcting direction and force, yet the task provides no discrete closure event (unlike grasping) to stabilize behavior. Consistent with this interpretation, Figure 5 shows that multiple pushing subtasks remain noticeably below ceiling, which may reflect the difficulty of learning persistent, progress-accumulating behaviors when success is only weakly indicated until a final spatial condition is met.

---

## Appendix C: Additional Results for the LIBERO Benchmark

### C.1 Overview
LIBERO (Liu et al., 2023) is a simulated benchmark for language-conditioned imitation learning on tabletop manipulation with a Franka Panda arm. It is organized into four suites that target complementary forms of generalization:
- **LIBERO-Spatial**: varies spatial relations.
- **LIBERO-Object**: introduces novel objects.
- **LIBERO-Goal**: evaluates transfer across goal specifications.
- **LIBERO-Long**: focuses on long-horizon multi-stage tasks.

Each suite contains 10 tasks, and each task has 50 demonstrations. Observations include two RGB views from a static camera `agentview_rgb` and a wrist camera `eye_in_hand_rgb`, together with low-dimensional robot state. In our setup, proprioception is a 9D vector consisting of 7 joint angles and a 2D gripper state, and actions are 7D relative actions from `rel_actions`. 

LightVLA (Jiang et al., 2025) targets efficient VLA inference by pruning visual tokens with instruction-guided differentiable selection before attention-heavy computation. It builds on OpenVLA-OFT and adds a lightweight selector that uses the instruction as the query signal to score visual tokens and retain a subset for downstream processing.

### C.2 Results
Figure 7 reports a task-level breakdown that makes clear where suite averages hide meaningful variation. In our setup, the selector operates over $H_v = 34$ available visual tokens per step on LIBERO, and we report token counts as kept/available. Averaged over suites, we keep 5/34 (Spatial), 7/34 (Object), 3/34 (Goal), and 9/34 (Long), which matches the observation that the Long suite benefits from retaining more interaction-relevant evidence.

Spatial and Goal tasks are often near ceiling, so differences among strong methods are necessarily small and mainly appear in a small set of remaining failure cases. In contrast, Long shows a much wider performance spread, because failures more often arise from compounding errors over multi-stage execution, progressing from approach to grasp or contact, then manipulation, and finally placement. This long-horizon regime is where proprioceptive grounding matters most: as the robot’s configuration changes during interaction, the selector must continue to prioritize the end-effector neighborhood, contact region, and target object, rather than drifting toward visually salient but control-irrelevant regions. Consistent with this view, gains in suite-level averages are typically driven by improvements on a subset of harder long-horizon tasks, while performance on already-solved tasks remains essentially unchanged. Table 13 reports the corresponding extended summary in the same format as our CALVIN extended table, including suite-level success rates together with high-level architecture attributes and the proprioception interface.

---

## Appendix D: Real-World Experiments

### D.1 Hardware Setup and Sensing
As shown in Figure 8, we evaluate on a real tabletop setup using an xArm robotic arm with a parallel-jaw gripper. The policy observes two synchronized RGB streams from Intel RealSense D455 cameras. One camera provides a third-person static view of the workspace, and the other is rigidly mounted on the gripper to provide a wrist view. This two-view sensing matches our simulated interface with static and gripper views, and it reduces occlusions during grasping and placement.

We evaluate a compact task suite with two categories, Pick-Place and Push, as listed in Table 14. All tasks are single-instruction, single-goal tabletop manipulation with common objects and simple receptacles or targets, including plates and bowls. We collect 50 real-world trajectories via teleoperation across the suite and fine-tune from a checkpoint pretrained on the CALVIN benchmark. We keep the same observation interface, which combines the two RGB views with proprioception, and we keep the same action parameterization as in simulation to minimize real-to-sim interface mismatch.

Our real-world evaluation is intentionally limited in scope and is not designed to provide a comprehensive measure of real-world performance. Since the underlying base model, FLOWER, is already strong on real-world tasks, we expect our approach to inherit much of this capability, although we do not claim broad real-world generalization from the limited evidence presented here. The main focus of this paper is improving the efficient use of proprioceptive information rather than maximizing scores on real-world benchmarks. We therefore include the real-world task primarily as a sanity check that verifies the full system can run end to end and produce coherent behavior under real sensor inputs and actuation in a physical setting. This evaluation serves to validate feasibility and basic robustness, while leaving extensive real-world benchmarking to future work.

### D.2 Results
Table 15 reports success rates for FLOWER and ThinkProprio on the real-world suite. We run 20 trials per task and aggregate results over four tasks per category, which gives 80 trials per category and 160 trials overall. Both methods perform reliably on this suite, and ThinkProprio improves over FLOWER in both Pick-Place and Push. This margin is consistent with the idea that state-aware evidence selection is most helpful when execution depends on maintaining contact and recovering from small deviations. Figure 9 provides representative rollouts, and more challenging variants with heavier occlusions, tighter placement tolerances, or longer multi-stage tasks may further differentiate methods.

Figure 9 shows representative real-world rollouts for both FLOWER and ThinkProprio from the third-person static camera and the wrist-mounted gripper camera. The examples cover both Pick-Place and Push tasks. They illustrate the progression from approach to grasp or contact and then to placement or object displacement, and they show how the static and wrist views together preserve visibility of the end-effector and target objects when one view is occluded.

---

## 📊 Summary of Main Results (Table Compilation)

Below is the consolidated set of quantitative results tables referenced throughout the paper:

### Table 2: CALVIN $ABC \to D$ Multi-Step Success Rates (%) and Average Length
| Method | LH-1 ↑ | LH-2 ↑ | LH-3 ↑ | LH-4 ↑ | LH-5 ↑ | Avg. Len. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OpenVLA | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| GR-1 | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| RoboFlamingo | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.47 |
| $\pi_0^*$ | 70.0 | 48.0 | 37.0 | 28.0 | 18.0 | 2.01 |
| $\pi_{0.5}^*$ | 71.0 | 56.0 | 45.0 | 37.0 | 29.0 | 2.38 |
| SuSIE | 87.0 | 69.0 | 49.0 | 38.0 | 26.0 | 2.69 |
| VPP | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| Seer | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.29 |
| FLOWER | 99.3 | 96.0 | 90.3 | 82.3 | 75.5 | 4.44 |
| FLOWER† | 99.4 | 95.8 | 90.7 | 84.9 | 77.8 | 4.53 |
| **ThinkProprio (Ours)** | **97.7** | **96.1** | **92.2** | **86.7** | **82.1** | **4.55** |

### Table 3: LIBERO Generalization Success Rates (%) across Suites
| Method | Spatial ↑ | Object ↑ | Goal ↑ | Long ↑ | Average ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| COA-VLA | 85.3 | 93.1 | 85.8 | 55.0 | 79.8 |
| LightVLA | **98.4** | 98.4 | 98.2 | 94.6 | **97.4** |
| FLOWER | 97.5 | 99.1 | 96.1 | 94.9 | 96.9 |
| **ThinkProprio (Ours)** | 97.6 | **98.4** | **98.0** | **95.2** | **97.3** |

### Table 4: Latency & VRAM Profiling (RTX 4090 GPU, bf16)
| Method | Visual Tokens ↓ | End-to-End Latency (ms) ↓ | Peak VRAM (MB) ↓ | CALVIN Avg. Len. ↑ |
| :--- | :---: | :---: | :---: | :---: |
| OpenVLA | 256 | 164 | 14,574 | 3.27 |
| $\pi_0$ | 256 | 104 | 6,692 | 2.01 |
| $\pi_{0.5}$ | 256 | 138 | 7,038 | 2.38 |
| FLOWER | 100 | 52 | **1,848** | 4.44 |
| **ThinkProprio (Ours)** | **15** | **22** | 1,899 | **4.55** |

### Table 15: Real-World Robot Manipulation Success Rates (%)
| Method | Pick-Place SR (%) ↑ | Push SR (%) ↑ | Overall Average SR (%) ↑ |
| :--- | :---: | :---: | :---: |
| FLOWER | 88.8 | 86.2 | 87.5 |
| **ThinkProprio (Ours)** | **92.5** | **90.0** | **91.3** |

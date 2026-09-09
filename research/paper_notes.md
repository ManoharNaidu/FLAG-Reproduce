# FLAG Paper Notes — implementation-level extraction

| Field | Value |
|---|---|
| Title | FLAG: Fraud Detection with LLM-enhanced Graph Neural Network |
| Venue | KDD '25, August 3-7 2025, Toronto, ON, Canada. Pages 5150-5160 |
| DOI | `10.1145/3711896.3737220` |
| Canonical URL | https://dl.acm.org/doi/10.1145/3711896.3737220 |
| Open-access PDF | https://dl.acm.org/doi/pdf/10.1145/3711896.3737220 (GOLD OA) |
| Author mirror | http://www.shichuan.org/doc/200.pdf (certificate expired; HTTP works) |
| Authors | Chengdong Yang (BUPT), Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang (BUPT), Chuan Shi (BUPT, corresponding) |
| arXiv | **none found** |
| Code | https://github.com/BUPT-GAMMA/FLAG |

**How to read this file.** Every row is tagged:

- `[PAPER]` — stated in the paper. Quoted where it matters.
- `[CODE]` — established from the official repository at commit `cb83944e`
  (see `flag_code_audit.md`, which is the authority for all `[CODE]` claims).
- `[CONFLICT]` — the paper and the code disagree. **Both values are recorded and
  both are exposed through configuration. Neither is silently preferred.**
- `UNKNOWN` — the paper does not state it. Not guessed.

---

## 1. Datasets

### 1.1 Which

`[PAPER]` Two public (Reddit, Instagram) + one industrial (Huabei, from Alipay).

Why these, verbatim:

> "for the public fraud detection datasets, we note that most of them, such as
> Yelp-Fraud [8], Amazon-Fraud [8], T-Finance [36] and T-Social [36], lack
> textual information... To overcome this limitation, we construct a dataset
> tailored to fraud detection by utilizing existing social network datasets that
> contain textual data. Specifically, we use two social network datasets:
> Reddit [25] and Instagram [25]."

This sentence is the basis of the Phase-9 integrity rule: the authors themselves
say Yelp/Amazon/T-Finance/T-Social have no text, so no run on those datasets may
be labelled canonical FLAG.

### 1.2 Construction

`[PAPER]` Reddit: nodes are users; edges are replies; text is the content of the
user's **last three posts**; labels are **popular vs normal**.
`[PAPER]` Instagram: nodes are users; edges are **following** relationships; text
is the user's **personal introduction**; labels are **commercial vs normal**.

`[PAPER]` Class imbalance:

> "Both datasets originally contain an approximately equal number of nodes in each
> class... we treat the popular category in Reddit and the commercial category in
> Instagram as the minority class. The minority class nodes are randomly selected
> so that the final ratio between the minority and majority classes is about 1:10."

Note the direction: **the minority class is downsampled**, the majority is untouched.

### 1.3 Statistics

**The paper publishes NO dataset statistics table.** Tables 1-2 are prompt
templates; Tables 3-5 are results. Node counts, edge counts, per-class counts and
the shallow-feature dimensionality are all `UNKNOWN` from the paper.

Counts are recoverable from GLBench (Reddit 33,434 / 198,448; Instagram
11,339 / 144,010) — see `dataset_notes.md`. FLAG's **post-downsampling** counts
remain `UNKNOWN` and must be regenerated.

### 1.4 Split protocol

`UNKNOWN`. The paper states only:

> "We follow the experimental setup used in BWGNN [36], using F1-macro and AUC [7]
> as evaluation metrics for public datasets. The model with the highest F1-macro
> score on the validation set is selected for testing on the test set."

No ratio, no random-vs-temporal statement, no per-dataset split.
Our default is **10/10/80**, inherited from GraphAdapter/GLBench and labelled as
such. Rationale in `dataset_notes.md` section 6.2.

`[PAPER]` Homophily analysis (Figure 5): homophily score per node = fraction of
edges connecting to same-class nodes, binned into **ten bins from 0.1 to 1.0**.
`[CODE]` `utils.py:generate_homo` implements a *binary* homophilous/heterophilous
split (`num_same_label > num_neighbors / 2`), not ten bins. The ten-bin version
is not in the repository.

---

## 2. Model and training

| Item | `[PAPER]` | `[CODE]` | Status |
|---|---|---|---|
| layers | **2** — "All models are configured with two layers and a hidden layer size of 64." | 2 for GCN/GAT/CARE/DGA/PMP; **4** for GeniePath (`layer_num = 4`) | `[CONFLICT]` |
| hidden dim | **64** (same sentence) | **32** (`--hidden` default; `x32` guards break at any other value) | `[CONFLICT]` |
| hops | **2-hop** subgraphs, "a training strategy similar to GraphSAGE" | sampler not shipped | consistent |
| top-N | **10** | directory `0_10_0` | consistent |
| similarity threshold | **0** | directory `0_10_0` | consistent |
| optimizer | **Adam**, fixed lr **0.01** | Adam lr 0.01 for the GNN; AdamW lr 1e-4 for LoRA | consistent |
| gradient accumulation | **10 subgraphs** | `accumulation_steps = 10` | consistent (but see `flag_code_audit.md` 5.4) |
| early stopping | "employing early stopping to prevent overfitting"; **criterion and patience UNKNOWN** | **not implemented** — `--patience` parsed, never read | `[CONFLICT]` |
| runs | **25** = 5 random seeds x 5 random initializations | **5** = `range(5)` x `range(1)` | `[CONFLICT]` |
| industrial runs | **1** ("only once... due to the time constraint") | n/a | — |
| weight decay | `UNKNOWN` | 0 in `test.py`; 5e-4 parsed-but-unused in `train.py` | code fills the gap |
| batch size | `UNKNOWN` (only the 10-subgraph accumulation) | one subgraph per step | code fills the gap |
| epochs | `UNKNOWN` | 5 (GNN); 3 outer x 10 inner (LoRA) | code fills the gap |
| dropout | `UNKNOWN` | 0.5 | code fills the gap |

`[PAPER]` Deployment (section 4.1.3): fine-tuned LLM served on **A100 80 GB**;
8 GPUs total across preproduction and online; GNN on a CPU cluster; predictions
"within a day"; monthly online prediction updates.

---

## 3. FLAG components

### 3.1 LLM and text encoder

`[PAPER]`:

> "We use Gemma-9b-it [37] as LLM model with LoRA [18] used for fine-tuning and
> Sentence-BERT [34] as LM."

- The literal string is `Gemma-9b-it`, which is **not a valid HuggingFace id**.
  The exact checkpoint is `UNKNOWN` from the paper.
  `[CODE]` resolves it: `model_name = "gemma-2-9b-it"` -> `google/gemma-2-9b-it`.
- Sentence-BERT checkpoint is `UNKNOWN` from the paper (cited only as [34],
  Reimers 2019). `[CODE]` resolves it: `SentenceTransformer("all-MiniLM-L6-v2")`,
  384-dim — independently corroborated by `models.py:DualGNN` hardcoding
  `Linear(384, out_channels)`.

`[PAPER]` The frozen LM also provides the embeddings used for similarity sampling:
*"with the pre-trained language models B, we first obtain its representation of
the text t_v."*

### 3.2 LoRA

`[PAPER]` **Nothing.** LoRA is named once and cited as [18]. No rank, alpha,
dropout, target modules, LLM learning rate, or step count appears anywhere,
including the appendices. All `UNKNOWN`.

`[CODE]` supplies every one of them:

```python
LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
           lora_dropout=0.1, bias="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

These are used as defaults and are tagged `source: code_not_paper` in config, so
no result ever implies the paper specified them.

### 3.3 Semantic similarity sampling

`[PAPER]` Eq. 3 — cosine similarity between the centre node and a k-hop neighbour:

```
sim(v, u) = (B(t_v) . B(t_u)) / (||B(t_v)|| ||B(t_u)||)
```

`[PAPER]` Eq. 4 — threshold filter, then top-N:

```
N_selected(v) = { u in N(v) : sim(v, u) >= delta }
```

> "we rank the neighboring nodes based on their similarity to the center node v's
> text t_v. We then select the top-N neighbors N_selected(v) with the highest
> similarity that exceed a predefined threshold delta... finally used to form a
> simplified subgraph G_v centered around node v."

`[PAPER]` Eq. 5 — average subgraph edge homophily, used in the Figure-3(a)
motivation study: `h = sum_{(u,v) in E} I(y(u) = y(v)) / |E|`.

`[PAPER]` Sampling strategies compared in Figure 3(a): **NS** (none), **RS**
(random), **FS** (shallow-feature similarity; shallow features = word2vec [30]),
**SS\*** (semantic similarity **without** threshold), **SS** (proposed).

**`[CODE]` has none of this.** The sampler construction is not in the repository
(`flag_code_audit.md` GAP-1). This is the single largest reimplementation task in
the project, and it is a *core contribution* of the paper.

### 3.4 Skip-GNN

`[PAPER]` Eq. 6, exactly as printed:

```
Z = GNN(X, A) + Linear(X)
```

> "we add a direct connection to the node feature X, so that the final node
> representation Z is a combination of both the aggregated features and the node
> features."

`[CODE]` `self.linear1 = Linear(in_channels, out_channels)`; `return ..., x + initial_x`.
**Applied in GCN, GeniePath, GeniePathLazy, BWGNN only.** GAT, CARE-GNN, DGA and
PMP compute `initial_x` and discard it. `[CONFLICT]` — the paper implies the skip
is part of FLAG for every backbone. Which behaviour produced Table 4's `+FLAG`
rows for those four backbones is `UNKNOWN`. Exposed as the `SG` ablation flag.

### 3.5 The three fine-tuning losses

`[PAPER]` Eq. 7 — Discriminative Text Loss (**BCE**):

```
L_Disc = - sum_i [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]
```

`[PAPER]` Eq. 8 — Residual Text Loss (**KL to uniform**):

```
L_Res = sum_i KL(p_hat_i || u_hat_i) = sum_i sum_c p_hat_i(c) log( p_hat_i(c) / u_hat_i(c) )
```

> "u_hat_i is the uniform distribution over the classes (i.e. 1/C)... This loss
> encourages the model to produce text representations that do not over-rely on
> contextual features."

`[PAPER]` Eq. 9 — Orthogonality Loss:

```
L_Orthog = || Z_D . Z_R ||_2^2
```

> "Z_D and Z_R are the representation matrices of the discriminative and residual
> text... This loss ensures that the two text representations are orthogonal."

`[PAPER]` Eq. 10 — total:

```
L = L_Disc + lambda_1 * L_Res + lambda_2 * L_Orthog
```

#### Code mapping and one substantive discrepancy

`[CODE]` uses *causal/non-causal* naming for the same three terms
(`utils.py`, applied in `train.py:155-157`):

| Paper | Code symbol | Code implementation | Match? |
|---|---|---|---|
| `L_Disc` (BCE) | `causal_loss` | `F.cross_entropy` over 2 classes | equivalent for binary |
| `L_Res` (KL to uniform) | `non_causal_loss` | `F.kl_div(log_softmax(x, dim=0), uniform)` | matches at the shape actually used |
| `L_Orthog` = `‖Z_D · Z_R‖²` | `orthogonal_loss` | `sum(normalize(a, dim=0) * normalize(b, dim=0))` — a **signed cosine** | **`[CONFLICT]`** |

**The orthogonality term does not implement Eq. 9.** The paper's squared form is
minimised at a dot product of **0** (true orthogonality). The code returns a raw
signed cosine, minimised at **-1** (anti-alignment). Verified numerically in
`tests/integration/test_flag_upstream_claims.py::test_3_2_orthogonal_loss_is_signed_cosine_not_squared_dot`:
an orthogonal pair scores 0.0, an anti-aligned pair scores -1.0, and the code
prefers the latter.

Both forms are implemented and selectable (`loss.orthogonality: squared_dot | signed_cosine`),
defaulting to the paper's Eq. 9 with the code's behaviour available for
reproducing the released implementation.

#### lambda_1, lambda_2

`[PAPER]` **UNKNOWN.** Only *"lambda_1 and lambda_2 are the hyper-parameters that
control the contribution of each loss term."* They are not in the sensitivity
study either (section 4.5 covers only top-N and delta).

`[CODE]` `--alpha 0.1` (residual) and `--beta 0.1` (orthogonality). So
**lambda_1 = lambda_2 = 0.1** in the released implementation.

`[PAPER]` also contradicts itself: **Appendix-A pseudocode line 14 writes the total
loss without the lambdas** — "Compute the total loss L = L_Disc. + L_Res. + L_Orthog."
That is an internal inconsistency in the paper, recorded here, not resolved.

### 3.6 Two-stage alternating training

`[PAPER]`, verbatim:

> "we employ an alternating two-stage approach. In the first stage, the LLM is
> fixed while the skip-GNN is trained on the generated text and graph structure.
> In the second stage, the GNN is fixed, and the LLM is fine-tuned to better
> distinguish between discriminative and residual text."

`[PAPER]` Training mechanics: two prompts per node produce `t_D` and `t_R`; both
are encoded by a **fixed** LM into `x_D` and `x_R`; both go through **two
shared-parameter skip-GNNs** yielding `z_D` and `z_R`.

`[CODE]` `train.py` matches the *structure* (outer loop = LLM, inner loop = GNN,
GNN pre-loaded from a checkpoint so Stage 1 precedes) but see the blocker below.

### 3.7 Inference

`[PAPER]`:

> "we utilize the fine-tuned LLM to extract discriminative text from each node's
> raw text. Then both the raw text and the discriminative text are encoded into
> text feature vectors using a frozen LM. These encoded features are independently
> processed through two shared-parameter skip-GNN modules. The outputs of the two
> skip-GNN modules are fed into an attention layer."

`[PAPER]` Appendix-A pseudocode: line 18 "Reset skip-GNN parameters";
line 23 `z = Attention(z_raw, z_D)`; line 24 "Update skip-GNN using BCE loss".

`[CODE]` This is exactly `models.py:DualGNN` — one shared `self.gnn` applied to
both branches, fused by a learned `attention_weights` parameter, driven by
`test_dual.py`. Verified in
`tests/integration/test_flag_upstream_claims.py::test_dualgnn_shares_one_backbone_across_both_branches`.

The attention layer's architecture is `UNKNOWN` from the paper ("an attention
layer"); `[CODE]` implements it as a softmax over a single learned weight vector
of shape `[1, out_channels]`.

### 3.8 THE FINE-TUNING BLOCKER

`[CODE]` As shipped, `train.py` **cannot fine-tune the LLM**. The LLM's output is
decoded to strings and re-encoded through Sentence-BERT into a fresh leaf tensor,
so no gradient can reach the LoRA parameters; `optimizer.step()` is a no-op.
Full evidence in `flag_code_audit.md` section 5.5, with a passing test.

The paper describes a coherent procedure. The released code does not implement it.
**We do not know which one produced the `+FLAG*` column of Table 4.**
This is the single largest open question in the reproduction and is escalated to
the maintainer in `reproduction_status.md`.

---

## 4. Prompts

### 4.1 The paper's prompts and the code's prompts are DIFFERENT TEXT

This is a `[CONFLICT]` that is easy to miss.

`[PAPER]` Table 1, discriminative prompt for Reddit, verbatim:

> "You are provided with a list of Reddit users' posts. Each user is classified as
> either popular or normal based on their interactions and content. Your task is to
> generate a brief **discriminative** text for each user that directly relates to
> **distinguishing** them as either popular or normal. Ensure that the generated
> text highlights specific features that help differentiate the user's
> classification while avoiding any general or irrelevant information."

`[CODE]` `chat.py:unique_prompt`, verbatim:

> "Your task is to generate a brief **causal** text for each user that directly
> relates to **classifying** them as either popular or normal. Ensure that the
> generated text is specific to the user and reflects features that help
> distinguish them between these two categories. The posts for each user are
> separated by semicolons. Here is the format for each user's posts: ..."

The paper uses *discriminative/residual*; the code uses *causal/non-causal*, and
the code carries batch formatting instructions the paper's version does not.
**These are not the same prompt.** Both are preserved:

```
prompts/
  reddit/    {system_instruction,global,discriminative,residual}.txt   <- from chat.py
  instagram/ {system_instruction,global,discriminative,residual}.txt   <- from chat1.py
```

extracted verbatim by AST (`scripts/preprocess/extract_prompts.py`) with SHA-256
hashes in `prompts/manifest.json`. The paper's variants will be added as a
separate `prompt_version` so both are runnable and comparable.

`[PAPER]` **No Instagram-specific prompts are shown.** `[CODE]` has them, and the
Instagram discriminative prompt differs from Reddit's in substance, not just
nouns — it asks the model to describe characteristics *"without predicting their
classification"*, the opposite emphasis to Reddit's. Preserved verbatim.

### 4.2 Worked example

`[PAPER]` Appendix C, a Reddit popular user:

- **Original:** "I've been experimenting with a new strategy in Civilization VI... ;
  I'm currently streaming my gameplay of Age of Empires on Twitch... ;
  Just came back from a hiking trip to the Rocky Mountains..."
- **Discriminative:** "This user frequently shares unique game strategies for
  Civilization VI and streams Age of Empires on Twitch, where they engage with a
  large community of viewers."
- **Residual:** "This user shares personal experiences, such as their hiking trip
  to the Rocky Mountains and the landscape photos they took."

The `;` separators confirm the three most-recent posts are concatenated into one
node text. Useful as a qualitative regression fixture.

`[CODE]` truncates each user's text at **1200 characters** and generates with
`max_new_tokens=550`. Both `UNKNOWN` in the paper.

---

## 5. Evaluation

| Item | Value |
|---|---|
| Public metrics | `[PAPER]` **F1-macro** and **AUC**, "following BWGNN [36]" |
| Industrial metric | `[PAPER]` additionally **KS** = `max abs(F1(x) - F2(x))`, "widely used in the financial industry" |
| Model selection | `[PAPER]` **highest F1-macro on validation**, then evaluated once on test |
| F1 decision threshold | `[PAPER]` **UNKNOWN** — no threshold policy stated anywhere |
| | `[CODE]` `argmax` over the 2 logits, i.e. a fixed 0.5-equivalent threshold |
| | BWGNN (whose setup the paper says it follows) sweeps the threshold over `np.linspace(0.05, 0.95, 19)` and picks the best macro-F1 |

The threshold gap matters: `argmax` on a 1:10 imbalanced problem yields very
different F1 from a swept threshold. Both policies are implemented
(`metrics.threshold_policy: argmax | validation_swept`), the swept policy fits on
validation only, and **the test set is never used to choose a threshold**.

`[CODE]` also computes **ECE** — but `utils.ECELoss` does not exist, so both
evaluation drivers fail at import (`flag_code_audit.md` 5.1). ECE is not a paper
metric.

---

## 6. Reference results

Transcribed into **`research/reported_results.csv`** (199 rows), generated by
`scripts/analyze/build_reported_results.py`. Every row carries
`verification_status=TRANSCRIBED` until a human re-checks it against the PDF.

- **Table 3** — industrial Huabei: 7 baselines (all `+text`) + FLAG + FLAG*,
  metrics KS / F1-macro / AUC, single run. FLAG's backbone is fixed to
  **GeniePath** "due to its resource efficiency". **Not reproducible** —
  proprietary Alipay data, 13M nodes / 120M edges.
- **Table 4** — Reddit and Instagram, 7 models x 4 variants x 2 metrics,
  mean±std over 25 runs. This is the reproduction target.
- **Table 5** — ablation over SS / LLM / SG. **Only 5 backbones** (GCN, GAT,
  GeniePath, CARE-GNN, BWGNN); **DGA-GNN and PMP are absent**. The full
  SS+LLM+SG row is not repeated — it is Table 4's `+FLAG` row.

`[PAPER]` Variant semantics, verbatim:

> "'baseline' indicates the vanilla GNN model utilizing shallow embeddings. The
> notation '+text' refers to the model trained with raw text embeddings. '+FLAG'
> signifies the performance of FLAG in a zero-shot context, whereas '+FLAG\*'
> represents the performance of FLAG after fine-tuning."

See `dataset_notes.md` section 7 — whether "shallow embeddings" really are shallow
is an open question, because the code's baseline input dimension is 4096.

`[PAPER]` Headline claims: average gains of **3.14% F1-macro / 6.97% AUC** on the
public datasets; zero-shot alone gives **1.58% F1 / 1.48% AUC**; fine-tuning adds
a further **0.84% F1 / 0.42% AUC**.

`[PAPER]` Ablation deltas: LLM **+1.01% F1 / +0.31% AUC**; SG **+0.33% / +0.74%**;
SS **+0.15% / +0.66%**.

> **Caveat, ours not the paper's.** Recomputing these deltas from the printed
> Tables 4-5 does not reproduce the stated percentages (we obtain roughly
> +0.47/+0.34 for LLM, +0.30/+0.27 for SG, +0.24/+0.42 for SS). The tabulated
> numbers themselves are transcribed exactly. Flagged as a reproduction risk; the
> discrepancy may come from a different averaging set than we assumed. **Not
> resolved.**

One further observation from the transcribed table: on Instagram/GAT/F1-macro the
`+FLAG` value (49.13) is **below** the `baseline` (49.21). It is the only such
cell in Table 4 and is a genuine feature of the published numbers.

---

## 7. Hyper-parameter sensitivity

`[PAPER]` Section 4.5, Figure 7. Backbone = **GCN**, metric = **AUC**, one
hyper-parameter varied at a time.

- **top-N in {5, 10, 15, 20}.** *"the AUC of GCN is slightly lower when N = 5
  compared to N = 10, and it decreases when N is set to 15 and 20. This is because
  a larger N leads to the sampled subgraph including more normal nodes, which
  diminishes the focus on fraudulent nodes."*
- **delta in {0.4, 0.2, 0, -0.2, -0.4}.** *"when delta = 0.2 or delta = -0.2, the
  performance remains similar to that of delta = 0. However, at delta = 0.4 and
  delta = -0.4, performance significantly declines."*
- Chosen: **top-10, delta = 0**.
- The underlying AUC values are **UNKNOWN** — Figure 7 is a plot with no table.

---

## 8. Visualization

`[PAPER]` Section 4.4, Figure 6:

> "we visualize how FLAG enhances the discriminability of node embeddings compared
> to baseline GNN models on the Reddit dataset. Specifically, we obtain
> **32-dimensional node embeddings** from each method and use **t-SNE** to project
> these embeddings into 2D space. Given the 1:10 class ratio in the test set, for
> better visualization, we select **all minority class nodes** and randomly sample
> **1/3 of the majority class nodes**. Notably, the LLM used in this experiment was
> not fine-tuned."

Panels (a)-(g) are the seven baselines, (h)-(n) are FLAG_GCN..FLAG_PMP.
Red = fraud, blue = normal. Perplexity, iterations and random state are `UNKNOWN`.

**Note the corroboration:** the paper says the visualised embeddings are
**32-dimensional**, and `[CODE]` returns exactly such a tensor — the variable is
literally named `x32` and equals the hidden layer output. This is independent
evidence that the code's hidden dimension of **32** is the operative value, and
strengthens the `[CONFLICT]` against the paper's stated hidden size of 64.
`[CODE]` `utils.t_sne` uses `TSNE(n_components=2, random_state=42)`.

---

## 9. Everything the paper does not state

Recorded so nobody fills these in by guessing. Where `[CODE]` supplies a value it
is used as a documented default tagged `source: code_not_paper`.

| Item | Filled by code? |
|---|---|
| Reddit/Instagram node, edge, per-class counts | no — regenerate |
| Shallow-embedding dimensionality | partially (4096 input dim observed) |
| Train/val/test split ratios and protocol | no — inherited from GLBench |
| Batch size, epochs, weight decay, dropout | yes |
| Early-stopping patience / criterion / metric | no — not implemented in code either |
| LoRA rank, alpha, dropout, target modules, LLM lr | **yes** |
| lambda_1, lambda_2 | **yes** (0.1, 0.1) |
| Sentence-BERT checkpoint | **yes** (`all-MiniLM-L6-v2`) |
| Exact HF id for "Gemma-9b-it" | **yes** (`gemma-2-9b-it`) |
| Decision-threshold policy for F1 | partially (`argmax`) |
| Attention-layer architecture | yes (`DualGNN`) |
| Max context length for concatenated subgraph text | yes (1200 chars/node, 550 new tokens) |
| Number of LLM fine-tuning steps | yes (3 outer x 10 inner) |
| Figures 3, 5, 7 underlying numbers | no |
| Instagram prompt wording | **yes** |

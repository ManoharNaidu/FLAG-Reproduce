# Dataset Notes

**Verification date:** 2026-09-09.

Confidence scale used throughout: `VERIFIED` (read from the source),
`LIKELY` (strong indirect evidence), `UNCERTAIN`, `UNKNOWN`.

---

## 1. Provenance chain for Reddit and Instagram

```
FLAG (KDD'25)
   |  cites [25]
   v
GLBench (NeurIPS'24 D&B, arXiv:2407.07457)     <-- DOWNLOAD FROM HERE
   |  cites [17]
   v
GraphAdapter, "Can GNN be Good Adapter for LLMs?" (WWW'24, arXiv:2402.12984)
   |                                            <-- datasets originate here
   +--> Instagram : Kim et al. WWW'20 influencer dataset + Instagram public API
   +--> Reddit    : ConvoKit Subreddit Corpus (Cornell)
```

**Confidence: VERIFIED.** The FLAG paper's bibliography entry [25] is:

> Yuhan Li, Peisong Wang, Xiao Zhu, Aochuan Chen, Haiyun Jiang, Deng Cai, Victor
> Wai Kin Chan, and Jia Li. 2024. Glbench: A comprehensive benchmark for graph
> with large language models. arXiv preprint arXiv:2407.07457 (2024).

and section 4.1.1 reads *"we use two social network datasets: Reddit [25] and
Instagram [25]."* GraphAdapter appears separately as FLAG reference [20], cited
as a **method**, not as the dataset origin.

---

## 2. Where to download

| Field | Value |
|---|---|
| Source | GLBench Google Drive |
| URL | https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj |
| Files | `reddit.pt` (552.5 MB), `instagram.pt` (181.3 MB) |
| Repo | https://github.com/NineAbyss/GLBench |
| Repo license | MIT, (c) 2024 NineAbyss |
| GLBench README instruction | *"All datasets in GLBench are available in this link. Please place them in the `datasets` folder."* |
| Confidence | **VERIFIED** — folder listing and file sizes read directly |

GLBench is **not** on HuggingFace (`huggingface.co/api/datasets?search=glbench`
returns `[]`). Google Drive is the only distribution channel, which is a supply-chain
risk: mirror the two files locally and record their checksums as soon as they are
downloaded (`scripts/download/prepare_reddit.py` does this).

---

## 3. File format

GLBench paper: *"we store the graph-type data in the .pt format using PyTorch.
This includes shallow embeddings commonly used in classical methods, raw text of
nodes, edge indices, node labels, label names, and masks for training,
validation, and testing procedures."*

Confirmed against GLBench's own loader
(`models/llm/llm_zeroshot/inference.py`):

```python
data = torch.load(f"datasets/{dataset_name}.pt")   # a single pickled PyG Data
data.test_mask = data.test_mask[0]                 # NOTE: may be a LIST of masks
data.label_name                                    # list[str]  (singular, not label_names)
data.raw_texts                                     # list[str], len == num_nodes
data.y
```

Fields: `x`, `edge_index`, `y`, `raw_texts`, `label_name`,
`train_mask` / `val_mask` / `test_mask`. **Confidence: VERIFIED.**

### 3.1 The mask gotcha

`data.{train,val,test}_mask` may be a **list of per-seed masks**, not a tensor —
GLBench's own loader does `data.test_mask = data.test_mask[0]`. FLAG's
`utils.py:generate_homo` indexes `data.test_mask` directly
(`mask_to_index(data.test_mask)`), which would break on a list. Our loader must
normalise this explicitly and record which split index was taken.
**Confidence: VERIFIED** (from GLBench source).

---

## 4. Published statistics (GLBench Table 3)

| Dataset | #Nodes | #Edges | Avg. Deg | Avg. Tok | #Classes | #Train | Node text | Domain |
|---|---|---|---|---|---|---|---|---|
| **Reddit** | 33,434 | 198,448 | 11.87 | 203.84 | 2 | 10.00% | user's posts | Social |
| **Instagram** | 11,339 | 144,010 | 25.40 | 59.25 | 2 | 10.00% | user's profile | Social |

**Confidence: VERIFIED.** GraphAdapter's own paper reports the identical node and
edge counts, which cross-validates the chain.

**The FLAG paper itself publishes NO dataset statistics table.** Tables 1-2 are
prompt templates and Tables 3-5 are results. So FLAG's post-downsampling node and
edge counts are **UNKNOWN** and must be regenerated, not looked up.

### 4.1 Measured stored structure (from the actual downloads)

Both files were downloaded and measured on 2026-09-09.

| | Reddit | Instagram |
|---|---|---|
| sha256 | `b655cc08693b59ff...d3d02be3` | `a42cff537fdb42b1...9f1cfeb2` |
| file size | 552.5 MB | 181.3 MB |
| **nodes** | **33,434** (published 33,434) | **11,339** (published 11,339) |
| `edge_index` stored | 302,876 | 155,349 |
| self-loops | 34,404 (= 33,434 + **970 pre-existing**) | 11,339 (= num_nodes exactly) |
| non-self-loop, directed | 268,472 | 144,010 |
| symmetric | yes | yes |
| **undirected edges** | **134,236** | **72,005** |
| published "#Edges" | 198,448 | 144,010 |
| expansion vs published | **x1.353** | x1.000 |
| `x` | (33434, **4096**) float32 | (11339, **4096**) float32 |
| `label_name` | `['Normal Users', 'Popular Users']` | `['Normal Users', 'Commercial Users']` |
| class counts | **16,717 / 16,717** (exactly 1:1) | 7,224 / 4,115 (1.76:1) |
| text chars min/mean/max | 9 / 767 / 21,035 | 0 / 110 / 523 |

### The two datasets relate to their published edge count differently

This tripped up an initial, too-strict verification and is worth stating plainly.

GraphAdapter applies `to_undirected` then `add_self_loops`. Symmetrising an edge
list of length *P* yields between *P* directed edges (if every edge was already
reciprocal) and *2P* (if none were). Measured:

- **Instagram**: 144,010 -> 144,010, expansion **x1.000**. The original list was
  already fully reciprocal, so the published figure equals the stored directed count.
- **Reddit**: 198,448 -> 268,472, expansion **x1.353**. Solving
  `2R + S = 198,448` and `2R + 2S = 268,472` gives **R = 64,212** reciprocal pairs
  and **S = 70,024** one-directional edges in the original list.

So the published "#Edges" is the **original edge-list length**, not a fixed
function of the stored graph. `verify()` therefore checks the
convention-independent property — stored count within `[P, 2P]`, plus actual
symmetry and at least one self-loop per node — rather than an exact equality.

### Consequences for downstream code

1. **Self-loops must be excluded** from degree, homophily and neighbour-sampling
   computations. Otherwise every node gains a spurious same-label neighbour and
   homophily is inflated — which would corrupt both FLAG's semantic sampling and
   the paper's Figure-5 homophily analysis. Reddit additionally has **970
   pre-existing self-loops** beyond the one-per-node that `add_self_loops` adds.
2. **State the convention** whenever an edge count is reported: stored-directed,
   loop-free-directed, or undirected. They differ by more than a factor of two.
3. **Reddit is exactly 50/50** (16,717 / 16,717), consistent with GraphAdapter
   defining `popular` as the top 50% by score. Instagram is **not** balanced
   (1.76:1), so the paper's "approximately equal number of nodes in each class"
   is accurate for Reddit and loose for Instagram.
4. Reddit's longest node text is **21,035 characters**, while FLAG truncates each
   node to **1,200 characters** before prompting. Truncation is therefore active
   for a substantial fraction of Reddit nodes, and is a real (documented) part of
   the method rather than an edge case.

### 4.2 Fields actually present

Beyond the documented ones, the shipped `Data` also carries GLBench's few-shot
splits:

```
x, y, edge_index, raw_texts, label_name,
train_mask, val_mask, test_mask,
one_shot_train,   one_shot_val,   one_shot_test,
three_shot_train, three_shot_val, three_shot_test,
five_shot_train,  five_shot_val,  five_shot_test
```

FLAG uses none of the few-shot splits. Recorded so they are not mistaken for the
standard split.

---

## 5. Label semantics

| | Reddit | Instagram |
|---|---|---|
| Node | user | user |
| Edge | reply between users | following relationship |
| Text | content of the user's **last three posts** (`;`-separated) | user's **personal introduction** |
| Labels | `popular` vs `normal` | `commercial` vs `normal` |
| Minority ("fraud") class | **popular** | **commercial** |
| Original balance | see below | **7,224 / 4,115** (1.76 : 1) |

GraphAdapter defines Reddit's `popular` as **the top 50% by score**.
**Confidence: VERIFIED** for the definition; the exact scoring window and
subreddit selection are **UNCERTAIN** (GraphAdapter describes it in one sentence).

**Measured class distribution, and a caveat about the paper's wording.** The FLAG
paper says both datasets *"originally contain an approximately equal number of
nodes in each class."* For Instagram the measured split is:

```
label_name = ['Normal Users', 'Commercial Users']
y == 0 (Normal)      7,224   63.7%
y == 1 (Commercial)  4,115   36.3%      -> 1.76 : 1, not 1 : 1
```

So the minority class index is **1**, and "approximately equal" is a loose
description of a 1.76:1 split. This is not a discrepancy that changes the method,
but it does change the arithmetic of the 1:10 construction: starting from 4,115
commercial and 7,224 normal nodes, a 1:10 ratio against the **untouched majority**
implies keeping about **722** commercial nodes, i.e. discarding ~82% of the
minority class. Our benchmark builder records the exact before/after counts in the
manifest rather than assuming a balanced start.

The `;` separators in the FLAG paper's Appendix-C worked example confirm the
three most-recent posts are concatenated into a single node text.

---

## 6. FLAG's transformation of these datasets

FLAG section 4.1.1, verbatim:

> "Both datasets originally contain an approximately equal number of nodes in
> each class, but fraud detection tasks generally require imbalanced datasets. To
> address this, we treat the popular category in Reddit and the commercial
> category in Instagram as the minority class. The minority class nodes are
> **randomly selected so that the final ratio between the minority and majority
> classes is about 1:10**."

Read precisely: the **minority class is downsampled**; the majority class is left
intact. So expected sizes are roughly `|majority|` unchanged and
`|minority| ~= |majority| / 10`.

**What is NOT published, and therefore must be regenerated:**

| Missing item | Status |
|---|---|
| The RNG seed for the 1:10 subsampling | **UNKNOWN** — unrecoverable |
| Post-downsampling node/edge counts | **UNKNOWN** |
| Train/val/test split ratios | **UNKNOWN in the FLAG paper** |
| Whether the split precedes or follows downsampling | **UNKNOWN** |
| The exact `reddit1.pt` vs `reddit2.pt` distinction | **UNKNOWN** (see 6.1) |

Consequence: **our reproduced numbers cannot be expected to match the paper's to
the last decimal**, because the benchmark construction itself is stochastic and
unseeded in the published record. This is stated plainly in
`reproduction_status.md` rather than hidden.

### 6.1 `reddit1.pt` vs `reddit2.pt`

`chat.py` (zero-shot) loads `Reddit/reddit1.pt`; `train.py` (fine-tuning) loads
`Reddit/reddit2.pt`. `test.py` loads `reddit1.pt`. Instagram has only
`instagram.pt`. Two readings are possible:

- (a) two independent 1:10 resamplings of the same GLBench graph, or
- (b) a zero-shot copy and a fine-tuning copy carrying different cached fields.

Both are consistent with the code. **UNCERTAIN.** Our pipeline generates
explicitly named, seed-stamped artefacts instead of numeric suffixes.

### 6.2 Split ratios — recommended default

The FLAG paper does not state a split. Two documented anchors:

1. **GLBench Table 3** lists 10.00% train for both datasets, and says these
   datasets *"follow the original split described in the paper"* (GraphAdapter).
2. **GraphAdapter's `finetune_utils.py`** calls
   `load_data_with_prompt_embedding(dataset, 10, 10, split)` -> **10% train /
   10% val / 80% test**, randomised per split seed.

So **10/10/80** is the defensible default. It is exposed as
`dataset.split_ratios` in config and recorded in the manifest, and is labelled
as *inherited from GraphAdapter/GLBench*, never as *stated by FLAG*.

---

## 7. `data.x` — the "baseline" features. **RESOLVED: 4096-d, not shallow.**

**Status: RESOLVED by direct inspection on 2026-09-09.**

The FLAG paper describes the `baseline` variant as using **"shallow embeddings"**,
and elsewhere identifies shallow features as **word2vec**. The released code
builds the baseline model with an input dimension of **4096**
(`methods/flag/test.py:201`, comment `# 4096 384`).

Downloading and inspecting the actual GLBench file settles it:

```
data/raw/instagram/instagram.pt
  x  -> torch.Size([11339, 4096])  torch.float32
```

**The stored node features are 4096-dimensional.** That is not a word2vec
dimensionality (typically 100-300). It is exactly **Llama-2-7B's hidden size**,
and GraphAdapter — the origin of these datasets — generates its node embeddings
with **Llama 2** (`token_embedding/{dataset}/sentence_embeddings.npy`, produced by
`preprocess.py`).

### What this means

The paper's `baseline` row is **not a shallow-feature baseline**. It is a GNN over
**LLM-derived node embeddings**. Consequently:

- `baseline` vs `+text` is **not** "shallow features vs text features". Both are
  text-derived; they differ in *which* encoder produced them — Llama-2 (4096-d)
  versus Sentence-BERT `all-MiniLM-L6-v2` (384-d).
- This makes the very low `baseline` scores in Table 4 more surprising, not less
  (e.g. Reddit GCN baseline F1-macro 45.46 with AUC 50.32 — essentially chance),
  and it is worth checking whether the baseline is under-trained rather than
  under-informed.
- Any write-up of ours must describe the `baseline` variant accurately. We label
  it `baseline (GLBench stored features, 4096-d, Llama-2-derived)` rather than
  repeating "shallow embeddings".

**Caveat, stated precisely.** We have verified the *dimensionality* (4096) and the
*origin pipeline* (GraphAdapter uses Llama 2). We have **not** verified that these
specific stored vectors are that exact Llama-2 output rather than some other
4096-d representation. That would need a byte-level comparison against
GraphAdapter's `sentence_embeddings.npy`. So: `LIKELY_LLM_EMBEDDING`, not
`CONFIRMED`. The classification is recorded automatically in every dataset
manifest by `flagbench.datasets.glbench._classify_features`.

## 8. Datasets with NO native text (Phase 9 integrity rule)

The FLAG paper itself states these lack textual information:

> "most of them, such as Yelp-Fraud [8], Amazon-Fraud [8], T-Finance [36] and
> T-Social [36], lack textual information"

| Dataset | Native text | Canonical FLAG possible? | Source |
|---|---|---|---|
| YelpChi / Yelp-Fraud | **NO** | **NO** | DGL `FraudYelpDataset`; also `YingtongDou/CARE-GNN` `data/YelpChi.zip` |
| Amazon-Fraud | **NO** | **NO** | DGL `FraudAmazonDataset`; also CARE-GNN `data/Amazon.zip` |
| T-Finance | **NO** | **NO** | BWGNN Google Drive |
| T-Social | **NO** | **NO** | BWGNN Google Drive (PMP issue #2 reports missing files) |
| Elliptic | **NO** | **NO** | DGA-GNN Google Drive `.7z` |

**Enforced rule.** For these datasets the registry reports
`FLAG canonical text mode: NOT AVAILABLE — dataset has no native text`, and
refuses to run `variant=flag` or `variant=flag_finetuned`. A text-augmented study
is permitted only under a *separate* experiment type that stamps
`native_text: false` and `text_source: engineered_or_external` into every result
row. It is never merged with the canonical FLAG reproduction, and never labelled
"FLAG".

---

## 9. WRONG SOURCES — do not use

### 9.1 `torch_geometric.datasets.Reddit` / DGL `RedditDataset`
This is **Hamilton et al.'s GraphSAGE Reddit**: 232,965 nodes, 41 communities,
**no raw text whatsoever**. It shares only the name. It will download without
error and silently produce meaningless results.

### 9.2 Kaggle "reddit fraud" / "instagram fake account" datasets
Wrong on every axis. Most have **no graph structure at all** — they are tabular
account-feature CSVs (follower count, has-profile-pic), which is a different
problem entirely.

### 9.3 Verification signature

Check **all five** before trusting any downloaded file:

1. Reddit = **33,434 nodes / 198,448 edges**; Instagram = **11,339 / 144,010**.
2. Labels are `popular`/`normal` and `commercial`/`normal` — **not**
   `fraud`/`spam`/`fake`/`bot`.
3. `raw_texts` exists, is `list[str]`, and `len(raw_texts) == num_nodes`.
4. Reddit edges are **replies**; Instagram edges are **following**.
5. Classes are roughly **balanced** in the original. If the download is already
   imbalanced, it is not the GLBench original.

`scripts/download/verify_datasets.py` asserts all five and refuses to proceed on
mismatch.

---

## 10. Upstream origins and their terms

### Instagram -> Kim et al., WWW 2020
*Multimodal Post Attentive Profiling for Influencer Marketing*, Kim, Jiang,
Nakada, Han, Wang. WWW 2020, 2878-2884. DOI `10.1145/3366423.3380052`.
Repo: https://github.com/ksb2043/instagram_influencer_dataset
Download is **gated behind an author request form**:
https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
As published: 33,935 influencers, 10,180,500 posts, ~37 GB JSON + ~189 GB images.

**Critical difference:** Kim et al.'s labels are **9 influencer categories**
(Beauty, Family, Fashion, Fitness, Food, Interior, Pet, Travel, Other) — **not**
commercial/normal. The binary labelling and the 11,339-node following graph were
constructed **by GraphAdapter** using Instagram's public API. Kim's raw dataset
will not reproduce FLAG's Instagram graph. Terms: research use only, no SPDX
licence, citation required.

### Reddit -> ConvoKit Subreddit Corpus (Cornell)
https://convokit.cornell.edu/documentation/subreddit.html — this exact URL is the
footnote in the GraphAdapter paper. Underlying data is Reddit comment dumps
(Pushshift-derived). ConvoKit's toolkit is MIT; the corpora are research-use and
subject to Reddit's content terms. GraphAdapter's exact subsetting procedure is
**UNCERTAIN**.

### GraphAdapter's own packaging
Repo: https://github.com/hxttkl/GraphAdapter
Drive: https://drive.google.com/drive/folders/13fqwSfY5utv8HibtEoLIAGk7k85W7b2d
Raw format is **NumPy `.npy`**, not `.pt`:
`edge_index.npy` (shape `(E,2)`, later transposed + `to_undirected` +
`add_self_loops`), `y.npy`, `train.npy`, `vaild.npy` (the typo is real).
**License: UNKNOWN** — no LICENSE file in the repo.

This is why we take GLBench's `.pt` repackaging rather than GraphAdapter's raw
`.npy`: it is the artefact FLAG's code actually consumes.

---

## 11. Directory contract

```
data/
  raw/            # exactly as downloaded. NEVER modified, NEVER overwritten.
    reddit/reddit.pt
    instagram/instagram.pt
  processed/      # normalised into our internal GraphDataset schema
  benchmark/      # FLAG's 1:10 fraud-detection transformation
    flag_reddit/
    flag_instagram/
```

`data/` is git-ignored. Every `benchmark/` artefact is accompanied by a
`dataset_manifest.json` recording: `dataset_name`, `source`, `source_url`,
`download_date`, `sha256`, original node/edge counts, original feature shape,
original label distribution, constructed label distribution, `sampling_seed`,
split ratios, split seed, and `preprocessing_version`.

Two rules are non-negotiable:

1. **`data/raw/` is written once and never touched again.** The original graph,
   text, and labels are always recoverable.
2. **Every stochastic choice is seeded and recorded.** A benchmark build that
   cannot be reproduced from its manifest is a bug.

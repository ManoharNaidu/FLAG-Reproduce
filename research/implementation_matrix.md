# Implementation Matrix

What exists, where it comes from, how faithful it is, and what has to be built.

**Status date:** 2026-09-09 — end of Phase 0-3 (research/audit). No training runs
have been executed yet. **Nothing in this file claims a reproduction.**

---

## 1. Legend

**Adaptation class** (Phase 18):

| Class | Meaning |
|---|---|
| `DIRECT` | Official code consumes benchmark data with minimal glue |
| `ADAPTER_REQUIRED` | Algorithm reusable; input/output/data-loading differs |
| `REIMPLEMENT_REQUIRED` | No usable official code, or fundamentally incompatible |
| `NOT_COMPATIBLE` | Dataset lacks assumptions the method requires |

**Implementation provenance** (Phase 35) — stamped into every result row as
`impl_source`, so a number can never be misattributed:

| Tag | Meaning |
|---|---|
| `OFFICIAL` | Unmodified code from the authors' repository |
| `ADAPTED` | Official code, wrapped; algorithm untouched |
| `FLAG_BUNDLED` | The FLAG authors' own re-implementation, shipped in `methods/flag/` |
| `REIMPLEMENTED` | Written here from the paper's equations |
| `APPROXIMATION` | Knowingly simplified; deviations documented |
| `UNKNOWN` | Provenance not established |

**Build status:** `AUDITED` (source read, claims test-backed) / `TODO` / `BLOCKED`.

---

## 2. Backbones

Two independent lineages per model. They are never merged: reproducing *the
paper's numbers* and reproducing *the baseline as its authors published it* are
different experiments.

| Model | Lineage A — as FLAG ran it | Fidelity of A | Lineage B — official | Class (B) | Status |
|---|---|---|---|---|---|
| **GCN** | `methods/flag/models.py:GCN` `FLAG_BUNDLED` | **Faithful.** Two `GCNConv` layers + skip. | PyG `GCNConv` | `DIRECT` | `AUDITED` |
| **GAT** | `models.py:GAT` `FLAG_BUNDLED` | **Not canonical GAT.** `conv1=GATConv(in,hid,8)` but **`conv2=SAGEConv`** — the second layer is GraphSAGE, not attention. No skip. | PyG `GATConv` x2 | `ADAPTER_REQUIRED` | `AUDITED` |
| **GeniePath** | `geniepath.py` `FLAG_BUNDLED` | **Closest to faithful.** Breadth(GAT)+Depth(LSTM) preserved; matches PyG's `examples/geniepath.py` globals exactly (`dim=256, lstm_hidden=256, heads=1, layer_num=4`). Skip added. | PyG example — **no official author repo exists** | `ADAPTER_REQUIRED` | `AUDITED` |
| **CARE-GNN** | `caregnn.py` `FLAG_BUNDLED` `APPROXIMATION` | **Substantially not CARE-GNN.** Missing all three defining components: label-aware similarity classifier, **RL-based adaptive neighbour filtering**, and **multi-relation** aggregation. It is a similarity-gated mean aggregator. No skip. | `YingtongDou/CARE-GNN` @ `a64ff752` (Apache-2.0) | `REIMPLEMENT_REQUIRED` | `AUDITED` |
| **BWGNN** | `bwgnn.py` `FLAG_BUNDLED` `APPROXIMATION` | **Beta-wavelet basis not applied.** `calculate_theta2` reproduces the official Beta coefficients correctly, but `PolyConv` evaluates the polynomial over **raw adjacency `A x`** instead of the normalised Laplacian `L = I - D^-1/2 A D^-1/2`. Wrong operator basis => not a beta wavelet. All `PolyConv(lin=False)`, so their linear+activation is dead. Skip present. | `squareRoot3/Rethinking-Anomaly-Detection` @ `de0631f0` (**no licence**) | `REIMPLEMENT_REQUIRED` | `AUDITED` |
| **DGA-GNN** | `dga.py` `FLAG_BUNDLED` `APPROXIMATION` | **Not DGA-GNN.** No decision-tree **dynamic grouping**, no bidirectional grouped aggregation. It is GraphSAGE-mean (`fc_self(x) + fc_neigh(mean(x_j))`). No skip. | `AtwoodDuan/DGA-GNN` @ `0907392f` (**no licence**) | `REIMPLEMENT_REQUIRED` | `AUDITED` |
| **PMP** | `pmp.py:LASAGE_S` `FLAG_BUNDLED` | **Partial.** The core idea is present — separate `fc_neigh_fraud` / `fc_neigh_benign` combined by a learned gate — but partitioning is applied to the **aggregated mean** rather than per-neighbour before aggregation, which is weaker than true partitioned message passing. No skip. | `Xtra-Computing/PMP` @ `3f7629f6` (**no licence**) | `ADAPTER_REQUIRED` | `AUDITED` |

### 2.1 Skip-GNN coverage in FLAG's own code

`[PAPER]` Eq. 6 says `Z = GNN(X, A) + Linear(X)`. `[CODE]` applies it to fewer
than half the backbones. Verified by test, not by reading:

| Skip active | Skip **dead** (computes `initial_x`, discards it) |
|---|---|
| GCN, GeniePath, GeniePathLazy, BWGNN | GAT, CARE-GNN, DGA, PMP, GraphSAGE |

Which behaviour produced the paper's `+FLAG` rows for the four "dead" backbones
is **UNKNOWN**. Exposed as the `SG` ablation flag rather than assumed.

### 2.2 Driver-compatibility traps

| Class | Returns | Works with the `emb32, out = model(...)` unpack? |
|---|---|---|
| `GeniePathLazy` | `(x32, out)` | yes |
| `GeniePath` (eager) | single tensor | **no** |
| `GraphSAGE` | single tensor | **no** |
| all others | `(x32, out)` | yes |

`DGA` and `LASAGE_S` additionally bind `x32` only when `hidden == 32`
(`if len(x[0]) == 32`), raising `UnboundLocalError` otherwise. Both verified.

---

## 3. FLAG pipeline components

| Component | In official repo? | Provenance | Class | Status |
|---|---|---|---|---|
| Prompts (4 per dataset) | **yes** | `OFFICIAL` — extracted verbatim by AST to `prompts/`, SHA-256 in `prompts/manifest.json` | `DIRECT` | **DONE** |
| LLM enhancement (`chat.py`) | **yes** | `OFFICIAL` | `ADAPTER_REQUIRED` (CPU + caching + batching) | `AUDITED` |
| Sentence-BERT encoding (`encode.py`) | **yes** | `OFFICIAL` | `ADAPTER_REQUIRED` | `AUDITED` |
| Skip-GNN (Eq. 6) | **yes**, partially | `OFFICIAL` | `ADAPTER_REQUIRED` | `AUDITED` |
| Attention fusion at inference | **yes** — `models.py:DualGNN` | `OFFICIAL` | `ADAPTER_REQUIRED` | `AUDITED` |
| Three fine-tuning losses | **yes** — `utils.py` | `OFFICIAL` (orthogonality **deviates from Eq. 9**) | `ADAPTER_REQUIRED` | `AUDITED` |
| LoRA config | **yes** — `train.py:37-43` | `OFFICIAL` | `DIRECT` | `AUDITED` |
| Two-stage alternating loop | **yes** — `train.py:main` | `OFFICIAL` but **non-functional** (5.5) | `REIMPLEMENT_REQUIRED` | **BLOCKED** |
| **Semantic similarity sampling** | **NO** | — | **`REIMPLEMENT_REQUIRED`** | `TODO` |
| 1:10 benchmark construction | **NO** | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| Train/val/test splitting | **NO** | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| Early stopping | **NO** (arg parsed, never used) | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| ECE metric (`utils.ECELoss`) | **NO** — imported but undefined | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| KS metric | **NO** | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| t-SNE visualization | **yes** — `utils.t_sne` | `OFFICIAL` | `ADAPTER_REQUIRED` | `AUDITED` |
| Ten-bin homophily analysis (Fig. 5) | **NO** — only a binary split | — | `REIMPLEMENT_REQUIRED` | `TODO` |
| Result storage / aggregation | **NO** | — | `REIMPLEMENT_REQUIRED` | `TODO` |

**The single most important row:** semantic similarity sampling is a *core
contribution* of the paper and **has no source in the repository**. Everything
downstream loads pre-built `*_sampler*.pt` files that nothing in the repo creates.

---

## 4. Variants (Phase 5)

| Variant | Meaning `[PAPER]` | How the official code selects it | Our mechanism |
|---|---|---|---|
| `baseline` | vanilla GNN on **shallow embeddings** | pass `data.x` (4096-d) to a model built with `in=4096` | config |
| `text` | same GNN on **raw text embeddings** | pass `embeddings[batch.subset]` (384-d), rebuild model with `in=384` | config |
| `flag` | FLAG **zero-shot** | pass `batch.unique_embeddings`; use `DualGNN` | config |
| `flag_finetuned` | FLAG **after LoRA fine-tuning** | as above, with LoRA-generated text | config |

In the official repo the variant is chosen by **hand-editing which tensor is
passed to `model(...)` and which constructor line is uncommented**
(`test.py:67-70` computes `text_embeddings` and then never uses it; line 82
hardcodes `data.x`). There is no flag. Building this switch is real integration
work, not a rewrite.

---

## 5. Dataset compatibility

`YES` = the canonical FLAG text pipeline can run.
`NO (no native text)` = **hard-blocked** by the Phase-9 integrity rule.

| Dataset | baseline | text | flag | flag_finetuned | Native text | Source |
|---|---|---|---|---|---|---|
| **Reddit** | YES | YES | YES | YES | yes | GLBench |
| **Instagram** | YES | YES | YES | YES | yes | GLBench |
| YelpChi | YES | NO | **NO** | **NO** | **no** | DGL / CARE-GNN |
| Amazon | YES | NO | **NO** | **NO** | **no** | DGL / CARE-GNN |
| T-Finance | YES | NO | **NO** | **NO** | **no** | BWGNN Drive |
| T-Social | YES | NO | **NO** | **NO** | **no** | BWGNN Drive |
| Elliptic | YES | NO | **NO** | **NO** | **no** | DGA-GNN Drive |
| Huabei (industrial) | — | — | — | — | yes | **proprietary Alipay — NOT OBTAINABLE** |

The paper itself is the authority for the "no native text" column: *"most of them,
such as Yelp-Fraud, Amazon-Fraud, T-Finance and T-Social, lack textual
information."* `benchmark.validate` refuses these combinations rather than
silently substituting something.

### 5.1 Multi-relation

CARE-GNN's official implementation requires a **multi-relation** graph
(YelpChi has 3 relations, Amazon 3). Reddit and Instagram are **single-relation**.
So official CARE-GNN on Reddit/Instagram is `NOT_COMPATIBLE` without collapsing
its inter-relation aggregator — which would change the algorithm. This is
precisely why FLAG's `caregnn.py` dropped those modules.

**Recorded as a genuine research decision, not an engineering one.** Options:
(a) run official CARE-GNN only on Yelp/Amazon and report FLAG's variant separately
on Reddit/Instagram; or (b) run official CARE-GNN with a single relation and
document the degeneration. Escalated in `reproduction_status.md`.

---

## 6. Capability declarations (Phase 41)

| Model | homog. | multi-rel. | native text | mini-batch | CPU | GPU |
|---|---|---|---|---|---|---|
| GCN | yes | no | no | yes | yes | yes |
| GAT | yes | no | no | yes | yes | yes |
| GeniePath | yes | no | no | yes | yes | yes |
| CARE-GNN (official) | yes | **required** | no | yes | partial (hardcoded `CUDA_VISIBLE_DEVICES`) | yes |
| CARE-GNN (FLAG) | yes | no | no | yes | yes | yes |
| BWGNN (official) | yes | optional | no | **no** (full-graph; open issue #15) | **yes** | yes |
| BWGNN (FLAG) | yes | no | no | yes | yes | yes |
| DGA-GNN (official) | yes | optional | no | yes | **no** | yes |
| PMP (official) | yes | optional | no | yes | **no** (`cuda.set_device` at startup) | yes |
| FLAG | yes | no | **required** | yes | partial — see below | yes |

**FLAG CPU status is deliberately split** (Phase 15):

- *CPU functional environment* — preprocessing, sampling, all GNN training,
  metrics, unit tests, small-scale runs: **fully supported**, and every backbone
  is verified to forward+backward on CPU.
- *Full reproduction environment* — `gemma-2-9b-it` in fp16 is ~18.5 GB of weights
  alone, plus KV cache for `max_new_tokens=550`. **Not feasible on CPU**, and this
  machine has no CUDA. Stated as a limitation, not worked around.

---

## 7. What Phase 0-3 actually produced

| Artefact | Status |
|---|---|
| `research/flag_code_audit.md` | **done** — 11 defects, 7 gaps, all test-backed |
| `research/paper_notes.md` | **done** — paper obtained, all tables extracted |
| `research/repository_provenance.md` | **done** — 9 repos pinned to SHAs |
| `research/dataset_notes.md` | **done** — chain verified to primary sources |
| `research/compatibility_notes.md` | **done** — a hard environment bug found and fixed |
| `research/reported_results.csv` | **done** — 199 reference rows |
| `research/implementation_matrix.md` | this file |
| `prompts/` + `manifest.json` | **done** — 8 prompts, verbatim, hashed |
| `tests/integration/test_flag_upstream_claims.py` | **done** — audit claims executable |
| `src/flagbench/compat/torch_scatter.py` + tests | **done** — environment fix, equivalence-tested |
| CPU environment | **done** — `.venv-cpu`, every backbone forward+backward verified |

### Not started (correctly — Phase 43 sequencing)

Dataset download, sampler implementation, adapters, registry, training loop,
metrics, result storage, config system, docs. The brief says *"Do not immediately
rewrite the project"*; the audit had to come first, and it changed several
assumptions that would have been baked into a premature abstraction — most
importantly that the sampler does not exist, that fine-tuning is non-functional,
and that four of the five bundled baselines are not their published algorithms.

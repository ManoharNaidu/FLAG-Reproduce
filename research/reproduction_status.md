# Reproduction Status

The ledger. Nothing is marked reproduced until it has actually run and been verified.

**Last updated:** 2026-09-09 — end of Phase 0-3 (research and audit).

---

## 1. Headline

**No experiment has been run. No metric has been produced. Nothing is reproduced.**

What exists so far is an audit: the paper obtained and extracted, the official
code read and its claims made executable, every repository pinned to a SHA,
dataset provenance traced to primary sources, and a working CPU environment
established. That is Phase 0-3 of 46.

| Status | Meaning |
|---|---|
| `NOT_STARTED` | Not attempted yet |
| `AUDITED` | Source understood; findings test-backed; not executed |
| `BLOCKED` | Attempted; cannot proceed; reason recorded |
| `PARTIAL` | Runs, but with documented deviations |
| `REPRODUCED` | Ran, verified, compared against the paper |

---

## 2. Experiment matrix

Legend: `-` = not started. **No cell is ticked.**

| Dataset | GCN | GAT | GeniePath | CARE-GNN | BWGNN | DGA-GNN | PMP | FLAG | FLAG* |
|---|---|---|---|---|---|---|---|---|---|
| Reddit | - | - | - | - | - | - | - | - | - |
| Instagram | - | - | - | - | - | - | - | - | - |
| YelpChi | - | - | - | - | - | - | - | **N/A** | **N/A** |
| Amazon | - | - | - | - | - | - | - | **N/A** | **N/A** |
| T-Finance | - | - | - | - | - | - | - | **N/A** | **N/A** |
| T-Social | - | - | - | - | - | - | - | **N/A** | **N/A** |
| Elliptic | - | - | - | - | - | - | - | **N/A** | **N/A** |
| Huabei (industrial) | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** |

- `N/A` for YelpChi/Amazon/T-Finance/T-Social/Elliptic under FLAG: **no native
  text**. The FLAG paper says so itself. Hard-blocked by the Phase-9 rule.
- `N/A` for Huabei: proprietary Alipay data (13M nodes / 120M edges).
  **Permanently unobtainable.** Table 3 can never be reproduced by anyone outside
  Alipay, and is recorded in `reported_results.csv` for completeness only.

---

## 3. Component status

| Component | Status | Note |
|---|---|---|
| FLAG paper obtained and extracted | **DONE** | All of Tables 3-5, all equations, both prompt tables |
| FLAG repo cloned and pinned | **DONE** | `cb83944e` |
| FLAG code audit | **DONE** | 11 defects, 7 gaps, **36/36 claims test-verified** |
| Original FLAG execution attempted | **DONE** | **0 of 7 entrypoints run** — see `FLAG_ORIGINAL_STATUS.md` |
| Baseline repo provenance | **DONE** | 9 repos pinned; licences recorded |
| Dataset provenance | **DONE** | FLAG -> GLBench -> GraphAdapter -> ConvoKit / Kim et al. |
| Prompts extracted verbatim | **DONE** | 8 prompts, AST-extracted, SHA-256 hashed |
| Reference results transcribed | **DONE** | 199 rows, all `TRANSCRIBED` (not yet re-checked) |
| CPU environment | **DONE** | torch 2.3.1+cpu pinned after finding a **numerically wrong** torch build |
| All 7 backbones forward+backward on CPU | **DONE** | verified at `hidden=32` |
| Dataset download | `NOT_STARTED` | GLBench Google Drive; 734 MB total |
| Semantic similarity sampler | `NOT_STARTED` | **No upstream source exists** — must be built from Eq. 3-4 |
| 1:10 benchmark construction | `NOT_STARTED` | No upstream source |
| Train/val/test splitting | `NOT_STARTED` | Ratios not in the paper; 10/10/80 inherited from GLBench |
| Internal data model + adapters | `NOT_STARTED` | |
| Backbone adapters | `NOT_STARTED` | |
| Metrics module | `NOT_STARTED` | AUC, F1-macro, KS, ECE, per-class |
| Training loop + early stopping | `NOT_STARTED` | Not implemented upstream either |
| Result storage / aggregation | `NOT_STARTED` | |
| Config system | `NOT_STARTED` | |
| LLM enhancement pipeline | `BLOCKED` | needs a GPU — see 5.1 |
| LoRA fine-tuning | `BLOCKED` | needs a GPU **and** an unresolved research question — see 4.1 |
| GPU support | **UNTESTED** | no GPU on this machine; will never be claimed as working |

---

## 4. Open questions requiring a human decision

These are escalated per Phase 44: each materially changes research validity and
cannot be settled by an engineering judgement call.

### 4.1 What produced the `+FLAG*` column? — **HIGHEST PRIORITY**

**The ambiguity.** The paper describes two-stage alternating fine-tuning with
three losses. The released `train.py` implements the structure, but its gradient
path to the LoRA parameters is **severed**: `model.generate()` is
non-differentiable, and the decode -> Sentence-BERT re-encode produces a fresh
leaf tensor. `optimizer.step()` on the PEFT parameters is a no-op with
`.grad is None`. Verified by test.

So the released code, run as written, produces `+FLAG*` numbers that are
**identical in expectation to `+FLAG`** — yet Table 4 reports `+FLAG*` as
consistently better (average +0.84% F1 / +0.42% AUC).

**Competing options.**

| Option | Description | Consequence |
|---|---|---|
| A | The authors ran a different, unreleased version with a working gradient path | Our reproduction must design that path ourselves. It is a genuine research reimplementation, and we cannot claim fidelity. |
| B | The gain comes from the GNN inner loop, which *does* train — the LLM is effectively frozen and `FLAG*` differs from `FLAG` only by extra GNN epochs plus residual/orthogonality regularisation | Reproducible exactly as published. `+FLAG*` would be renamed to reflect what it is. |
| C | Ask the authors | Slowest, most authoritative. |

**Recommendation: implement B first** (it is what the code does, is cheap, and is
honestly labelled), **and pursue C in parallel.** Reserve A until we hear back.
B is the only option that reproduces the released artefact rather than
substituting our own design for it.

**Why it matters.** `+FLAG*` is one of the paper's four headline variants. Getting
this wrong means either fabricating a mechanism the authors never used, or
under-reporting their method.

### 4.2 Which CARE-GNN is "the CARE-GNN baseline"?

**The ambiguity.** Official CARE-GNN needs a **multi-relation** graph; Reddit and
Instagram are single-relation. FLAG's bundled `caregnn.py` resolves this by
dropping CARE-GNN's three defining components (RL neighbour filtering, label-aware
similarity, inter-relation aggregation) — so it is not CARE-GNN.

| Option | Description |
|---|---|
| A | Run FLAG's `caregnn.py` on Reddit/Instagram (reproduces the paper) **and** official CARE-GNN on Yelp/Amazon (reproduces the baseline), reported as separate rows |
| B | Run official CARE-GNN on Reddit/Instagram with a single relation, documenting the degeneration |
| C | Reimplement CARE-GNN faithfully for single-relation graphs |

**Recommendation: A.** It is the only option where every number means what its
label says. B silently degrades a published algorithm; C invents a variant the
authors never proposed.

The same question applies in weaker form to **BWGNN** (whose bundled version
evaluates the polynomial over raw adjacency rather than the normalised Laplacian,
so it is not a beta wavelet) and **DGA-GNN** (whose bundled version has no dynamic
grouping at all). The same recommendation follows.

### 4.3 hidden = 32 or 64?

**The ambiguity.** The paper says *"two layers and a hidden layer size of 64."*
The code's default is **32**, and `dga.py:63` / `pmp.py:110` guard on
`if len(x[0]) == 32`, so those two backbones **crash** at any other value.
Independent corroboration for 32: the paper's own t-SNE section says the
visualised embeddings are **32-dimensional**, and the code's variable is literally
named `x32` and equals the hidden layer output.

**Recommendation: run both.** It is a single config axis and cheap. Report 32 as
`config: code` and 64 as `config: paper`. Do not silently pick one.

This is a *reported* decision rather than a blocking one — it is already exposed
in config — but it is listed here because it changes which numbers are comparable
to Table 4.

### 4.4 Are the "shallow embeddings" actually shallow?

The paper's `baseline` variant is described as using *"shallow embeddings"*, and
elsewhere identifies shallow features as **word2vec**. But the code builds the
baseline model with an input dimension of **4096** — not a word2vec size, and
exactly Llama-2-7B's hidden size. GraphAdapter (the dataset's origin) produces its
sentence embeddings **with Llama 2**.

If the baseline features are LLM embeddings, then `baseline` vs `+text` in Table 4
does not mean what the labels suggest.

**Status: resolvable by inspection, not a decision.** Reading `data.x.shape` and
its value distribution from the downloaded `instagram.pt` settles it. Flagged here
so it is not forgotten; `scripts/download/` will record the observed shape into
the dataset manifest automatically.

### 4.5 Licensing — redistribution is not currently permitted

**FLAG, BWGNN, DGA-GNN and PMP ship no LICENSE file.** Under default copyright
that means no rights to copy, modify or redistribute are granted.

Handled for now by git-ignoring `methods/` and reconstructing it from upstream at
pinned SHAs, so this repository re-hosts nobody's code. **But if this project is
ever published with vendored copies, permission must be obtained first.** Decision
required before any public release.

---

## 5. Known limitations of this machine

### 5.1 No GPU

`torch.cuda.is_available() == False`; `nvidia-smi` is not installed. Consequences:

- **Every GPU claim in this project is UNTESTED** and is recorded as such. GPU
  tests will skip rather than pass.
- `gemma-2-9b-it` is ~18.5 GB in fp16, plus KV cache for `max_new_tokens=550`.
  The paper reports deployment on an **A100 80 GB**.
- **Full FLAG / FLAG\* reproduction is not possible here.** This is stated, not
  worked around. Per Phase 15 the project splits into:
  - *CPU functional environment* — preprocessing, sampling, all GNN training,
    metrics, tests, small-scale runs, and a small mock LLM for pipeline validation.
    **Real and already exercised**: all 7 backbones forward+backward on CPU.
  - *Full reproduction environment* — requires a GPU. Not available.

**We will not claim that Gemma-2-9B fine-tuning on CPU is practical.**

### 5.2 A torch build on this machine returns wrong numbers

`torch==2.4.0+cpu` here produces both hard crashes and, at 2 threads, **silently
incorrect scatter results**. Pinned away to `2.3.1+cpu`; full bisection in
`compatibility_notes.md` section 3.

Recorded prominently because a silently-wrong numeric build would invalidate every
experiment with no visible symptom. `scripts/smoke_test.py` re-checks numerical
equivalence on every run so the environment can never regress unnoticed.

---

## 6. Deviations from the paper, declared in advance

Every one will be stamped into result rows so no number can be misread.

| # | Deviation | Why | Avoidable? |
|---|---|---|---|
| 1 | Benchmark 1:10 downsampling uses **our** seed | The paper's seed is unpublished and unrecoverable | no |
| 2 | Split ratios **10/10/80** inherited from GLBench/GraphAdapter | The FLAG paper states no split | no |
| 3 | Semantic sampler **reimplemented** from Eq. 3-4 | No upstream source exists | no |
| 4 | Early stopping **implemented by us** | Paper says "early stopping" with no criterion; code does not implement it | no |
| 5 | Orthogonality loss defaults to the paper's Eq. 9 (squared dot), **not** the code's signed cosine | The two differ; the code's form rewards anti-alignment, not orthogonality | selectable |
| 6 | Skip-GNN applied uniformly or per-backbone | Upstream applies it to only 4 of 9 backbones | selectable |
| 7 | 5 runs vs 25 | Paper says 25 (5 seeds x 5 inits); code does 5 | selectable |
| 8 | hidden 32 vs 64 | See 4.3 | selectable |
| 9 | F1 threshold policy | Paper unspecified; code uses argmax; BWGNN (which the paper says it follows) sweeps on validation | selectable |
| 10 | ECE binning scheme **chosen by us** | `utils.ECELoss` does not exist upstream | no (not a paper metric) |

---

## 7. Next steps

In Phase-43 order:

1. Download `reddit.pt` / `instagram.pt` from GLBench; checksum them; **resolve
   question 4.4** by inspecting `data.x`.
2. Build the internal `GraphDataset` model + adapters; verify against GLBench's
   published node/edge counts.
3. Implement the 1:10 benchmark construction with recorded seeds and a manifest.
4. Implement semantic similarity sampling (Eq. 3-4) with unit tests on synthetic
   graphs where the correct answer is known by construction.
5. Implement metrics with an explicit, validation-only threshold policy.
6. Reproduce **one** baseline end-to-end (GCN on Reddit) before generalising.
7. Then the remaining baselines, then FLAG zero-shot, then the ablations.

Fine-tuning stays blocked on question 4.1 and on GPU access.

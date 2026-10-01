# Reproduction Status

The ledger. Nothing is marked reproduced until it has actually run and been verified.

**Last updated:** 2026-09-18 — `flag` and `flag_finetuned` now have real GPU
results for 4 of 7 backbones; `+text` is currently missing from the tracked
aggregation.

---

## 1. Headline

**The pipeline runs end to end, on CPU for baselines and on a rented GPU for the
LLM stage.** `google/gemma-2-9b-it` has been run on vast.ai (decision D-003) to
produce real `flag` and `flag_finetuned` results for GAT, CARE-GNN, BWGNN and
DGA-GNN on both Reddit and Instagram (208 `+FLAG` runs, 200 `+FLAG*` runs — see
`results/tables/comparison.md`). GCN, GeniePath and PMP still only have
`baseline`. One paper claim (Figure 3(a)) has also been tested directly.

Still true, and important:

- **The `+text` variant has no results in the currently tracked aggregation.**
  `results/raw/` is git-ignored, and `results/aggregated/results.{csv,json}`
  (the committed source of truth) now contains only `baseline` (15 rows),
  `flag` (208 rows) and `flag_finetuned` (200 rows) — no `text` rows. Whatever
  produced the `+text` numbers an earlier revision of this ledger quoted is not
  part of the current committed record. Recorded as a gap to re-run, not
  silently dropped.
- **`flag`/`flag_finetuned` are not run for GCN, GeniePath or PMP.** Only GAT,
  CARE-GNN, BWGNN and DGA-GNN have them, on both datasets.
- **The LLM text cache has low node coverage** (0.6%-12.4%, decision D-004,
  generated at a reduced decode budget to control vast.ai cost), so most
  `flag`/`flag_finetuned` subgraphs use the documented raw-text fallback, not
  genuine LLM-generated text. A materially weaker test than Table 4's.
- **`flag_finetuned` is not LLM fine-tuning as reproduced here.** Decision D-001
  reproduces the released code's actual behaviour — the LoRA gradient path is
  severed, so the LLM is effectively frozen and only the GNN inner loop trains.
  Every `flag_finetuned` row carries `llm_finetuned: false`.
- **No result should be read as "reproduced".** Our benchmark uses our own
  downsampling seed, our own splits, and a reimplemented sampler, so exact
  agreement with Table 4 is not achievable in principle. See section 6.

| Status | Meaning |
|---|---|
| `NOT_STARTED` | Not attempted yet |
| `AUDITED` | Source understood; findings test-backed; not executed |
| `BLOCKED` | Attempted; cannot proceed; reason recorded |
| `PARTIAL` | Runs, with documented deviations |
| `REPRODUCED` | Ran, verified, compared against the paper |

---

## 1.1 What has actually been run

**Figure 3(a)** — semantic sampling raises subgraph homophily.
Full write-up: `research/figure3a_reproduction.md`.

| graph | verdict |
|---|---|
| original GLBench Instagram | **SUPPORTED** — SS > SS\* > RS ≈ FS' > NS, the paper's ordering |
| our 1:10 benchmark (both datasets) | **NOT SUPPORTED** — diagnosed, see below |

Diagnosed, not hand-waved: after downsampling only 1.2% of Reddit nodes have
degree > 10, so top-10 selection is a no-op for 98.8% of them and every strategy
picks the same neighbours; and at 1:10 a randomly wired graph already scores
0.835 homophily, leaving almost no headroom.

**Table 4, all 56 `baseline`/`flag`/`flag_finetuned` cells** (`+text` is
currently UNAVAILABLE — see §1), `threshold_policy=argmax`,
`impl_source=flag_bundled`. Full tables: `results/tables/comparison_vs_reported_*.md`.

| metric | MATCH (±1pp) | CLOSE (±2pp) | DEVIATION | UNAVAILABLE |
|---|---:|---:|---:|---:|
| F1-macro | 1 | 1 | **28** | 26 |
| AUC | 5 | 6 | **19** | 26 |

Three patterns, none of them flattering to a naive reading:

1. **Every available `+FLAG`/`+FLAG*` AUC is above the paper's**, by +0.35 to
   +5.52 points across GAT, CARE-GNN, BWGNN and DGA-GNN on both datasets — same
   direction as the `baseline` gap below, so this looks like the same protocol
   difference (split, downsampling draw, reimplemented sampler) propagating
   through `+FLAG`, not something specific to the LLM stage.
2. **The smallest `+FLAG*` deltas are close to MATCH** (DGA-GNN/reddit +0.46 AUC,
   BWGNN/instagram +0.35 AUC) while `baseline` deltas run much larger (e.g. GCN
   reddit baseline +9.42 AUC) — consistent with `+FLAG*`'s low text-cache
   coverage (D-004) diluting whatever the protocol difference is, since most of
   its "text" is actually the `baseline`/`text` fallback embedding.
3. **Our baselines are stronger on Reddit** (GCN +9.42 AUC, GeniePath +5.88,
   PMP +1.63, BWGNN +1.79) than on Instagram, where results are mixed (some
   `DEVIATION` in either direction). Whatever differs is dataset-specific.

Our baselines are systematically stronger than the paper's, most sharply on
Reddit AUC (GCN +9.42) where the paper's baseline sits essentially at chance
(50.32) while ours reaches 59.74. This is consistent with the unresolved
question about what the paper's "shallow embeddings" are: the stored features
we use are 4096-d and Llama-2-derived (`dataset_notes.md` section 7), not
shallow.

The decision-threshold policy alone moves F1 by 2-4 points. The paper states no
policy, so every result row records which one produced it.

**The `+text` gap.** `results/raw/` is git-ignored and the committed
`results/aggregated/results.{csv,json}` now contains only `baseline` (15 rows),
`flag` (208 rows) and `flag_finetuned` (200 rows) — no `text` rows survive in
the tracked record, even though this section previously reported `+text`
numbers from an earlier run. Not fabricated here and not silently dropped:
`+text` needs to be re-run and re-aggregated before it can be reported again.

---

## 2. Experiment matrix

Legend: `-` = not started.

Cells show which variants have been RUN and are in the currently tracked
aggregation, not whether they reproduce the paper. `b` = baseline, `t` = +text,
`f` = flag, `f*` = flag_finetuned. A cell is only marked when a run actually
completed and its result is in `results/aggregated/results.csv`.

| Dataset | GCN | GAT | GeniePath | CARE-GNN | BWGNN | DGA-GNN | PMP |
|---|---|---|---|---|---|---|---|
| Reddit | b | b, f, f* | b | b, f, f* | b, f, f* | b, f, f* | b |
| Instagram | b | b, f, f* | b | b, f, f* | b, f, f* | b, f, f* | b |
| YelpChi | - | - | - | - | - | - | - |
| Amazon | - | - | - | - | - | - | - |
| T-Finance | - | - | - | - | - | - | - |
| T-Social | - | - | - | - | - | - | - |
| Elliptic | - | - | - | - | - | - | - |
| Huabei (industrial) | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** | **N/A** |

`+text` (`t`) is not shown as run on any dataset/model: earlier `+text` result
files exist only in the git-ignored `results/raw/` and are not part of the
tracked aggregation (see §1). `flag`/`flag_finetuned` (`f`/`f*`) are `N/A` for
YelpChi/Amazon/T-Finance/T-Social/Elliptic (no native text, see below) and are
simply `-` (not yet run) for GCN, GeniePath and PMP on Reddit/Instagram.

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
| Dataset download | **DONE** | both verified against GLBench's published signature; sha256 recorded |
| Semantic similarity sampler | **DONE** | REIMPLEMENTED from Eq. 3-4 (no upstream source); 25 unit tests; reproduces Figure 3(a) on the original graph |
| 1:10 benchmark construction | **DONE** | REIMPLEMENTED; 20 unit tests; seeds + before/after counts in every manifest |
| Train/val/test splitting | **DONE** | stratified 10/10/80, inherited from GraphAdapter/GLBench and labelled as such |
| Internal data model | **DONE** | `BenchmarkGraph` + `Subgraph`; `original_node_ids` preserves provenance through re-indexing |
| Backbone adapters | **DONE** | all 7 wrapped without altering their mathematics; upstream constraints surfaced, not smoothed |
| Metrics module | **DONE** | AUC, F1-macro, KS, ECE, per-class; 30 tests; test-set tuning prevented by the API |
| Training loop + early stopping | **DONE** | subgraph loop, accumulation over 10, real early stopping (upstream parses `--patience` and never reads it) |
| Result storage / aggregation | **DONE** | one JSON per run; `impl_source` required; failures recorded, not dropped |
| Config system | `PARTIAL` | full CLI surface; YAML experiment configs still to come |
| Model/variant/dataset registry | **DONE** | refuses 133 of 224 combinations with reasons (Phase 24/41) |
| Reported-result comparison | **DONE** | `python -m analysis.compare_reported` (Phase 25) |
| LLM enhancement pipeline | **DONE** | ran on rented vast.ai GPU (D-003); reduced decode budget (D-004), low node coverage 0.6%-12.4% |
| LoRA fine-tuning | `PARTIAL` | reproduces the released code's actual (severed-gradient) behaviour per D-001, not LLM fine-tuning; `flag_finetuned` results exist for 4/7 backbones — see 4.1 |
| GPU support | **DONE, remote only** | no GPU on this development machine; `google/gemma-2-9b-it` run successfully on a rented vast.ai instance — see `docs/vastai_gpu_workflow.md` |

---

## 4. Open questions requiring a human decision

These are escalated per Phase 44: each materially changes research validity and
cannot be settled by an engineering judgement call.

### 4.1 What produced the `+FLAG*` column? — **DECIDED (D-001), action item still open**

**Resolved 2026-09-09, implemented since.** Option B below was decided and is
what the `flag_finetuned` GPU runs in this repo now report. The ambiguity and
options are kept here as the record of that decision; see `research/decisions.md`
D-001 for the full consequences and §1 above for current run counts. **Still
open:** contacting the authors (option C) has not been done.

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

### 5.1 No local GPU — mitigated by renting one

`torch.cuda.is_available() == False` on this development machine;
`nvidia-smi` is not installed here. Per decision D-003, the LLM stage is run on
a **rented vast.ai instance** instead, and its output (a checksummed text cache)
is brought back to this machine for everything downstream. Consequences:

- **Local GPU tests still skip rather than pass** — this machine itself remains
  untested for CUDA. The vast.ai run is real, but it is remote and ephemeral, not
  a standing capability of this repository's default environment.
- `gemma-2-9b-it` is ~18.5 GB in fp16, plus KV cache. The paper reports
  deployment on an **A100 80 GB**; the actual generation runs used ≥24 GB
  instances (`docs/vastai_gpu_workflow.md`).
- **`flag`/`flag_finetuned` have been run this way for 4 of 7 backbones**
  (GAT, CARE-GNN, BWGNN, DGA-GNN) on both datasets, at a reduced decode budget
  (D-004) that gives low node coverage (0.6%-12.4%). GCN, GeniePath and PMP
  remain not run under `flag`/`flag_finetuned`.
  - *CPU functional environment* — preprocessing, sampling, all GNN training,
    metrics, tests, small-scale runs. **Real and already exercised**: all 7
    backbones forward+backward on CPU.
  - *Full reproduction environment* — the GPU stage, now exercised remotely.

**We will not claim that Gemma-2-9B fine-tuning on CPU is practical** — it
isn't, and it was never attempted; the GPU stage always ran remotely.

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

Steps 1-6 of the original Phase-43 plan (dataset download, internal data model,
1:10 benchmark construction, semantic sampling, metrics, first baseline
end-to-end) are **done** — see §3. Remaining, in priority order:

1. **Re-run and re-aggregate `+text`.** `results/raw/` is git-ignored, so
   whatever produced the earlier `+text` numbers is not in the current tracked
   `results/aggregated/results.{csv,json}`. Re-run `baseline,text` for all 7
   backbones on both datasets and commit the refreshed aggregation.
2. **Run `flag`/`flag_finetuned` for GCN, GeniePath and PMP** on Reddit and
   Instagram — the same GPU pipeline already used for GAT, CARE-GNN, BWGNN and
   DGA-GNN.
3. **Consider regenerating the LLM cache at the paper-faithful decode budget**
   (`max_new_tokens=550, truncate_chars=1200`, D-003's original intent) — the
   current production cache (D-004) has low node coverage (0.6%-12.4%), so most
   `flag`/`flag_finetuned` subgraphs use the raw-text fallback rather than
   genuine LLM-generated text.
4. **Contact the authors** (D-001's still-open action item) about what produced
   the paper's `+FLAG*` column, since the released code's LoRA gradient path is
   severed.
5. Resolve question 4.4 (are the "shallow embeddings" actually shallow) by
   inspecting `data.x` on the downloaded tensors.
6. Revisit the licensing question (4.5) before any public release.

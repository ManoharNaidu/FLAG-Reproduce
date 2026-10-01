# Dataset Integration Audit

**Purpose.** Establish what the existing FLAG implementation actually consumes,
before any YelpChi/Amazon integration code is written. Nothing here proposes a
change; this is the contract that a later adapter must satisfy.

**Date:** 2026-09-18. **Method:** direct reading of the repository at the commit
in `git log -1`. Every claim below carries a `file:line` citation. Claims that
could not be settled by reading are marked **UNKNOWN** and were delegated to the
Loop-1 investigation agents (`research/_evidence/loop1_*.md`).

---

## 1. Executive summary

The canonical **graph** side of this task is already done and verified by prior
work in this repository. The canonical **text** side does not exist yet, and
whether it can exist at all is an open provenance question, not an engineering
one.

| Layer | YelpChi | Amazon | Status |
|---|---|---|---|
| canonical `.mat` obtained | `data/raw/yelpchi/YelpChi.mat` (207.7 MB) | `data/raw/amazon/Amazon.mat` (222.6 MB) | **present locally**, sha256 recorded |
| parsed graph payload | `data/benchmark/native_yelpchi/graph.pt` | `data/benchmark/native_amazon/graph.pt` | **built + manifested** |
| features / labels / relations | 45,954 x 32, 3 relations | 11,944 x 25, 3 relations | **verified against published stats** |
| raw node text | — | — | **absent; sourcing under investigation** |
| node -> original review/reviewer mapping | — | — | **UNKNOWN — the crux of this task** |
| wired into `flagbench` registry/runner | no | no | deliberately not done |

The blocking question is item 5-6, not item 1-4: the CARE-GNN `.mat` release is
believed to contain only features, labels and adjacency matrices — no review id,
no reviewer id, no text. If that is confirmed, then **no text can be attached to
a node without an alignment argument**, and per the brief's §55/§56 the honest
outcome may be `UNRESOLVED` rather than a mapping.

---

## 2. What the existing FLAG implementation consumes

### 2.1 The benchmark payload

`flagbench.experiments.runner.load_benchmark()` (runner.py:41-54) reads exactly
one path and attaches the sidecar manifest:

```
data/benchmark/flag_<dataset>/graph.pt      ->  payload: dict
data/benchmark/flag_<dataset>/dataset_manifest.json -> payload["_manifest"]
```

The payload is a plain `dict` written by
`scripts/preprocess/build_benchmark.py:133-142`:

| Key | Type | Consumed by |
|---|---|---|
| `x` | `Tensor [N, F]` | runner.py:97 (`baseline` variant features) |
| `edge_index` | `Tensor [2, E]` | sampling / subgraph construction |
| `y` | `Tensor [N]` | runner.py:334 (labels into trainer) |
| `raw_texts` | `list[str]`, len N | encode_text.py:58, generate_text.py:58 |
| `train_mask` / `val_mask` / `test_mask` | `Tensor [N]` bool | runner.py:312-314 |
| `original_node_ids` | `Tensor [N]` | provenance only |
| `label_names` | `list[str] \| None` | reporting only |

**Load-bearing:** `x`, `y`, the three masks, `raw_texts`, `edge_index`.
**Informational:** `original_node_ids`, `label_names`.

### 2.2 The derived caches

Three caches are built *from* the payload and keyed by dataset name. A new
dataset must produce all three before the corresponding variant can run:

| Cache | Producer | Consumer | Needed for |
|---|---|---|---|
| `cache/embeddings/<ds>__all-MiniLM-L6-v2__raw.pt` | `scripts/preprocess/encode_text.py` (reads `payload["raw_texts"]`, encode_text.py:58) | `runner.load_text_embeddings` (runner.py:78-86) | `text` variant |
| `cache/sampling/<ds>__<config.cache_key()>.pt` | `scripts/preprocess/sample_subgraphs.py` | `runner.load_subgraphs` (runner.py:57-75) | every variant |
| `cache/llm/...` | `scripts/llm/generate_text.py` (reads `payload["raw_texts"]`, generate_text.py:58) | `runner._llm_embeddings_path` (runner.py:130+) | `flag`, `flag_finetuned` |

The embedding cache is fingerprinted over the exact text corpus
(`encode_text.py:71-80`), so a text change correctly invalidates it. The LLM
cache key covers dataset, sampling config, prompt hashes, model id and decode
params (runner.py:130-140).

### 2.3 The variant switch

`runner.make_feature_fn` (runner.py:89-116) is the single point of divergence
between variants: `baseline` -> `payload["x"]`, `text` -> the Sentence-BERT
cache, `flag`/`flag_finetuned` -> a dual-branch path. **The only thing a variant
changes is which node features feed the GNN** (runner.py:4-7). This is a
structural fairness guarantee and must not be worked around.

### 2.4 The text encoder

`all-MiniLM-L6-v2`, defaulted in `runner.load_text_embeddings` (runner.py:78)
and in `scripts/preprocess/encode_text.py`. It is a parameter, not a hard-coded
constant, and the cache path embeds the model name (encode_text.py:43-45), so
substituting an encoder is safe and traceable. **Do not swap it silently.**

---

## 3. Current Reddit/Instagram assumptions that do NOT hold for YelpChi/Amazon

These are the reasons this cannot be a copy-paste of the GLBench path.

1. **Single relation.** The payload carries one `edge_index`. YelpChi and Amazon
   have three relations each. The existing native payload keeps them separate as
   `edge_index_relations: dict[str, Tensor]` plus a union `edge_index_homo`
   (`build_native_benchmark.py:190-204`) — a *different key set* from what the
   runner reads.
2. **1:10 downsampling.** `flagbench.datasets.benchmark` exists to *create*
   imbalance in the near-balanced Reddit/Instagram graphs (benchmark.py:1-19).
   YelpChi/Amazon are natively imbalanced (14.5% / 6.9% fraud) and carry their
   own split convention from CARE-GNN. Running them through that construction
   would invent an unpublished variant — prior work documented this refusal in
   `experiments/yelpchi_amazon/README.md:13-21`.
3. **Splits.** Reddit/Instagram use a 10/10/80 stratified split inherited from
   GLBench (benchmark.py:51-56). The native payload instead reproduces
   CARE-GNN's own `train_test_split(test_size=0.60, random_state=2)` and carves
   a validation fold out of it (documented as *our* addition in the native
   manifests).
4. **Unlabeled nodes.** Amazon has 3,305 unlabeled nodes at indices `[0, 3305)`
   that CARE-GNN excludes. The GLBench path has no concept of an unlabeled node,
   and the payload schema has no `unlabeled_mask`.
5. **Node semantics.** Reddit/Instagram nodes are users with one text each.
   YelpChi nodes are **reviews** (one text each — a clean match). Amazon nodes
   are **users** (many reviews each — text must be *aggregated*, which is a
   derived construct and must be labelled as such).
6. **Native text.** GLBench ships `raw_texts` inside the source `.pt`
   (glbench.py:205). The `.mat` files ship none.

## 4. The registry gate

`DATASET_REGISTRY` (registry.py:284-320) marks both datasets
`has_native_text=False` with the note *"the FLAG paper states it lacks textual
information"*, which hard-blocks the `text`, `flag` and `flag_finetuned`
variants at validation time.

This gate is **research policy, not a bug**. README §4 and operating principle 5
state that fabricating text for a text-free dataset and calling it FLAG is a
research-integrity failure, and that a text-augmented study is permitted only as
a separate experiment type stamped `native_text: false`, never merged with the
canonical reproduction.

Consequence for this task: sourcing *genuine* Yelp/Amazon review text is not
fabrication and is legitimate — but it does **not** make these datasets part of
the canonical FLAG reproduction, because the FLAG paper never ran them with
text. Any result obtained this way is a **new experiment**, not a reproduction of
a published number. Whether the gate should be opened, and how the resulting
runs should be stamped, is a decision for the maintainer (raised in
`research/decisions.md` as a proposed D-005), not something to flip silently.

## 5. Safe integration point

The lowest-risk design, requiring **no change to FLAG's core**:

```
 .mat (canonical, untouched)          raw reviews (new)
          |                                   |
          v                                   v
   build_native_benchmark            text sourcing + node mapping
          |                                   |
          +-----------------+-----------------+
                            v
                  unified dataset object
          (x, y, raw_texts, relation_edges, edge_index, masks,
           original ids, mapping metadata, provenance labels)
                            |
                  +---------+---------+
                  v                   v
        native/relation-aware    flag_<ds>/graph.pt-shaped payload
        consumers (CARE-GNN,     -> existing encode_text / sample_subgraphs
        PMP, future baselines)      / generate_text / runner, unchanged
```

The adapter writes a payload with the keys the runner already reads, deriving a
single `edge_index` from the union of relations **while preserving
`relation_edges` alongside it** — the homogeneous view is a derived artefact and
is documented as such, never a replacement.

## 6. Files that must remain UNCHANGED

Verified as core; an adapter must work around them, not through them:

- `src/flagbench/experiments/runner.py` — the variant/fairness contract
- `src/flagbench/training/trainer.py`, `src/flagbench/training/flag_losses.py`
- `src/flagbench/sampling/semantic.py`
- `src/flagbench/adapters/backbone.py` and everything under `methods/`
- `src/flagbench/metrics/classification.py`
- `src/flagbench/datasets/benchmark.py` — encodes the FLAG 1:10 construction
- `experiments/yelpchi_amazon/build_native_benchmark.py` and the two
  `data/benchmark/native_*/` payloads — already verified; treat as read-only
  inputs

Two files will need *additive* change and are flagged for approval rather than
edited silently:

| File | Why | Proposed minimal change |
|---|---|---|
| `src/flagbench/registry/registry.py` | `has_native_text=False` blocks every text variant for these datasets | add an explicit, separately-named text-augmented experiment type; do **not** flip the flag in place |
| `research/decisions.md` | a research decision of this size must be recorded | add D-005 covering the text-augmented study and its labelling |

## 7. Environment blockers found during the audit

1. **`.venv-cpu` is broken.** Its base interpreter
   (`C:\Users\hp\AppData\Local\Programs\Python\Python311`) has been uninstalled;
   every `.venv-cpu/Scripts/python.exe` invocation fails with *"No Python at ..."*.
   The machine now has Python 3.12 and a miniforge base, neither with `torch`.
   **Consequence:** the repo's documented CPU pipeline cannot run at all until a
   venv is rebuilt, and `environment/cpu.lock.txt` is pinned for 3.11
   (`torch==2.3.1+cpu`), so a rebuild is a re-pinning decision, not a re-install.
2. **A usable interpreter exists for inspection work**:
   `C:\Users\hp\miniforge3\envs\dgp-bl-consisgad\python.exe` (Python 3.9, torch
   1.13.1+cpu, scipy 1.9.3, numpy 1.23.5, sklearn, pandas, dgl 1.1.0). Sufficient
   for `.mat` parsing, mapping work and validation; **insufficient** for the FLAG
   forward-pass compatibility test (§48 of the brief), which needs
   `torch_geometric` and `sentence_transformers`.
3. **`https://jmcauley.ucsd.edu` fails TLS** with `SEC_E_CERT_EXPIRED`. That is
   the canonical host for the Amazon review corpus, so canonical-source
   acquisition may require a documented provenance caveat.

---

## 8. Open questions handed to Loop 1

| # | Question | Agent |
|---|---|---|
| Q1 | Does either `.mat` contain any node -> review/reviewer identifier? | A |
| Q2 | Can the canonical raw Amazon Musical Instruments file be obtained, and does its unique-reviewer count match 11,944? | B |
| Q3 | Is the original YelpChi review text obtainable, and does any join key exist? | C |
| Q4 | Exact FLAG input contract, relation handling, minimal adapter surface | D |
| Q5 | Version discrepancies (Amazon 24 vs 25 features), and has anyone ever *proved* a node<->review mapping? | E |

Answers land in `research/_evidence/` and are consolidated into
`research/evidence_matrix.md` before any loader is written.

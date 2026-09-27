# Final Dataset Integration Report — YelpChi & Amazon

**Date:** 2026-09-19. **Scope:** make the canonical YelpChi and Amazon fraud
benchmarks consumable by the FLAG implementation through one dataset
abstraction, with provenance strong enough to defend.

---

## 1. Executive summary

Both datasets are integrated and pass a FLAG forward pass. The load-bearing
result is not the plumbing — it is that **the node -> raw-review mapping was
proven, not assumed**, which no published work had done before.

| | YelpChi | Amazon |
|---|---|---|
| node type | review | user |
| nodes / features | 45,954 / 32 | 11,944 / 25 |
| relations | rur, rtr, rsr | upu, usu, uvu |
| text mapping | **EXACT**, all 45,954 | **EXACT** for 8,639; 3,305 UNRESOLVED |
| text coverage | 100% | 72.3% |
| evaluated nodes lacking text | **0** | **0** |
| FLAG forward pass | PASS | PASS (all 7 backbones) |

**The single most important caveat:** the FLAG paper reports **no** YelpChi or
Amazon results and states both datasets lack textual information. Nothing here
reproduces a published number. Every artifact is stamped
`experiment_type: text_augmented_study`, `is_flag_reproduction: false`.

## 2. Sources

| Artefact | Source | Provenance |
|---|---|---|
| `YelpChi.mat`, `Amazon.mat` | CARE-GNN release (already local) | VERIFIED, sha256 recorded |
| Yelp metadata + review text | Mukherjee et al. ICWSM 2013 corpus, via mirror | MIRROR, authenticated by counts |
| `reviews_Musical_Instruments.json.gz` (2014 full) | McAuley, via `cseweb.ucsd.edu` -> `snap.stanford.edu` | AUTHORITATIVE |

`jmcauley.ucsd.edu` is dead (cert expired 2026-05-21). Reached the data through
McAuley's own live `cseweb` page instead. **No certificate verification was
bypassed anywhere.** `shebuti.com` is email-gated and `odds.cs.stonybrook.edu`
is unreachable, hence the Yelp mirror — authenticated by matching the author's
published counts exactly (67,395 / 38,063 / 201 / 8,919).

## 3. The mapping problem, and how it was solved

Neither `.mat` contains any identifier — 6 variables each, all numeric or
sparse, no struct/cell/char anywhere. So text cannot be looked up; it must be
reconstructed and proven.

**YelpChi.** Dropping the 19 products with >800 reviews and ordering by user
first-appearance reproduces the canonical node set. Verification compared
**sparsity patterns entry by entry**, not totals:

| check | result |
|---|---|
| labels | 45,954 / 45,954 |
| R-U-R | 98,630 edges, **0 mismatched** |
| R-S-R | 6,805,486 edges, **0 mismatched** |
| R-T-R | 412,824 vs 1,147,232 — NOT reproduced |

Only the date-dependent relation fails; the mirror's `date` column is from a
different snapshot. Two date-free relations plus the label vector already pin
the ordering, so this does not weaken the proof.

**Amazon.** The generator is in the repo (`methods/care_gnn/amazon_preprocess.py`).
It builds the cohort as `{**sampled_reviews, **labeled_reviews}`, which explains
the all-zero `[0,3305)` prefix, and fills `labeled_reviews` in reviewerID
first-appearance order. From 339,231 reviewers: 8,639 labelled, and
`int(330,592 x 0.01)` = 3,305 sampled -> **11,944 exactly**.

| check | result |
|---|---|
| cohort arithmetic | 3,305 + 8,639 = 11,944 |
| label sequence | 8,639 / 8,639 (chance ~82.8%) |
| induced U-P-U | 294,764 edges, **0 mismatched** |

## 4. What could NOT be resolved

**Amazon nodes [0, 3305) — UNRESOLVED, deliberately.** They are a 1% sample
drawn by `rd.sample` over a *set-difference* list, whose iteration order depends
on `PYTHONHASHSEED`; `rd.seed(1)` does not make it reproducible. Fingerprint
recovery over the 16 exactly-computable feature columns gave only 562/3,305
unique (17.0%), because the 330,592-user pool collapses to 90,129 distinct
fingerprints — most users have one review. These nodes carry **empty text and an
UNRESOLVED stamp**, never a guess. They are excluded from every split, so no
evaluated node is affected.

**YelpChi feature semantics — UNKNOWN.** A feature-name file exists, but no
column behaves like a monotone function of review length or rating (a rating
control, provably aligned, also fails). So the 32 columns could not be used to
corroborate the text join, and the join rests on the statistical test instead.

**R-T-R — NOT reproduced.** ~35 date definitions tried.

## 5. Unified format

`flagbench.fraud_text.FLAGDataset` — one container, both datasets:

```
dataset_name, node_type, node_features, labels, raw_texts, text_status,
edge_index, relation_edges, relation_names, train_idx, val_idx, test_idx,
unlabeled_mask, node_metadata, mapping_metadata, dataset_metadata, provenance
```

Canonical tensors are reproduced exactly. `edge_index` is a DERIVED single-graph
view kept **alongside** `relation_edges`, never replacing it. Every field carries
a provenance label (CANONICAL / DERIVED / VERIFIED / UNRESOLVED).

```python
from flagbench.fraud_text import load_dataset
ds = load_dataset("yelpchi")   # node_type == "review"
ds = load_dataset("amazon")    # node_type == "user"
```

## 6. FLAG compatibility

`flagbench.flag_adapter` writes the exact payload
`flagbench.experiments.runner` already reads. **No FLAG core file was
modified** — not the runner, trainer, sampler, metrics, adapters, or any bundled
backbone.

One core file did change, with prior approval: `registry/registry.py` gained a
`text_source` field and two new dataset keys (decision **D-005**). A new dataset
key cannot exist without a registry row, so this was unavoidable. Every existing
row was set explicitly and behaviour was verified unchanged.

Forward pass, 512-node induced subgraph, real MiniLM embeddings:

| dataset | backbones | result |
|---|---|---|
| yelpchi | gcn, gat, bwgnn | logits (512, 2), finite, PASS |
| amazon | all 7 | logits (512, 2), finite, PASS |

**Three relations are collapsed** for any FLAG run, because nothing in this repo
consumes a multi-relation graph. Recorded in `edge_index_provenance` on every
payload.

## 7. Problems encountered

1. **`.venv-cpu` is broken** — its base Python 3.11 was uninstalled. Work was
   done in `miniforge3/envs/dgp-bl-consisgad` (inspection) and a new
   `.venv-fraudtext` (forward pass).
2. **`.venv-fraudtext` runs torch 2.8.0**, not the repo's pinned 2.3.1 — pip
   upgraded it as a `sentence-transformers` dependency. The repo's numerical
   equivalence test (`tests/unit/test_compat_torch_scatter.py`, the one that
   caught the torch 2.4.0 miscomputation) **passes 9/9** there, but this is not
   the validated environment and no numeric result should be taken from it.
3. **`datasets/` shadows the HuggingFace `datasets` package** whenever cwd is
   the repo root — confirmed, and it already broke a `sentence-transformers`
   import. See section 9.
4. Subagent investigations were cut short twice by API spend limits; the
   remaining work was completed in-session.

## 8. Reproduction

```bash
# canonical graphs (already built)
python -m experiments.yelpchi_amazon.build_native_benchmark --dataset all

# prove the mappings, emit mapping csvs
python -m scripts.verify_yelpchi_alignment
python -m scripts.verify_amazon_alignment

# build, validate, and emit FLAG payloads
python -m scripts.prepare_dataset --dataset all
python -m scripts.validate_dataset --dataset all
python -m scripts.smoke_test_datasets

# compatibility
python -m scripts.test_flag_dataset_compatibility --dataset yelpchi
python -m scripts.test_flag_dataset_compatibility --dataset amazon --models gcn,gat,geniepath,care_gnn,bwgnn,dga_gnn,pmp
python -m pytest tests/test_fraud_text_datasets.py     # 43 passed
```

## 9. Unresolved issues / recommended next steps

1. **Rename `datasets/`.** It shadows HuggingFace `datasets` from the repo root.
   The brief specified this layout, but `pyproject.toml` already documents this
   exact collision class as the reason the repo avoids flat top-level packages.
   Recommend `data_sources/` or folding it under `data/`. **Needs your call.**
2. **Rebuild a pinned environment.** `.venv-cpu` is dead and `.venv-fraudtext`
   is on an unvalidated torch. Nothing numeric should be run until this is
   settled.
3. **Correct `data/benchmark/native_amazon/dataset_manifest.json`** — it calls
   `homo` the "union" of the three relations. True for YelpChi, false for
   Amazon (Jaccard 0.626). Logged, not silently edited.
4. **Restate three Amazon numbers** in the native manifest: "11,123 benign" is
   `label==0` *including* 3,305 unlabelled (true benign is 7,818); the 6.9%
   fraud rate is over all nodes where the literature quotes 9.5% over labelled
   nodes; edge counts are directed (2x the published undirected table).
5. **Amazon 24-vs-25 features** — no source claiming 24 was found; our file has
   25 and DGL does not slice. Left UNKNOWN rather than resolved.
6. Semantic sampling and LLM caches have **not** been built for these datasets,
   so `flag`/`flag_finetuned` runs are not yet possible — only `baseline` and
   `text`. YelpChi's 7.7M-edge homo view may make sampling expensive; untested.

## 10. Files generated

```
research/  dataset_integration_audit.md, evidence_matrix.md,
           FINAL_DATASET_INTEGRATION_REPORT.md, _evidence/loop1_{a..e}*.md
           decisions.md (D-005 appended)
scripts/   verify_yelpchi_alignment.py, verify_amazon_alignment.py,
           build_flag_dataset.py, prepare_dataset.py, validate_dataset.py,
           smoke_test_datasets.py, test_flag_dataset_compatibility.py
src/flagbench/fraud_text/    base, yelpchi, amazon, loaders, validation, serialization
src/flagbench/flag_adapter/  dataset_adapter, graph_adapter, text_adapter
src/flagbench/registry/registry.py   (D-005: text_source + 2 new keys)
tests/     test_fraud_text_datasets.py (43 tests)
datasets/  raw/, processed/{yelpchi,amazon}/, manifests/
data/benchmark/flag_{yelpchi,amazon}_text/
```

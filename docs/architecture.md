# Architecture

How the pieces fit, and — more usefully — *why* they are arranged this way. Most
of these decisions are consequences of what the audit found, not preferences.

## The pipeline

```
data/raw/            GLBench .pt, written once, never modified
    |                verified against published node/edge counts on download
    v
data/benchmark/      FLAG's 1:10 fraud transformation
    |                REIMPLEMENTED; every stochastic choice seeded + manifested
    v
cache/embeddings/    Sentence-BERT of raw node text  (B(.) of Eq. 3)
    |
    v
cache/sampling/      semantic similarity subgraphs   (Eq. 3-4)
    |                REIMPLEMENTED; no upstream source exists
    |
    +---------------> [ GPU BOUNDARY ] -------------------------+
    |                 cache/llm/  discriminative + residual text |
    |                 gemma-2-9b-it, rented instance             |
    v                                                            v
results/raw/         one JSON per run  <---- training loop <-----+
    |
    v
results/aggregated/  results.csv / results.json
results/tables/      comparison.md, comparison_vs_reported_*.md
```

## Directory map

| Path | Contents |
|---|---|
| `methods/` | **Unmodified upstream source.** Git-ignored; fetched at pinned SHAs. Never edited. |
| `src/flagbench/` | Everything we wrote |
| `src/flagbench/compat/` | `sys.path` shim for the crashing `torch_scatter` wheel |
| `scripts/` | CLI entry points, one per pipeline stage |
| `analysis/` | Comparison against the paper's reported numbers |
| `research/` | The audit. The most important directory here. |
| `configs/` | Declarative dataset / model / experiment definitions |
| `tests/` | 129 tests; the unit tests are the spec for reimplemented components |

## Load-bearing design decisions

### 1. Upstream is never edited

`methods/` is git-ignored and reconstructed by `scripts/setup/fetch_methods.sh`
at pinned commits. Two reasons:

- **Licensing.** FLAG, BWGNN, DGA-GNN and PMP ship no LICENSE, so we have no
  redistribution rights. Pinning SHAs preserves provenance without re-hosting.
- **Falsifiability.** `tests/integration/test_flag_upstream_claims.py` probes the
  *pristine* source. If upstream changes, those tests tell us which of the 36
  audit findings went stale. That only works if nobody has edited it.

Adaptations live in `src/flagbench/adapters/`, environment shims in
`src/flagbench/compat/`. The one behavioural repair we apply (DGA's dropped
`dim_size`) patches the class **in memory** at construction time.

### 2. Two implementation lineages, never merged

`impl_source` is a required field on every result row.

| value | reproduces |
|---|---|
| `flag_bundled` | **the paper's numbers** — Table 4 came from FLAG's own rewrites |
| `official` | **the baseline as its authors published it** |

Four of the five bundled baselines are not their published algorithms — CARE-GNN
has no RL filtering, BWGNN uses raw adjacency instead of the normalised
Laplacian, DGA-GNN has no grouping, GAT's second layer is a `SAGEConv`. Reporting
one as the other would misattribute performance, so the comparison tool groups by
`impl_source` and never averages across it.

### 3. A variant changes features, and nothing else

Phase 11's fair-comparison rule is enforced structurally rather than by
discipline. `runner.make_feature_fn` is the *single* point of divergence between
`baseline`, `+text`, `+FLAG` and `+FLAG*`. The dataset, splits, subgraphs,
metrics, selection rule, seeding and training loop are identical.

The one deliberate exception is neighbour sampling, which is a *per-variant*
property: semantic sampling is a FLAG contribution (Table 5 ablates it as a
component), so `baseline` and `+text` use plain 2-hop neighbourhoods. Giving the
baseline SS would hand it part of the method it is a baseline for — a bug we
actually shipped for one commit before catching it.

### 4. The test set cannot be used to pick a threshold

Enforced by the API, not by convention:

```python
threshold, record = fit_threshold(val_labels, val_scores, policy=...)  # val only
test_eval = evaluate(test_labels, test_scores, threshold)              # no policy arg
```

`evaluate()` has no `policy` parameter, so it *cannot* search. A test asserts
that signature. `evaluate_val_and_test()` wires the correct order so the right
thing is the easy thing.

### 5. Two seed axes, kept independent

The paper runs 25 = 5 seeds x 5 initialisations. If both axes derived from one
seed, `(seed=0, init=1)` and `(seed=1, init=0)` would not be independent samples
and the reported standard deviation would understate true variance.
`RunIdentity.stream(name)` hashes a stream name with the *relevant* axis only —
`init` never depends on `seed`.

### 6. Refuse, never substitute

`registry.validate()` rejects 133 of 224 (dataset, model, variant) combinations
with a reason and suggested alternatives. The important one: FLAG variants on
text-free datasets. The paper itself says Yelp/Amazon/T-Finance/T-Social lack
text, so running "FLAG" there would require fabricating it.

The same principle covers the GPU boundary — requesting an LLM variant on CPU
raises an explanatory error rather than quietly falling back to a smaller model
whose numbers could not be compared to the paper's.

### 7. Failures are recorded

A crashed run still writes a row with `status="failed"` and the error. Silently
missing rows are how a benchmark ends up reporting only the runs that worked.
This is how the DGA `dim_size` bug surfaced: 20% of Reddit nodes are isolated
after downsampling, and the failed rows said so.

### 8. Namespaced under `flagbench`

`pyproject.toml` sets `package-dir = {"" = "src"}`, so anything directly under
`src/` becomes a top-level importable name. A leftover empty `src/datasets/`
shadowed HuggingFace's `datasets` and broke `sentence_transformers` with a
misleading error. `src/utils/` or `src/models/` would collide with upstream
FLAG's own modules once `methods/flag` is on the path.
`tests/unit/test_package_layout.py` guards this and was confirmed to fire when
the bug is reintroduced.

## Adding a dataset

1. Add a `DatasetSpec` to `DATASET_REGISTRY` — `has_native_text` and
   `num_relations` drive compatibility checking.
2. Add a loader + verification signature (see `datasets/glbench.py`).
3. Add `configs/datasets/<name>.yaml`.
4. If it has no native text, that is all: the FLAG variants are refused
   automatically.

No model code changes.

## Adding a model

1. Add a `ModelSpec` to `MODEL_REGISTRY` with an **honest** `fidelity` string and
   the correct `impl_source`.
2. Add construction to `FlagBundledBackbone.__init__`, or write an adapter if the
   signature differs materially.
3. Declare `Capabilities` — especially `requires_multi_relation`.
4. Add `configs/models/<name>.yaml`.
5. Add a forward/backward test.

Do not alter the model's mathematics to fit the interface. If it will not fit,
write an adapter and document why.

## Known structural limits

- **Full-graph methods.** Official BWGNN trains full-graph (its own issue #15
  asks about minibatching). The subgraph loop cannot host it unchanged; the
  `official` lineage will need its own runner.
- **Multi-relation.** `BenchmarkGraph` carries a single `edge_index`. Official
  CARE-GNN needs per-relation edges, so that lineage needs a multi-relation
  container before it can run.
- **`flag_finetuned`.** Blocked on decision D-001 as well as on GPU access —
  upstream's LoRA gradient path is severed, so what the paper's `+FLAG*` column
  measures is genuinely unknown.

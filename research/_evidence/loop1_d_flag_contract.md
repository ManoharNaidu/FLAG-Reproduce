# Loop 1 / Subagent D — FLAG input contract & YelpChi/Amazon integration specification

**Scope.** Static reading only. The project venv is broken (torch / PyG unavailable), so
nothing here was executed. Every claim is cited `file:line`. Where reading alone cannot
settle a question it is marked **UNKNOWN**.

**Repo root.** `D:\Campus Courses\Sem 4\Major Project\Codes\FLAG Reproduce`
All paths below are relative to that root unless absolute.

---

## 1. The full input contract of `graph.pt`

### 1.1 Where the payload is written

`scripts/preprocess/build_benchmark.py:134-146` constructs and saves the dict:

```
payload = {
    "x": graph.x,                              # build_benchmark.py:135
    "edge_index": graph.edge_index,            # :136
    "y": graph.y,                              # :137
    "raw_texts": graph.raw_texts,              # :138
    "train_mask": graph.train_mask,            # :139
    "val_mask": graph.val_mask,                # :140
    "test_mask": graph.test_mask,              # :141
    "original_node_ids": graph.original_node_ids,  # :142
    "label_names": graph.label_names,          # :143
}
torch.save(payload, out_dir / "graph.pt")      # :145-146
```

Destination is `data/benchmark/flag_<dataset>/graph.pt` (`build_benchmark.py:68`, `:145`).
A sibling `dataset_manifest.json` is written at `build_benchmark.py:158-160`.

### 1.2 Where it is read

| Reader | Line | Keys touched |
|---|---|---|
| `flagbench.experiments.runner.load_benchmark` | `src/flagbench/experiments/runner.py:41-54` | loads whole dict; injects `_manifest` at `:50-53` |
| `runner.make_feature_fn` | `runner.py:96-102` | `x` |
| `runner.run_single` | `runner.py:288-291` | `_manifest` → `preprocessing_version`, `output_sha256` |
| `runner.run_single` | `runner.py:312-314` | `train_mask`, `val_mask`, `test_mask` |
| `runner.run_single` | `runner.py:333-336` | `y` (passed to `SubgraphTrainer` as `labels`) |
| `scripts/llm/generate_text.py` | `:57-58` | `raw_texts`; `:132-135` `_manifest` |
| `scripts/preprocess/encode_text.py` | `:54-58` | `raw_texts` |
| `scripts/preprocess/sample_subgraphs.py` | `:55-63` (`load_inputs`), `:129-133`, `:210-213` | `edge_index`, `y`, `test_mask`, and `x` (only for `strategy="feature"`, `:158`) |

### 1.3 Per-field contract

| Key | Load-bearing? | Dtype / shape | Evidence |
|---|---|---|---|
| `x` | **Yes** for `variant=baseline` only | float tensor `[N, F]`. Written as `data.x[kept]` (`benchmark.py:283`) — dtype inherited from the source graph, not cast. Consumed as `features[subgraph.subset]` (`runner.py:113-114`); `in_dim = features.shape[1]` (`runner.py:116`). Also used by sampler strategy `feature` (`sample_subgraphs.py:158`). | `benchmark.py:283`, `runner.py:96-116` |
| `edge_index` | **Yes** for sampling; **not read by the runner at all** | long `[2, E]`, contiguous 0..N-1 re-indexed (`benchmark.py:243-246`). Only consumed by `sample_subgraphs.load_inputs` → `semantic.build_adjacency` (`sample_subgraphs.py:129-130`, `:211-213`). At train time the runner uses only `Subgraph.edge_index` from the sampling cache (`trainer.py:168`). | `benchmark.py:230-260`, `sample_subgraphs.py:129`, `trainer.py:168` |
| `y` | **Yes** | **long**, shape `[N]`, values in {0,1}. Produced as `data.y.long().flatten()[kept]` (`benchmark.py:273`, `:282`). Must be long: it feeds `CrossEntropyLoss` targets via `self.labels[sg.central].reshape(1)` (`trainer.py:209-210`, criterion at `trainer.py:151`) and `torch.bincount(self.y.long(), minlength=2)` (`benchmark.py:102`). `minlength=2` hard-assumes **binary** labels. | `benchmark.py:102`, `:273`, `:282`, `trainer.py:151`, `:209-210` |
| `raw_texts` | **Yes** for `text` / `flag` / `flag_finetuned`; unused for `baseline` | `list[str]`, `len == N`. Source-side invariant enforced at `glbench.py:264-270` (length == num_nodes, element is `str`). Consumed by `encode_text.py:58`, `generate_text.py:58`, `:98`, `enhance.py:296`, `:323`. | `benchmark.py:284`, `glbench.py:264-270` |
| `train_mask` / `val_mask` / `test_mask` | **Yes** | **boolean** tensors of shape `[N]`, NOT index tensors. Allocated `torch.zeros(n, dtype=torch.bool)` (`benchmark.py:192-194`). Runner converts with `torch.nonzero(mask).flatten().tolist()` (`runner.py:309`) — that would *silently mis-behave* on an index tensor (it would return positions of non-zero entries, dropping index 0). Invariants asserted at build time: pairwise disjoint and exhaustive (`benchmark.py:213-216`). | `benchmark.py:192-194`, `:213-216`, `runner.py:308-314` |
| `original_node_ids` | **Informational only** | long `[N]`, sorted, maps new index → original index (`benchmark.py:156-157`, `:296`). **No reader anywhere in `src/`, `scripts/` or `experiments/`** — grep finds only the write site and the dataclass declaration (`benchmark.py:87`, `:296`; `build_benchmark.py:142`). Pure provenance. | grep over `src scripts experiments` |
| `label_names` | **Informational only** | `list[str] | None` (`benchmark.py:90`, `:297`). Never read downstream; `build_benchmark.minority_class_for` reads `label_name` from the **raw** GLBench object, not from this payload (`build_benchmark.py:48-62`). | `build_benchmark.py:143`, no reader |
| `_manifest` | Informational, injected at load | dict; `{}` if the JSON is absent (`runner.py:50-53`). Used for `dataset_version` and `dataset_manifest_sha256` result columns (`runner.py:290-291`) and for the LLM manifest (`generate_text.py:132-135`). Missing file is tolerated. | `runner.py:49-53` |

**Summary of the hard requirements.** A payload is runnable by `run_single` iff it has
`x` (for `baseline`), `y` (long, binary), and the three **boolean** masks. `edge_index` is
required only to *build* the sampling cache. `raw_texts` is required for every non-baseline
variant. `original_node_ids` and `label_names` are inert.

**Not validated at load.** `load_benchmark` (`runner.py:41-54`) performs **no** schema
check: no key presence check, no dtype check, no length agreement check. A malformed
payload surfaces as a `KeyError`/`IndexError` inside `run_single`'s try block
(`runner.py:287`, `:412-415`), which records `status="failed"` rather than raising. The
only cross-file consistency check anywhere is in the sampler script:
`embeddings.shape[0] != payload["y"].shape[0]` → `RuntimeError` (`sample_subgraphs.py:99-104`).

---

## 2. `flagbench.datasets.benchmark.BenchmarkGraph`

### 2.1 The dataclass

`src/flagbench/datasets/benchmark.py:76-118`. Mutable `@dataclass` (not frozen).

Fields, in declaration order (`:80-91`):

```
x: torch.Tensor                       # :80
edge_index: torch.Tensor              # :81
y: torch.Tensor                       # :82
raw_texts: list[str]                  # :83
train_mask: torch.Tensor              # :84
val_mask: torch.Tensor                # :85
test_mask: torch.Tensor               # :86
original_node_ids: torch.Tensor       # :87  (new index -> original index)
label_names: list[str] | None = None  # :90
metadata: dict = field(default_factory=dict)   # :91
```

Methods:
- `num_nodes` → `int(self.y.shape[0])` (`:93-95`) — **node count is defined by `y`**, not by `x`.
- `num_edges` → `int(self.edge_index.shape[1])` (`:97-99`).
- `class_counts()` → `torch.bincount(self.y.long(), minlength=2)` (`:101-103`) — binary assumption.
- `split_counts()` → per-split `n` and per-class counts (`:105-118`); indexes `self.y[mask]`, so masks must be boolean.

`metadata` is assigned the full manifest after construction (`:335`), and is **not** part of
the saved payload (`build_benchmark.py:134-144` omits it).

### 2.2 How `build()` constructs it

`benchmark.py:263-336`. Input `data` is a GLBench PyG `Data` (docstring `:270`).

1. `y_all = data.y.long().flatten()`; `n_original = y_all.shape[0]` (`:273-274`).
2. `kept, downsample_record = select_minority_subset(y_all, config)` (`:276`).
3. `edge_index, edge_record = induce_subgraph(data.edge_index, kept, n_original, config.drop_self_loops)` (`:278-280`).
4. `y = y_all[kept]`, `x = data.x[kept]`, `raw_texts = [data.raw_texts[i] for i in kept.tolist()]` (`:282-284`).
5. `train/val/test = make_splits(y, config)` (`:286`).
6. `BenchmarkGraph(...)` with `original_node_ids=kept` and
   `label_names=list(getattr(data, "label_name", []) or []) or None` (`:288-298`).
7. Manifest assembled (`:300-334`), assigned to `graph.metadata` (`:335`).

### 2.3 Validation performed

There is **no** `__post_init__` and no field validator on `BenchmarkGraph`. All checking is
procedural, inside the helpers:

| Check | Where | Behaviour on failure |
|---|---|---|
| `split_ratios` sum to 1.0 ±1e-6 | `benchmark.py:185-188` | `ValueError` |
| Splits pairwise disjoint | `benchmark.py:213-215` | bare `assert` (disabled under `python -O`) |
| Splits exhaustive (`|train∪val∪test| == n`) | `benchmark.py:216` | bare `assert` |
| Minority target ≥ available → keep all, log a warning | `benchmark.py:143-150` | **warns, does not fail** |
| `Subgraph` centre appears exactly once in `subset` | `semantic.py:143-148` | `ValueError` (sampler-side, not benchmark-side) |
| Source-graph signature / field presence / text length | `glbench.py:205`, `:264-270` | `DatasetVerificationError`; invoked at `build_benchmark.py:78` **before** `build()` |

Note `glbench.verify` (`build_benchmark.py:78`) is a *GLBench-only* gate: it is keyed off
`glbench.SIGNATURES`, which contains only `reddit` and `instagram`
(`src/flagbench/datasets/glbench.py:59-87`). So `build_benchmark.py` cannot be pointed at a
non-GLBench source without either adding a signature or bypassing `process()`.

### 2.4 Assumptions that only hold for GLBench-style data

1. **1:10 downsampling of the minority class.** `select_minority_subset`
   (`benchmark.py:121-177`) keeps the majority whole and subsamples the minority to
   `|majority| / imbalance_ratio` (`:140`). This exists because Reddit/Instagram are
   near-balanced. YelpChi (39,277 / 6,677 ≈ 5.9:1) and Amazon are *already* imbalanced;
   applying this would fabricate a third dataset variant. `build_native_benchmark.py:246-250`
   and `experiments/yelpchi_amazon/README.md` say exactly this and deliberately skip it.
2. **Self-loop dropping.** `drop_self_loops=True` default (`benchmark.py:64-68`), applied in
   `induce_subgraph` (`:250-251`). Justified by GLBench shipping `add_self_loops` graphs.
   The `.mat` relations are not self-loop-augmented in the same way — **UNKNOWN** whether
   YelpChi/Amazon `homo` contains self-loops (cannot execute to check; `is_symmetric` is
   recorded in the manifest but self-loop count is not).
3. **Contiguous re-indexing + `original_node_ids`.** `remap` at `benchmark.py:243-246`
   produces 0..len(kept)-1. A payload built by `build_native_benchmark.py` is *already*
   0..N-1 with no subsetting, so `original_node_ids` would be the identity — correct but
   vacuous.
4. **`label_name` attribute on the source object** (`benchmark.py:297`,
   `build_benchmark.py:48-62`). `.mat` files carry no such field.
5. **`raw_texts` attribute on the source object** (`benchmark.py:284`). `.mat` files carry
   none — this is the root of the whole problem.
6. **Binary labels.** `minlength=2` at `benchmark.py:102` and `:113`, and `out_dim=2`
   hardcoded at `runner.py:326`. Fine for both target datasets.
7. **10/10/80 split.** `benchmark.py:51-56`, explicitly labelled as inherited from
   GraphAdapter/GLBench, *not* FLAG's. The native builder instead reproduces CARE-GNN's
   60% test split (`build_native_benchmark.py:126-133`).

---

## 3. Relation handling — definitive verdict

### 3.1 Verdict

**Nothing in the pipeline can consume a multi-relation graph. Every stage assumes exactly
one homogeneous `edge_index`.** There is no relation-aware code path anywhere in
`src/flagbench`, `scripts/`, or the bundled `methods/flag/` backbones.

### 3.2 Evidence — bundled backbones (`methods/flag/`)

Every backbone's `forward` signature is `(self, x, edge_index)`. A single tensor. None
accepts a relation list, a relation-type vector, or a dict of adjacencies:

| Model | Class | `forward` signature | Line |
|---|---|---|---|
| GCN | `models.GCN` | `forward(self, x, edge_index)` | `methods/flag/models.py:14` |
| GAT | `models.GAT` | `forward(self, x, edge_index)` | `methods/flag/models.py:65` |
| DualGNN | `models.DualGNN` | `forward(self, x1, x2, edge_index)` — **one** `edge_index`, reused for both branches (`:32-33`) | `methods/flag/models.py:31` |
| CARE-GNN | `caregnn.CAREGNN` | `forward(self, x, edge_index)` | `methods/flag/caregnn.py:43` |
| CARE-GNN layer | `caregnn.CAREGNNLayer` | `forward(self, x, edge_index)`; `add_self_loops` then one `propagate` | `methods/flag/caregnn.py:19-22` |
| BWGNN | `bwgnn.BWGNN` | `forward(self, x, edge_index)` | `methods/flag/bwgnn.py:55` |
| DGA | `dga.DGA` | `forward(self, x, edge_index)` | `methods/flag/dga.py:58` |
| PMP | `pmp.LASAGE_S` | `forward(self, x, edge_index, batch=None)` | `methods/flag/pmp.py:104` |
| GeniePath | `geniepath.GeniePathLazy` | `forward(self, x, edge_index)` | `methods/flag/geniepath.py:76` |

**Specifically on CARE-GNN and PMP, as asked:**

- **CARE-GNN (`methods/flag/caregnn.py`) takes a single `edge_index`.** It is a
  similarity-gated mean aggregator: `CAREGNNLayer.message` concatenates `x_i, x_j`, runs an
  MLP, sigmoids it, and scales `x_j` (`caregnn.py:24-29`). There is **no** per-relation
  aggregation, no RL neighbour filter, no label-aware similarity — exactly as the registry
  records (`registry.py:203-211`, fidelity string `"no multi-relation aggregation"`).
- **PMP (`methods/flag/pmp.py:LASAGE_S`) takes a single `edge_index`** (`pmp.py:104`), and
  its layer `LASAGESConv.forward` also takes one (`pmp.py:74-79`). Its fraud/benign
  partition (`pmp.py:84-88`) operates on the already-aggregated mean, and is **not** a
  relation partition — the registry says so at `registry.py:232-241`.

### 3.3 Evidence — adapter, trainer, sampler

- **Adapter** `src/flagbench/adapters/backbone.py`:
  `BaseBackbone.forward(self, x, edge_index)` (`:148-154`);
  `FlagBundledBackbone.forward(self, x, edge_index)` → `self.net(x, edge_index)` (`:253-264`);
  `DualBranchBackbone.forward(self, x_raw, x_disc, edge_index)` (`:309-310`).
  `build_backbone` (`:313-328`) has no relation parameter.
- **Trainer** `src/flagbench/training/trainer.py`:
  `_forward_center` uses `subgraph.edge_index.to(self.device)` — one tensor (`:168`), passed
  to the model at `:172-176`. `finetune_extra` likewise (`:409`, `:413-414`).
- **Sampler** `src/flagbench/sampling/semantic.py`:
  `build_adjacency(edge_index, num_nodes, drop_self_loops)` → one flat neighbour list per
  node (`:152-172`). `Subgraph.edge_index` is a single `[2, E']` tensor (`:127-128`).
  `sample_subgraph` builds edges from that one adjacency (`:290-305`).
- **Sampling cache format** stores exactly `{central, subset, edge_index, hop}`
  (`sample_subgraphs.py:240-250`), reconstructed identically at `runner.py:69-75`. There is
  no slot for a relation type.

### 3.4 Registry's relation fields are inert

`Capabilities.multi_relation` and `.requires_multi_relation` default `False`
(`registry.py:72-73`). **No `ModelSpec` in `MODEL_REGISTRY` overrides `capabilities`** —
every entry (`registry.py:174-242`) relies on `field(default_factory=Capabilities)`
(`registry.py:96`). Therefore the check at `registry.py:417-421`
(`requires_multi_relation and num_relations < 2`) can **never fire**. `DatasetSpec.num_relations`
(`registry.py:164`, values at `:296`, `:301`) is currently decorative metadata.

### 3.5 Consequence for YelpChi/Amazon

`data/benchmark/native_{yelpchi,amazon}/graph.pt` stores
`edge_index_homo` plus `edge_index_relations: dict[str, Tensor]` and `relation_names`
(`build_native_benchmark.py:197-204`). **Nothing outside `experiments/yelpchi_amazon/smoke_test.py`
reads either key** (grep over `src scripts experiments`). Any integration must pick **one**
`edge_index`. The only honest choice available without new modelling code is
`edge_index_homo`, which is the union view CARE-GNN itself ships — and that fact must be
stamped on every resulting row, because it collapses the three relations the dataset is
famous for.

Scale warning (from the manifest, `data/benchmark/native_yelpchi/dataset_manifest.json`):
YelpChi `homo` has **7,693,958** edges over 45,954 nodes (mean degree ≈ 167; `rsr` alone is
6.8 M). `build_adjacency` is O(E) memory (`semantic.py:152-172`) which is fine, but
`sample_subgraph`'s induced-edge construction is a Python double loop over
`order × adjacency[node]` (`semantic.py:292-300`) — with `top_k=10, hops=2` the subgraph is
small so this is bounded, but the *candidate* scan `select_neighbors` computes a dot product
over every neighbour of the centre (`semantic.py:217-218`), i.e. up to ~10⁵ candidates for
hub nodes. **UNKNOWN** whether full-graph sampling of 45,954 YelpChi centres is tractable in
practice; it cannot be measured here.

---

## 4. The sampler — `src/flagbench/sampling/semantic.py`

### 4.1 What it requires

From the **graph**: only `edge_index` and `num_nodes`, via
`build_adjacency(edge_index, num_nodes, drop_self_loops=True)` (`semantic.py:152-172`,
called at `sample_subgraphs.py:129-130` and `:211-213`). Note the script passes
`drop_self_loops=True` unconditionally at both call sites.

From **text**: **embeddings only — never raw text.** `sample_all` takes
`embeddings: torch.Tensor | None` (`semantic.py:315`) and L2-normalises rows
(`:320-321`, `normalize_embeddings` at `:175-186`). `select_neighbors` does
`neighbor_vecs @ center_vec` (`semantic.py:216-218`). The string `raw_texts` appears nowhere
in `semantic.py`.

The *script* wrapper, however, needs the embedding **cache file** to exist:
`sample_subgraphs.load_inputs` reads
`cache/embeddings/{dataset}__{safe_model}__raw.pt` (`sample_subgraphs.py:86-97`) and hard-fails
if absent (`:90-96`), and hard-fails if `embeddings.shape[0] != payload["y"].shape[0]`
(`:99-104`).

Strategies `none` and `random` need **no** embeddings at all (`semantic.py:204-212`;
`sample_subgraphs.py:159` passes `emb = None` for those). So `variant=baseline` and
`variant=text`, which default to `strategy="none"` (`registry.py:143-146` prose;
`VariantSpec.default_sampling_strategy` default `"none"` at `registry.py:135`), can have
their sampling cache built **without any text at all**.

### 4.2 `SamplingConfig.cache_key()`

`semantic.py:92-107`. Underscore-joined:

```
f"{strategy}" _ f"h{hops}" _ f"k{top_k}" _ f"t{similarity_threshold:g}" _ ("perhop"|"total")
[ _ f"s{seed}"  only when strategy == "random" ]
```

Fields entering the key: `strategy`, `hops`, `top_k`, `similarity_threshold`, `per_hop`, and
`seed` (random only). Fields **not** in the key: `threshold_first` (`:82-83`) and
`include_center` (`:88-90`) — changing either silently reuses a stale cache. Worth knowing;
not a blocker.

Defaults (`semantic.py:66-90`): `hops=2, top_k=10, similarity_threshold=0.0,
strategy="semantic", per_hop=True, threshold_first=True, seed=0, include_center=True`.
So the two cache keys the pipeline actually uses are:
- `none_h2_k10_t0_perhop` (baseline, +text)
- `semantic_h2_k10_t0_perhop` (flag, flag_finetuned)

Consumed by `runner.load_subgraphs` at `runner.py:57-75`, path
`cache/sampling/{dataset}__{cache_key}.pt` (`:58-60`).

### 4.3 Preconditions for `sample_subgraphs` on a new dataset

Via the existing script `scripts/preprocess/sample_subgraphs.py`, in order:

1. `data/benchmark/flag_<dataset>/graph.pt` must exist with `edge_index`, `y`, `test_mask`
   (`sample_subgraphs.py:56-63`; `test_mask` only for `--compare-strategies`, `:133`).
2. `cache/embeddings/<dataset>__all-MiniLM-L6-v2__raw.pt` must exist and have exactly
   `N` rows (`:86-104`) — required for `semantic`/`semantic_nothreshold`/`feature`, not for
   `none`/`random`.
3. The dataset name must pass argparse: `choices=["reddit", "instagram", "all"]`
   (`sample_subgraphs.py:284-285`). **This is a hard gate on the CLI only** — the underlying
   `build_cache(dataset, args)` (`:207`) and `load_inputs` (`:49`) take a plain string and
   have no such restriction, so a new driver can import and call them directly.

### 4.4 Single-relation / homophily assumptions in the sampler

- **Single relation: yes, assumed.** One `edge_index` → one adjacency list (§3.3).
- **Homophily: not assumed, but motivated by it.** The algorithm itself makes no homophily
  assumption; it ranks by cosine similarity of *text* embeddings (`semantic.py:216-237`).
  The paper's justification (higher subgraph homophily) is measured, not assumed, by
  `subgraph_homophily` (`semantic.py:341-359`) and the Figure 3(a) driver
  (`sample_subgraphs.py:125-204`), which explicitly prints `NOT SUPPORTED` and records it
  as-is if the claim fails (`:183-187`).
- **Reddit/Instagram-specific choices baked in:** self-loop exclusion is *always* applied to
  candidates (`semantic.py:200`, and module docstring `:45-48` justifies it by GLBench's
  `add_self_loops`); zero-norm rows get similarity 0 and therefore sit exactly on the default
  threshold (`semantic.py:175-186`, justified by Instagram's empty texts). Neither is wrong
  for YelpChi/Amazon, but both were chosen for GLBench reasons.
- **`threshold=0.0` + cosine on SBERT embeddings**: SBERT raw-text cosines are almost always
  positive, so `threshold=0` is near-vacuous. For *derived/templated* text (§8) the
  distribution would be even tighter (near-duplicate templates → cosine ≈ 1 for everything),
  which would make semantic sampling degenerate to arbitrary top-10. **This is the single
  biggest scientific risk of the whole integration** and must be measured with
  `--compare-strategies` before any FLAG-variant numbers are reported.

---

## 5. The trainer and the dual-branch path

### 5.1 `SubgraphTrainer` (`src/flagbench/training/trainer.py:120-458`)

Constructor (`:123-163`): `(model, config: TrainConfig, device, feature_fn, labels, dual_branch=False)`.
- `labels` is `payload["y"]`, moved to device (`:148`), indexed by **original node id**
  (`trainer.py:209`, `:251`), so it must be the full-graph label tensor, not a per-split one.
- Criterion is plain `CrossEntropyLoss` (`:151`); class weighting off by default
  (`TrainConfig.class_weighted_loss=False`, `:60-62`).

Single-branch path: `feature_fn(subgraph) -> Tensor[n_sub, in_dim]`, model called as
`model(features, edge_index)` returning `(hidden, logits)` (`:174-176`).

Dual-branch path: `feature_fn(subgraph) -> (x_raw, x_disc)`, model called as
`model(x_raw, x_disc, edge_index)` (`:170-173`). Both branches share **one** `edge_index`.

`fit` (`:273-332`): epochs from `TrainConfig.epochs` (default 5, `:47`), Adam lr 0.01
(`:48`), accumulation over 10 subgraphs (`:53`, `:221-224`), selection by validation
`f1_macro` (`:57`, `:290-294`), best state deep-copied and restored (`:313`, `:329-331`).

### 5.2 Exact inputs required for `flag` / `flag_finetuned`

`runner.run_single` at `:296-305` routes dual-branch variants to `make_dual_feature_fn`.

`make_dual_feature_fn` (`runner.py:171-217`) needs **two caches**:

1. **Raw-text SBERT embeddings** — `load_text_embeddings(dataset)` (`runner.py:192` → `:78-86`),
   file `cache/embeddings/{dataset}__all-MiniLM-L6-v2__raw.pt`, a dense `Tensor[N, 384]`.
   `in_dim` is taken from this (`runner.py:217`).
2. **LLM discriminative embeddings** — `load_llm_embeddings(dataset, "discriminative")`
   (`runner.py:193` → `:152-168`), a `dict[int, Tensor[k, 384]]` keyed by central node id.

For `flag_finetuned` a **third** cache is required:

3. **LLM residual embeddings** — `load_llm_embeddings(dataset, "residual")` (`runner.py:203`),
   same format, used by `extra_feature_fn` (`:205-211`).

`extra_feature_fn` returns `None` unless *both* `disc` and `common` exist for that centre
**and** both have exactly `len(subgraph.subset)` rows (`runner.py:209-210`).
`finetune_extra` then drops every subgraph for which it returns `None`
(`trainer.py:364`), matching upstream's drop-not-substitute behaviour (`trainer.py:346-349`).

The `x_disc` fallback for the plain `flag` variant is all-or-nothing:
`x_disc = d if d is not None and d.shape[0] == len(subgraph.subset) else x_raw`
(`runner.py:197-198`) — deliberately mirroring upstream `test_dual.py` (`runner.py:176-179`).

### 5.3 Cache-key derivation for the LLM embeddings

`runner._llm_embeddings_path` (`:130-149`) **recomputes** the key rather than globbing:

```
variant  = get_variant("flag")                                   # :143
sampling = SamplingConfig(strategy=variant.default_sampling_strategy)   # :144  -> "semantic_h2_k10_t0_perhop"
prompts  = PromptSet.load(dataset)                               # :145
config   = LLMConfig(model_id="google/gemma-2-9b-it", **PRODUCTION_LLM_CONFIG)  # :146
key      = llm_cache_key(dataset, sampling.cache_key(), prompts, config, kind)  # :147
path     = cache/embeddings/{key}__all-MiniLM-L6-v2.pt           # :148-149
```

`PRODUCTION_LLM_CONFIG = {"max_new_tokens": 64, "truncate_chars": 300}` (`runner.py:127`) —
decision D-004, deliberately *not* the paper-faithful default of 550/1200 (`enhance.py:48-49`).

`cache_key` itself (`enhance.py:189-209`) is
`f"{dataset}__{kind}__{model_slug}__{sampling_key}__{sha256(payload)[:16]}"` where the hashed
payload includes dataset, sampling key, kind, prompt version and **all four prompt file
hashes**, plus every `LLMConfig` field.

**Consequence:** `PromptSet.load(dataset)` (`enhance.py:93-114`) requires
`prompts/<dataset>/{system_instruction,global,discriminative,residual}.txt` to exist and
raises `FileNotFoundError` otherwise (`:96-100`). Only `prompts/reddit/` and
`prompts/instagram/` exist today. **A new dataset therefore needs four new prompt files
before `make_dual_feature_fn` can even compute a path.**

`generate_text.py` additionally uses a dataset-specific noun:
`DATASET_NOUN = {"reddit": "posts", "instagram": "introductions"}` with fallback `"posts"`
(`generate_text.py:53`, `:73`). The fallback means a new dataset silently gets "posts" —
acceptable but should be set explicitly.

### 5.4 Full cache dependency chain for `flag_finetuned`

```
data/benchmark/flag_<ds>/graph.pt                              [build]
  -> cache/embeddings/<ds>__all-MiniLM-L6-v2__raw.pt           [encode_text]
     -> cache/sampling/<ds>__semantic_h2_k10_t0_perhop.pt      [sample_subgraphs]
        -> cache/llm/<key_disc>.json      + .manifest.json     [generate_text, GPU]
        -> cache/llm/<key_resid>.json     + .manifest.json     [generate_text, GPU]
           -> cache/embeddings/<key_disc>__all-MiniLM-L6-v2.pt   [encode_llm_text]
           -> cache/embeddings/<key_resid>__all-MiniLM-L6-v2.pt  [encode_llm_text]
  -> cache/sampling/<ds>__none_h2_k10_t0_perhop.pt             [sample_subgraphs, for baseline/+text]
  + prompts/<ds>/{system_instruction,global,discriminative,residual}.txt
```

The two GPU steps are gated: `generate_text.py:183-190` returns exit code 2 with no CUDA.

### 5.5 Orthogonality / residual losses

`finetune_extra` (`trainer.py:335-458`) builds the three-term loss at `:416-422`:

```
CE(disc_logits[pos], label)
  + alpha * non_causal_loss(common_logits[pos])      # flag_losses.py:32-40, KL to uniform
  + beta  * orthogonal_loss(disc_hidden[pos], common_hidden[pos], mode)   # flag_losses.py:43-58
```

Crucially these run the **shared single-backbone GNN** `self.model.backbone`
(`trainer.py:387`, `:413-414`), **not** the attention-fused forward — so `finetune_extra`
requires `self.model` to be a `DualBranchBackbone` (asserted at `trainer.py:360-361`) purely
so that `.backbone` exists (`adapters/backbone.py:306`).

`FinetuneConfig` defaults (`trainer.py:72-96`): `outer_epochs=3, inner_epochs=10, lr=1e-4,
alpha=0.1, beta=0.1, accumulation_steps=10, orthogonality="squared_dot"`. The phase is
monotone-guarded: it only replaces the checkpoint if `ft_outcome.best_epoch > 0`
(`runner.py:348-350`), and `best_epoch` starts at 0 with the pre-finetune validation score as
the bar (`trainer.py:370-381`).

`result.llm_finetuned` is hardcoded `False` on every row (`runner.py:279`) — decision D-001.

---

## 6. The registry gate

### 6.1 `validate()` in full — `src/flagbench/registry/registry.py:372-439`

Signature: `validate(dataset, model, variant="baseline", device="cpu", hidden_dim=None) -> ValidationResult`.
It accumulates `reasons`; `ok = not reasons` (`:439`). `ValidationResult.__bool__` returns
`ok` (`:331-332`), so callers use `if not verdict:`.

The five refusal conditions, in order:

| # | Condition | Line | Message |
|---|---|---|---|
| 1 | `not ds.obtainable` | `:391-394` | `"<name> is not obtainable (<source>). <notes>"` — fires only for `huabei` (`:318`) |
| 2 | `vs.requires_llm and not ds.has_native_text` | `:397-408` | `"FLAG canonical text mode: NOT AVAILABLE. <name> has no native text."` **+ three suggestions**, one of which is the text-augmented escape hatch (`:406-407`) |
| 3 | `vs.feature_source == "lm_text" and not ds.has_native_text` | `:410-414` | `"variant '+text' needs raw node text, which <name> does not have"` |
| 4 | `ms.capabilities.requires_multi_relation and ds.num_relations < 2` | `:417-421` | **dead code** — no ModelSpec sets it (§3.4) |
| 5a | `device.startswith("cuda") and not ms.capabilities.gpu` | `:423-424` | dead — all default `gpu=True` (`:77`) |
| 5b | `device == "cpu" and not ms.capabilities.cpu` | `:425-426` | dead — all default `cpu=True` (`:76`) |
| 6 | `hidden_dim is not None and ms.hidden_dim_locked is not None and hidden_dim != locked` | `:428-437` | fires for `dga_gnn` and `pmp` at any hidden ≠ 32 (`:229`, `:239`) |

Callers: `runner.run_single` at `:254-258` (raises `ExperimentNotAvailable` — note this is
raised **outside** the try block that swallows other exceptions, `:287`, `:410-411`), and
`scripts/train/run.py:140-141` which pre-validates the whole matrix and prints refusals
(`:143-153`).

### 6.2 Exactly what `has_native_text` blocks

`DatasetSpec.has_native_text` (`registry.py:163`) is read in exactly two places, `:397` and
`:410`. It blocks:

- `variant=flag` (because `requires_llm=True`, `registry.py:263`) — via condition 2
- `variant=flag_finetuned` (`requires_llm=True`, `registry.py:270`) — via condition 2
- `variant=text` (because `feature_source="lm_text"`, `registry.py:258`) — via condition 3

It does **not** block `variant=baseline` (`feature_source="stored"`, `requires_llm=False`,
`registry.py:249-256`). With `has_native_text=False`, YelpChi/Amazon are permitted for
`baseline` on every model today — *if* a payload existed at `data/benchmark/flag_yelpchi/`,
which it does not (`runner.py:42`; only `native_yelpchi/` exists).

### 6.3 The "separate experiment type stamped `native_text: false`" — does it exist in code?

**No. It does not exist. It is prose only.**

- README.md:123-126 states the policy.
- `research/dataset_notes.md:335-341` states it in more detail, adding a second required
  stamp `text_source: engineered_or_external`.
- `registry.py:406-407` repeats it as a *suggestion string* printed on refusal.

Searching `src scripts experiments configs tests` for `experiment_type`, `native_text`,
`text_augmented`: the only hits are `Capabilities.requires_native_text` (`registry.py:74`,
itself dead — no ModelSpec sets it), `DatasetSpec.has_native_text`, and the prose above.

**`RunResult` has no such field.** Its full field list is `results.py:51-112`: there is no
`experiment_type`, no `native_text`, no `text_source`. `RunResult.validate()`
(`results.py:125-135`) requires only `impl_source`, `dataset`, `model`, `variant`, and a
legal `status`.

There *are* two usable free-form escape hatches that need **no** code change:
- `RunResult.extra: dict` (`results.py:112`), which is flattened into the CSV as
  `extra.<key>` columns (`results.py:198-202`).
- `RunResult.notes: str` (`results.py:111`).

And `RunResult.save(directory=...)` accepts an alternate output directory
(`results.py:140-142`), while `aggregate_to_files(raw_dir=...)` accepts an alternate input
directory (`results.py:186-188`). So an augmented study can be written to a *separate* raw
directory and aggregated separately — physically unmergeable with the canonical set.

This matters because `build_comparison_table` groups only by
`(dataset, model, variant, impl_source)` (`results.py:246`) and `aggregate_to_files` globs
**all** of `results/raw/*.json` (`results.py:170-179`). A run saved into the default
directory with a distinct dataset key would still land in the same `comparison.md`.

### 6.4 The minimal, honest change to permit a text-augmented YelpChi/Amazon study

Ranked from least invasive to most:

**Option A — separate dataset key, no semantic change to `has_native_text` (recommended).**
Register `yelpchi_text` / `amazon_text` as *new* `DatasetSpec` rows whose
`display_name` makes the derivation explicit (e.g. `"YelpChi (derived text, NOT FLAG)"`),
`source="derived from CARE-GNN .mat features; text is engineered, not native"`, and
`notes=` the full caveat. Setting `has_native_text=True` on such a row is the part that
requires judgement: it is *literally* true that the payload carries `raw_texts`, and *false*
in the sense the field was created for ("the dataset ships text"). Because the flag is
overloaded, setting it True on a derived-text row silently weakens the integrity gate for
anyone reading the registry later.

**Option B — disambiguate the field (the honest fix).** Add one field to `DatasetSpec`:
`text_source: str = "native"` with values `native | none | derived`, keep
`has_native_text` untouched, and change `registry.py:397` and `:410` from
`not ds.has_native_text` to `ds.text_source == "none"`. Then `yelpchi_text` gets
`has_native_text=False, text_source="derived"` — the gate opens, and the row still says
plainly that the text is not native. **This edits `registry.py` (a core file) and requires
human approval** (§8.4, change C-1).

**Stamping, either way, needs no code change**: a dedicated driver calls
`run_single(..., save_result=False)`, then sets
`result.extra.update({"native_text": False, "text_source": "engineered_or_external",
"experiment_type": "text_augmented", "relation_view": "homo_union", ...})` and
`result.save(directory=ROOT/"results"/"raw_text_augmented")`. That satisfies both the README
requirement and the "never merged" requirement at once, since
`aggregate_to_files(raw_dir=..., out_dir=...)` keeps the two corpora physically separate.

**What must NOT be done:** flipping `yelpchi`/`amazon`'s existing `has_native_text` to `True`
(`registry.py:295`, `:300`). That would retroactively make the canonical `yelpchi`/`amazon`
keys look like FLAG-eligible datasets and would directly contradict the FLAG paper, which the
registry docstring cites as the authority (`registry.py:6-9`).

---

## 7. SentenceTransformer usage

**Model id: `all-MiniLM-L6-v2` (384-dimensional).**

| Aspect | Answer | Evidence |
|---|---|---|
| Declared default | `DEFAULT_MODEL = "all-MiniLM-L6-v2"` | `scripts/preprocess/encode_text.py:35`; identically `scripts/preprocess/encode_llm_text.py:35` |
| CLI override | Yes — `--model` with no `choices` restriction | `encode_text.py:170`; `encode_llm_text.py:158`; `sample_subgraphs.py:286` |
| Runner-side default | `load_text_embeddings(dataset, model="all-MiniLM-L6-v2")` | `runner.py:78` |
| Runner-side override | **Effectively no.** `runner.make_feature_fn` calls `load_text_embeddings(dataset)` with no model arg (`runner.py:104`), and `make_dual_feature_fn` likewise (`runner.py:192`). `_llm_embeddings_path` defaults `sbert_model="all-MiniLM-L6-v2"` (`runner.py:130`) and is called with no override (`runner.py:159`). `run_single` exposes no SBERT parameter (`runner.py:220-232`). | `runner.py:78`, `:104`, `:130`, `:159`, `:192` |
| Instantiation | `SentenceTransformer(args.model, device=str(device_info.device))` | `encode_text.py:116-118`; `encode_llm_text.py:107-109` |
| Encoding settings | `batch_size=args.batch_size` (default 64, `:173`), `convert_to_numpy=True`, **`normalize_embeddings=False`** — normalisation is deferred to the sampler | `encode_text.py:120-126`, comment at `:125`; mirrored at `encode_llm_text.py:112-115` |
| Output dtype | `torch.from_numpy(...).float()` → float32 `[N, 384]` | `encode_text.py:128` |
| Why this checkpoint | Paper says only "Sentence-BERT"; upstream `encode.py` pins it, corroborated by `DualGNN` hardcoding `Linear(384, ...)` | `encode_text.py:11-13`; the hardcode is at `methods/flag/models.py:29` — **note it is constructed but never used in `DualGNN.forward` (`models.py:31-39`), so 384 is a dead parameter, not a live shape constraint** |

**Cache path (not a hash).** `cache_path(dataset, model, text_kind)` returns
`cache/embeddings/{dataset}__{model.replace("/","_")}__{text_kind}.pt`
(`encode_text.py:42-44`). `text_kind` is `"raw"` when `--source benchmark` and
`"raw_original"` when `--source original` (`encode_text.py:89`).

**The fingerprint is a separate, sidecar mechanism.** `texts_fingerprint`
(`encode_text.py:69-78`) is `sha256` over `str(len(texts))` then each text UTF-8-encoded and
`b"\x00"`-delimited. It is stored in the sidecar `.json` as `texts_sha256`
(`encode_text.py:140`, written at `:154-156`) and checked on cache hit at
`encode_text.py:92-102`: if it differs, the script re-encodes and prints
`"cache present but the text fingerprint changed (different benchmark build); re-encoding"`.
**The fingerprint is not part of the filename**, so a stale `.pt` with no sidecar `.json`
would be treated as a hit only if `--force` is absent *and* the sidecar exists — with no
sidecar the `if meta_path.exists()` guard (`:94`) falls through and it re-encodes. Safe.

**Dataset gate.** `encode_text.py:168-169` restricts `--dataset` to
`["reddit", "instagram", "all"]`, and `main` expands `"all"` to `["instagram", "reddit"]`
(`:181`). Again, `encode(dataset, args)` (`:81`) and `load_texts(dataset, source)` (`:47`)
take plain strings — the gate is argparse-only and is bypassable by importing the function.

**LLM-text encoder.** `encode_llm_text.py:76` builds the output path as
`cache/embeddings/{llm_key}__{args.model}.pt` — **without** the `replace("/", "_")` that
`runner._llm_embeddings_path:148` applies to the same component. For the default
`all-MiniLM-L6-v2` (no slash) the two agree; for any org-prefixed SBERT id
(e.g. `sentence-transformers/all-mpnet-base-v2`) the writer and the reader would disagree and
the runner would report the file missing. **Latent bug; not triggered by current defaults.**

---

## 8. Safest integration point

### 8.1 The key structural fact

`runner.load_benchmark` hardcodes the path template
`data/benchmark/flag_<dataset>/graph.pt` (`runner.py:42`). So does
`encode_text.load_texts` for `--source benchmark` (`encode_text.py:55`) and
`sample_subgraphs.load_inputs` (`sample_subgraphs.py:56`). **All three agree on the same
template, and all three take the dataset name as a plain string.**

Therefore: *if a new payload is written to `data/benchmark/flag_<newname>/graph.pt` in the
GLBench-shaped format, the entire existing pipeline consumes it with zero modification to any
core file.* The whole integration reduces to "produce a correctly-shaped payload under a new
dataset key" — exactly the constraint the task set.

### 8.2 Diagram of the minimal new artifacts

```
 EXISTING, UNCHANGED                         NEW ARTIFACTS (all under experiments/)
 ───────────────────                         ──────────────────────────────────────

 data/benchmark/native_yelpchi/graph.pt
   x, y, edge_index_homo,                 ┌─ [NEW-1] experiments/yelpchi_amazon/
   edge_index_relations{rur,rtr,rsr},     │      make_text_payload.py
   train/val/test/unlabeled_mask          │        • pick ONE edge_index (homo)
            │                             │        • synthesise raw_texts[N] from
            └──────────────────────────►  │          the 32/25 numeric features
                                          │          (+ label_names, original_node_ids)
                                          │        • re-emit the 9-key GLBench payload
                                          │        • write dataset_manifest.json
                                          └────────────────┬─────────────────────
                                                           ▼
                            data/benchmark/flag_yelpchi_text/graph.pt   ◄── [NEW-2] artifact
                            data/benchmark/flag_yelpchi_text/dataset_manifest.json
                                                           │
   scripts/preprocess/encode_text.py  ◄────────────────────┤  (called via [NEW-3] driver,
     (function `encode`, imported not edited)              │   bypassing argparse choices)
                                                           ▼
                            cache/embeddings/yelpchi_text__all-MiniLM-L6-v2__raw.pt
                                                           │
   scripts/preprocess/sample_subgraphs.py ◄────────────────┤  (function `build_cache`)
     (unchanged)                                           ▼
                            cache/sampling/yelpchi_text__none_h2_k10_t0_perhop.pt
                            cache/sampling/yelpchi_text__semantic_h2_k10_t0_perhop.pt
                                                           │
   prompts/yelpchi_text/*.txt  ◄── [NEW-4] four prompt files (required by PromptSet.load)
                                                           │
   scripts/llm/generate_text.py (GPU) ◄────────────────────┤  (function `run`)
   scripts/preprocess/encode_llm_text.py ◄─────────────────┤  (function `encode`)
                                                           ▼
                            cache/llm/<key>.json  →  cache/embeddings/<key>__all-MiniLM-L6-v2.pt
                                                           │
   src/flagbench/registry/registry.py  ◄── [CHANGE C-1] one/two DatasetSpec rows (+ gate fix)
   src/flagbench/experiments/runner.py ◄── UNCHANGED        │
   src/flagbench/training/trainer.py   ◄── UNCHANGED        ▼
   src/flagbench/sampling/semantic.py  ◄── UNCHANGED   run_single(dataset="yelpchi_text", ...)
   src/flagbench/adapters/backbone.py  ◄── UNCHANGED        │
   methods/flag/*.py                   ◄── UNCHANGED        ▼
                                          [NEW-5] experiments/.../run_text_augmented.py
                                            • run_single(save_result=False)
                                            • stamp result.extra{native_text:false, ...}
                                            • result.save(dir=results/raw_text_augmented/)
                                            • aggregate_to_files(raw_dir=..., out_dir=...)
```

### 8.3 The new artifacts, enumerated

- **[NEW-1] `experiments/yelpchi_amazon/make_text_payload.py`** (new file). Reads
  `data/benchmark/native_<ds>/graph.pt`, emits the 9-key GLBench-shaped payload under a new
  dataset key. Must:
  - choose `edge_index = payload["edge_index_homo"]` and record that choice in the manifest;
  - drop self-loops itself if present, or accept that `build_adjacency`'s
    `drop_self_loops=True` (`sample_subgraphs.py:130`, `:213`) handles it at sampling time;
  - handle Amazon's 3,305 unlabeled prefix (`build_native_benchmark.py:88`,
    `:120-123`) — those nodes are in **none** of train/val/test, so `run_single`'s
    `split_subgraphs` simply never selects them (`runner.py:308-314`). They can stay in the
    graph as structure-only context. `y` for them is whatever the `.mat` says; it is never
    used as a target. **Recommended: keep them, and record the decision.**
  - synthesise `raw_texts: list[str]` of length N. **This is the scientifically load-bearing
    step and is out of scope for this document** — see §4.4 for why templated text risks
    collapsing semantic sampling.
  - set `label_names` and `original_node_ids` (identity) for schema completeness even though
    both are inert (§1.3).
- **[NEW-2]** the resulting `data/benchmark/flag_<newkey>/{graph.pt,dataset_manifest.json}`.
  The manifest should carry `preprocessing_version` (read at `runner.py:290`) and
  `output_sha256` (`runner.py:291`).
- **[NEW-3] `experiments/yelpchi_amazon/build_text_caches.py`** (new file). Imports
  `scripts.preprocess.encode_text.encode` and `scripts.preprocess.sample_subgraphs.build_cache`
  and calls them with an `argparse.Namespace`, sidestepping the four argparse `choices` gates
  (`encode_text.py:169`, `sample_subgraphs.py:285`, `generate_text.py:165`,
  `encode_llm_text.py:155`) **without editing any of those files**.
- **[NEW-4] `prompts/<newkey>/{system_instruction,global,discriminative,residual}.txt`**.
  Required by `PromptSet.load` (`enhance.py:93-100`) for any FLAG variant. Also decide the
  noun; `DATASET_NOUN` falls back to `"posts"` (`generate_text.py:53`, `:73`), which would be
  wrong for reviews/users — but changing that dict is an edit to `generate_text.py`, so
  either accept `"posts"` or pass a custom noun through a new driver that calls
  `enhancer.enhance(..., noun=...)` directly (`enhance.py` / `generate_text.py:113-115`).
- **[NEW-5] `experiments/yelpchi_amazon/run_text_augmented.py`** (new file). The stamping and
  segregation driver described in §6.4.

### 8.4 Files that must remain UNCHANGED

Core model / sampling / training / evaluation — the task's hard rule:

```
methods/flag/models.py          methods/flag/caregnn.py     methods/flag/bwgnn.py
methods/flag/dga.py             methods/flag/pmp.py         methods/flag/geniepath.py
src/flagbench/adapters/backbone.py
src/flagbench/sampling/semantic.py
src/flagbench/training/trainer.py
src/flagbench/training/flag_losses.py
src/flagbench/metrics/classification.py
src/flagbench/experiments/runner.py
src/flagbench/experiments/results.py
src/flagbench/datasets/benchmark.py
src/flagbench/datasets/glbench.py
src/flagbench/llm/enhance.py
src/flagbench/utils/seeding.py          src/flagbench/utils/device.py
scripts/preprocess/build_benchmark.py   scripts/preprocess/encode_text.py
scripts/preprocess/encode_llm_text.py   scripts/preprocess/sample_subgraphs.py
scripts/llm/generate_text.py            scripts/train/run.py
experiments/yelpchi_amazon/build_native_benchmark.py
experiments/yelpchi_amazon/smoke_test.py
```

All of these are consumed by the plan above either as library functions (imported, not
edited) or not at all.

### 8.5 Where a core change is genuinely unavoidable

**C-1 — `src/flagbench/registry/registry.py`. REQUIRES HUMAN APPROVAL.**

*Reason.* `run_single` calls `get_dataset(dataset)` at `runner.py:238` and `validate(...)` at
`:254`. `get_dataset` raises `RegistryError` for any key not in `DATASET_REGISTRY`
(`registry.py:364-369`). A new dataset key **cannot exist** without a registry row. And even
with a row, `validate` condition 2/3 (`registry.py:397`, `:410`) blocks every non-baseline
variant unless the honesty flag is addressed.

*Proposed minimal change (two parts, both small and both in the data/policy layer, not the
algorithm layer):*

1. Add one field to `DatasetSpec` (`registry.py:159-167`):
   ```python
   text_source: str = "native"   # native | none | derived
   ```
2. Change two conditions to key off it instead of the overloaded boolean:
   - `registry.py:397`: `if vs.requires_llm and ds.text_source == "none":`
   - `registry.py:410`: `if vs.feature_source == "lm_text" and ds.text_source == "none":`
   Every existing row keeps its behaviour, because `has_native_text=True` rows are
   `text_source="native"` and `has_native_text=False` rows become `text_source="none"` —
   these must be set explicitly on all eight existing rows (`registry.py:285-322`) so the
   default `"native"` never applies silently.
3. Add the two new rows:
   ```python
   "yelpchi_text": DatasetSpec(
       key="yelpchi_text", display_name="YelpChi (derived text — NOT canonical FLAG)",
       has_native_text=False, text_source="derived", num_relations=1,
       source="derived from CARE-GNN YelpChi.mat; homo relation only; text engineered",
       notes="The FLAG paper states YelpChi lacks textual information. Text here is "
             "engineered, not native. Results are a text-augmented study, never FLAG.",
   ),
   ```
   and the Amazon analogue.

*Why this is the minimal honest version.* It touches no model, no sampler, no trainer, no
metric. It does not weaken the gate — it makes the gate express what it always meant
("the dataset has no text at all" vs "the text is not the dataset's own"). It keeps the
canonical `yelpchi` / `amazon` rows blocked exactly as today.

*The alternative that avoids the edit, and why it is worse.* `DATASET_REGISTRY` is a plain
module-level dict (`registry.py:285`), so a driver in `experiments/` could
`DATASET_REGISTRY["yelpchi_text"] = DatasetSpec(..., has_native_text=True)` at import time.
This requires **zero** file changes and works. It is nonetheless **not recommended**: it makes
the integrity gate mutable from outside, leaves no trace in the file a reader would inspect,
and sets `has_native_text=True` on derived text — precisely the misrepresentation the gate
exists to prevent. If it is used anyway, the stamping in [NEW-5] becomes mandatory rather
than merely good practice.

**C-2 — `src/flagbench/experiments/results.py`. OPTIONAL; can be avoided.**
Adding first-class `experiment_type` / `native_text` / `text_source` columns to `RunResult`
(`results.py:51-112`) would match the README's language literally. It is **avoidable** via
`result.extra` (§6.4), which is already flattened into the CSV (`results.py:198-202`).
Recommendation: do not edit; use `extra` plus a separate raw directory. Flag for human
decision only if the maintainer wants the columns promoted.

**No other core change is required.** In particular the trainer, sampler, adapter and all
bundled backbones work unmodified on a single-`edge_index` payload, which is what
`edge_index_homo` already is.

### 8.6 Residual risks to state alongside any result

1. **Three relations collapsed into one.** `edge_index_homo` is the union; the relation
   structure that makes YelpChi/Amazon interesting is discarded (§3.5). No model in this repo
   can use it (§3.2).
2. **The text is engineered.** Every semantic-sampling and LLM claim rests on it. Near-duplicate
   templated text will make cosine similarity uninformative (§4.4).
3. **Split protocol differs from the canonical FLAG datasets.** CARE-GNN's 60% test split
   (`build_native_benchmark.py:126-133`) vs FLAG's 10/10/80 (`benchmark.py:51-56`), and the
   val fold is this repo's own addition (`build_native_benchmark.py:135-142`, `:259-266`).
4. **No 1:10 downsampling.** Deliberately (§2.4.1), so the class balance differs from every
   canonical FLAG number.
5. **YelpChi scale.** 7.69 M homo edges vs Reddit's ~198 K. Sampling cost is **UNKNOWN**
   without execution.
6. **D-004 decode budget.** Any LLM cache built here inherits `max_new_tokens=64,
   truncate_chars=300` (`runner.py:127`), already far below the paper's 550/1200.

---

## 9. UNKNOWNs (stated rather than guessed)

- Whether YelpChi/Amazon `homo`/relation matrices contain self-loops, and how many. The
  manifest records symmetry but not self-loops (`build_native_benchmark.py:212-217`).
- Actual dtype of `x` in `data/benchmark/flag_reddit/graph.pt` — `benchmark.py:283` does not
  cast, so it is whatever GLBench shipped. (The native builder *does* cast to float32,
  `build_native_benchmark.py:183`.)
- Runtime feasibility of `sample_all` over 45,954 YelpChi centres at 167 mean degree.
- Whether the authors applied `top_k` per-hop or as a total budget — already recorded as
  UNKNOWN upstream in `semantic.py:36-43`.
- Whether the four existing argparse `choices` gates were deliberate integrity guards or
  incidental. They are bypassable by import, which suggests incidental, but this is an
  inference.

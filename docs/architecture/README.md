# FLAG architecture documentation

Code-grounded workflow documentation for this repository. Every claim was read
off the source (file + function named next to it). Where something could not be
confirmed from code the text says `UNVERIFIED` or `NOT FOUND IN REPOSITORY`.

## Read this first: three things that differ from the usual mental model

The repository contains **two code lineages**, and the pipeline you might expect
from the paper is not exactly what either of them does.

1. **Two lineages.**
   - `src/flagbench/` + `scripts/` = the **reproduction** written in this repo.
     It is the only lineage that actually runs end to end, and the one every
     diagram here treats as the *main execution path*.
   - `methods/flag/` = **unmodified upstream FLAG** (BUPT-GAMMA/FLAG @ `cb83944`,
     git-ignored, fetched by `scripts/setup/fetch_methods.sh`). Its GNN classes
     (`models.py`, `geniepath.py`, `bwgnn.py`, `caregnn.py`, `dga.py`, `pmp.py`)
     are imported by the reproduction. Its *drivers* (`chat.py`, `encode.py`,
     `train.py`, `test_dual.py`, ...) are **not** called by the reproduction.
2. **The subgraph is 2-hop by default, not 1-hop** (`SamplingConfig.hops = 2`,
   `semantic.py:66`; the cached files in `cache/sampling/` are `h2`).
   A 1-hop subgraph is the special case `--hops 1`, and the hop-1 nodes of any
   subgraph are identifiable through `Subgraph.hop == 1`.
3. **Cosine similarity is used for neighbour *selection* on raw-text embeddings,
   before any GNN runs.** It is not computed between GNN representations, and it
   is never turned into a feature. There is no "similarity -> feature vector"
   stage. The score is used to filter/rank neighbours and then discarded.
   Details: [04](04-representation-and-similarity.md).

Also: "anomaly detection" here is **supervised binary node classification**
(2-class cross-entropy, fraud score = `softmax(logits)[1]` of the centre node,
threshold tuned on validation). There is no unsupervised anomaly score.

## Map

```text
README (you are here)
  |
  00-master-workflow        entrypoint, master diagram, execution trace
  |    |
  |    +-- 01-repository-map        what every directory/file is, tagged
  |    +-- 02-flag-pipeline         stage-by-stage narrative + variants + artifacts
  |    |      +-- 03-subgraph-generation              (1/2-hop subgraph)
  |    |      +-- 04-representation-and-similarity    (embeddings, cosine)
  |    |      +-- 05-feature-and-detection-pipeline   (features, GNN, scoring, eval)
  |    +-- 06-function-call-graph   signature-level records per function
  |    +-- 07-data-flow             tensor/object transformations
  |    +-- 08-paper-to-code         paper claim vs repository behaviour
```

| Doc | Answers |
|---|---|
| [00-master-workflow](00-master-workflow.md) | "If I start running FLAG, what happens next?" |
| [01-repository-map](01-repository-map.md) | "Which files matter, and which are unused?" |
| [02-flag-pipeline](02-flag-pipeline.md) | "What are the stages, variants and cached artefacts?" |
| [03-subgraph-generation](03-subgraph-generation.md) | "Exactly how is a subgraph built?" |
| [04-representation-and-similarity](04-representation-and-similarity.md) | "Where do vectors come from, and where is cosine computed?" |
| [05-feature-and-detection-pipeline](05-feature-and-detection-pipeline.md) | "How does a subgraph become a fraud score and a metric?" |
| [06-function-call-graph](06-function-call-graph.md) | "Who calls whom, with what, returning what?" |
| [07-data-flow](07-data-flow.md) | "What shape/type is the data at each step?" |
| [08-paper-to-code](08-paper-to-code.md) | "Where does the code differ from the paper?" |

## Rendering

Diagrams are Mermaid code blocks. In VS Code use Markdown preview with a Mermaid
extension (e.g. *Markdown Preview Mermaid Support*, or the built-in preview in
recent VS Code builds).

- Every diagram has `click` directives (node -> doc page or source file). Whether
  a click is honoured depends on the extension's Mermaid `securityLevel`; in the
  stock preview it may be ignored.
- **Because of that, every diagram is followed by a plain Markdown link table.**
  Those links always work (Ctrl+Click in the preview / editor).
- Source links are relative (`../../src/...`). `methods/` is git-ignored, so
  links into it only resolve if you have run `scripts/setup/fetch_methods.sh`.

## Status labels used throughout

| Label | Meaning |
|---|---|
| **MAIN PATH** | Traced from `scripts/train/run.py` (or a prep script it depends on) through actual calls |
| **UPSTREAM REFERENCE** | Code in `methods/flag/` that the reproduction imports or mirrors |
| **Present but not observed in main execution path** | Exists, but no call to it was found in the traced path |
| `UNVERIFIED` | Could not be confirmed from code (usually a third-party library internal or a secondary document) |
| `NOT FOUND IN REPOSITORY` | Looked for, does not exist |

## Verification log (second pass)

Each row: an arrow/claim in the diagrams and the code that proves it.

| Claim | Evidence |
|---|---|
| `run.py:main` -> `run_single` | `scripts/train/run.py:173` |
| `run_single` -> `load_benchmark`, `load_subgraphs` | `runner.py:307`, `runner.py:312` |
| Feature source differs only through `make_feature_fn` / `make_dual_feature_fn` | `runner.py:317-324` |
| `run_single` -> `build_backbone` -> `SubgraphTrainer.fit` | `runner.py:345`, `runner.py:355-359` |
| `flag_finetuned` -> `finetune_extra` | `runner.py:361-366` (`requires_finetuned_llm`) |
| Test scores -> threshold from **validation** -> test metrics | `runner.py:387` -> `classification.py:305-319` |
| Subgraph is produced by `sample_all` -> `sample_subgraph` -> `sampler.select` | `semantic.py:416-419`, `semantic.py:330-331` |
| Cosine is `normalized[nbrs] @ normalized[center]` | `semantic.py:249-251` |
| `sims` is not returned or stored | `semantic.py:222-270` returns only `np.sort(candidates)` |
| Hop-2 nodes are ranked by similarity to the **hop-1 node**, not the original centre | `semantic.py:346-347` (`select(node)` for each frontier node) |
| Induced edges = all adjacency edges among selected nodes | `semantic.py:367-382` |
| Baseline (`none`) applies no top-k cap | `semantic.py:237-238` |
| `SamplingConfig.include_center` is never read | `grep include_center` -> only its definition, `semantic.py:91` |
| `configs/*.yaml` are not read by any Python file | `grep yaml/configs/` over `src scripts analysis benchmark experiments tests` -> no loader |
| Early stopping cannot fire at default settings | `TrainConfig.epochs=5 < patience=10`, `trainer.py:47,56`, check at `trainer.py:317-319` |
| `run.py`'s `except FileNotFoundError` is not reachable for missing caches | every loader runs inside `run_single`'s `try` (`runner.py:306-437`), whose `except Exception` records a `failed` row instead |
| `cache/llm/` is empty in this checkout | directory listing; so `flag` / `flag_finetuned` runs would fail here with a `failed` row |

## Mechanical checks performed on these files

- All 16 Mermaid blocks were parsed with `mermaid.parse()` (Node + jsdom) - 16/16 valid; a deliberately broken block was rejected, so the check is real. This checks syntax, not visual layout.
- All relative Markdown links and `click` targets resolve to existing files.
- Not done: rendering in your VS Code extension, and running any pipeline code (nothing was executed; all behaviour statements come from reading source and the JSON manifests already in `cache/`).

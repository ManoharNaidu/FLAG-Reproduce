# YelpChi / Amazon native benchmark

Prepares `data/raw/yelpchi/YelpChi.mat` and `data/raw/amazon/Amazon.mat`
(official CARE-GNN release) for **native baseline evaluation only**.

## Why this is separate from `src/flagbench`

`flagbench.registry.DATASET_REGISTRY` already marks both datasets
`has_native_text=False` (registry.py:294-303), which hard-blocks the `flag` and
`+text` variants at validation time -- correctly, since the FLAG paper itself
says these datasets lack textual information. Nothing here changes that gate.

`flagbench.experiments.runner.load_benchmark()` reads
`data/benchmark/flag_<dataset>/graph.pt`, a payload built by
`flagbench.datasets.benchmark`'s FLAG-specific 1:10 downsampling construction.
That construction exists to artificially imbalance Reddit/Instagram, which are
naturally near-balanced. YelpChi/Amazon are already naturally imbalanced with
an established split convention from their own literature (CARE-GNN), so
running them through that construction again would not reproduce anything --
it would invent an unpublished fourth variant of these datasets. This
directory intentionally does not touch that code path.

## What's here

- `build_native_benchmark.py` -- parses the `.mat` files (verified keys, see
  module docstring) into `data/benchmark/native_<dataset>/graph.pt` plus a
  `dataset_manifest.json`, using CARE-GNN's own official train/test split
  (exact reproduction of `methods/care_gnn/train.py:50-61`). A validation fold
  is additionally carved out of their train split -- that part is **our
  addition**, not upstream's protocol, and is labelled as such in the
  manifest.
- `smoke_test.py` -- loads the payload, checks internal consistency (mask
  coverage, edge index bounds), and runs a few epochs of a throwaway 2-layer
  GCN as a plumbing check. **Not a benchmark result.**

Both have been run against the actual local `.mat` files; the manifests
contain real observed counts (45,954 / 11,944 nodes, matching the published
CARE-GNN statistics), not assumed ones.

## What's NOT here

- No wiring into `flagbench.registry` / `flagbench.experiments.runner`. Doing
  that properly means deciding how the runner should consume a payload with
  three relation-specific `edge_index` tensors instead of one (CARE-GNN,
  PMP-style models want the relations kept separate; GCN/GAT/GeniePath only
  need `edge_index_homo`) -- an architecture decision for the maintained
  codebase, not something to fold in silently from an experiments/ sandbox.
- No real (tuned, multi-seed) baseline numbers. `smoke_test.py`'s accuracy is
  not meaningful on this class imbalance -- see
  `research/degenerate_baselines.md`.

## Usage

```bash
python -m experiments.yelpchi_amazon.build_native_benchmark --dataset all
python -m experiments.yelpchi_amazon.smoke_test --dataset yelpchi
python -m experiments.yelpchi_amazon.smoke_test --dataset amazon
```

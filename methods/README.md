# Imported Implementations

This directory holds **unmodified upstream source**. It is **git-ignored** and is
reconstructed from upstream at pinned commits:

```bash
bash scripts/setup/fetch_methods.sh            # fetch all
bash scripts/setup/fetch_methods.sh --verify   # check existing clones against pins
```

Why ignored rather than vendored: **four of these repositories ship no LICENSE
file**, so redistributing their source is not permitted. Pinning exact SHAs
preserves provenance without re-hosting anyone's code. See
`research/repository_provenance.md` section 11.

**Nothing in this directory may be edited.** Adaptations live in `src/adapters/`;
environment-level shims live in `src/compat/`. `fetch_methods.sh --verify` warns
if a clone is dirty. Keeping upstream pristine is what lets
`tests/integration/test_flag_upstream_claims.py` tell us when an audit finding has
gone stale.

---

## Provenance table

| Model | Source | Commit | Framework | Native dataset | Integrated | Licence |
|---|---|---|---|---|---|---|
| **flag** | [BUPT-GAMMA/FLAG](https://github.com/BUPT-GAMMA/FLAG) | `cb83944e` | PyTorch + PyG + transformers + peft | Reddit, Instagram (files not shipped) | **audited, not yet integrated** | **NONE** |
| **care_gnn** | [YingtongDou/CARE-GNN](https://github.com/YingtongDou/CARE-GNN) | `a64ff752` | PyTorch (no DGL/PyG) | YelpChi, Amazon (in-repo) | no | Apache-2.0 |
| **bwgnn** | [squareRoot3/Rethinking-Anomaly-Detection](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) | `de0631f0` | PyTorch + DGL | Yelp, Amazon, T-Finance, T-Social | no | **NONE** |
| **dga_gnn** | [AtwoodDuan/DGA-GNN](https://github.com/AtwoodDuan/DGA-GNN) | `0907392f` | PyTorch + DGL + Lightning + Hydra | Elliptic, T-Finance, T-Social, YelpChi, Amazon | no | **NONE** |
| **pmp** | [Xtra-Computing/PMP](https://github.com/Xtra-Computing/PMP) | `3f7629f6` | PyTorch + DGL + PyG | Yelp, Amazon, T-Finance, T-Social | no | **NONE** |
| **geniepath** | [shuowang-ai/GeniePath-pytorch](https://github.com/shuowang-ai/GeniePath-pytorch) | `143f07cc` | PyTorch + PyG 1.x | PPI | no | MIT |
| **glbench** | [NineAbyss/GLBench](https://github.com/NineAbyss/GLBench) | unpinned | PyTorch | source of Reddit + Instagram | no | MIT |

`Integrated` means "wired into this framework's registry and runnable through our
config system". Only `flag` has been *audited*; nothing is integrated yet.

### Deliberately absent: GCN and GAT

No repository is imported for these. We use PyG's `GCNConv` / `GATConv`.

- `tkipf/gcn` is TensorFlow 1.x (119 open issues, uninstallable on Python 3.11).
- `tkipf/pygcn`'s own README states it is *"not intended for reproduction of the
  results"*.
- GAT's author explicitly recommends PyG/DGL over his TF1 repo, which requires
  Python 3.5.2 / CUDA 9.
- FLAG's own code uses PyG convolutions, so PyG is also the faithful choice.

Full rationale: `research/repository_provenance.md` sections 7-8.

---

## Two lineages, never merged

Critical: **FLAG's repository contains its own re-implementations of five
baselines** (`geniepath.py`, `bwgnn.py`, `caregnn.py`, `dga.py`, `pmp.py`) written
in PyG by the FLAG authors. They are **not** the official baseline code, and
several differ substantially from the published algorithms.

Every result therefore carries an `impl_source` field:

| `impl_source` | What it reproduces |
|---|---|
| `flag_bundled` | **the FLAG paper's numbers** — Table 4 was produced by these files |
| `official` | **the baseline as its authors published it** |

Fidelity assessment (evidence in `research/flag_code_audit.md` section 7, each
claim backed by a passing test):

| Baseline | FLAG's bundled version |
|---|---|
| GCN | faithful — two `GCNConv` layers |
| GAT | **not canonical** — second layer is `SAGEConv`, not `GATConv` |
| GeniePath | closest to faithful — matches PyG's example exactly |
| CARE-GNN | **not CARE-GNN** — no RL neighbour filtering, no multi-relation, no label-aware similarity |
| BWGNN | **beta-wavelet basis not applied** — polynomial evaluated over raw adjacency `A`, not the normalised Laplacian |
| DGA-GNN | **not DGA-GNN** — no dynamic grouping, no decision tree; it is GraphSAGE-mean |
| PMP | partial — fraud/benign partition present, but applied post-aggregation |

Reporting `flag_bundled` numbers as though they were the official baselines would
misattribute performance. The framework keeps them distinct by construction.

---

## Licensing

| Licence | Repositories | Redistribution |
|---|---|---|
| **NONE** | flag, bwgnn, dga_gnn, pmp | **not permitted** — default copyright |
| Apache-2.0 | care_gnn | permitted with attribution + NOTICE |
| MIT | geniepath, glbench | permitted with attribution |

Local cloning for research is standard practice and is all this project does.
**Before publishing any artefact containing code from the four unlicensed
repositories, permission must be sought from those authors.** Open item, tracked
in `research/reproduction_status.md` section 4.5.

---

## Known upstream defects

Recorded so they are not rediscovered. Full detail in `research/`.

**FLAG** (`research/flag_code_audit.md`, `research/FLAG_ORIGINAL_STATUS.md`) —
**0 of 7 entrypoints run**:
- `test.py` / `test_dual.py` import `ECELoss` from `utils`, which **does not
  exist** -> `ImportError` at import time.
- `train.py` indexes a tuple with a bool tensor -> `TypeError`.
- `train.py`'s LoRA gradient path is **severed**; fine-tuning is a no-op.
- `dga.py` / `pmp.py` raise `UnboundLocalError` at any hidden size but 32.
- ~40 hardcoded `.cuda()` calls; no CPU support.

**DGA-GNN** — the README's config names are wrong (`yelpchit` -> `yelpchi`, etc.;
the documented names 404). W&B logging is on by default. Zero issues ever filed.

**PMP** — README says `--train_ratio`, but the CLI defines `--train_size`.
`torch.cuda.set_device()` at startup crashes on CPU-only machines.
`requirements.txt` pins `PyYAML` twice, and its `+cu118` pins break plain `pip install`.

**CARE-GNN** — the authors document that the paper's **Table 2 similarity scores
are incorrect** and that **Figure 3's relation-weight subfigure is wrong**
(a bug in the CARE-Weight variant).

**GeniePath (community)** — all 5 open issues are reproducibility failures
(NaN loss, F1 0.45, missing skip connection). Validation and test are the same
split. **No official author repository exists** — verified, not assumed.

**BWGNN** — T-Finance / T-Social depend on a single Google Drive link. No licence.

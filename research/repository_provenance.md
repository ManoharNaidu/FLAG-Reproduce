# Repository Provenance

Every repository considered for this benchmark, classified and pinned.

**Verification date:** 2026-09-09. All commit SHAs captured with `git ls-remote`.
Every URL in this file was resolved at that date.

**Classification scale** (Phase 1 priority order):

| Class | Meaning |
|---|---|
| `OFFICIAL_AUTHOR` | Repository owned by a named author of the paper |
| `OFFICIAL_LAB` | Owned by the lab/org of a paper author, not a personal author account |
| `RECOGNIZED_AUTHOR_IMPL` | Author-endorsed implementation hosted elsewhere |
| `REPUTABLE_BENCHMARK` | Maintained by a major framework/benchmark project (PyG, DGL, GADBench) |
| `INDEPENDENT` | Third-party implementation |
| `NONE_EXISTS` | No official implementation exists (verified, not assumed) |

---

## 0. Summary table

| Model | Selected source | Class | Branch @ HEAD | License |
|---|---|---|---|---|
| **FLAG** | `BUPT-GAMMA/FLAG` | OFFICIAL_AUTHOR | `main` @ `cb83944e` | **NONE** |
| GCN | PyG `GCNConv` (+ `tkipf/gcn` as reference) | REPUTABLE_BENCHMARK | PyG 2.3.1 | MIT |
| GAT | PyG `GATConv` (+ `PetarV-/GAT` as reference) | REPUTABLE_BENCHMARK | PyG 2.3.1 | MIT |
| GeniePath | PyG `examples/geniepath.py` | REPUTABLE_BENCHMARK | PyG master | MIT |
| CARE-GNN | `YingtongDou/CARE-GNN` | OFFICIAL_AUTHOR | `master` @ `a64ff752` | Apache-2.0 |
| BWGNN | `squareRoot3/Rethinking-Anomaly-Detection` | OFFICIAL_AUTHOR | `master` @ `de0631f0` | **NONE** |
| DGA-GNN | `AtwoodDuan/DGA-GNN` | OFFICIAL_AUTHOR | `main` @ `0907392f` | **NONE** |
| PMP | `Xtra-Computing/PMP` | OFFICIAL_LAB | `master` @ `3f7629f6` | **NONE** |
| *(datasets)* | `NineAbyss/GLBench` | OFFICIAL_AUTHOR | `main` | MIT |

**Four of the nine carry no licence at all** (FLAG, BWGNN, DGA-GNN, PMP). Under
default copyright that means no redistribution rights. See section 11.

---

## 1. FLAG (the paper under reproduction)

| Field | Value |
|---|---|
| Paper | FLAG: Fraud Detection with LLM-enhanced Graph Neural Network |
| Venue | KDD '25, Aug 3-7 2025, Toronto. pp. 5150-5160 |
| DOI | `10.1145/3711896.3737220` |
| Authors | Chengdong Yang (BUPT), Hongrui Liu, Daixin Wang, Zhiqiang Zhang, Cheng Yang (BUPT), Chuan Shi (BUPT, corresponding) |
| Repository | https://github.com/BUPT-GAMMA/FLAG |
| Owner | BUPT-GAMMA — the GAMMA Lab at Beijing Univ. of Posts and Telecommunications, Chuan Shi's lab |
| Classification | **OFFICIAL_AUTHOR** |
| Branch @ HEAD | `main` @ `cb83944ed8a8a9b070a3f5a167d363973369fc80` |
| Commit | single commit, "Add files via upload", 2025-06-04 |
| Framework | PyTorch + PyTorch Geometric + transformers + peft + sentence-transformers |
| Python version | **UNKNOWN** — not stated anywhere |
| Dependency versions | **UNKNOWN** — no requirements.txt (see `compatibility_notes.md` for inferred bounds) |
| Native datasets | Reddit, Instagram (as `.pt` files that are **not** shipped) |
| Training entrypoint | `python test.py` (GNN), `python train.py` (LoRA). No README, no example commands. |
| Inference entrypoint | none separate — `test.py:189` evaluates the test split inline |
| README / LICENSE | **both absent** (verified: raw.githubusercontent 404 for each) |
| CPU support | **NO** — `.cuda()` hardcoded at ~40 sites |
| GPU support | yes, single GPU, hardcoded |
| Directly consumes Reddit/Instagram? | Only after unshipped preprocessing (samplers, 1:10 downsampling) |
| Adaptation required? | **YES** — see `flag_code_audit.md` sections 5 and 6 |

Full audit: **`research/flag_code_audit.md`**. Two import-time blockers and one
severed-gradient finding are documented there, each backed by a passing test in
`tests/integration/test_flag_upstream_claims.py`.

### 1.1 FLAG bundles its own baseline re-implementations

`methods/flag/` contains `geniepath.py`, `bwgnn.py`, `caregnn.py`, `dga.py`,
`pmp.py` — **PyG re-implementations written by the FLAG authors**, not the
official baseline code. Several diverge substantially from the original
algorithms (CARE-GNN has no RL selector; BWGNN applies raw adjacency instead of
the normalised Laplacian; DGA has no dynamic grouping). Details and evidence in
`flag_code_audit.md` section 7.

**This creates two legitimate but distinct experiments**, tracked separately by
an `impl_source` field:

- `impl_source=flag_bundled` — reproduces *the paper's numbers*.
- `impl_source=official` — reproduces *the baselines as their authors published them*.

Conflating them would misattribute the baselines' performance. They are never merged.

---

## 2. GeniePath

| Field | Value |
|---|---|
| Paper | GeniePath: Graph Neural Networks with Adaptive Receptive Paths — Liu et al., **AAAI 2019** (arXiv 1802.00910) |
| Official repository | **NONE EXISTS** |

**Verified negative result.** A GitHub repo search for `GeniePath` returns 5
results, none owned by Ziqi Liu, Ant Financial/antgroup, or any co-author. No
Ant Financial release was found. This is recorded as `NONE_EXISTS`, not as
"not found yet".

Candidates, in the order we prefer them:

| # | Source | Class | Pin | License | Notes |
|---|---|---|---|---|---|
| A | `pyg-team/pytorch_geometric` `examples/geniepath.py` | REPUTABLE_BENCHMARK | PyG master | MIT | **SELECTED.** `dim=256, lstm_hidden=256, layer_num=4, heads=1`, Adam lr 0.005, PPI. Matches FLAG's `geniepath.py` globals exactly, so it is also the closest thing to what the paper ran. |
| B | `shuowang-ai/GeniePath-pytorch` (formerly `shawnwang-tech/...`, old URL redirects) | INDEPENDENT | `master` @ `143f07cc49ef9eb9fe176cc355ba2ccba609e57a` | MIT (c) 2019 Shuo Wang | **All 5 open issues are reproducibility failures**: loss NaN after epoch 1, F1 0.45 for the eager variant, missing skip connection in Lazy, paper's "Efficient Numerical Computation" unimplemented, PPI results diverge. Also `val_dataset = PPI(split='test')` — val and test are the same split. |
| C | `dmlc/dgl` `examples/pytorch/geniepath` | REPUTABLE_BENCHMARK | DGL master | Apache-2.0 | Written by Kay Liu (AWS intern), not an author. DGL's own README reports Pubmed **73.0% vs the paper's 78.5%** — a 5.5-point shortfall the DGL maintainers document themselves. Deps: Python 3.7.10, PyTorch 1.8.1, dgl 0.7.0. |
| D | `safe-graph/DGFraud` `algorithms/GeniePath/` | REPUTABLE_BENCHMARK | `master` @ `22b72d75f81dd057762f0c7225a4558a25095b8f` | Apache-2.0 | TensorFlow 1.x, `networkx<=1.11` — effectively uninstallable on modern Python. Supports only DBLP, **not Yelp/Amazon**. Absent from DGFraud-TF2 despite its README table. Defaults: `dim=128, lstm_hidden=128, heads=1, layer_num=4, lr=0.001`. |

**Decision:** use (A). Record in every GeniePath result that no official
implementation exists and that the paper's numbers came from FLAG's own port.

---

## 3. CARE-GNN

| Field | Value |
|---|---|
| Paper | Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters — Dou, Liu, Sun, Deng, Peng, Yu, **CIKM 2020** |
| Repository | https://github.com/YingtongDou/CARE-GNN |
| Owner | `YingtongDou` = Yingtong Dou, **first author** |
| Classification | **OFFICIAL_AUTHOR** |
| Branch @ HEAD | `master` @ `a64ff7523e187a24251f7ca88435d2c9d8f7dcd9` |
| Framework | PyTorch only (custom GraphSAGE-style; no DGL/PyG) |
| Versions | `torch>=1.4.0, numpy>=1.16.4, scipy>=1.2.1, scikit_learn>=0.21rc2` |
| Python | ">= 3.6" (README) |
| Native datasets | **YelpChi, Amazon** (`data/YelpChi.zip`, `data/Amazon.zip`, in-repo) |
| Entrypoint | `unzip data/*.zip` -> `python data_process.py` -> `python train.py` |
| License | **Apache-2.0** |
| CPU support | partial — `os.environ["CUDA_VISIBLE_DEVICES"]="0"` hardcoded at top of `train.py` |
| Consumes Reddit/Instagram? | **NO** — expects multi-relation Yelp/Amazon `.mat` structure |
| Adaptation required? | **YES** — see `compatibility_notes.md` |

Official hyperparameters (argparse defaults):
`--data yelp --model CARE --inter GNN --batch-size 1024` (256 for amazon)
`--lr 0.01 --lambda_1 2 --lambda_2 1e-3 --emb-size 64 --num-epochs 31`
`--test-epochs 3 --under-sample 1 --step-size 2e-2 --seed 72`.
Split: `train_test_split(test_size=0.60, random_state=2, stratify=labels)`;
Amazon skips nodes 0-3304 as unlabelled.

**Known issues affecting reproducibility (author-acknowledged):**
- README "Bug Fixes and Update (06/2021)": the feature/label similarity scores in
  the paper's **Table 2 are incorrect**; corrected values are in the README and
  `simi_comp.py`.
- Per issue #5, the **CARE-Weight variant's weighted aggregation had an error**,
  so **Figure 3's relation-weight subfigure and its conclusion are wrong**.
- Open issue #21 "Problems with the dataset" (2025-11-23).

Alternative: `dmlc/dgl` `examples/pytorch/caregnn` (REPUTABLE_BENCHMARK). Note its
documented deviation — the sampling version uses the previous epoch's
current-layer embedding for Eq. 2 instead of the last layer's.

---

## 4. BWGNN

| Field | Value |
|---|---|
| Paper | Rethinking Graph Neural Networks for Anomaly Detection — Tang, Li, Gao, Li, **ICML 2022** (PMLR v162) |
| Repository | https://github.com/squareRoot3/Rethinking-Anomaly-Detection |
| Owner | `squareRoot3` = Jianheng Tang (HKUST), **first author**. README says "official implementation" |
| Classification | **OFFICIAL_AUTHOR** |
| Branch @ HEAD | `master` @ `de0631f039bbd19c1890b483cc01f1007f596af7` |
| Framework | PyTorch + **DGL** |
| Versions | README: `pytorch 1.9.0`, `dgl 0.8.1`, `sympy`, `sklearn`. No requirements.txt |
| Python | **UNKNOWN** |
| Native datasets | `amazon`, `yelp` (auto-download via DGL), `tfinance`, `tsocial` (manual Google Drive) |
| Entrypoint | `python main.py --dataset amazon --train_ratio 0.4 --hid_dim 64 --order 2 --homo 1 --epoch 100 --run 1` |
| License | **NONE** (LICENSE 404; GitHub API `license: null`) |
| CPU support | **YES** — `main.py` has no `.cuda()` calls at all; AUC computed on CPU |
| Consumes Reddit/Instagram? | **NO** — expects DGL `FraudDataset` graphs |

Official hyperparameters: argparse defaults `dataset=amazon, train_ratio=0.4,
hid_dim=64, order=2, homo=1, epoch=100, run=1`. Hard-coded (not exposed):
`Adam(lr=0.01)`; class-weighted cross-entropy with `weight = #neg/#pos`;
**decision threshold swept over `np.linspace(0.05, 0.95, 19)` for best macro-F1**.
Per-dataset: Amazon 40% / Yelp 1% -> `hid_dim 64, order 2`; T-Social 40% ->
`hid_dim 10, order 5`. BWGNN(hetero) supports only Yelp and Amazon.
Split: `train_test_split(train_size=train_ratio, random_state=2)` then
`test_size=0.67` of the remainder; Amazon uses index range `3305:` only.

**Note for Phase 12:** BWGNN's threshold sweep is exactly the
"choose the threshold on validation" policy the FLAG paper leaves unspecified,
and the FLAG paper says it follows BWGNN's setup. This is the most defensible
basis for our F1 thresholding policy. Recorded as a decision input, not as a
claim about what FLAG did.

Open issues: #16 "issue about dataset", #15 "How to train with minibatch?".
T-Finance/T-Social depend on a single Google Drive link — mirror early.

The author points to **GADBench** (`squareRoot3/GADBench` @ `f9aa021ce9b6c6580427fb633b596843be76ddc6`)
as BWGNN's maintained home.

---

## 5. DGA-GNN

| Field | Value |
|---|---|
| Paper | DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection — Duan, Zheng, Gao, Wang, Feng, Wang, **AAAI 2024** |
| Repository | https://github.com/AtwoodDuan/DGA-GNN |
| Owner | `AtwoodDuan` = Mingjiang Duan (Zhejiang University), **first author**. Description: "The official implementation of the DGA-GNN algorithm." |
| Classification | **OFFICIAL_AUTHOR** |
| Branch @ HEAD | `main` @ `0907392f6060e18230339ca30eaca0c917414820` |
| Framework | PyTorch + DGL + PyTorch Lightning + Hydra + Weights&Biases |
| Versions | `torch==1.13.1, dgl==1.1.2, toad==0.1.1, pandas==1.3.5, numpy==1.21.5, scikit-learn==1.0.2, pytorch-lightning==1.9.4, wandb==0.13.10, hydra-core==1.3.2` |
| Python | **UNKNOWN** |
| Native datasets | Elliptic, T-Finance, T-Social, YelpChi, Amazon |
| Entrypoint | Google Drive `fraud_graph_rawdata.7z` -> `7z x` -> `cd code` -> `python data_handle.py` -> `python train.py --config-name <name>` |
| License | **NONE** |
| CPU support | **NO** — configs set `usegpu: True, gpuid: 0` |
| Consumes Reddit/Instagram? | **NO** |

Official hyperparameters (Hydra YAMLs in `code/configs/`):

| config | n_head | n_hidden | p | k | z | bs | lr | wd | max_epochs | patience | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `amazon.yaml` | 1 | 128 | 0.3 | 4 | 0.05 | 32 | 1e-3 | 5e-4 | 10000 | 25 | 321 |
| `yelpchi.yaml` | 2 | 64 | 0.3 | 32 | 0.025 | 256 | 1e-3 | 5e-4 | 10000 | 25 | 321 |
| `tfinance.yaml` | 1 | 64 | 0.3 | 4 | 0.025 | 64 | 1e-3 | 5e-4 | 10000 | 25 | 321 |
| `tsocial.yaml` | 2 | 128 | 0.2 | 32 | 0.2 | 10240 | 1e-3 | 5e-4 | 10000 | 10 | 321 |
| `elliptic_of_amnet.yaml` | 2 | 128 | 0.3 | 4 | 0.1 | 64 | 1e-3 | 5e-4 | 10000 | 25 | 321 |

All also set `bin_encoding: True, model: dga, usegpu: True, gpuid: 0, nowandb: False`.

**Known issues:**
- **README config names are wrong.** It instructs `--config-name yelpchit / amazont /
  tfinancet / tsocialt`; those files 404. The real names have no trailing `t`.
- `nowandb: False` in every config means **W&B logging is ON by default**; expect a
  login prompt or network call on a fresh run.
- **Zero issues filed** (open or closed) — no community troubleshooting exists.
- Source comments are in Chinese.

---

## 6. PMP

| Field | Value |
|---|---|
| Paper | Partitioning Message Passing for Graph Fraud Detection — Zhuo, Liu, Hooi, He, Tan, Fathony, Chen, **ICLR 2024** |
| Repository | https://github.com/Xtra-Computing/PMP |
| Owner | Xtra Computing Group, **National University of Singapore** — the group led by co-author Bingsheng He |
| Classification | **OFFICIAL_LAB** (org account, not a personal author account) |
| Branch @ HEAD | `master` @ `3f7629f6c180891a0bc1bba3c66d94d288a1ddae` |
| Framework | PyTorch + DGL + PyG |
| Versions | `torch==2.0.1, dgl==1.1.1+cu118, torch_geometric==2.3.1, torch_scatter==2.1.1+pt20cu118, torch_sparse==0.6.17+pt20cu118, numpy==1.23.5, scipy==1.10.0, scikit_learn==1.2.1, torchmetrics==0.11.4, ogb==1.3.6` |
| Python | **UNKNOWN** |
| Native datasets | `amazon`, `yelp` (DGL auto-download), `tfinance`, `tsocial` (BWGNN's Drive), `grab` (proprietary, unusable) |
| Entrypoint | `python main.py --dataset yelp --gpu_id 0`; pretrained: `python test.py --dataset yelp` |
| License | **NONE** |
| CPU support | **NO** — `torch.cuda.set_device(args.gpu_id)` unconditionally at startup; crashes on a CPU-only machine |
| Consumes Reddit/Instagram? | **NO** |

Official hyperparameters (`config/*.yml`, model key `LA-SAGE-S` = PMP):

| | yelp | amazon |
|---|---|---|
| hid_dim | 48 | 256 |
| batch_size | 512 | 128 (`test_batch_size 4096`) |
| dropout | 0.0 | 0.6 |
| epochs / patience | 500 / 100 | 100 / 20 |
| homo | false | true |
| lr / weight_decay | 0.01 / 0.0 | 0.01 / 0.0 |
| n_layer / num_trans | 1 / 1 | 1 / 1 |
| resi | 0.2 | 0.2 |
| train_size / val_size | 0.4 / 0.2 | 0.4 / 0.2 |
| threshold_moving / thres | true / 0.5 | true / 0.5 |
| optimizer / seed / dataset_seed | Adam / 1234 / 717 | same |

Inline comments record the authors' own AUC targets: yelp `0.9397/0.0015 40%`,
amazon `0.9757/0.0012 40%`.

**Known issues:**
- **README/CLI mismatch:** README says `--train_ratio 0.4`; `main.py` defines
  `--train_size`. `--train_ratio` errors out.
- Open issue #2 "T-Social dataset files missing" — corroborated by a missing
  `config/tsocial.yml`. Open issue #3 "How are the labelled nodes chosen?". Both
  unanswered; zero closed issues.
- `+cu118` pins make a plain `pip install -r requirements.txt` fail.
- `requirements.txt` pins **both** `PyYAML==6.0` and `PyYAML==6.0.1`.

**Note:** PMP's official pin `torch_geometric==2.3.1` matches the version this
project independently selected for CPU stability — see `compatibility_notes.md`.

---

## 7. GCN

| Field | Value |
|---|---|
| Paper | Semi-Supervised Classification with Graph Convolutional Networks — Kipf & Welling, **ICLR 2017** |
| Canonical (TF) | https://github.com/tkipf/gcn — **OFFICIAL_AUTHOR**, `master` @ `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`, MIT (file spelled `LICENCE`) |
| Canonical (PyTorch) | https://github.com/tkipf/pygcn — **OFFICIAL_AUTHOR**, `master` @ `1600b5b748b3976413d1e307540ccc62605b4d6d`, MIT |
| **Selected** | **PyG `torch_geometric.nn.GCNConv`** — REPUTABLE_BENCHMARK |

Official TF hyperparameters: `learning_rate=0.01, epochs=200, hidden1=16,
dropout=0.5, weight_decay=5e-4, early_stopping=10, max_degree=3, seed=123`.
Datasets: cora, citeseer, pubmed.

**Why not the author repos:**
- `tkipf/gcn` is TensorFlow 1.x (`tensorflow>=1.15.2,<2.0`), with **119 open issues**
  including installation failures. Effectively dead on modern Python.
- `tkipf/pygcn`'s own README **explicitly disclaims reproduction**: *"This
  re-implementation serves as a proof of concept and is not intended for
  reproduction of the results reported in [1]."* That is decisive.

FLAG's own GCN is `torch_geometric.nn.GCNConv`, so PyG is also the
consistent choice for reproducing the paper.

---

## 8. GAT

| Field | Value |
|---|---|
| Paper | Graph Attention Networks — Velickovic, Cucurull, Casanova, Romero, Lio, Bengio, **ICLR 2018** (arXiv 1710.10903) |
| Canonical | https://github.com/PetarV-/GAT — **OFFICIAL_AUTHOR**, `master` @ `5af87e7fce2b90ae1cbd621cd58059036a3c7436`, MIT |
| **Selected** | **PyG `torch_geometric.nn.GATConv`** — REPUTABLE_BENCHMARK, **author-endorsed** |

Official hyperparameters (hard-coded in `execute_cora.py`, no argparse):
`batch_size=1, nb_epochs=100000, patience=100, lr=0.005, l2_coef=0.0005,
hid_units=[8], n_heads=[8, 1], residual=False, nonlinearity=elu`.
Environment: Python 3.5.2, TensorFlow-GPU 1.6.0, CUDA 9.0 — unusable on modern hardware.

**The author himself redirects users elsewhere.** The README states GAT has
"optimised implementations within virtually all standard GRL libraries" and
*"We recommend using either one of those ... as their implementations have been
more readily battle-tested"*, listing PyTorch Geometric and DGL. Using PyG here
is therefore following the author's own instruction, not a shortcut.

---

## 9. Datasets — GLBench

| Field | Value |
|---|---|
| Paper | GLBench: A Comprehensive Benchmark for Graph with Large Language Models — Li, Wang, Zhu, Chen, Jiang, Cai, Chan, Li, **NeurIPS 2024 D&B** (arXiv 2407.07457) |
| Repository | https://github.com/NineAbyss/GLBench |
| Classification | **OFFICIAL_AUTHOR** (of GLBench) |
| License | **MIT**, (c) 2024 NineAbyss |
| Why this one | **FLAG cites GLBench as reference [25] for both Reddit and Instagram.** Verified from the paper text. |

Full provenance chain and the download links are in **`research/dataset_notes.md`**.

---

## 10. Rejected / ruled out

| Candidate | Verdict |
|---|---|
| `torch_geometric.datasets.Reddit` / DGL `RedditDataset` | **WRONG DATASET.** That is Hamilton et al.'s GraphSAGE Reddit: 232,965 nodes, 41 communities, **no raw text**. Same name, different graph. It will download happily and silently give meaningless results. |
| Kaggle "reddit fraud" / "instagram fake account" datasets | **WRONG.** No graph structure, wrong labels (`fraud`/`fake` vs `popular`/`commercial`), wrong node counts. Explicitly warned against in `dataset_notes.md`. |
| HuggingFace `Graph-COM/Text-Attributed-Graphs` | **NOT THE SOURCE.** Contains Cora, CiteSeer, PubMed, BookHis, BookChild, SportsFit, WikiCS, Cornell, Texas, Wisconsin, Washington. No Instagram, no Reddit. Its `raw_texts.pt` naming is coincidentally similar. |
| GLBench on HuggingFace | Does not exist — `huggingface.co/api/datasets?search=glbench` returns `[]`. Google Drive is the only channel. |
| `tkipf/pygcn` for GCN numbers | Author's README disclaims reproduction. |
| GADBench | Useful cross-check (hosts GCN/GAT/CARE-GNN/BWGNN) but **does not contain GeniePath, DGA-GNN, or PMP**, so it cannot cover our matrix. `master` @ `f9aa021ce9b6c6580427fb633b596843be76ddc6`. |

---

## 11. Licensing — open risk

| Repository | License | Redistribution |
|---|---|---|
| `BUPT-GAMMA/FLAG` | **NONE** | **NOT PERMITTED** by default copyright |
| `squareRoot3/Rethinking-Anomaly-Detection` | **NONE** | **NOT PERMITTED** |
| `AtwoodDuan/DGA-GNN` | **NONE** | **NOT PERMITTED** |
| `Xtra-Computing/PMP` | **NONE** | **NOT PERMITTED** |
| `YingtongDou/CARE-GNN` | Apache-2.0 | permitted with attribution + NOTICE |
| `safe-graph/DGFraud` | Apache-2.0 | permitted with attribution |
| `tkipf/gcn`, `tkipf/pygcn`, `PetarV-/GAT`, `shuowang-ai/GeniePath-pytorch` | MIT | permitted with attribution |
| `NineAbyss/GLBench` | MIT | permitted; **upstream data terms still apply** (see `dataset_notes.md`) |

**Consequences enforced in this repository:**

1. `methods/` is **git-ignored**. No third-party source is committed here.
   `scripts/setup/fetch_methods.sh` reconstructs it from upstream at the pinned SHAs.
   Provenance is preserved without re-hosting anyone's code.
2. Local cloning for research use is standard practice and is what we do.
3. Before publishing any artefact that *includes* code from the four unlicensed
   repositories, permission must be sought from those authors. **This is an open
   item for the maintainer, not a resolved one.**
4. GLBench's MIT grant covers GLBench's packaging. It does **not** override
   Reddit's content terms or the research-use-only terms of the upstream
   Instagram influencer dataset.

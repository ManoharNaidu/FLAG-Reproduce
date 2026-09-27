# Loop 1 / Subagent A — Exhaustive characterisation of `YelpChi.mat` and `Amazon.mat`

**Scope:** research only. No repo code or data was modified.
**Date of analysis:** 2026-09-18
**Interpreter used:** `C:\Users\hp\miniforge3\envs\dgp-bl-consisgad\python.exe`
(numpy 1.23.5, scipy 1.9.3 — the project venv `.venv-cpu` is broken and was not used.)

Every number below came out of a script that was actually executed. Scripts live in
the session scratchpad and are reproduced verbatim in [§10](#10-reproduction).
Anything not established by execution is marked **UNKNOWN**.

---

## 0. Files analysed

| | YelpChi | Amazon |
|---|---|---|
| Path | `data/raw/yelpchi/YelpChi.mat` | `data/raw/amazon/Amazon.mat` |
| Size (bytes) | 207,676,376 | 222,634,416 |
| sha256 | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` | `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63` |
| `__header__` | `MATLAB 5.0 MAT-file Platform: posix, Created on: Wed Aug 19 20:09:02 2020` | `MATLAB 5.0 MAT-file Platform: posix, Created on: Wed Aug 19 20:20:49 2020` |
| `__version__` | `1.0` | `1.0` |
| `__globals__` | `[]` (empty list) | `[]` (empty list) |

Both sha256 values match `data/benchmark/native_*/dataset_manifest.json` as recorded by
`experiments/yelpchi_amazon/build_native_benchmark.py`. The identical creation date
(2020-08-19, 11 minutes apart, `Platform: posix`) is consistent with a single CARE-GNN
release export.

**Corroborating local copies.** `methods/care_gnn/data/YelpChi.zip` and
`methods/care_gnn/data/Amazon.zip` are the upstream archives, still zipped. Their
central directories were listed: each archive contains **exactly one member and nothing
else**, and the member's uncompressed size is byte-identical to the `.mat` on disk.

| Archive | Member | Uncompressed size | CRC32 | Zip timestamp |
|---|---|---|---|---|
| `YelpChi.zip` (17,980,652 B) | `YelpChi.mat` | 207,676,376 | `0DC1B2C4` | 2020-08-19 20:09:02 |
| `Amazon.zip` (26,122,297 B) | `Amazon.mat` | 222,634,416 | `D8E636F0` | 2020-08-19 20:20:50 |

There is **no README, no id file, no side-car mapping** in either archive.

---

## 1. Every key in each file

`scipy.io.whosmat` reports the MAT-file v5 top-level variable table directly (name,
shape, MATLAB class). This is authoritative for "what variables exist": a v5 MAT-file
has a flat top-level variable list, so nothing can be hiding above this level.

### 1.1 YelpChi — `whosmat` output

```
('homo',     (45954, 45954), 'sparse')
('net_rur',  (45954, 45954), 'sparse')
('net_rtr',  (45954, 45954), 'sparse')
('net_rsr',  (45954, 45954), 'sparse')
('features', (45954, 32),    'sparse')
('label',    (1, 45954),     'int64')
```

### 1.2 Amazon — `whosmat` output

```
('homo',     (11944, 11944), 'sparse')
('net_upu',  (11944, 11944), 'sparse')
('net_usu',  (11944, 11944), 'sparse')
('net_uvu',  (11944, 11944), 'sparse')
('features', (11944, 25),    'sparse')
('label',    (1, 11944),     'double')
```

### 1.3 Full per-key detail (from `loadmat`)

**YelpChi**

| Key | Python type | Sparse? | Format | dtype | Shape | nnz | Value range / uniques |
|---|---|---|---|---|---|---|---|
| `__header__` | `bytes` | – | – | – | – | – | MATLAB banner string (above) |
| `__version__` | `str` | – | – | – | – | – | `'1.0'` |
| `__globals__` | `list` | – | – | – | – | – | `[]`, length 0 |
| `homo` | `csc_matrix` | yes | csc | float64 | (45954, 45954) | 7,693,958 | all data `== 1.0` (1 unique) |
| `net_rur` | `csc_matrix` | yes | csc | float64 | (45954, 45954) | 98,630 | all data `== 1.0` |
| `net_rtr` | `csc_matrix` | yes | csc | float64 | (45954, 45954) | 1,147,232 | all data `== 1.0` |
| `net_rsr` | `csc_matrix` | yes | csc | float64 | (45954, 45954) | 6,805,486 | all data `== 1.0` |
| `features` | `csc_matrix` | yes | csc | float64 | (45954, 32) | 1,469,088 | dense min 0.0, max 1.0, mean 0.6311111426021925; 72,086 distinct stored values |
| `label` | `ndarray` | no | – | int64 | (1, 45954) | – | uniques `{0, 1}` |

**Amazon**

| Key | Python type | Sparse? | Format | dtype | Shape | nnz | Value range / uniques |
|---|---|---|---|---|---|---|---|
| `__header__` | `bytes` | – | – | – | – | – | MATLAB banner string (above) |
| `__version__` | `str` | – | – | – | – | – | `'1.0'` |
| `__globals__` | `list` | – | – | – | – | – | `[]`, length 0 |
| `homo` | `csc_matrix` | yes | csc | float64 | (11944, 11944) | 8,796,784 | all data `== 1.0` |
| `net_upu` | `csc_matrix` | yes | csc | float64 | (11944, 11944) | 351,216 | all data `== 1.0` |
| `net_usu` | `csc_matrix` | yes | csc | float64 | (11944, 11944) | 7,132,958 | all data `== 1.0` |
| `net_uvu` | `csc_matrix` | yes | csc | float64 | (11944, 11944) | 2,073,474 | all data `== 1.0` |
| `features` | `csc_matrix` | yes | csc | float64 | (11944, 25) | 174,488 | dense min −1.0, max 5525.0, mean 16.947474677171016; 5,319 distinct stored values |
| `label` | `ndarray` | no | – | **float64** | (1, 11944) | – | uniques `{0.0, 1.0}` |

Note the dtype asymmetry: YelpChi's `label` is `int64`, Amazon's is `float64` (MATLAB
class `double`). Anything that consumes these must cast; `build_native_benchmark.py`
already does (`.astype(np.int64)`).

---

## 2. Feature matrices

### 2.1 YelpChi features

| Property | Value |
|---|---|
| Storage | `scipy.sparse.csc_matrix`, dtype float64 |
| Shape | **(45954, 32)** — feature dimension = **32** |
| nnz | 1,469,088 (of 45954×32 = 1,470,528 → density 0.99902) |
| Dense min / max / mean | 0.0 / 1.0 / 0.6311111426021925 |
| NaNs | 0 (`np.isnan(...).any() == False`) |
| Infs | 0 (`np.isinf(...).any() == False`) |
| Constant columns | **none** |
| All-zero columns | **none** |
| Distinct feature rows | **45,954 / 45,954 — every row is unique** |

Per-column statistics (`intlike` = every value is an integer to 1e-9; `frac0` = fraction
of rows equal to 0):

| col | min | max | mean | std | n_uniq | intlike | frac0 |
|---|---|---|---|---|---|---|---|
| 0 | 0.00317531 | 1 | 0.487706 | 0.290194 | 1782 | False | 0.000 |
| 1 | 0.000237406 | 0.999985 | 0.497164 | 0.289082 | 818 | False | 0.000 |
| 2 | 0.428682 | 0.999985 | 0.757261 | 0.282408 | **2** | False | 0.000 |
| 3 | 0.0443208 | 0.999985 | 0.956293 | 0.199616 | **2** | False | 0.000 |
| 4 | 0.022791 | 0.999985 | 0.97787 | 0.145333 | **2** | False | 0.000 |
| 5 | 0.398457 | 0.999985 | 0.751894 | 0.296116 | **2** | False | 0.000 |
| 6 | 0 | 0.999985 | 0.513187 | 0.303623 | 4493 | False | 0.000 |
| 7 | 4.45137e-05 | 0.999985 | 0.500954 | 0.289246 | 15086 | False | 0.000 |
| 8 | 1.48379e-05 | 1 | 0.503624 | 0.288161 | 847 | False | 0.000 |
| 9 | 0.0980191 | 1 | 0.592509 | 0.362081 | 560 | False | 0.000 |
| 10 | 0.0188145 | 0.999985 | 0.637028 | 0.385068 | 363 | False | 0.000 |
| 11 | 0.000103865 | 0.999985 | 0.524359 | 0.318117 | 1828 | False | 0.000 |
| 12 | 0.000801246 | 1 | 0.524283 | 0.317951 | 1828 | False | 0.000 |
| 13 | 0.381482 | 0.99997 | 0.578257 | 0.206066 | 29232 | False | 0.000 |
| 14 | 0.381646 | 1 | 0.578327 | 0.206032 | 29199 | False | 0.000 |
| 15 | 0 | 0.999974 | 0.900338 | 0.28435 | 13 | False | 0.000 |
| 16 | 0.643092 | 0.999974 | 0.727838 | 0.139254 | 110 | False | 0.000 |
| 17 | 0.122481 | 0.999974 | 0.831301 | 0.340964 | 73 | False | 0.000 |
| 18 | 0.000288995 | 0.999974 | 0.505725 | 0.286882 | 9381 | False | 0.000 |
| 19 | 0.000288995 | 0.999974 | 0.506984 | 0.287117 | 10564 | False | 0.000 |
| 20 | 0.755169 | 0.999974 | 0.811773 | 0.102222 | 29 | False | 0.000 |
| 21 | 0.770512 | 1 | 0.806443 | 0.0691329 | 324 | False | 0.000 |
| 22 | 0.94806 | 1 | 0.949713 | 0.00763359 | 151 | False | 0.000 |
| 23 | 2.62722e-05 | 0.999947 | 0.505795 | 0.286474 | 3145 | False | 0.000 |
| 24 | 0.00497512 | 0.995025 | 0.777519 | 0.244746 | 8 | False | 0.000 |
| 25 | 0.0895522 | 0.995025 | 0.525344 | 0.32356 | 156 | False | 0.000 |
| 26 | 0 | 0.995025 | 0.509241 | 0.353805 | 156 | False | 0.015 |
| 27 | 0 | 0.995025 | 0.51548 | 0.326225 | 170 | False | 0.008 |
| 28 | 0 | 0.995025 | 0.52798 | 0.31545 | 175 | False | 0.008 |
| 29 | 0.0348259 | 1 | 0.464007 | 0.323113 | 168 | False | 0.000 |
| 30 | 0.139303 | 1 | 0.442475 | 0.278123 | 157 | False | 0.000 |
| 31 | 0.00995025 | 1 | 0.506883 | 0.296719 | 182 | False | 0.000 |

**No column is a raw rating, date, or count.** Every column is a real number in [0, 1].
See §7 for what the values *are* quantised on — that is the interesting part.

### 2.2 Amazon features

| Property | Value |
|---|---|
| Storage | `scipy.sparse.csc_matrix`, dtype float64 |
| Shape | **(11944, 25)** — feature dimension = **25** |
| nnz | 174,488 (of 11944×25 = 298,600 → density 0.5844) |
| Dense min / max / mean | −1.0 / 5525.0 / 16.947474677171016 |
| NaNs | 0 |
| Infs | 0 |
| Constant columns | none |
| All-zero columns | none |
| All-zero feature **rows** | 0 |
| Distinct feature rows | **10,616 / 11,944 — 1,328 rows are duplicates of another row** |

> **The 24-vs-25 literature discrepancy.** The file on disk contains **25** columns.
> Not 24. This is read straight from the MAT-file variable header
> (`('features', (11944, 25), 'sparse')`) and confirmed by the dense shape.
> I did **not** resolve *why* some of the literature says 24 — see §8 for a
> code-level hypothesis with supporting evidence, explicitly flagged as a hypothesis.

Per-column statistics:

| col | min | max | mean | std | n_uniq | intlike | frac0 |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 483 | 3.32619 | 9.88692 | 81 | **True** | 0.000 |
| 1 | 0 | 49 | 14.1024 | 8.681 | 50 | **True** | 0.003 |
| 2 | 0 | 7 | 0.16569 | 0.458884 | 8 | **True** | 0.857 |
| 3 | 0 | 14 | 0.126842 | 0.460135 | 10 | **True** | 0.898 |
| 4 | 0 | 119 | 0.273359 | 1.47019 | 20 | **True** | 0.826 |
| 5 | 0 | 149 | 0.694993 | 2.75002 | 38 | **True** | 0.680 |
| 6 | 0 | 429 | 2.0653 | 6.74262 | 70 | **True** | 0.297 |
| 7 | 0 | 1 | 0.0951235 | 0.274176 | 90 | False | 0.857 |
| 8 | 0 | 1 | 0.0484689 | 0.188278 | 92 | False | 0.898 |
| 9 | 0 | 1 | 0.0811891 | 0.233003 | 142 | False | 0.826 |
| 10 | 0 | 1 | 0.185103 | 0.331122 | 201 | False | 0.680 |
| 11 | 0 | 1 | 0.590116 | 0.432705 | 226 | False | 0.297 |
| 12 | 0 | 1 | 0.143592 | 0.321014 | 128 | False | 0.774 |
| 13 | 0 | 1 | 0.775219 | 0.373967 | 225 | False | 0.166 |
| 14 | −9.99995e-06 | 1.60939 | 0.218462 | 0.371783 | 497 | False | 0.000 |
| 15 | 1 | 5 | 4.18909 | 1.2255 | 9 | False | 0.000 |
| 16 | 1 | 5 | 4.34913 | 1.211 | 5 | **True** | 0.000 |
| 17 | 1 | 5 | 3.72597 | 1.45282 | 5 | **True** | 0.000 |
| 18 | 1 | 5 | 4.12662 | 1.19533 | 337 | False | 0.000 |
| 19 | 0 | 201 | 3.15807 | 10.9229 | 113 | **True** | 0.604 |
| 20 | 0 | 5525 | 352.486 | 733.075 | 2044 | **True** | 0.643 |
| 21 | 0 | 2.44527 | 0.262991 | 0.455403 | 630 | False | 0.712 |
| 22 | 0 | 1 | 0.642749 | 0.47919 | **2** | **True** | 0.357 |
| 23 | 1 | 128 | 27.0426 | 14.3073 | 1343 | False | 0.000 |
| 24 | **−1** | 1 | 0.811788 | 0.569875 | **3** | **True** | 0.016 |

Full unique lists for the small-cardinality columns:

```
col  2 (8 uniq):  [0. 1. 2. 3. 4. 5. 6. 7.]
col  3 (10 uniq): [0. 1. 2. 3. 4. 5. 6. 9. 12. 14.]
col  4 (20 uniq): [0. 1. 2. ... 21. 32. 56. 119.]
col 15 (9 uniq):  [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]
col 16 (5 uniq):  [1. 2. 3. 4. 5.]
col 17 (5 uniq):  [1. 2. 3. 4. 5.]
col 22 (2 uniq):  [0. 1.]
col 24 (3 uniq):  [-1. 0. 1.]
```

Unlike YelpChi, Amazon features are **raw, un-normalised, human-readable quantities**.
§8 decodes all 25 of them and proves the decoding arithmetically.

---

## 3. Labels

| | YelpChi | Amazon |
|---|---|---|
| Key | `label` | `label` |
| Shape | (1, 45954) → flatten to (45954,) | (1, 11944) → flatten to (11944,) |
| dtype | int64 | float64 |
| Unique values | `{0, 1}` | `{0.0, 1.0}` |
| count(0) | 39,277 (85.4703 %) | 11,123 (93.1263 %) |
| count(1) | **6,677 (14.5297 %)** | **821 (6.8737 %)** |

Amazon, split at CARE-GNN's documented unlabeled boundary:

| Index range | label 0 | label 1 |
|---|---|---|
| `[0, 3305)` | 3,305 | 0 |
| `[3305, 11944)` | 7,818 | 821 |

First index with label 1 = 3306. Last = 11940. `3305 + 8639 = 11944`, and
`821 + 7818 = 8639`.

### 3.1 Which value denotes fraud/spam? → **1 = fraud/spam. Both datasets.**

Evidence, strongest first:

**(E1) DIRECT SOURCE EVIDENCE — Amazon only, definitive.**
`methods/care_gnn/amazon_preprocess.py` (the upstream script that *created* the Amazon
graph) builds the ground truth itself:

```python
helpful, votes = sum([single['helpful'][0] for single in total]), sum([single['helpful'][1] for single in total])
if votes >= 20:
    if helpful/votes > 0.8:
        labeled_reviews[u] = total
        user_labels.append(0)
    elif helpful/votes < 0.2:
        labeled_reviews[u] = total
        user_labels.append(1)
```

Users whose votes are **overwhelmingly "unhelpful"** (helpful ratio < 0.2) get label **1**.
Those are the fraudulent/spam reviewers. This is not an inference — it is the label
definition, in the code, in this repo.

**(E2) DIRECT SOURCE EVIDENCE — both datasets, from the consumer side.**
`methods/care_gnn/utils.py::pos_neg_split`:

```python
for idx, label in enumerate(labels):
    if label == 1:
        pos_nodes.append(aux_nodes[idx])
        neg_nodes.remove(aux_nodes[idx])
```

Label 1 → *positive* class. And `utils.py::test_care` / `test_sage` score the model with
`roc_auc_score(labels, gnn_prob[:, 1])` — i.e. class index 1 is the fraud score that
CARE-GNN's reported AUC/AP is computed against. `methods/care_gnn/README.md` reinforces
this: *"we only compute the similarity scores for positive nodes to demonstrate the
camouflage of fraudsters (positive nodes)"* — fraudsters ≡ positive ≡ label 1.

**(E3) NODE-ORDERING CORROBORATION — Amazon, computational.**
`amazon_preprocess.py` assembles `new_reviews = {**sampled_reviews, **labeled_reviews}`,
i.e. the randomly-sampled *unlabeled* users first, then the labeled ones. The file shows
exactly that layout: the first 3,305 nodes are all label 0 and contain **zero** positives,
and the remaining 8,639 = 821 + 7,818 carry the whole labeled set. This independently
confirms both CARE-GNN's "0-3304 are unlabeled nodes" comment in `train.py` and the fact
that the 0s in the prefix are a *default fill for unlabeled users*, not genuine benign
labels.

**(E4) CIRCUMSTANTIAL — class asymmetry.** 1 is the minority class in both datasets
(14.53 % Yelp, 6.87 % Amazon), which is what a fraud positive class should look like.

**Confidence.** Amazon: **certain** (E1+E2+E3 are all direct source evidence).
YelpChi: **high but strictly weaker** — E2 and E4 only. There is **no YelpChi
preprocessing script in this repository** (CARE-GNN ships `amazon_preprocess.py` but no
Yelp equivalent), so the Yelp labels' construction rule is **UNKNOWN from local
evidence**. The determination "1 = spam" for Yelp rests on CARE-GNN's own downstream
treatment of label 1 as positive/fraudster, not on the labelling code.

---

## 4. Relation / adjacency matrices

All eight matrices: dtype `float64`, `csc_matrix`, **every stored value is exactly 1.0**
(1 unique data value), **perfectly symmetric** (`(A != A.T).nnz == 0`), and **zero
self-loops** (no nonzero on the diagonal). Because they are symmetric with an empty
diagonal, undirected edges = nnz / 2 exactly.

### 4.1 YelpChi

| `.mat` key | Canonical name | Shape | nnz (directed entries) | Self-loops | Symmetric | Undirected edges | deg min/max/mean | Isolated nodes |
|---|---|---|---|---|---|---|---|---|
| `net_rur` | **R-U-R** | (45954, 45954) | 98,630 | 0 | yes | 49,315 | 0 / 46 / 2.146 | 22,123 |
| `net_rtr` | **R-T-R** | (45954, 45954) | 1,147,232 | 0 | yes | 573,616 | 0 / 118 / 24.965 | 522 |
| `net_rsr` | **R-S-R** | (45954, 45954) | 6,805,486 | 0 | yes | 3,402,743 | 0 / 465 / 148.093 | 40 |
| `homo` | union of the three | (45954, 45954) | 7,693,958 | 0 | yes | 3,846,979 | 0 / 501 / 167.427 | 13 |

### 4.2 Amazon

| `.mat` key | Canonical name | Shape | nnz (directed entries) | Self-loops | Symmetric | Undirected edges | deg min/max/mean | Isolated nodes |
|---|---|---|---|---|---|---|---|---|
| `net_upu` | **U-P-U** | (11944, 11944) | 351,216 | 0 | yes | 175,608 | 0 / 511 / 29.405 | 1,720 |
| `net_usu` | **U-S-U** | (11944, 11944) | 7,132,958 | 0 | yes | 3,566,479 | 0 / 6311 / 597.200 | 90 |
| `net_uvu` | **U-V-U** | (11944, 11944) | 2,073,474 | 0 | yes | 1,036,737 | 0 / 6477 / 173.600 | 81 |
| `homo` | **NOT the union** (see §5) | (11944, 11944) | 8,796,784 | 0 | yes | 4,398,392 | 3 / 6991 / 736.502 | 0 |

### 4.3 Key → semantic relation: the naming/source evidence

**This mapping is established by key name and by source code, never by array order.**
The task explicitly warned against ordering-based inference; note that array *order* in
the file is `homo, net_rur, net_rtr, net_rsr` for Yelp — i.e. **RUR, RTR, RSR**, which is
*not* the order the CARE-GNN README table prints (`rur | rtr | rsr` — actually it matches
here) nor the order `build_native_benchmark.py` declares its dict in
(`rur, rtr, rsr` — also matches). Ordering is therefore not load-bearing anywhere, but
the evidence below is what actually fixes the mapping.

**Evidence tier 1 — the key names are the relation names.** `net_rur` / `net_rtr` /
`net_rsr` and `net_upu` / `net_usu` / `net_uvu` literally spell R-U-R, R-T-R, R-S-R and
U-P-U, U-S-U, U-V-U.

**Evidence tier 2 — CARE-GNN's own code binds key → name, by key not by position.**
`methods/care_gnn/data_process.py`:

```python
net_rur = yelp['net_rur'];  sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
net_rtr = yelp['net_rtr'];  sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
net_rsr = yelp['net_rsr'];  sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
net_upu = amz['net_upu'];   sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
net_usu = amz['net_usu'];   sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
net_uvu = amz['net_uvu'];   sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
```

and `methods/care_gnn/README.md` reports per-relation similarity scores under the headings
`rur | rtr | rsr | homo` and `upu | usu | uvu | homo`.

**Evidence tier 3 — Amazon only: the relation *construction code* is in this repo.**
`methods/care_gnn/amazon_preprocess.py::build_graph` contains three builders, one active
and two commented out, each labelled:

| Comment in `build_graph` | Rule implemented | Relation |
|---|---|---|
| `# user-product-user` (ACTIVE) | edge iff `len(set(products[u1]) & set(products[u2])) >= 1` — two users reviewed at least one common `asin` | **U-P-U** |
| `# user-star&time-user` (commented) | edge iff `star_judge(...)` **and** `time_judge(..., diff=7)` — same star rating within 7 days | **U-S-U** |
| `# user-textsim_user` (commented) | edge iff TF-IDF cosine ≥ the top-5 % threshold over all pairs | **U-V-U** |

So for Amazon the semantics of all three relation keys are pinned by upstream source, and
`v` in U-V-U is the **text-similarity** ("view"/TF-IDF) relation.

**Evidence tier 4 — an independent local consumer agrees on node type.**
`methods/dga_gnn/code/data_handle.py` builds heterographs with canonical etypes
`("review", "net_rur", "review")`, `("review", "net_rtr", "review")`,
`("review", "net_rsr", "review")` for YelpChi and
`("user", "net_upu", "user")`, `("user", "net_usu", "user")`, `("user", "net_uvu", "user")`
for Amazon — i.e. **YelpChi nodes are reviews, Amazon nodes are users/reviewers**.
`methods/glbench/.../ZeroG/code/dataset_benchmark.py` lists the same key sets.

**What is NOT established locally.** For **YelpChi** there is **no relation-construction
source in this repo** — only the names. The usual prose definitions ("R-U-R = reviews by
the same user; R-T-R = reviews of the same product in the same month; R-S-R = reviews of
the same product with the same star rating") appear **nowhere** in the local tree
(`grep -rniE "same (user|star|month|product|rating)|posted by|top.?5%" methods src docs`
returned no matches). I therefore attempted a structural check and it **did not confirm
those definitions** — see §7.4. **The precise semantics of `net_rtr` and `net_rsr` are
UNKNOWN from local evidence**, beyond the names themselves.

---

## 5. Is `homo` the union of the three relations?

Verified computationally on sparsity patterns (`(M != 0)`), with set algebra over all
subsets of the three relations.

### 5.1 YelpChi — **YES, exactly.**

```
union(net_rur, net_rtr, net_rsr) nnz = 7,693,958
homo                             nnz = 7,693,958
(homo_pattern != union_pattern).nnz  = 0        -> EXACT MATCH
```

### 5.2 Amazon — **NO. Not even close.** ⚠️

```
|homo|             = 8,796,784
|union(all three)| = 8,835,152
|homo AND union|   = 6,786,522
|homo MINUS union| = 2,010,262     <-- edges in homo that no relation has
|union MINUS homo| = 2,048,630     <-- edges in a relation that homo lacks
Jaccard(homo, union) = 0.625750
```

Every subset was tested; none matches:

| union of | nnz | symmetric difference vs `homo` | exact? |
|---|---|---|---|
| `{upu}` | 351,216 | 8,551,192 | no |
| `{usu}` | 7,132,958 | 4,501,502 | no |
| `{uvu}` | 2,073,474 | 7,943,546 | no |
| `{upu, usu}` | 7,308,074 | 4,428,354 | no |
| `{upu, uvu}` | 2,392,996 | 7,727,048 | no |
| `{usu, uvu}` | 8,666,342 | 4,128,026 | no |
| `{upu, usu, uvu}` | 8,835,152 | 4,058,892 | no |

Stronger still, **no individual relation is even a subset of `homo`**:

| relation | nnz | ∩ homo | minus homo | subset of homo? |
|---|---|---|---|---|
| `net_upu` | 351,216 | 298,404 | **52,812** | **no** |
| `net_usu` | 7,132,958 | 5,714,120 | **1,418,838** | **no** |
| `net_uvu` | 2,073,474 | 1,463,356 | **610,118** | **no** |

Degree comparison (is `homo` a *permutation* of the union?):

| | min | max | mean |
|---|---|---|---|
| `homo` degree | 3 | 6991 | 736.50 |
| union degree | 3 | 6981 | 739.71 |

`Pearson corr(deg_homo, deg_union) = 0.952569`; sorted degree sequences are **not**
identical; only **399 / 11,944** nodes have `deg_homo == deg_union`. Block counts:
`homo[0:3305, 0:3305].nnz = 267,106` vs union `272,554`;
`homo[3305:, 3305:].nnz = 6,597,068` vs union `6,598,582`.

**Interpretation:** `homo` and the union are highly correlated but genuinely different
graphs — the totals are close (0.4 % apart) while ~23 % of each side's edges are absent
from the other. **The cause is UNKNOWN.** Candidate explanations I could *not* test with
local data: the released `homo` was exported from a different (earlier/later) build of
the three relations than the ones shipped alongside it; or one relation was regenerated
after `homo` was frozen. Nothing in `amazon_preprocess.py` constructs `homo` at all.

**Practical consequence for this repo:** `build_native_benchmark.py` stores
`edge_index_homo` from the `homo` key and the three relations separately, and its manifest
describes `homo` as *"one 'homo' (union) edge_index"*. **That description is wrong for
Amazon.** The manifest field `graph_structure` should be corrected — but per the
research-only scope, I made no change. For YelpChi the description is exactly right.

---

## 6. ⚠️ THE CENTRAL QUESTION: is there ANY node → original-review/reviewer identifier?

# **NO. Unambiguously no.**

**Neither `.mat` file contains a single identifier of any kind that links a graph node
back to an original review, reviewer, product, or text.**

Specifically, there is **no** `review_id`, `reviewerID`, `reviewerName`, `asin`,
`user_id`, `product_id`, `date`, `unixReviewTime`, `rating`/`overall`, `summary`,
`reviewText`, or any raw text anywhere in either file.

How this was established exhaustively:

1. **The complete top-level variable table was enumerated** with `scipy.io.whosmat`,
   which reads the MAT-file v5 variable directory directly. YelpChi has **exactly 6**
   variables; Amazon has **exactly 6**. They are listed in full in §1.1 / §1.2. There is
   nothing else in either file.

2. **Every variable's MATLAB class was checked.** `whosmat`'s third field reports the
   class. For both files every entry is either `'sparse'` (the 4 adjacency-like
   matrices + `features`) or a numeric class (`'int64'` for Yelp `label`, `'double'` for
   Amazon `label`). **Not one variable is `struct`, `cell`, `char`, `object`, or
   `logical`.** So there is no struct/cell that `loadmat` could be flattening or hiding.

3. **A defensive scan was run over every `loadmat` value** looking for
   `dtype == object`, a structured dtype (`dtype.names is not None`), or a
   string/void kind (`dtype.kind in "USV"`). Result for both files:

   ```
   suspicious (object/struct/char) keys = []
   ```

4. **The three `__`-prefixed keys carry no payload.** `__header__` is the MATLAB banner
   string (creation timestamp + platform — metadata about the *file*, not about nodes);
   `__version__` is `'1.0'`; `__globals__` is an **empty list** in both files.

5. **The upstream archives contain nothing else.** `methods/care_gnn/data/YelpChi.zip`
   and `Amazon.zip` each hold exactly one member — the `.mat` itself. No id map, no
   README, no index file (§0).

**Conclusion:** the files are *only* features + labels + adjacencies. Every node is an
anonymous integer index in `[0, N)`. **Direct raw-text mapping via an identifier stored
in the `.mat` is impossible.** Any node → review/reviewer mapping must be reconstructed
by matching feature values against externally-obtained raw data — which is what §7 and §8
assess.

---

## 7. Can the 32 YelpChi features act as a join key?

**Verdict: plausibly yes in principle, but unproven, and no local artefact enables it.
Confidence: LOW–MEDIUM.**

### 7.1 Row uniqueness (necessary condition)

`np.unique(F, axis=0).shape[0] == 45954` out of 45,954 rows. **Every YelpChi node has a
unique 32-dim fingerprint.** So a join is not blocked by collisions — unlike Amazon (§8.4).

### 7.2 The features are exactly quantised — and by three different denominators

This is the strongest structural finding on the Yelp side. Reducing each distinct value
with `fractions.Fraction(v).limit_denominator(200000)` shows the columns split into three
clean blocks, each with one base denominator (the other denominators observed are its
divisors: 22465 = 67395/3, 13479 = 67395/5, 4493 = 67395/15, 2239 = 38063/17, 67 = 201/3):

| Block | Columns | Base denominator **D** | max over all uniques of \|v·D − round(v·D)\| |
|---|---|---|---|
| A | 0 – 14 (15 cols) | **67,395** | 7.28e−12 |
| B | 15 – 23 (9 cols) | **38,063** | 3.64e−12 |
| C | 24 – 31 (8 cols) | **201** | 2.84e−14 |

Cross-check that the assignment is not coincidence — applying the *wrong* denominator
fails catastrophically:

| Block | D = 67395 | D = 38063 | D = 201 |
|---|---|---|---|
| cols 0–14 | **7.3e−12 ✓** | 0.5 ✗ | 0.5 ✗ |
| cols 15–23 | 0.5 ✗ | **3.6e−12 ✓** | 0.5 ✗ |
| cols 24–31 | 0.4925 ✗ | 0.4975 ✗ | **2.8e−14 ✓** |

So **every value in the YelpChi feature matrix is an exact integer multiple of 1/67395,
1/38063 or 1/201**, depending on which of three column blocks it is in. Supporting
observations: `1/min_positive` is exactly 67395.0000 for cols 6 and 8, exactly 38063.0000
for cols 15 and 23, exactly 201.0000 for cols 24, 26, 27, 28; and `1/(1−max)` is exactly
67395 for cols 1–7 and 10–11, exactly 38063 for cols 15–20, exactly 201 for cols 24–28.

**Interpretation (confidence: MEDIUM for the fact, LOW for the semantics).** Three
distinct integer denominators partitioning 32 columns into 15 / 9 / 8 is the signature of
a rank- or count-normalisation performed separately over three different entity
populations of sizes 67,395, 38,063 and 201. The natural reading is
*review-level / reviewer-level / product-level* feature blocks — which matches the
three-tier structure of the standard YelpChi behavioural feature set. **I did not verify
that 67,395 / 38,063 / 201 are the YelpChi review / reviewer / product counts**, because
verifying that would require the original Rayana & Akoglu metadata, which is not in this
repo. Treat the *numbers* as established fact and the *entity interpretation* as a
hypothesis.

### 7.3 Latent group structure implied by each block

| Block | Columns | Distinct row-tuples over 45,954 rows | Chance collision Σpᵢ² |
|---|---|---|---|
| A (D=67395) | 0–14 | **45,951** (essentially one per node) | 0.000022 |
| B (D=38063) | 15–23 | **27,261** | 0.000074 |
| C (D=201) | 24–31 | **182** | 0.010570 |

Block C collapsing 45,954 rows onto only **182** distinct tuples — with 182 ≤ 201, its own
denominator — is a strong hint that block C is a per-entity constant over ~182 entities.
If that entity is the product, block C is effectively a **latent product grouping**
recoverable from the features alone (up to relabelling). Confidence: **MEDIUM**.

### 7.4 Structural cross-check against the relations — **it did not confirm the textbook definitions**

If R-S-R and R-T-R were product-restricted and block C were product-level, block C would
be *constant* on every R-S-R / R-T-R edge. It is not:

| Relation | P(endpoints share block A) | block B | block C |
|---|---|---|---|
| `net_rur` | 0.000000 | **0.421616** | 0.073629 |
| `net_rtr` | 0.000000 | 0.002213 | 0.110370 |
| `net_rsr` | 0.000000 | 0.000702 | 0.089517 |
| `homo` | 0.000000 | 0.006182 | 0.091750 |
| *(chance)* | *0.000022* | *0.000074* | *0.010570* |

Component-level (multi-node connected components only):

| Relation | # components | multi-node | block A const | block B const | block C const |
|---|---|---|---|---|---|
| `net_rur` | 29,431 (largest 47, 22,123 singletons) | 7,308 | 0 / 7308 | **2,808 / 7308 (38.4 %)** | 906 / 7308 (12.4 %) |
| `net_rsr` | 803 (largest 466, 40 singletons) | 763 | 0 / 763 | 0 / 763 | **325 / 763 (42.6 %)** |
| `net_rtr` | 1,835 (largest 331, 522 singletons) | 1,313 | 0 / 1313 | 0 / 1313 | **635 / 1313 (48.4 %)** |
| `homo` | 26 (largest 45,900, 13 singletons) | – | – | – | – |

Per-column constancy within multi-node `net_rur` components, against a per-column chance
baseline (`mean over components of Σ pᵢ^{|component|}`) — only the high-lift rows are
informative, since low-cardinality columns are constant by accident:

| col | observed const | chance | lift |
|---|---|---|---|
| 0 | 0.0029 | 0.000783 | 3.7× |
| 1 | 0.1097 | 0.002003 | **54.8×** |
| 2–5 | 0.40 / 0.89 / 0.95 / 0.44 | 0.36 / 0.86 / 0.93 / 0.36 | **~1.0× (chance only)** |
| 6–12 | 0.001–0.19 | ~same | 1.0–1.7× |
| 13 | 0.3555 | 0.085027 | 4.2× |
| 14 | 0.3555 | 0.085104 | 4.2× |
| 15 | 0.8807 | 0.710660 | 1.2× |
| 16 | 0.6448 | 0.330842 | 1.9× |
| 17 | 0.7677 | 0.541850 | 1.4× |
| **18** | 0.4212 | 0.000973 | **432.7×** |
| **19** | 0.4158 | 0.000786 | **529.2×** |
| 20 | 0.7512 | 0.451040 | 1.7× |
| 21 | 0.7567 | 0.454476 | 1.7× |
| 22 | 0.9535 | 0.846059 | 1.1× |
| **23** | 0.3848 | 0.001436 | **267.9×** |
| 24 | 0.3678 | 0.214802 | 1.7× |
| 25–31 | 0.12–0.19 | 0.006–0.042 | 4.4× – **20.9×** |

**Honest reading.** Block-B columns 18, 19, 23 are 268–529× more likely than chance to be
constant across an `net_rur` component — `net_rur` is unmistakably grouping nodes that
share reviewer-associated feature values. But **nothing reaches 1.0**: not one column is
*deterministically* constant on any relation. So the clean story ("block B = reviewer-level
constants, R-U-R = same reviewer") is **elevated far above chance but not confirmed**. I
could not reconcile the 38.4 % / 42.6 % / 48.4 % figures with exact per-entity constancy
and I am **not** going to assert a definition the data does not support.
**Status: UNKNOWN.** Candidate explanations I could not test: the blocks mix per-entity
constants with per-review deviations computed *relative to* that entity (the standard
behavioural-feature design), so only a subset of each block is entity-constant; or the
normalisation was computed over a **superset** corpus and then subset to these 45,954
rows, which also fits the empirical-CDF test failing by small margins
(max\|ECDF(v)−v\| ≈ 0.0002–0.027 for cols 0–23, i.e. close but never 0).

### 7.5 Verdict for YelpChi joinability

- **No column is a literal rating, date, or count** — everything is rank/ratio-normalised
  into [0, 1]. There is **no direct join key**.
- Rows are 100 % unique, so an **indirect** join is not blocked in principle: if one
  obtains the original YelpChi metadata and re-derives the same 32 features *with the same
  normalisation constants* (67395 / 38063 / 201), an exact match is conceivable.
- **But the feature-extraction code is not in this repository** and the semantics of the
  32 columns are **UNKNOWN** locally. Attempting this reconstruction is a research project,
  not a lookup. **Confidence that it would succeed: LOW.**

---

## 8. Can the 25 Amazon features act as a join key?

**Verdict: YES, far more plausibly than YelpChi — because all 25 columns are decoded and
proven, and they are raw un-normalised quantities. Confidence: MEDIUM–HIGH for the
semantics (proven); MEDIUM for an actual successful join.**

### 8.1 The decoding is proven, not guessed

`methods/care_gnn/amazon_preprocess.py::build_features` builds a **36-dim** matrix with
numbered comments, then filters:

```python
new_features = features[:, :19]
new_features = np.hstack([new_features, features[:, 30:]])
# sp.save_npz('amz_features_25.npz', sp.csr_matrix(new_features))
```

`19 + 6 = 25`, and the file has exactly 25 columns — **slice check: match = True**.

Every arithmetic relation implied by that code was then tested against the actual bytes.
All passed **exactly** (`max |difference| = 0`, not merely `allclose`):

| # | Hypothesis | Result |
|---|---|---|
| H1 | `col0 == sum(col2..col6)` (n rated products == total ratings) | max\|diff\| = **0**; 11944/11944 exact |
| H2 | `col(7+k) == col(2+k) / col0` (ratio == count / n) | max\|diff\| = **0**; allclose True |
| H3 | `col12 == col7 + col8` (ratio of 1★ + 2★) | max\|diff\| = **0** |
| H3b | `col13 == col10 + col11` (ratio of 4★ + 5★) | max\|diff\| = **0** |
| H4 | `col14 == −Σₖ rₖ·log(rₖ + 1e−5)` (the literal entropy expression, `1e-5` fudge and all) | max\|diff\| = **0**; allclose True |
| H4b | `col14.min() == −log(1 + 1e−5)` | observed `−9.999950000398841e−06`, predicted `−9.999950000398841e−06` — **bit-identical** |
| H5 | reconstruct the rating multiset from counts `col2..col6`, then compare | `col15 == median` **0**, `col16 == max` **0**, `col17 == min` **0**, `col18 == mean` **0** |

H4b is decisive on its own: the residual `1e-5` smoothing constant from
`cal_rating()` is visible in the published file's minimum value to full float precision.
These features were produced by exactly this code.

### 8.2 The 25 columns

Positions in the original 36-dim build are given because the mapping is
`[0..18] → [0..18]` and `[30..35] → [19..24]`.

| col | orig | Semantics | Confidence | Evidence |
|---|---|---|---|---|
| 0 | 0 | Number of rated products (== number of reviews by this user) | **certain** | H1 exact; range 1–483 int |
| 1 | 1 | Length of `reviewerName` in characters (0 if absent) | **high** | code; range 0–49 int, 0.3 % zeros |
| 2 | 2 | Count of 1★ ratings | **certain** | H1/H2 exact |
| 3 | 3 | Count of 2★ ratings | **certain** | H1/H2 exact |
| 4 | 4 | Count of 3★ ratings | **certain** | H1/H2 exact |
| 5 | 5 | Count of 4★ ratings | **certain** | H1/H2 exact |
| 6 | 6 | Count of 5★ ratings | **certain** | H1/H2 exact |
| 7 | 7 | Ratio of 1★ | **certain** | H2 exact |
| 8 | 8 | Ratio of 2★ | **certain** | H2 exact |
| 9 | 9 | Ratio of 3★ | **certain** | H2 exact |
| 10 | 10 | Ratio of 4★ | **certain** | H2 exact |
| 11 | 11 | Ratio of 5★ | **certain** | H2 exact |
| 12 | 12 | **Negative** ratio = ratio(1★) + ratio(2★) | **certain** | H3 exact |
| 13 | 13 | **Positive** ratio = ratio(4★) + ratio(5★) | **certain** | H3b exact |
| 14 | 14 | Rating entropy, `−Σ rₖ log(rₖ + 1e−5)` | **certain** | H4 + H4b bit-exact |
| 15 | 15 | **Median** rating | **certain** | H5 exact; uniques `{1, 1.5, …, 5}` |
| 16 | 16 | **Max** rating | **certain** | H5 exact; uniques `{1,2,3,4,5}` |
| 17 | 17 | **Min** rating | **certain** | H5 exact; uniques `{1,2,3,4,5}` |
| 18 | 18 | **Mean** rating | **certain** | H5 exact |
| 19 | **30** | **Min number of unhelpful votes** across the user's reviews | **medium** | code position (`cal_votes` fills cols 19–30; the 12th slot is `min(unhelp)`); range 0–201 int, 60.4 % zero; see §8.3 |
| 20 | 31 | **Day gap** — days between the user's first and last review | **high** | code; range 0–5525 int = 15.13 years, matching the Amazon dump's span |
| 21 | 32 | **Time entropy** over review **years** | **high** | code; range 0–2.44527, and ln(12) = 2.4849 bounds it |
| 22 | 33 | **Same-date indicator** (1 iff day gap == 0) | **high** | code; uniques exactly `{0, 1}`, mean 0.6427 vs `frac(col20 == 0) = 0.643` — **consistent** |
| 23 | 34 | Average feedback-**summary length** in characters | **high** | code; range 1–128, non-integer (it is a mean) |
| 24 | 35 | Review-text **sentiment sign** (VADER compound → −1 / 0 / +1) | **certain** | uniques exactly `{−1, 0, +1}`, which only the `sentiment()` sign function produces |

### 8.3 The 24-vs-25 discrepancy — hypothesis, plus a label-leakage finding

**The file contains 25 columns. That is a fact.** The reason the literature sometimes says
24 is **UNKNOWN**. But a specific, testable hypothesis falls out of the code:

`cal_votes()` writes **12** vote-derived columns into positions **19–30** of the 36-dim
build. The comment above the filter reads `# filter polluted features` — the vote columns
are "polluted" because the **label is itself defined from the helpful/unhelpful vote
ratio** (§3.1 E1). Dropping all 12 would require `features[:, :19]` + `features[:, 31:]`
= 19 + 5 = **24** columns. The shipped code instead writes `features[:, 30:]` = 19 + 6 =
**25**, retaining position 30 — the last vote column, `min(unhelpful votes)`.

An off-by-one that leaves one leaky column behind would explain both numbers: **24** =
the intended clean count, **25** = what was actually released.

Supporting evidence that column 19 really is vote-derived and really is leaky:

| | n | mean of col19 (min unhelpful votes) | mean of col0 (n reviews) |
|---|---|---|---|
| label 0 (benign) | 11,123 | **1.2053** | 3.4596 |
| label 1 (fraud) | 821 | **29.6151** | 1.5189 |

A **24.6×** separation on a single raw feature, in exactly the direction the labelling
rule (`helpful/votes < 0.2 → 1`) would produce. **This is label leakage present in the
canonical Amazon benchmark**, and it is worth flagging to whoever consumes
`data/benchmark/native_amazon/graph.pt`. I did not act on it — research-only scope.

**Status: the 24-vs-25 question is NOT resolved by assumption.** The file has 25. The
off-by-one above is a hypothesis with strong circumstantial support, not a proof.

### 8.4 Amazon joinability

| Probe | Result |
|---|---|
| Distinct feature rows | 10,616 / 11,944 → **1,328 duplicate rows** |
| Collision groups | 776, covering 2,104 nodes; largest group = 11 nodes |
| Collision groups with disagreeing labels | **0** |
| Collision groups entirely inside `[0, 3305)` | 257 |
| Distinct on rating-count cols 2–6 alone | 1,025 |
| Distinct on (col20 day-gap, col23 summary-len, col24 sentiment) | 4,601 |

So ~17.6 % of Amazon nodes are **not uniquely identified by their own features** — an
exact 1-to-1 join is impossible for those. But the join route is otherwise real and
concrete: `amazon_preprocess.py` names its input, `reviews_Musical_Instruments.json.gz`
(the Amazon product-review dump), and all 25 features are exactly reproducible per
`reviewerID` from it. Recomputing the 25-dim vector for every reviewer in that dump and
matching would recover node → `reviewerID` for the ~82 % with unique fingerprints, and to
within a small ambiguity set for the rest.

**Caveat:** `amazon_preprocess.py` draws its unlabeled users with
`rd.sample(unlabel_user, int(len(unlabel_user)*0.01))` under `rd.seed(1)`, but the sampled
set depends on dict/set iteration order, so the prefix of 3,305 nodes is **not** reliably
reproducible. That does not block feature-based matching. Note also that
`3305 / 0.01 ≈ 330,500` implies an unlabeled pool of roughly that size, versus
`821 + 7818 = 8,639` labeled users — recorded as an observation, not a verified count.

---

## 9. Summary of what is established vs UNKNOWN

**Established by execution:**
- Complete key inventory for both files (6 variables each; nothing else).
- Feature dims: YelpChi **32**, Amazon **25**. No NaN/Inf, no constant or all-zero columns.
- Label: 1 = fraud/spam, with direct source evidence for Amazon and code-level (not
  labelling-code) evidence for YelpChi.
- All 8 relation matrices: symmetric, binary (all 1.0), zero self-loops; exact nnz and
  undirected-edge counts.
- YelpChi `homo` **is** exactly the union. Amazon `homo` **is not**, and no relation is
  even a subset of it.
- **No node → review/reviewer identifier exists in either file.**
- All 25 Amazon feature columns decoded and arithmetically proven against upstream code.
- YelpChi features are exactly quantised by 1/67395, 1/38063, 1/201 in three column blocks.

**UNKNOWN / not determined:**
- Why Amazon `homo` differs from the union of its own relations.
- The precise semantic definitions of `net_rtr` and `net_rsr` (no YelpChi construction
  code exists locally; the structural test did not confirm the textbook definitions).
- The semantics of the individual 32 YelpChi feature columns.
- Whether 67,395 / 38,063 / 201 are the YelpChi review / reviewer / product counts.
- The origin of the 24-vs-25 Amazon feature-count discrepancy in the literature (§8.3
  gives a hypothesis with supporting evidence, not a resolution).
- YelpChi's labelling rule (no Yelp preprocessing script in this repo).

---

## 10. Reproduction

All scripts were run with:

```bash
PYTHONIOENCODING=utf-8 "/c/Users/hp/miniforge3/envs/dgp-bl-consisgad/python.exe" <script>.py
```

from the session scratchpad
`C:\Users\hp\AppData\Local\Temp\claude\d--Campus-Courses-Sem-4-Major-Project-Codes-FLAG-Reproduce\1c76c6b7-ce98-4218-b54a-bc143a8b6b15\scratchpad\`
(`inspect_mat.py`, `inspect2.py`, `inspect3.py`, `inspect4.py`, `inspect5.py`,
`inspect6.py`, with outputs `inspect_out.txt` … `inspect5_out.txt`). Scratch files are
session-scoped; the load-bearing snippets are inlined below so the results can be
regenerated from this document alone.

### 10.1 Key inventory, identifier hunt, labels, features, adjacencies, homo-vs-union

```python
import hashlib, pathlib
import numpy as np, scipy.io as sio, scipy.sparse as sp

p = pathlib.Path(r"D:\...\data\raw\yelpchi\YelpChi.mat")   # and .../amazon/Amazon.mat

# sha256
h = hashlib.sha256()
with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
print(h.hexdigest())

# complete top-level variable table (name, shape, MATLAB class)
for e in sio.whosmat(str(p)): print(e)

md = sio.loadmat(str(p))
for k, v in md.items():
    print(k, type(v).__name__,
          (v.format, v.dtype, v.shape, v.nnz) if sp.issparse(v)
          else (getattr(v, "dtype", None), getattr(v, "shape", None)))

# IDENTIFIER HUNT: nothing object / struct / char anywhere
susp = [k for k, v in md.items()
        if not k.startswith("__") and not sp.issparse(v)
        and (np.asarray(v).dtype == object
             or np.asarray(v).dtype.names is not None
             or np.asarray(v).dtype.kind in "USV")]
print("suspicious:", susp)          # -> []  for BOTH files

# labels
lf = np.asarray(md["label"]).ravel()
print(np.unique(lf, return_counts=True))

# features
D = np.asarray(md["features"].todense(), dtype=np.float64)
print(D.shape, D.min(), D.max(), D.mean(), np.isnan(D).any(), np.isinf(D).any())
print("constant cols:", [j for j in range(D.shape[1]) if np.unique(D[:, j]).size == 1])
print("distinct rows:", np.unique(D, axis=0).shape[0], "of", D.shape[0])

# adjacencies
for k in [c for c in md if sp.issparse(md[c]) and c != "features"]:
    A = md[k].tocsr()
    nsl = int((A.diagonal() != 0).sum())
    print(k, A.shape, A.nnz, "sym:", (A != A.T).nnz == 0,
          "selfloops:", nsl, "undirected:", (A.nnz - nsl) // 2,
          "uniq data:", np.unique(A.data))

# homo == union ?
rels = [c for c in md if sp.issparse(md[c]) and c not in ("features", "homo")]
U = None
for k in rels:
    B = (md[k].tocsr() != 0).astype(np.int8)
    U = B if U is None else (U + B)
U = (U != 0).astype(np.int8)
H = (md["homo"].tocsr() != 0).astype(np.int8)
print("union nnz", U.nnz, "homo nnz", H.nnz, "symdiff", (H != U).nnz)
```

### 10.2 Amazon feature-semantics proof

```python
F = np.asarray(sio.loadmat(amazon_path)["features"].todense(), dtype=np.float64)
counts, ratios, nprod = F[:, 2:7], F[:, 7:12], F[:, 0]

assert np.abs(nprod - counts.sum(1)).max() == 0                      # H1
assert np.abs(counts / nprod[:, None] - ratios).max() == 0           # H2
assert np.abs(F[:, 12] - (F[:, 7] + F[:, 8])).max() == 0             # H3
assert np.abs(F[:, 13] - (F[:, 10] + F[:, 11])).max() == 0           # H3b
ent = -(ratios * np.log(ratios + 1e-5)).sum(1)
assert np.abs(ent - F[:, 14]).max() == 0                             # H4
assert F[:, 14].min() == -np.log(1 + 1e-5)                           # H4b, bit-exact

levels = np.array([1., 2., 3., 4., 5.])
for i in range(F.shape[0]):                                          # H5
    r = np.repeat(levels, counts[i].astype(np.int64))
    assert (F[i, 15], F[i, 16], F[i, 17]) == (np.median(r), r.max(), r.min())
    assert abs(F[i, 18] - r.mean()) == 0
```

### 10.3 YelpChi quantisation-denominator test

```python
F = np.asarray(sio.loadmat(yelp_path)["features"].todense(), dtype=np.float64)
for name, cols, Dm in [("0-14", range(0, 15), 67395),
                       ("15-23", range(15, 24), 38063),
                       ("24-31", range(24, 32), 201)]:
    err = max(np.abs(np.unique(F[:, j]) * Dm - np.round(np.unique(F[:, j]) * Dm)).max()
              for j in cols)
    print(name, Dm, err)          # 7.28e-12 / 3.64e-12 / 2.84e-14
```

### 10.4 Relation-vs-feature structural check

```python
from scipy.sparse.csgraph import connected_components
A = sio.loadmat(yelp_path)["net_rur"].tocsr()
nc, comp = connected_components(A, directed=False)
sizes = np.bincount(comp)
groups = [np.where(comp == c)[0] for c in np.where(sizes >= 2)[0]]
sub = F[:, 15:24]
print(sum(1 for g in groups if np.unique(sub[g], axis=0).shape[0] == 1), "/", len(groups))
```

### 10.5 Upstream zip inventory

```python
import zipfile
for z in ["methods/care_gnn/data/YelpChi.zip", "methods/care_gnn/data/Amazon.zip"]:
    with zipfile.ZipFile(z) as zf:
        for i in zf.infolist():
            print(z, i.filename, i.file_size, "%08X" % i.CRC, i.date_time)
```

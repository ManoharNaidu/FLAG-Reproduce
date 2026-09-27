# Evidence Matrix — YelpChi / Amazon text integration

Consolidates Loop 1 (five parallel investigations) and Loop 2 (mapping
resolution). Every row is either backed by something executed on the actual
bytes, or marked UNKNOWN. Per-agent detail lives in `research/_evidence/`.

**Date:** 2026-09-18. **Status:** both mappings resolved; see §6 for the verdicts.

---

## 1. Sources acquired

| Artefact | Source | How obtained | Provenance |
|---|---|---|---|
| `YelpChi.mat` | CARE-GNN release | already local, sha256 in `data/benchmark/native_yelpchi/dataset_manifest.json` | VERIFIED |
| `Amazon.mat` | CARE-GNN release | already local, sha256 recorded | VERIFIED |
| Yelp raw metadata + text | Mukherjee et al. ICWSM 2013 corpus | mirror `github.com/zyni2001/Anomaly-detection-LLM` | **MIRROR** — authenticated by counts (see §3) |
| `reviews_Musical_Instruments.json.gz` (2014 full) | McAuley, via `cseweb.ucsd.edu/~jmcauley` -> `snap.stanford.edu` | direct download, valid TLS | AUTHORITATIVE |
| `Musical_Instruments_5.json.gz` (2018) | `mcauleylab.ucsd.edu` | downloaded, **not used** | wrong vintage (§4) |
| `reviews_Musical_Instruments_5.json.gz` (2014 5-core) | snap.stanford.edu | downloaded, **not used** | wrong subset (§4) |

**Access notes.** `jmcauley.ucsd.edu` is dead — TLS cert expired 2026-05-21
(`SEC_E_CERT_EXPIRED`). Worked around via McAuley's own live `cseweb` page,
which links to snap.stanford.edu. **Zero certificate bypasses were used.**
`odds.cs.stonybrook.edu` is unreachable (TLS alert 40, no certificate
presented). `shebuti.com/yelpchi-dataset/` is up but the download is
email-gated, hence the mirror.

## 2. The central structural fact

**Neither `.mat` file contains any node -> review/reviewer identifier.**
Exhaustively verified: each file has exactly 6 top-level variables, all sparse
or numeric — no struct, cell, char or object array anywhere, `__globals__`
empty, and the upstream zips contain only the `.mat`. So every text attachment
in this work is a *reconstruction with proof*, never a lookup.

| | YelpChi | Amazon |
|---|---|---|
| variables | `homo`, `net_rur`, `net_rtr`, `net_rsr`, `features`, `label` | `homo`, `net_upu`, `net_usu`, `net_uvu`, `features`, `label` |
| features | 45,954 x **32** float64 | 11,944 x **25** float64 |
| fraud label | **1** (6,677) | **1** (821) |
| `homo` == union of relations? | **YES, exactly** | **NO** — Jaccard 0.626 ⚠ |

⚠ **Correction required:** `data/benchmark/native_amazon/dataset_manifest.json`
describes `homo` as the "(union)" of the three relations. That is true for
YelpChi and **false for Amazon** (2,010,262 homo-only vs 2,048,630 union-only
entries; no relation is even a subset). Flagged, not silently edited.

## 3. YelpChi — chain of evidence

| Claim | Method | Result |
|---|---|---|
| Mirror is authentic | raw counts vs shebuti.com's published figures | 67,395 reviews / 38,063 users / 201 products / 8,919 spam — **all match** |
| Node subset rule | drop products with >800 reviews | **45,954 nodes exactly** |
| Label vector | per-row compare after `-1 -> 1` recode | **45,954/45,954 EXACT** |
| R-U-R | rebuild, compare **sparsity pattern** | 98,630 edges, **0 mismatched** |
| R-S-R | rebuild (same product + same rating) | 6,805,486 edges, **0 mismatched** |
| R-T-R | ~35 date definitions tried | **412,824 vs 1,147,232 — NOT REPRODUCED** |
| Text join | TF-IDF within-product vs same-block/diff-product | 0.0747 vs 0.0433 (**1.72x**); decays to control under shift |

**R-T-R note.** The only relation that fails is the only date-dependent one; the
two that succeed are both date-free. The mirror's `date` column evidently comes
from a different snapshot than the `.mat`'s source. This does **not** weaken the
ordering proof — two independent relations plus the label vector already pin it.

## 4. Amazon — chain of evidence

The generator is available locally (`methods/care_gnn/amazon_preprocess.py`),
which is what made this tractable.

| Claim | Method | Result |
|---|---|---|
| Correct source file | reviewer counts vs 11,944 | 2014 5-core=1,429; 2018 5-core=27,530; **2014 FULL=339,231** -> full file |
| Feature decoding | 25 columns vs raw arithmetic | **max\|diff\| = 0**, incl. bit-exact `-log(1+1e-5)` |
| Labelling rule | `votes>=20`, `>0.8` benign / `<0.2` fraud | fraud **821**, benign **7,818**, labeled **8,639** — all EXACT |
| Cohort arithmetic | `int(330,592 x 0.01) + 8,639` | **= 11,944 exactly** |
| Node ordering | `{**sampled, **labeled}` -> unlabeled prefix | prefix `[0,3305)` all-zero labels ✓ |
| Label sequence | labeled users in file-appearance order | **8,639/8,639 EXACT** (chance ~82.8%) |
| U-P-U adjacency | rebuild, compare induced subgraph | 294,764 edges, **0 mismatched** |
| Unlabeled prefix | 16-column fingerprint vs 330,592-user pool | 562 unique (17.0%), 2,642 ambiguous, 101 unmatched |

**Why the prefix is unrecoverable.** `rd.sample(list(all_user - label_user), …)`
draws from a list built from a **set difference**; its iteration order depends on
`PYTHONHASHSEED`, so the draw is not reproducible even with `rd.seed(1)`. The
pool's 330,592 users collapse to only 90,129 distinct fingerprints because most
have a single review. This is structural, not a shortfall of effort.

## 5. Conflicts and corrections logged

| # | Item | Status |
|---|---|---|
| C1 | Amazon `homo` described as "union" in our own manifest | **WRONG for Amazon** — needs correcting |
| C2 | Amazon feature dim 24 vs 25 | No source claiming 24 was found; our file has **25**. DGL does not slice. Rumour likely from SEFraud's table adjacency. **UNKNOWN, not resolved** |
| C3 | Amazon "11,123 benign" in our manifest | Should read **label==0 including 3,305 unlabeled**; true benign = 7,818 |
| C4 | Amazon fraud rate 6.9% (ours) vs 9.5% (literature) | Same numbers, different denominator — ours is all nodes, literature is labeled-only. **State the denominator** |
| C5 | Our relation edge counts are 2x CARE-GNN's published table | Directed vs undirected counting. Ours match DGL's docs exactly |
| C6 | DGL `FraudYelp/FraudAmazon` vs `.mat` | Tensors identical, node *i* = row *i*; but DGL injects an unpublished 70/10/20 split at seed 717. **Runs across loaders are not comparable** |
| C7 | YelpChi origin attributed to Rayana & Akoglu in my brief | Corpus is **Mukherjee et al. ICWSM 2013**; Rayana & Akoglu (KDD 2015) contributed the 32 features; the graph + 45,954 subset are CARE-GNN's |
| C8 | YelpChi 32-column semantics | Feature-name file exists, but **no column behaves like a monotone function of review length or rating**. Column semantics **UNKNOWN** |

## 6. Verdicts

| Dataset | Node type | Nodes | Text mapping | Status |
|---|---|---|---|---|
| YelpChi | review | 45,954 | node -> raw review, all rows | **EXACT** (ordering proven; text join statistically verified) |
| Amazon | user | 11,944 | 8,639 labeled -> reviewerID | **EXACT** for labeled block |
| Amazon | user | (of those) 3,305 unlabeled | — | **UNRESOLVED** — not guessed |

The Amazon `UNRESOLVED` band is the **unlabeled** prefix, which CARE-GNN's own
protocol excludes from train/val/test. Every node that participates in
evaluation has a proven identity.

## 7. What this means scientifically

**No published work has ever demonstrated this mapping.** Surveyed LLM/graph
fraud papers either rebuild a different graph from raw text (DGP,
arXiv:2507.21653 — Amazon *Video*, review nodes) or discard text entirely
(LGSPF, arXiv:2605.28524 — *"all unstructured attributes are discarded"*). The
single artefact claiming a YelpChi text mapping is an unpublished repo doing a
**purely positional** join that never loads the `.mat` to check; only 12 of
45,954 of its indices coincide with the proven ordering.

**FLAG itself reports zero YelpChi/Amazon numbers.** Its tables cover Huabei
(Table 3) and Reddit + Instagram (Tables 4-5) only, and the paper explicitly
says these datasets *"lack textual information"* — citing CARE-GNN, i.e. exactly
our `.mat` release. So a text-augmented YelpChi/Amazon run is **necessarily a
novel experiment**; there is no published FLAG number to reproduce or compare
against. It must never be presented as part of the FLAG reproduction.

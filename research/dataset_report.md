# Dataset Report

Status of the two text-augmented fraud datasets (brief section 47).
Full evidence: `research/evidence_matrix.md`. Decision: `research/decisions.md` D-005.

## Table 1 — What each dataset is

| Dataset | Node Type | Nodes | Features | Text | Relations |
|---|---|---:|---:|---|---|
| YelpChi | review | 45,954 | 32 | 45,954 / 45,954 (100%) | rur 98,630 / rtr 1,147,232 / rsr 6,805,486 |
| Amazon | user | 11,944 | 25 | 8,639 / 11,944 (72.3%) | upu 351,216 / usu 7,132,958 / uvu 2,073,474 |

Edge counts are **directed entries** (the `.mat` matrices are symmetric), i.e.
2x the undirected figures published in CARE-GNN's table. They match DGL's
documented numbers exactly.

Labels: 1 = fraud, unchanged from the canonical release.
YelpChi 39,277 / 6,677. Amazon 11,123 / 821 — but "11,123" is `label==0`
*including* 3,305 unlabelled nodes; true benign is 7,818, and the literature's
9.5% fraud rate is over labelled nodes only (821 / 8,639), not all nodes.

## Table 2 — Provenance and mapping

| Dataset | Graph Source | Text Source | Mapping | Status |
|---|---|---|---|---|
| YelpChi | CARE-GNN `YelpChi.mat` | Mukherjee et al. ICWSM 2013 corpus (mirror) | adjacency proof — R-U-R + R-S-R entry-for-entry, 0 mismatches | **VERIFIED** |
| Amazon (labelled) | CARE-GNN `Amazon.mat` | McAuley 2014 `reviews_Musical_Instruments.json.gz` (full, not 5-core) | generator order + induced U-P-U, 0 mismatches | **VERIFIED** |
| Amazon (unlabelled 3,305) | same | — | `rd.sample` over a set-difference list; not reproducible | **UNRESOLVED** |

## Not a reproduction

The FLAG paper reports **no** YelpChi or Amazon results and states both datasets
lack textual information. There is no published number to compare against, so
every run on these keys is a novel experiment. Registry keys are
`yelpchi_text` / `amazon_text`; the canonical `yelpchi` / `amazon` keys remain
blocked for all text variants.

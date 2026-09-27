# Validation report — amazon

Generated from the built dataset. Reproduce: `python -m scripts.validate_dataset --dataset amazon`

## Dataset statistics

- node type: **user**
- nodes: **11,944**
- features: **25**
- derived edge_index: 8,796,784 directed entries
- relations: `upu` 351,216, `usu` 7,132,958, `uvu` 2,073,474

## Label distribution

- {0: 11123, 1: 821}  (1 = fraud)
- unlabeled nodes: **3,305** (counted in class 0 above; true benign = 7,818)

## Split statistics

| split | n | per class |
|---|---:|---|
| train | 2,591 | {0: 2345, 1: 246} |
| val | 864 | {0: 782, 1: 82} |
| test | 5,184 | {0: 4691, 1: 493} |

## Mapping statistics

- text VERIFIED: **8,639 / 11,944** (72.3%)
- text UNRESOLVED: **3,305**
- evaluated nodes lacking text: **0** (enforced by validator + test)
- method: generator_order + upu_adjacency_proof

## Text statistics

- median length: 1,809 chars
- min / max length: 83 / 1,314,979 chars
- nodes sharing a text with another: 0 (informational)
- missing text (empty but VERIFIED): 0

## Validation checks

| status | check | detail |
|---|---|---|
| PASS | node count > 0 | 11,944 nodes |
| PASS | feature rows == num_nodes | (11944, 25) |
| PASS | labels == num_nodes |  |
| PASS | len(raw_texts) == num_nodes | 11,944 |
| PASS | len(text_status) == num_nodes |  |
| PASS | no NaN/Inf in features |  |
| PASS | labels in {0,1} | values=[0, 1], 1=fraud |
| PASS | edge_index in bounds | 8,796,784 edges |
| PASS | relation indices in bounds | upu=351,216, usu=7,132,958, uvu=2,073,474 |
| PASS | relation_names match relation_edges |  |
| PASS | no duplicate node in splits | 8,639 split assignments |
| PASS | no train/val/test leakage | 0 overlapping nodes |
| PASS | unlabeled nodes excluded from splits | 3,305 unlabeled, 0 leaked into a split |
| PASS | text status values valid | 8,639 VERIFIED / 3,305 UNRESOLVED |
| PASS | no VERIFIED node with empty text | 0 offending nodes |
| PASS | no UNRESOLVED node carrying text | 0 offending nodes |
| PASS | every train/val/test node has VERIFIED text | 0 evaluated nodes lack text |
| PASS | duplicate-text report | 3,305 nodes share a text with another (informational) |
| PASS | provenance recorded for every core field |  |

**RESULT: PASS** (19 checks)

## Caveat

This is a text-augmented **study** dataset. The FLAG paper reports no
amazon results and states the dataset lacks textual information, so no
number produced here has a published counterpart.

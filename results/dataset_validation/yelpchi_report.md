# Validation report — yelpchi

Generated from the built dataset. Reproduce: `python -m scripts.validate_dataset --dataset yelpchi`

## Dataset statistics

- node type: **review**
- nodes: **45,954**
- features: **32**
- derived edge_index: 7,693,958 directed entries
- relations: `rur` 98,630, `rtr` 1,147,232, `rsr` 6,805,486

## Label distribution

- {0: 39277, 1: 6677}  (1 = fraud)

## Split statistics

| split | n | per class |
|---|---:|---|
| train | 13,785 | {0: 11782, 1: 2003} |
| val | 4,596 | {0: 3928, 1: 668} |
| test | 27,573 | {0: 23567, 1: 4006} |

## Mapping statistics

- text VERIFIED: **45,954 / 45,954** (100.0%)
- text UNRESOLVED: **0**
- evaluated nodes lacking text: **0** (enforced by validator + test)
- method: adjacency_proof

## Text statistics

- median length: 569 chars
- min / max length: 7 / 4,996 chars
- nodes sharing a text with another: 25 (informational)
- missing text (empty but VERIFIED): 0

## Validation checks

| status | check | detail |
|---|---|---|
| PASS | node count > 0 | 45,954 nodes |
| PASS | feature rows == num_nodes | (45954, 32) |
| PASS | labels == num_nodes |  |
| PASS | len(raw_texts) == num_nodes | 45,954 |
| PASS | len(text_status) == num_nodes |  |
| PASS | no NaN/Inf in features |  |
| PASS | labels in {0,1} | values=[0, 1], 1=fraud |
| PASS | edge_index in bounds | 7,693,958 edges |
| PASS | relation indices in bounds | rur=98,630, rtr=1,147,232, rsr=6,805,486 |
| PASS | relation_names match relation_edges |  |
| PASS | no duplicate node in splits | 45,954 split assignments |
| PASS | no train/val/test leakage | 0 overlapping nodes |
| PASS | unlabeled nodes excluded from splits | 0 unlabeled, 0 leaked into a split |
| PASS | text status values valid | 45,954 VERIFIED / 0 UNRESOLVED |
| PASS | no VERIFIED node with empty text | 0 offending nodes |
| PASS | no UNRESOLVED node carrying text | 0 offending nodes |
| PASS | every train/val/test node has VERIFIED text | 0 evaluated nodes lack text |
| PASS | duplicate-text report | 13 nodes share a text with another (informational) |
| PASS | provenance recorded for every core field |  |

**RESULT: PASS** (19 checks)

## Caveat

This is a text-augmented **study** dataset. The FLAG paper reports no
yelpchi results and states the dataset lacks textual information, so no
number produced here has a published counterpart.

# Reproduction attempt: Figure 3(a) — semantic sampling raises subgraph homophily

**Status:** first genuine reproduction attempt in this project.
**Date:** 2026-09-09. **Device:** CPU.

## Why this claim first

Figure 3(a) is the paper's *motivation* study: it argues that semantic
similarity sampling (SS) produces subgraphs with higher edge homophily than no
sampling (NS), random sampling (RS), shallow-feature similarity (FS), or
semantic sampling without a threshold (SS\*).

It is the only substantive claim in the paper that can be checked **without
training any model and without the 9B LLM** — so it is reproducible on CPU today,
while everything involving `gemma-2-9b-it` is blocked on GPU access (decision
D-003). It is therefore the cheapest honest test of whether our reimplemented
sampler behaves like the authors'.

Quantity measured is the paper's Eq. 5, averaged over sampled subgraphs:

```
h = sum_{(u,v) in E} I( y(u) = y(v) ) / |E|
```

Settings are the paper's: 2 hops, top-10, threshold delta = 0, Sentence-BERT
(`all-MiniLM-L6-v2`) embeddings of raw node text, 2,000 test-node centres.

## Result

### On the **original** GLBench graph — **SUPPORTED**

| strategy | homophily | nodes/subgraph | edges/subgraph |
|---|---:|---:|---:|
| NS  (no sampling) | 0.5811 | 229.24 | 3571.74 |
| RS  (random) | 0.5888 | 48.12 | 318.64 |
| FS' (stored-feature similarity) | 0.5887 | 40.31 | 234.78 |
| SS\* (semantic, no threshold) | 0.5942 | 44.59 | 292.00 |
| **SS  (semantic + threshold)** | **0.5943** | 44.53 | 291.65 |

SS is highest, and the ordering **SS > SS\* > RS ≈ FS' > NS** is exactly the
pattern the paper reports. SS beats NS by **+0.0132**, RS by **+0.0055**, FS' by
**+0.0056**, SS\* by **+0.0001**.

### On our **1:10 benchmark** — **NOT SUPPORTED**

| strategy | Instagram | Reddit |
|---|---:|---:|
| NS  (no sampling) | **0.8659** | 0.6620 |
| RS  (random) | 0.8582 | **0.7014** |
| FS' (stored-feature sim) | 0.8599 | 0.6963 |
| SS\* (semantic, no threshold) | 0.8606 | 0.6944 |
| SS  (semantic + threshold) | 0.8606 | 0.6953 |

On Instagram SS beats RS and FS' but loses to NS. On Reddit SS beats NS
(+0.0332) and SS\*, but loses to RS (−0.0062). Neither reproduces the full
ordering.

## Diagnosis — two mechanisms, both measured

### 1. After 1:10 downsampling the graph is too sparse for top-10 to bind

| dataset | nodes | edges | avg degree | median degree | degree = 0 | **degree > 10** |
|---|---:|---:|---:|---:|---:|---:|
| instagram (benchmark) | 7,946 | 76,058 | 9.57 | 5 | 1,424 | **30.8%** |
| reddit (benchmark) | 18,389 | 42,824 | 2.33 | 2 | 3,108 | **1.2%** |

For a node with degree ≤ 10, "select the top 10 most similar neighbours" keeps
**all** of them. SS, RS, FS' and SS\* then make the *identical* selection. On
Reddit that is true for **98.8% of nodes**, so the comparison is measuring noise
from the remaining 1.2%. The strategies' near-identical Reddit scores
(0.6944–0.7014) are exactly what that predicts.

Downsampling is what causes it: Reddit's average degree falls from 8.03 on the
original graph to 2.33 on the benchmark, because discarding 90% of the minority
class also discards the edges attached to those nodes (302,876 → 42,824, −86%).

### 2. Class imbalance dominates the homophily number

With a 1:10 split, a *randomly wired* graph already has homophily

```
p0^2 + p1^2 = 0.909^2 + 0.091^2 = 0.835
```

| dataset (benchmark) | chance-level | actual full-graph |
|---|---:|---:|
| instagram | 0.8348 | 0.8648 |
| reddit | 0.8347 | **0.7510** |

Instagram's 0.865 is barely above chance, leaving almost no headroom for any
sampler to demonstrate an effect. Reddit's 0.751 is **below** chance — the
benchmark graph is genuinely *heterophilous* relative to its class distribution,
which is the classic fraud-detection setting (fraudsters attach to normal users)
and is presumably why FLAG targets fraud at all.

On the original Instagram graph (7,224 / 4,115) chance-level homophily is only
0.534, so the measured 0.58–0.59 sits well above it and differences between
strategies are meaningful.

## What this means

1. **Our sampler reimplementation behaves as the paper describes.** It
   reproduces the claimed ordering on the graph where the comparison is
   well-posed. That is meaningful evidence that Eq. 3–4 were implemented
   correctly, which matters because there is no upstream source to diff against
   (flag_code_audit.md GAP-1).
2. **The motivation study and the experiments are on different graphs.** Figure
   3(a) reproduces on the *original* graph; the models in Table 4 are trained on
   the *1:10 downsampled* graph, where — by our measurement — semantic sampling
   has little room to help and top-10 rarely binds at all. We cannot verify which
   graph the authors used for Figure 3(a); the paper does not say. Our evidence
   is that it reproduces on one and not the other.
3. **This is a hypothesis about the paper, not a refutation of it.** Confounds we
   have not eliminated:
   - our 1:10 downsampling uses **our** seed, since the authors' is unpublished;
     a different draw changes the induced graph;
   - the paper may define top-*N* as a total budget rather than per-hop
     (documented as UNKNOWN in `flagbench/sampling/semantic.py`);
   - the paper's FS uses word2vec, which GLBench does not ship, so our FS' is a
     different baseline and is labelled as such;
   - Figure 3(a) reports no numbers, only a plot, so we compare orderings and
     not values.

## Honest labelling of FS

The paper's **FS** uses *shallow* features, identified as word2vec. GLBench ships
no word2vec features; its stored `x` is **4096-dimensional**, which is Llama-2's
hidden size (`research/dataset_notes.md` section 7). Our "FS'" is therefore
similarity on the **stored 4096-d features**, not the paper's shallow-feature
baseline. It is renamed `FS'` everywhere rather than borrowing the paper's label
for a different quantity.

## Reproducing this

```bash
python -m scripts.download.glbench       --dataset all
python -m scripts.preprocess.build_benchmark --dataset all
python -m scripts.preprocess.encode_text --dataset all
python -m scripts.preprocess.encode_text --dataset all --source original

python -m scripts.preprocess.sample_subgraphs --dataset all --compare-strategies
python -m scripts.preprocess.sample_subgraphs --dataset all --compare-strategies --source original
```

Outputs: `results/tables/figure3a_homophily__benchmark.json` and
`…__original.json`.

## Outstanding

- **Reddit on the original graph**: encoding 33,434 long texts on CPU; result
  pending. Instagram-original is the only `original` measurement so far, so the
  "SUPPORTED" verdict currently rests on **one dataset**.
- Sensitivity of the verdict to the downsampling seed (re-run with
  `--downsample-seed 1..4`).
- Whether top-*N* per-hop vs total-budget changes the ordering.

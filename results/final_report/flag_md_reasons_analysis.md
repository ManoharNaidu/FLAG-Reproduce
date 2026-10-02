# FLAG-MD (Markov-diffusion sampler) vs cosine — why it did or didn't change results

[Back to final report](README.md) · See also: [flag_vs_flag_md.md](flag_vs_flag_md.md) (the paired-stats comparison this document explains)

Generated from existing repository artefacts only — **no new training was run.** Sources (read in full before writing this):

- [`research/flag_md.md`](../../research/flag_md.md)
- [`results/tables/flag_md_text.md`](../tables/flag_md_text.md), [`flag_md_flag.md`](../tables/flag_md_flag.md)
- `results/tables/sampler_analysis__{reddit,instagram}__*.json` (produced by [`scripts/analyze/compare_samplers.py`](../../scripts/analyze/compare_samplers.py) `analyze`)
- [`flag_vs_flag_md.md`](flag_vs_flag_md.md) — my own paired t-test re-analysis
- Backbone forward() code: `methods/flag/{models,geniepath,bwgnn,caregnn,dga,pmp}.py`

## Bottom line

**FLAG-MD does not reliably beat or lose to cosine sampling for any backbone on either dataset.** The repo's own research doc states this outright — `research/flag_md.md` line 3: *"No improvement over cosine sampling was observed."* My paired t-test (cosine vs `md_K2_matched`, matched per seed/init pair, exactly as `run_flag_md_matrix.sh`'s own design intends) found **1 of 28 tests significant at p<0.05 — chance level.** Everything below explains *why*, mechanistically.

## 1. Why the two samplers produce almost the same subgraphs

| | Reddit | Instagram |
|---|---:|---:|
| Nodes where cosine's top-10/threshold actually trims anything ("contested") | 2,994 / 18,389 = **16.3%** | 2,481 / 7,946 = **31.2%** |
| For every other node | cosine and MD select the **identical** set by construction (degree ≤ budget) | same |
| 1-hop Jaccard overlap, contested nodes only | 0.74 | 0.72 |
| Mean churned neighbours per centre (cosine-only + MD-only) vs mean subgraph size | 1.63 / ~9 (≈18%) | 15.5 / ~38 (≈42%) |
| Subgraph edge homophily, cosine → MD | 0.693 → 0.704 | 0.862 → 0.861 |
| Fraud-neighbour fraction of fraud centres, cosine → MD | 0.189 → 0.177 | 0.105 → 0.107 |

**Root cause — mathematical, not accidental.** `H = Z(K)X` (the Markov-diffusion embedding, `markov_diffusion.py:diffusion_embeddings()`) is a neighbourhood-*smoothed* version of the exact same Sentence-BERT matrix `X` that cosine compares directly. Over only K=2 hops, smoothing barely perturbs relative distances on a graph where the entire premise of semantic sampling is that nearby nodes already have similar text — so the two rankings agree on "who's relevant" most of the time and diverge only at the margin. That margin is small (16–31% of nodes) **by construction**, because `md_selection="matched_cosine"` always keeps exactly as many neighbours as cosine would — it only changes *which* ones, never *how many*.

Reddit's own degree distribution shrinks the margin further: only 1.2% of Reddit nodes have degree > 10, so cosine's real lever there is its delta=0 **threshold** (drops 10.4% of edges), not the top-10 ranking itself — and MD inherits that same filtering because it matches the budget cosine produced.

> **Counter-intuitive finding:** Instagram has 2.6x the structural churn (42% vs 18% of each subgraph swapped) but does *not* show larger or more consistent metric shifts than Reddit. That's a sign the downstream numbers are dominated by training noise, not a reproducible content effect.

## 2. Per-backbone reasons, grouped by aggregation mechanism

### Group A — content-agnostic structural aggregators (no internal re-weighting)
**GCN · BWGNN · DGA-GNN** aggregate by a plain sum/mean over whichever neighbours happen to be in the subgraph (`GCNConv`'s degree-normalised sum, `PolyConv`'s repeated `propagate()` over raw adjacency, `IntraConv`'s GraphSAGE-mean). No feature-dependent gate can compensate for a "wrong" neighbour — the outcome is driven purely by set overlap.

| Model | Reddit (text) Δ | Instagram (text) Δ | Reading |
|---|---|---|---|
| **gcn** | F1 +0.0025, AUC −0.0026 | F1 −0.0053, AUC +0.0030 | Sign-flips between datasets → noise, not a real effect. |
| **bwgnn** | F1 **+0.0210**, F1-macro **+0.0091**, AUC +0.0078, Recall **+0.0348** (largest Reddit gain in the whole table) | F1 **−0.0238**, F1-macro −0.0031, AUC −0.0047 (reverses) | Aggregates over *raw* (un-normalised) adjacency (audit: not a true beta-wavelet) → output scale is sensitive to subgraph *size*, which also shifts (Reddit 8.95→8.74, Instagram 38.7→36.4 nodes/subgraph). Most volatile backbone, in **both** directions. |
| **dga_gnn** | F1 +0.0124, Recall **+0.0536** (largest single recall gain, Reddit) | F1 +0.0144, Recall +0.0429 | Positive on Reddit `text`; **flips negative** under the full `flag` variant on Reddit (Recall −0.0302) — direction isn't even stable across variants of the *same* model/dataset. Strongest evidence of noise. |

### Group B — learned, feature-dependent re-weighting (can self-correct)
**GAT · CARE-GNN · PMP**: `GATConv`'s attention, `CAREGNNLayer`'s `sigmoid(MLP(x_i,x_j))` edge gate, and PMP's `balance_w` fraud/benign gate all let the model down-weight a less-useful neighbour regardless of which sampler put it there — giving these architectures some built-in insulation from the sampler swap.

| Model | Reddit (flag) Δ | Instagram (flag) Δ | Reading |
|---|---|---|---|
| **gat** | AUC −0.0024, F1-macro −0.0024 | AUC −0.0027, F1-macro −0.0030 | Small, consistent-sign but tiny. GAT's 2nd layer is a `SAGEConv`, not attention (audit finding) — only layer 1 gets the self-correction benefit. |
| **care_gnn** | AUC +0.0008, F1-macro −0.0010 | AUC −0.0021, F1-macro +0.0003 | Near-zero both ways — the learned gate appears to absorb most of the neighbour-set difference. |
| **pmp** | AUC −0.0073, F1-macro −0.0024 | AUC **+0.0022**, F1-macro **+0.0058** (`text`: AUC **+0.0070**, largest Instagram AUC gain) | The *one* model with a consistent-direction small positive trend — but only on Instagram, and only ~0.006–0.007, still inside typical run-to-run std (~0.01–0.02). |

### Group C — the outlier: GeniePath
The only backbone where the data shows something that looks like a real, structural effect rather than noise — and only on **Reddit**:

- `flag` variant: AUC 0.6144 → 0.6074 (Δ = −0.0070). This is the **one** paired comparison significant at p=0.016 in my t-test (still just 1-of-28 overall, i.e. chance-level across the whole study, but the single worst case).
- `text` variant: AUC 0.6073 ± 0.0081 → 0.5929 ± **0.0392**. The standard deviation nearly **quintuples**. MD didn't just shift GeniePath's mean on Reddit, it destabilised its run-to-run variance.
- On Instagram, GeniePath is untouched: 0.5687 → 0.5690.

**Why GeniePath specifically:** it's the only backbone with 4 message-passing layers (vs 2 everywhere else) plus an LSTM "Depth" component (`geniepath.py`) that processes the 4 layers as a pseudo-sequence. A changed neighbour at hop 1 (one of Reddit's 16.3% contested nodes) propagates and compounds through 4 rounds instead of 2, and an LSTM is more sensitive to compounding perturbations across steps than a 2-layer sum. It also **ignores `--hidden-dim`** and always runs at its own hardcoded 256-dim hidden state — a higher-capacity model on an 18K-node, 1:10-imbalanced graph is already the most training-variance-prone backbone in this suite, and MD's per-node neighbour churn is one more source of instability stacked on top of that.

## 3. The honest summary

**Mechanistically:** most of the graph cannot be affected at all (degree ≤ budget means cosine and MD pick the identical set). Where it *can* be affected, MD mostly agrees with cosine anyway (Jaccard 0.72–0.74), because both rankings are computed from the same underlying text embeddings — diffusion just smooths them.

**Empirically:** no backbone wins or loses consistently across both datasets and both variants (`text`, `flag`):
- BWGNN and DGA-GNN flip sign between Reddit/Instagram, or between the `text` and `flag` variants of the *same* model/dataset.
- PMP shows the closest thing to a real small positive trend, and only on Instagram.
- GeniePath shows the only concerning destabilisation, and only on Reddit.

**Repo's own conclusion** (stated before I re-derived it): `research/flag_md.md` line 3 — *"No improvement over cosine sampling was observed."* My paired t-test (1/28 significant, chance-level) and the mechanistic neighbourhood analysis above are independent confirmations of that conclusion, not a new finding.

## 4. Caveats

Carried over from `research/flag_md.md`'s own "Limitations and departures" section — not invented here:

1. Top-N almost never binds on the Reddit benchmark graph (1.2% of nodes have degree > 10; Instagram 30.8%). Cosine's effect there is mostly its delta=0 threshold. The matched-budget MD variant was chosen as primary specifically to isolate the ranking criterion from the budget size.
2. DGP Eq. 6 (k=0..K, 1/K) is used literally; a `walk_only` ablation (dropping k=0) is available and *does* change the ranking, but was not the primary comparison.
3. X is MiniLM raw-text embeddings (what FLAG's cosine already uses), not DGP's original DeBERTa + numeric features; there are no metapaths since FLAG's graph is homogeneous.
4. LLM text (the `flag` variant's discriminative text) covers only 0.6–12.9% of nodes; Instagram `flag` is almost entirely the raw-text fallback for both samplers, which limits how much the `flag`-variant comparison can say about the LLM-text pathway specifically.
5. 8 runs per cell (4 seeds × 2 inits) under-represent variance. The repo's own research doc explicitly claims no significance test; I added one (paired t-test) in a separate pass and it agrees with the "no effect" reading.
6. K=2 was fixed before results were seen; K in {1,3,5} and top-N selection were only run for the LLM-free `text` variant, not `flag`. `flag_finetuned` was never run under any sampler (needs a GPU-generated LLM cache not present on this machine).
7. The neighbourhood analysis's "no sampling" homophily reference used only the first 4,000 nodes, not the full graph.

## Source files

Nothing in this document was asserted without being traceable to one of these:

- [`research/flag_md.md`](../../research/flag_md.md)
- [`results/tables/flag_md_text.md`](../tables/flag_md_text.md), [`flag_md_flag.md`](../tables/flag_md_flag.md)
- [`sampler_analysis__reddit__*.json`](../tables/sampler_analysis__reddit__markov_diffusion_h2_k10_t0_perhop_K2_matched_cosine.json), [`sampler_analysis__instagram__*.json`](../tables/sampler_analysis__instagram__markov_diffusion_h2_k10_t0_perhop_K2_matched_cosine.json)
- [`flag_vs_flag_md.md`](flag_vs_flag_md.md), [`data/flag_vs_flag_md_paired_stats.csv`](data/flag_vs_flag_md_paired_stats.csv)
- `methods/flag/models.py` (GCN, GAT, DualGNN) · `geniepath.py` (GeniePathLazy) · `bwgnn.py` (BWGNN, PolyConv) · `caregnn.py` (CAREGNN, CAREGNNLayer) · `dga.py` (DGA, IntraConv) · `pmp.py` (LASAGE_S, LASAGESConv)
- [`src/flagbench/sampling/markov_diffusion.py`](../../src/flagbench/sampling/markov_diffusion.py), [`semantic.py`](../../src/flagbench/sampling/semantic.py)
- [`scripts/analyze/compare_samplers.py`](../../scripts/analyze/compare_samplers.py)

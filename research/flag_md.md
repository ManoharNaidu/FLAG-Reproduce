> **Superseded (2026-10-07):** this file covers the earlier 64-token / 4 seeds x 2 inits run (`results/flag_md/`). The current results are in [`results/2026-10-02-flag-cosine-vs-md-main-run-report.md`](../results/2026-10-02-flag-cosine-vs-md-main-run-report.md). Kept for history.

# FLAG-MD: Markov-diffusion neighbour sampling (isolated ablation of FLAG's sampler)

Status: experiments complete (2026-09-19). **No improvement over cosine sampling was observed.**

## What changed

Only the criterion that ranks a node's candidate neighbours in FLAG's semantic sampler.

| | FLAG (cosine) | FLAG-MD |
|---|---|---|
| ranking | `cos(B(t_w), B(t_u))`, high = relevant | `‖H_w − H_u‖₂`, low = relevant |
| candidates | 1-hop neighbours of the expanding node (frontier expansion, 2 hops) | identical |
| budget | threshold δ=0, then top-10 | `matched_cosine` (default): as many as cosine keeps for that node; or `top_n`: 10 smallest, no threshold |
| everything downstream (LLM, dual-branch, Skip-GNN, losses, optimiser, splits, metrics) | unchanged | unchanged |

## Mathematics and where it lives

`src/flagbench/sampling/markov_diffusion.py`

```
A      binary adjacency of the sampled graph (self-loops dropped, duplicate edges collapsed)   adjacency_to_csr    :85
T      = D^-1 A, zero-degree rows stay zero (no division by zero)                              transition_matrix   :111
Z(K)   = (1/K) * sum_{k=0..K} T^k        DGP Eq. 6 literal (k=0 kept; 1/K vs 1/(K+1) is       _operator_terms     :72
         a positive constant and provably does not change the ranking - tested)
X      L2-normalised Sentence-BERT (all-MiniLM-L6-v2) embeddings of the RAW node text:
         the same matrix the cosine sampler compares
H      = Z(K) X, computed as K sparse-dense products; Z(K) and T^k are never formed            diffusion_embeddings:118
delta  = ||H_w - H_u||_2, smaller = more relevant; ties broken by lower node id                diffusion_distances :133
```

`MarkovDiffusionNeighborSampler` (:141) implements `select(center, candidates)`, the same
interface as `semantic.CosineNeighborSampler`. Row w of H depends only on w's K-hop ball, so
computing H globally is exactly equal to running the diffusion locally per target.

Hooks in `sampling/semantic.py`: `SamplingConfig` fields (:97-107), `cache_key` (:129),
`CosineNeighborSampler` (:273), `make_sampler` (:290), `sample_subgraph(sampler=)` (:307),
`sample_all(sampler=)` (:389). Cosine keys, records and caches are unchanged; regenerating the
cosine subgraphs with the new code is tensor-identical to the existing caches.

## Commands

```bash
# subgraphs
python -m scripts.preprocess.sample_subgraphs --dataset all                              # FLAG (cosine)
python -m scripts.preprocess.sample_subgraphs --dataset all --strategy markov_diffusion --diffusion-steps 2

# one run
python -m scripts.train.run --dataset reddit --model gcn --variant flag --sampling-strategy semantic
python -m scripts.train.run --dataset reddit --model gcn --variant flag --sampling-strategy markov_diffusion \
    --diffusion-steps 2 --md-selection matched_cosine --results-dir results/flag_md/x

# Gemma text for MD subgraphs (sharded, resumable, reuses cosine text where the node list is identical)
bash scripts/setup/run_md_llm.sh
python -m scripts.preprocess.encode_llm_text --dataset all --kind both --strategy markov_diffusion --diffusion-steps 2

# full matrices (4 seeds x 2 inits, paired)
VARIANT=text bash scripts/reproduce/run_flag_md_matrix.sh
VARIANT=flag SAMPLERS="cosine md_K2_matched" bash scripts/reproduce/run_flag_md_matrix.sh
python -m scripts.analyze.summarize_flag_md --variant text|flag

# debug / sanity / neighbourhood analysis
python -m scripts.analyze.compare_samplers debug|sanity|analyze --dataset reddit --diffusion-steps 2
```

## Verification

144 unit tests pass (25 original sampler tests + 17 new). 11 sanity checks pass on both
datasets: dimensions; T row-stochastic; sparse H equals dense Z(K)X to 3e-17 for K in
{1,2,3,5} and 3 operators; no NaN/inf; no division by zero; selected nodes are in the candidate
set; budget respected; centre never selected; smallest (not largest) distances chosen for all
2,994 (Reddit) / 2,481 (Instagram) contested nodes; MD differs from cosine for 1,353 / 2,230
nodes; no label reaches the sampler.

## Results (4 seeds x 2 inits = 8 paired runs per cell, 7 backbones, mean over backbones)

`text` = LLM-free (same GNN, raw-text features, only the sampler differs).
`flag` = full pipeline (dual-branch, Gemma discriminative text with fallback).

| variant | dataset | sampler | F1 | F1-macro | AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| text | reddit | cosine | 0.1601 | 0.5320 | 0.6154 | 0.1481 | 0.1844 |
| text | reddit | MD K=2 | 0.1652 | 0.5320 | 0.6151 | 0.1463 | 0.2008 |
| text | instagram | cosine | 0.1715 | 0.5405 | 0.6010 | 0.1683 | 0.1913 |
| text | instagram | MD K=2 | 0.1692 | 0.5396 | 0.6007 | 0.1671 | 0.1873 |
| flag | reddit | cosine | 0.1657 | 0.5354 | 0.6243 | 0.1559 | 0.1910 |
| flag | reddit | MD K=2 | 0.1644 | 0.5346 | 0.6230 | 0.1533 | 0.1890 |
| flag | instagram | cosine | 0.1656 | 0.5372 | 0.5946 | 0.1626 | 0.1844 |
| flag | instagram | MD K=2 | 0.1713 | 0.5383 | 0.5949 | 0.1626 | 0.1973 |

Paired MD - cosine deltas are within +-0.006 on every metric, change sign across backbones, and
are much smaller than run-to-run std (0.01-0.03). Per-backbone tables: `results/tables/flag_md_{text,flag}.md`.

Diffusion depth (LLM-free, matched budget, mean over 7 backbones; cosine reference in brackets):

| K | Reddit F1 / F1-macro / AUC | Instagram F1 / F1-macro / AUC |
|---|---|---|
| ref | (0.1601 / 0.5320 / 0.6154) | (0.1715 / 0.5405 / 0.6010) |
| 1 | 0.1638 / 0.5324 / 0.6159 | 0.1726 / 0.5403 / 0.6009 |
| 2 | 0.1652 / 0.5320 / 0.6151 | 0.1692 / 0.5396 / 0.6007 |
| 3 | 0.1622 / 0.5323 / 0.6160 | 0.1678 / 0.5394 / 0.5975 |
| 5 | 0.1603 / 0.5308 / 0.6139 | 0.1687 / 0.5403 / 0.5980 |

Selection rule at K=2 (LLM-free): literal top-N without threshold scores F1 0.1631 (Reddit) and
0.1656 (Instagram), also indistinguishable from cosine.

## Runtime and memory

| | cosine | MD K=2 matched | MD K=2 top-N |
|---|---|---|---|
| H = Z(K)X precompute | - | 0.2 s | 0.2 s |
| sampling, whole graph (one-off) | 11.7 s | 21.1 s (includes evaluating cosine to size budgets) | 3.4 s |
| training / inference per run (CPU, `flag`) | 66.4 s / 26.1 s | 64.6 s / 26.0 s | - |
| GCN `flag` on GPU, peak memory | 66.1 MB (Reddit), 69.1 MB (Instagram) | 66.1 MB, 68.8 MB | - |
| Gemma regeneration | (existing) | 5,049 Reddit + 5,490 Instagram subgraphs per kind, ~8 s each, ~10.5 h wall-clock on 2xA100 (4 workers) | |

## Neighbourhood analysis (K=2, matched)

| | Reddit | Instagram |
|---|---|---|
| centres whose subgraph node set differs from cosine's | 27.5% | 69.0% |
| 1-hop Jaccard, contested nodes only | 0.74 | 0.72 |
| mean unique neighbours per centre (cosine-only / MD-only / shared) | 0.92 / 0.71 / 7.03 | 8.92 / 6.58 / 28.81 |
| mean hop distance of selected nodes | 1.746 vs 1.739 | 1.861 vs 1.851 |
| subgraph edge homophily, cosine -> MD | 0.693 -> 0.704 | 0.862 -> 0.861 |
| homophily on contested centres only, cosine -> MD | 0.678 -> 0.703 | 0.861 -> 0.859 |
| fraud-neighbour fraction of FRAUD centres, cosine -> MD | 0.189 -> 0.177 | 0.105 -> 0.107 |

Labels were used only to describe these neighbourhoods, after selection.

## Limitations and departures

1. Top-N almost never binds on the Reddit benchmark graph (1.2% of nodes have degree > 10; Instagram 30.8%). Cosine's
   effect there is mostly its delta=0 threshold (drops 10.4% of Reddit edges). Literal same-top-N MD without a threshold
   therefore keeps nearly everyone (~= no sampling); the matched budget was chosen as primary to isolate the ranking.
   By construction, nodes whose whole neighbourhood is kept get identical sets under both samplers.
2. DGP Eq. 6 (k=0..K, 1/K) is used literally; the task text left the lower limit open. `walk_only` (no k=0) is available
   as an ablation and does change the ranking.
3. X is MiniLM raw-text embeddings (what FLAG's cosine uses), not DGP's DeBERTa + numeric features; the graph is FLAG's
   homogeneous one, so there are no metapaths. Each step scores candidates against the *expanding* node, like cosine.
4. LLM text covers only 0.6-12.9% of nodes (decode budget 64 tokens / 300 chars, decision D-004). None of the 5,490
   regenerated Instagram discriminative subgraphs passed the format check, so Instagram `flag` is almost entirely the
   raw-text fallback for both samplers. Cosine text is reused only where the MD subgraph's node list is identical.
5. `seed` varies batch order (and init varies weights); the split masks are fixed. 8 runs per cell under-represent variance.
   No significance test is claimed. Absolute performance is near chance (F1 ~0.17, AUC ~0.6).
6. K = 2 was fixed before results were seen. K in {1,3,5} and top-N were run LLM-free only. `flag_finetuned` (+FLAG*) was not run.
   GPU peak memory was profiled on GCN only (one run per dataset/sampler); the matrices ran on CPU.
7. The neighbourhood analysis's "no sampling" homophily reference used only the first 4,000 nodes.

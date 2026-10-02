# FLAG vs FLAG-MD: main 5 × 5 run, full report

**Repository:** `FLAG-Reproduce` · **Run window:** 2026-10-01 02:45 UTC → 2026-10-02 06:15 UTC (all 3,500 runs complete)
**Hardware:** 4 × NVIDIA A100-SXM4-80GB, 256 logical CPUs (container quota ≈ 122 cores), 1 TB RAM (vast.ai instance)
**Status:** **complete**: 3,500 / 3,500 runs, 0 failed (see [§1](#1-progress)). Sections marked *auto-generated* are rebuilt from the result files with

```bash
.venv-gpu/bin/python -P -m scripts.analyze.build_main_run_report
```

Everything between `<!-- AUTO:… -->` markers is overwritten by that command; everything else is hand-written. Numbers quoted in
the hand-written text were checked against the final tables (2026-10-02 06:15 UTC, 100% complete); the tables are
authoritative.

---

## Contents

0. [Executive summary](#0-executive-summary)
1. [Progress](#1-progress)
2. [Goals](#2-goals)
3. [Starting point: what the repository held before this run](#3-starting-point)
4. [Decisions](#4-decisions)
5. [What was changed](#5-what-was-changed)
6. [Execution timeline](#6-execution-timeline)
7. [Stage results: sampling and LLM text](#7-sampling-and-llm-text)
8. [Results: F1-macro and AUC per cell](#8-results-per-cell)
9. [Cosine vs FLAG-MD](#9-cosine-vs-flag-md)
10. [FLAG variants vs baseline](#10-flag-vs-baseline)
11. [Did the bigger LLM budget matter?](#11-llm-budget)
12. [Compute, timing and scheduling](#12-compute-timing-and-scheduling)
13. [Progress dashboard](#13-progress-dashboard)
14. [Incidents and fixes](#14-incidents-and-fixes)
15. [Caveats and threats to validity](#15-caveats)
16. [Conclusions](#16-conclusions)
17. [Open items and next steps](#17-next-steps)
18. [How to reproduce](#18-how-to-reproduce)
19. [Appendix: file inventory](#19-file-inventory)

---

## 0. Executive summary

**What was run.** 5 variant·sampler groups (baseline; flag and flag_finetuned, each with the paper's cosine sampler and with
FLAG-MD) × 7 GNN backbones × 4 datasets (Reddit, Instagram, Amazon, YelpChi) × 5 seeds × 5 initialisations = **3,500 runs**,
after regenerating all LLM text at the paper's decode budget with batched vLLM on 4 A100s.

**Headline findings (final).**

1. **FLAG-MD vs cosine: equal on three datasets, clearly better on Amazon.** Across all 56 paired cells, FLAG-MD's average
   Δ F1-macro is **+0.005** (AUC +0.001). The gain is concentrated on **Amazon: Δ F1 +0.015**, with FLAG-MD ahead beyond the
   run-to-run spread in 9 of 14 cells and a paired Wilcoxon test significant (p < 0.05) in favour of FLAG-MD in 12 of 14.
   On Reddit, Instagram and YelpChi every cell is within noise (|Δ| below the larger std); Reddit is ±0.000 on average.
   FLAG-MD loses beyond noise **nowhere**.
2. **FLAG beats the baseline on the two native-text datasets.** Reddit: +0.030 to +0.034 F1 and +0.064 to +0.072 AUC averaged
   over backbones; Instagram: +0.024 to +0.029 F1, +0.053 to +0.061 AUC. All 7 backbones improve.
3. **On Amazon and YelpChi the picture is mixed and must not be read as "FLAG hurts".** Amazon F1 drops (−0.040 to −0.059)
   while AUC rises (+0.046 to +0.053); YelpChi is slightly positive on average but the strong spectral/linear backbones (BWGNN,
   GCN, GeniePath, PMP) score *below* baseline there. The baseline on these two datasets uses their engineered fraud features
   (on Amazon these very likely include helpful-vote statistics, the quantity the labels are derived from), while the FLAG
   variants replace node features with text embeddings. See §10 and §15.
4. **The 12× larger LLM text coverage barely moved results.** Reddit discriminative-text coverage went from 4.4% → 55.5% of
   nodes, but flag's mean F1 changed by only +0.001; a raw-text-only reference (previous run) is within ±0.006 of flag on both
   Reddit and Instagram. The LLM branch contributes little beyond the raw-text embedding in this reproduction (§11).
5. **flag_finetuned ≈ flag.** As designed in this repository (decision D-001: the LLM stays frozen; extra GNN epochs with the
   residual + orthogonality losses), the fine-tuned variant is within ±0.01 F1 of flag in 55 of 56 cells (largest gap: Amazon
   CARE-GNN −0.013), at 3–6× the compute.

**Engineering outcomes.** Batched vLLM generation (≈ 7 h for all 16 caches on 4 GPUs instead of an estimated 10–14
GPU-days); Amazon and YelpChi integrated as text-augmented datasets with a dense-graph sampling budget; a read-only progress
dashboard (v0.3.1) with a cosine-vs-MD comparison view; a wall-clock-aware ETA after the original estimate was found to be
3–6× too optimistic for flag_finetuned; and a parallel scheduler that cut the tail from ~17 h to ~6 h.

---

## 1. Progress

*Auto-generated.*

<!-- AUTO:progress BEGIN -->
_Generated 2026-10-02 06:16 UTC._

| Variant · sampler | Reddit | Instagram | Amazon | YelpChi | Total |
|---|---:|---:|---:|---:|---:|
| baseline | 175/175 | 175/175 | 175/175 | 175/175 | **700/700** |
| flag · cosine | 175/175 | 175/175 | 175/175 | 175/175 | **700/700** |
| flag · FLAG-MD | 175/175 | 175/175 | 175/175 | 175/175 | **700/700** |
| flag_finetuned · cosine | 175/175 | 175/175 | 175/175 | 175/175 | **700/700** |
| flag_finetuned · FLAG-MD | 175/175 | 175/175 | 175/175 | 175/175 | **700/700** |
| **All** | | | | | **3500/3500 (100.0%)** |

**Complete: every cell has 25/25 runs.**
<!-- AUTO:progress END -->

---

## 2. Goals

Stated by the user at the start of the session (2026-10-01):

- Run **FLAG, FLAG-finetuned, FLAG-MD and FLAG-MD-finetuned** with **all baseline backbones**, **5 seeds × 5 initialisations**.
- Datasets: **Reddit, Instagram, Amazon, YelpChi**.
- Understand how the discriminative / residual LLM text is produced (per node or per subgraph, per run or cached) and **cache
  it once, reuse it for every run**.
- Get results **in the least wall-clock time**, using all GPUs and CPUs.
- Primary scientific question (dashboard brief, v0.3): **compare the paper's cosine sampler against the Markov-diffusion
  sampler (FLAG-MD)** within the same variant, backbone and dataset.

---

## 3. Starting point

State of the repository when this session began (audit of README, `research/`, `results/`, caches):

| Item | State before this run |
|---|---|
| Upstream FLAG code | Audited earlier: 0/7 entrypoints run as shipped; semantic sampler has no source; LoRA path is a no-op; 4 of 5 bundled baselines are not their published algorithms (README §2). |
| Datasets built | Reddit, Instagram (GLBench, 1:10 downsampled). Amazon/YelpChi: manifests only, no `graph.pt` on this machine. |
| LLM text caches | Generated at a **reduced budget** (decision D-004: `max_new_tokens=64`, `truncate_chars=300`) with the HF one-prompt loop. **Node coverage 0.6–12.9%**: only 1–3-node subgraphs could emit one line per node in 64 tokens, so `flag` was mostly the raw-text fallback. |
| Previous results | `results/flag_md/`: flag and text variants, cosine vs 5 MD settings, **4 seeds × 2 inits**, Reddit + Instagram only. `flag_finetuned` results quoted in the README were **not on disk**. |
| Prompts | Upstream's verbatim prompts for Reddit and Instagram only. |

How the LLM text works (answered for the user, unchanged by this run): text is generated **per sampled subgraph** (one prompt
listing every node's text, one output line per node), separately for *discriminative* and *residual* kinds, and **cached
once** keyed by dataset, sampler config, prompt hashes and decode settings. Every backbone, seed, init and both FLAG variants
replay the same cache; only a different sampler (cosine vs MD) needs its own cache. FLAG-MD reuses the cosine text for every
subgraph whose node list is identical.

---

## 4. Decisions

| # | Decision | Chosen | Rationale / who |
|---|---|---|---|
| 1 | LLM engine | **vLLM**, batched, one process per GPU | User: speed. HF path was estimated at 10–14 GPU-days. |
| 2 | LLM decode budget | **Paper-faithful: 550 new tokens, 1,200 chars/node** | User. The 64/300 caches were near-useless (§3). |
| 3 | dtype | **bfloat16** (upstream used float16) | Forced: vLLM refuses float16 for Gemma-2 ("numerical instability"); bf16 is Gemma-2's training dtype. |
| 4 | Splits | **Fixed benchmark build**; *seed* = batch order, *init* = weight init | Recommended and accepted; a new split would require regenerating all LLM text. |
| 5 | Amazon | **CARE-GNN Amazon** (Musical Instruments) now; **Amazon Video later** | User: "both if possible". Amazon Video feasibility checked (§17), deferred to after the 4 main datasets. |
| 6 | Amazon/YelpChi sampling budget | **top_k = 3** per hop (≤ 13-node subgraphs) | User approved. Average degree 736 / 167 made top-10 subgraphs 56–75 nodes; 72–87% of prompts exceeded Gemma's 8k context. |
| 7 | Baseline sampler on Amazon/YelpChi | **random, same k** (paper's RS) instead of full 2-hop | User approved. Full 2-hop ≈ the whole graph per node (sampling estimated at 7.5 h, memory-prohibitive). |
| 8 | Prompts for Amazon/YelpChi | **Drafted here**, mirroring Reddit's structure; recorded in `prompts/manifest.json` | Upstream has none. |
| 9 | flag_finetuned semantics | **Unchanged: decision D-001** (LLM frozen; extra GNN epochs with residual + orthogonality losses) | Upstream LoRA gradient path is severed. Dashboard labels the stage "not tracked". |
| 10 | Training hyper-parameters | Code defaults, identical to the previous run (5 epochs, lr 0.01, hidden 32, dropout 0.5, accumulation 10) | Comparability with earlier results. |
| 11 | Where results live | `results/main/<variant>/<sampler>/`, plus `results/main_gpu/` for runs moved to GPU | Old 4 × 2 low-coverage runs kept untouched in `results/flag_md/`. |

All research-relevant decisions are recorded as **D-005** in `research/decisions.md`.

---

## 5. What was changed

### 5.1 Code (tracked files)

| File | Change |
|---|---|
| `src/flagbench/llm/enhance.py` | `LLMConfig.engine` ("hf" \| "vllm"), included in the cache key **only when not "hf"** so every older key is unchanged; `GenerationStats.context_overflows`; new `VLLMEnhancer` (same raw prompt string, greedy, per-prompt token budget clipped to the 8,192 context, prompts over the context counted as format failures, `parse_response` fed prompt + completion exactly as HF's decode); `make_enhancer()` caches one engine per process; spawn start method for vLLM workers. |
| `scripts/llm/generate_text.py` | `--engine`; `--dataset` accepts comma lists; context-overflow counts carried through shard merge; review noun for the new datasets. |
| `scripts/preprocess/encode_llm_text.py` | `--engine`, `--llm-dtype` (defaults from `PRODUCTION_LLM_CONFIG`); dataset-aware sampler defaults. |
| `src/flagbench/experiments/runner.py` | `PRODUCTION_LLM_CONFIG = 550 / 1200 / vllm / bfloat16` (D-005); `default_sampling_config()` resolves per-dataset top_k and baseline sampler. |
| `src/flagbench/registry/registry.py` | New datasets `amazon_text`, `yelpchi_text` (text-augmented study, `native_text: false` in manifests); `DatasetSpec.default_top_k`, `baseline_sampling_strategy`. |
| `src/flagbench/sampling/cli.py` | `config_from_args(..., dataset=)`: unset `--top-k` → dataset default; `none` → dataset's baseline sampler. |
| `scripts/preprocess/sample_subgraphs.py`, `encode_text.py` | New dataset choices; dataset-aware config. |
| `scripts/train/run.py` | `--top-k` default from dataset; per-dataset sampling config; **`--seed-list` / `--init-list`** to split a cell across processes. |
| `scripts/reproduce/run_flag_md_matrix.sh` | `default` sampler (variant's own); `-P` to avoid the repo's `datasets/` folder shadowing HF `datasets`. |
| `prompts/manifest.json`, `prompts/{amazon_text,yelpchi_text}/` | Drafted prompts + hashes. |
| `research/decisions.md` | D-005 (items 1–5). |
| `data/benchmark/*/dataset_manifest.json` | Rebuilt (only `built_at` changed). |

All 144 unit tests passed after the changes (`tests/unit`).

### 5.2 New files

| File | Purpose |
|---|---|
| `scripts/setup/run_vllm_llm.sh` | 4-GPU vLLM launcher: cosine → merge → FLAG-MD (reuse from cosine) → Sentence-BERT encode. |
| `scripts/reproduce/run_main_flag.sh` | flag + flag_finetuned × (cosine, FLAG-MD) matrices on CPU. |
| `scripts/analyze/build_main_run_report.py` | Generates this report's tables. |
| `logs/main/parallel/{plan.py, jobs.txt, procs.py, run_queue.sh, stop_gpu_originals.py, scheduler.py}` | The per-(seed, init) parallel queue and scheduler (§12). |
| `tools/flag-dashboard/` | Progress dashboard (§13). |
| `.venv-vllm/` | Separate venv for vLLM 0.30 (torch 2.13 cu130), so `.venv-gpu` stays untouched. |
| `.env` | `HF_TOKEN` (git-ignored). |

### 5.3 Data and caches built

- **Amazon / YelpChi sources** downloaded and SHA-256-verified against `datasets/manifests/` (CARE-GNN `.mat`, McAuley 2014
  Musical Instruments reviews, YelpChi review text). Alignment proofs re-run: YelpChi text join statistically supported
  (as before); Amazon labelled block **proven exact** (8,639 users), 3,305 unlabelled prefix nodes **unresolved** (empty text,
  excluded from every split).
- **FLAG payloads** `data/benchmark/flag_{amazon,yelpchi}_text/graph.pt` (19/19 validation checks each).
- **Sampling caches** at k = 3 for Amazon/YelpChi (semantic, random, markov_diffusion K2 matched).
- **16 LLM text caches** (4 datasets × disc/resid × cosine/FLAG-MD) + their Sentence-BERT encodings.
- **Gemma-2-9b-it** weights (18 GB) in `/workspace/.hf_home`.

---

## 6. Execution timeline

All times UTC.

| When | Event |
|---|---|
| 10-01 02:49 | vLLM venv installed; vLLM backend written. |
| 02:53–02:56 | Amazon/YelpChi sources downloaded + verified; payloads built. |
| 03:07 | k = 3 sampling caches for Amazon/YelpChi (k = 10 attempt showed 72–87% context overflow). |
| 03:10 | **Baseline matrix started** (CPU, 700 runs). |
| 03:16–03:26 | vLLM smoke test (fp16 refused → bf16; fork → spawn; GPU memory target 0.9 → 0.8). |
| 03:27 | **Full LLM generation started** on 4 GPUs. |
| 07:56 | Cosine caches merged (all 4 datasets). |
| 11:22 | FLAG-MD caches merged; 11:25 Sentence-BERT encoding done. |
| 11:38 | Baseline complete (700/700). |
| 12:04 | **FLAG training started** (the auto-start chain was ~40 min late; §14). |
| 16:08 | YelpChi GeniePath flag cells moved to GPUs, split per seed (each run ~46 min). |
| 20:12 | ETA found to be 3–6× too optimistic for flag_finetuned; true remaining ≈ 17 h. |
| 20:16 | Sequential jobs replaced by a **per-(seed, init) parallel queue** (52 CPU slots). |
| 22:00 | flag complete for every dataset; GPU originals stopped after their first flag_finetuned run. |
| 23:25 | Scheduler adds **16 GPU slots** (GPUs had been idle since 22:00). |
| 10-02 05:26 | Amazon and Reddit done; last YelpChi flag_finetuned GeniePath runs finishing (~7–8.5 h each). |
| **10-02 06:15** | **Queue finished: all 3,500 runs complete, 0 failed.** Report tables regenerated. |

---

## 7. Sampling and LLM text

**Sampling.** Reddit/Instagram keep the paper's setting (2 hops, top-10 per hop, threshold 0). Amazon/YelpChi use top-3
(≤ 13 nodes). FLAG-MD keeps the same number of neighbours as cosine for each node and changes only their ranking
(Markov diffusion sampler, adapted from DGP; `md_selection = matched_cosine`, K = 2).

**LLM text.** Gemma-2-9b-it, bf16, greedy, 550 new tokens, 1,200 characters per node, upstream prompts (Reddit, Instagram) or
the drafted ones (Amazon, YelpChi). A subgraph's text is kept only if the model returns exactly one line per node; otherwise its
nodes fall back to the raw-text embedding (upstream behaviour).

*Auto-generated.*

<!-- AUTO:coverage BEGIN -->
| Dataset | Kind | Sampler | top-k | Subgraphs | Succeeded | Node coverage | Context overflows | Reused from cosine | GPU time (s, max shard) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Reddit | discriminative | cosine | 10 | 18,389 | 14,015 (76.2%) | 55.5% | 226 | — | 1663 |
| Reddit | discriminative | FLAG-MD | 10 | 18,389 | 14,005 (76.2%) | 55.9% | 188 | 10908 | 919 |
| Reddit | residual | cosine | 10 | 18,389 | 15,525 (84.4%) | 72.5% | 231 | — | 1290 |
| Reddit | residual | FLAG-MD | 10 | 18,389 | 15,568 (84.7%) | 74.0% | 190 | 11511 | 742 |
| Instagram | discriminative | cosine | 10 | 7,946 | 3,367 (42.4%) | 9.1% | 0 | — | 1290 |
| Instagram | discriminative | FLAG-MD | 10 | 7,946 | 3,404 (42.8%) | 9.8% | 0 | 2365 | 1202 |
| Instagram | residual | cosine | 10 | 7,946 | 4,235 (53.3%) | 20.7% | 0 | — | 1222 |
| Instagram | residual | FLAG-MD | 10 | 7,946 | 4,278 (53.8%) | 21.8% | 0 | 2416 | 1155 |
| Amazon | discriminative | cosine | 3 | 11,944 | 11,644 (97.5%) | 97.5% | 0 | — | 1432 |
| Amazon | discriminative | FLAG-MD | 3 | 11,944 | 11,625 (97.3%) | 97.1% | 0 | 489 | 1387 |
| Amazon | residual | cosine | 3 | 11,944 | 11,829 (99.0%) | 99.1% | 0 | — | 1158 |
| Amazon | residual | FLAG-MD | 3 | 11,944 | 11,815 (98.9%) | 98.8% | 0 | 501 | 1112 |
| YelpChi | discriminative | cosine | 3 | 45,954 | 45,850 (99.8%) | 99.8% | 0 | — | 4704 |
| YelpChi | discriminative | FLAG-MD | 3 | 45,954 | 45,848 (99.8%) | 99.8% | 0 | 12677 | 3381 |
| YelpChi | residual | cosine | 3 | 45,954 | 45,006 (97.9%) | 98.0% | 0 | — | 3267 |
| YelpChi | residual | FLAG-MD | 3 | 45,954 | 44,902 (97.7%) | 97.9% | 0 | 12447 | 2328 |
<!-- AUTO:coverage END -->

**Reading.** Coverage is driven by subgraph size: ≤ 13-node subgraphs (Amazon, YelpChi) succeed 97–100% of the time;
Reddit (mean 9 nodes, long tail) 76–84%; Instagram (median 35 nodes) only 42–54% of subgraphs and **9–22% of nodes**: most
large subgraphs cannot produce 35–100 lines in 550 tokens, exactly as under upstream's own code. About 1.2% of Reddit prompts
exceeded the 8,192-token context. FLAG-MD's subgraph was identical to cosine's for 72.5% of Reddit nodes, 31% of Instagram,
28% of YelpChi and 4% of Amazon; those reused the cosine text, which roughly halved FLAG-MD's generation time on Reddit.

---

## 8. Results per cell

Mean ± sample std over the 25 runs (5 seeds × 5 inits) of the **test** metrics, threshold chosen on validation
(`validation_swept`). *Auto-generated.*

### 8.1 F1-macro

<!-- AUTO:results_f1 BEGIN -->
**Reddit** (F1-macro, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.527 ± 0.012 | 0.550 ± 0.009 | 0.552 ± 0.009 | 0.551 ± 0.011 | 0.551 ± 0.009 |
| care_gnn | 0.489 ± 0.022 | 0.531 ± 0.005 | 0.527 ± 0.005 | 0.521 ± 0.010 | 0.520 ± 0.006 |
| dga_gnn | 0.482 ± 0.012 | 0.536 ± 0.005 | 0.534 ± 0.006 | 0.534 ± 0.008 | 0.530 ± 0.006 |
| gat | 0.490 ± 0.012 | 0.546 ± 0.007 | 0.548 ± 0.007 | 0.544 ± 0.009 | 0.548 ± 0.008 |
| gcn | 0.517 ± 0.015 | 0.530 ± 0.004 | 0.527 ± 0.002 | 0.522 ± 0.006 | 0.525 ± 0.004 |
| geniepath | 0.522 ± 0.012 | 0.531 ± 0.007 | 0.531 ± 0.004 | 0.529 ± 0.007 | 0.530 ± 0.008 |
| pmp | 0.492 ± 0.019 | 0.532 ± 0.004 | 0.530 ± 0.004 | 0.528 ± 0.008 | 0.528 ± 0.006 |

**Instagram** (F1-macro, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.518 ± 0.013 | 0.526 ± 0.012 | 0.527 ± 0.008 | 0.527 ± 0.011 | 0.532 ± 0.010 |
| care_gnn | 0.493 ± 0.024 | 0.542 ± 0.006 | 0.543 ± 0.007 | 0.544 ± 0.006 | 0.545 ± 0.006 |
| dga_gnn | 0.506 ± 0.015 | 0.536 ± 0.010 | 0.539 ± 0.006 | 0.535 ± 0.013 | 0.542 ± 0.007 |
| gat | 0.516 ± 0.009 | 0.538 ± 0.010 | 0.542 ± 0.008 | 0.539 ± 0.006 | 0.545 ± 0.007 |
| gcn | 0.518 ± 0.012 | 0.534 ± 0.006 | 0.533 ± 0.007 | 0.538 ± 0.008 | 0.540 ± 0.007 |
| geniepath | 0.513 ± 0.014 | 0.526 ± 0.007 | 0.527 ± 0.004 | 0.525 ± 0.007 | 0.528 ± 0.008 |
| pmp | 0.505 ± 0.015 | 0.538 ± 0.007 | 0.539 ± 0.006 | 0.536 ± 0.012 | 0.540 ± 0.005 |

**Amazon** (F1-macro, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.909 ± 0.011 | 0.777 ± 0.009 | 0.791 ± 0.005 | 0.774 ± 0.010 | 0.785 ± 0.009 |
| care_gnn | 0.645 ± 0.164 | 0.714 ± 0.017 | 0.738 ± 0.007 | 0.701 ± 0.016 | 0.735 ± 0.009 |
| dga_gnn | 0.703 ± 0.110 | 0.762 ± 0.012 | 0.784 ± 0.009 | 0.760 ± 0.015 | 0.782 ± 0.007 |
| gat | 0.744 ± 0.043 | 0.747 ± 0.013 | 0.770 ± 0.009 | 0.748 ± 0.016 | 0.769 ± 0.008 |
| gcn | 0.912 ± 0.003 | 0.791 ± 0.006 | 0.798 ± 0.004 | 0.786 ± 0.008 | 0.789 ± 0.009 |
| geniepath | 0.912 ± 0.008 | 0.777 ± 0.016 | 0.786 ± 0.005 | 0.769 ± 0.014 | 0.783 ± 0.009 |
| pmp | 0.912 ± 0.004 | 0.784 ± 0.007 | 0.791 ± 0.006 | 0.786 ± 0.009 | 0.790 ± 0.009 |

**YelpChi** (F1-macro, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.649 ± 0.001 | 0.596 ± 0.004 | 0.596 ± 0.004 | 0.595 ± 0.003 | 0.598 ± 0.003 |
| care_gnn | 0.546 ± 0.013 | 0.588 ± 0.003 | 0.589 ± 0.003 | 0.586 ± 0.003 | 0.587 ± 0.004 |
| dga_gnn | 0.469 ± 0.030 | 0.579 ± 0.007 | 0.583 ± 0.011 | 0.581 ± 0.006 | 0.582 ± 0.008 |
| gat | 0.465 ± 0.011 | 0.581 ± 0.004 | 0.585 ± 0.004 | 0.582 ± 0.004 | 0.585 ± 0.004 |
| gcn | 0.649 ± 0.001 | 0.598 ± 0.004 | 0.598 ± 0.005 | 0.597 ± 0.004 | 0.599 ± 0.002 |
| geniepath | 0.648 ± 0.002 | 0.598 ± 0.004 | 0.597 ± 0.004 | 0.597 ± 0.003 | 0.597 ± 0.004 |
| pmp | 0.629 ± 0.015 | 0.592 ± 0.005 | 0.594 ± 0.004 | 0.592 ± 0.004 | 0.595 ± 0.003 |
<!-- AUTO:results_f1 END -->

### 8.2 AUC

<!-- AUTO:results_auc BEGIN -->
**Reddit** (AUC, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.597 ± 0.019 | 0.630 ± 0.014 | 0.630 ± 0.016 | 0.630 ± 0.020 | 0.630 ± 0.015 |
| care_gnn | 0.533 ± 0.069 | 0.621 ± 0.012 | 0.619 ± 0.011 | 0.604 ± 0.022 | 0.599 ± 0.018 |
| dga_gnn | 0.507 ± 0.019 | 0.622 ± 0.008 | 0.621 ± 0.011 | 0.621 ± 0.011 | 0.621 ± 0.009 |
| gat | 0.519 ± 0.025 | 0.648 ± 0.009 | 0.649 ± 0.012 | 0.644 ± 0.010 | 0.646 ± 0.011 |
| gcn | 0.583 ± 0.018 | 0.611 ± 0.006 | 0.608 ± 0.003 | 0.591 ± 0.014 | 0.598 ± 0.011 |
| geniepath | 0.592 ± 0.013 | 0.607 ± 0.013 | 0.602 ± 0.011 | 0.604 ± 0.018 | 0.605 ± 0.018 |
| pmp | 0.518 ± 0.028 | 0.619 ± 0.007 | 0.614 ± 0.010 | 0.605 ± 0.020 | 0.609 ± 0.014 |

**Instagram** (AUC, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.547 ± 0.016 | 0.570 ± 0.011 | 0.571 ± 0.011 | 0.570 ± 0.010 | 0.568 ± 0.012 |
| care_gnn | 0.516 ± 0.039 | 0.605 ± 0.006 | 0.604 ± 0.007 | 0.597 ± 0.015 | 0.591 ± 0.018 |
| dga_gnn | 0.517 ± 0.028 | 0.602 ± 0.009 | 0.606 ± 0.009 | 0.595 ± 0.029 | 0.599 ± 0.015 |
| gat | 0.529 ± 0.020 | 0.604 ± 0.012 | 0.610 ± 0.009 | 0.586 ± 0.015 | 0.585 ± 0.019 |
| gcn | 0.545 ± 0.021 | 0.593 ± 0.008 | 0.594 ± 0.008 | 0.594 ± 0.008 | 0.586 ± 0.013 |
| geniepath | 0.535 ± 0.028 | 0.568 ± 0.009 | 0.566 ± 0.009 | 0.567 ± 0.007 | 0.562 ± 0.010 |
| pmp | 0.523 ± 0.025 | 0.589 ± 0.012 | 0.588 ± 0.016 | 0.578 ± 0.018 | 0.594 ± 0.010 |

**Amazon** (AUC, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.927 ± 0.026 | 0.915 ± 0.003 | 0.915 ± 0.002 | 0.912 ± 0.005 | 0.913 ± 0.006 |
| care_gnn | 0.704 ± 0.195 | 0.850 ± 0.015 | 0.858 ± 0.011 | 0.832 ± 0.022 | 0.854 ± 0.013 |
| dga_gnn | 0.712 ± 0.102 | 0.906 ± 0.005 | 0.912 ± 0.004 | 0.905 ± 0.005 | 0.912 ± 0.004 |
| gat | 0.836 ± 0.042 | 0.878 ± 0.007 | 0.884 ± 0.007 | 0.876 ± 0.013 | 0.886 ± 0.005 |
| gcn | 0.916 ± 0.017 | 0.918 ± 0.002 | 0.916 ± 0.002 | 0.917 ± 0.003 | 0.914 ± 0.007 |
| geniepath | 0.920 ± 0.026 | 0.913 ± 0.007 | 0.916 ± 0.004 | 0.910 ± 0.007 | 0.911 ± 0.007 |
| pmp | 0.933 ± 0.012 | 0.916 ± 0.004 | 0.918 ± 0.002 | 0.917 ± 0.004 | 0.918 ± 0.008 |

**YelpChi** (AUC, mean ± std over runs; n shown when below 25)

| Backbone | baseline | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---:|---:|---:|---:|---:|
| bwgnn | 0.767 ± 0.002 | 0.691 ± 0.003 | 0.691 ± 0.003 | 0.691 ± 0.003 | 0.692 ± 0.002 |
| care_gnn | 0.615 ± 0.002 | 0.671 ± 0.004 | 0.671 ± 0.003 | 0.669 ± 0.002 | 0.670 ± 0.003 |
| dga_gnn | 0.514 ± 0.049 | 0.670 ± 0.009 | 0.674 ± 0.010 | 0.669 ± 0.007 | 0.670 ± 0.013 |
| gat | 0.513 ± 0.025 | 0.661 ± 0.006 | 0.667 ± 0.005 | 0.664 ± 0.004 | 0.666 ± 0.006 |
| gcn | 0.767 ± 0.002 | 0.694 ± 0.004 | 0.695 ± 0.005 | 0.694 ± 0.003 | 0.694 ± 0.003 |
| geniepath | 0.767 ± 0.002 | 0.692 ± 0.003 | 0.691 ± 0.005 | 0.692 ± 0.004 | 0.691 ± 0.003 |
| pmp | 0.755 ± 0.008 | 0.692 ± 0.004 | 0.693 ± 0.005 | 0.691 ± 0.004 | 0.693 ± 0.003 |
<!-- AUTO:results_auc END -->

---

## 9. Cosine vs FLAG-MD

Comparison only within the same variant, dataset and backbone. Runs are **paired**: the seed/init streams do not depend on the
sampler, so run (s, i) under cosine and under FLAG-MD share data order and weight initialisation. *Auto-generated.*

<!-- AUTO:compare BEGIN -->
Δ = FLAG-MD − cosine (positive = FLAG-MD better). W/L/T uses the dashboard rule: tie when |Δ| < the larger of the two stds. *sig+ / sig−* = cells where a paired Wilcoxon signed-rank test over the matched (seed, init) runs gives p < 0.05 in that direction (no multiple-comparison correction; read as indicative). Partial cells are excluded.

| Slice | pairs | Avg Δ F1 | F1 W/L/T | F1 sig+/sig− | Avg Δ AUC | AUC W/L/T | AUC sig+/sig− |
|---|---:|---:|---|---|---:|---|---|
| **All** | 56 | +0.005 | 9/0/47 | 21/3 | +0.001 | 2/0/54 | 12/6 |
| flag | 28 | +0.004 | 5/0/23 | 9/2 | +0.001 | 1/0/27 | 6/2 |
| flag_finetuned | 28 | +0.005 | 4/0/24 | 12/1 | +0.002 | 1/0/27 | 6/4 |
| Reddit | 14 | ±0.000 | 0/0/14 | 0/3 | ±0.000 | 0/0/14 | 0/1 |
| Instagram | 14 | +0.003 | 0/0/14 | 4/0 | ±0.000 | 0/0/14 | 2/2 |
| Amazon | 14 | +0.015 | 9/0/5 | 12/0 | +0.004 | 2/0/12 | 8/2 |
| YelpChi | 14 | +0.002 | 0/0/14 | 5/0 | +0.001 | 0/0/14 | 2/1 |
| bwgnn | 8 | +0.004 | 2/0/6 | 4/0 | ±0.000 | 0/0/8 | 0/0 |
| care_gnn | 8 | +0.007 | 2/0/6 | 2/1 | +0.002 | 0/0/8 | 2/0 |
| dga_gnn | 8 | +0.007 | 2/0/6 | 4/1 | +0.003 | 2/0/6 | 2/0 |
| gat | 8 | +0.008 | 2/0/6 | 5/0 | +0.004 | 0/0/8 | 4/0 |
| gcn | 8 | +0.002 | 1/0/7 | 1/1 | −0.001 | 0/0/8 | 0/4 |
| geniepath | 8 | +0.003 | 0/0/8 | 3/0 | −0.001 | 0/0/8 | 1/2 |
| pmp | 8 | +0.003 | 0/0/8 | 2/0 | +0.003 | 0/0/8 | 3/0 |

Complete pairs: **56 of 56**.

Per-cell detail (complete pairs; ✓ = paired p < 0.05):

| Variant | Dataset | Backbone | Δ F1 | p(F1) | Δ AUC | p(AUC) |
|---|---|---|---:|---:|---:|---:|
| flag | Amazon | bwgnn | +0.014 | 0.000 ✓ | ±0.000 | 0.474 |
| flag_finetuned | Amazon | bwgnn | +0.011 | 0.000 ✓ | +0.001 | 0.134 |
| flag | Amazon | care_gnn | +0.025 | 0.000 ✓ | +0.008 | 0.003 ✓ |
| flag_finetuned | Amazon | care_gnn | +0.034 | 0.000 ✓ | +0.022 | 0.001 ✓ |
| flag | Amazon | dga_gnn | +0.022 | 0.000 ✓ | +0.006 | 0.000 ✓ |
| flag_finetuned | Amazon | dga_gnn | +0.021 | 0.000 ✓ | +0.007 | 0.000 ✓ |
| flag | Amazon | gat | +0.022 | 0.000 ✓ | +0.006 | 0.003 ✓ |
| flag_finetuned | Amazon | gat | +0.021 | 0.000 ✓ | +0.009 | 0.001 ✓ |
| flag | Amazon | gcn | +0.007 | 0.000 ✓ | −0.002 | 0.001 ✓ |
| flag_finetuned | Amazon | gcn | +0.003 | 0.200 | −0.004 | 0.026 ✓ |
| flag | Amazon | geniepath | +0.009 | 0.039 ✓ | +0.002 | 0.004 ✓ |
| flag_finetuned | Amazon | geniepath | +0.014 | 0.000 ✓ | +0.001 | 0.182 |
| flag | Amazon | pmp | +0.007 | 0.001 ✓ | +0.002 | 0.059 |
| flag_finetuned | Amazon | pmp | +0.004 | 0.063 | +0.001 | 0.027 ✓ |
| flag | Instagram | bwgnn | +0.002 | 0.672 | +0.002 | 0.653 |
| flag_finetuned | Instagram | bwgnn | +0.005 | 0.010 ✓ | −0.002 | 0.411 |
| flag | Instagram | care_gnn | +0.001 | 0.458 | −0.001 | 0.085 |
| flag_finetuned | Instagram | care_gnn | ±0.000 | 0.958 | −0.006 | 0.220 |
| flag | Instagram | dga_gnn | +0.003 | 0.367 | +0.004 | 0.080 |
| flag_finetuned | Instagram | dga_gnn | +0.007 | 0.011 ✓ | +0.005 | 0.791 |
| flag | Instagram | gat | +0.004 | 0.101 | +0.006 | 0.039 ✓ |
| flag_finetuned | Instagram | gat | +0.006 | 0.045 ✓ | −0.001 | 0.895 |
| flag | Instagram | gcn | −0.001 | 0.634 | +0.001 | 0.692 |
| flag_finetuned | Instagram | gcn | +0.002 | 0.107 | −0.008 | 0.006 ✓ |
| flag | Instagram | geniepath | +0.001 | 0.653 | −0.002 | 0.325 |
| flag_finetuned | Instagram | geniepath | +0.003 | 0.009 ✓ | −0.005 | 0.001 ✓ |
| flag | Instagram | pmp | +0.002 | 0.525 | −0.001 | 0.812 |
| flag_finetuned | Instagram | pmp | +0.005 | 0.134 | +0.016 | 0.000 ✓ |
| flag | Reddit | bwgnn | +0.001 | 0.525 | ±0.000 | 0.791 |
| flag_finetuned | Reddit | bwgnn | ±0.000 | 0.596 | ±0.000 | 0.853 |
| flag | Reddit | care_gnn | −0.004 | 0.008 ✓ | −0.002 | 0.312 |
| flag_finetuned | Reddit | care_gnn | −0.001 | 0.751 | −0.005 | 0.731 |
| flag | Reddit | dga_gnn | −0.002 | 0.312 | −0.001 | 0.958 |
| flag_finetuned | Reddit | dga_gnn | −0.004 | 0.048 ✓ | ±0.000 | 0.895 |
| flag | Reddit | gat | +0.002 | 0.491 | +0.001 | 0.596 |
| flag_finetuned | Reddit | gat | +0.004 | 0.173 | +0.002 | 0.853 |
| flag | Reddit | gcn | −0.003 | 0.022 ✓ | −0.003 | 0.032 ✓ |
| flag_finetuned | Reddit | gcn | +0.003 | 0.052 | +0.007 | 0.120 |
| flag | Reddit | geniepath | −0.001 | 0.367 | −0.005 | 0.085 |
| flag_finetuned | Reddit | geniepath | +0.001 | 0.653 | ±0.000 | 0.692 |
| flag | Reddit | pmp | −0.001 | 0.200 | −0.004 | 0.107 |
| flag_finetuned | Reddit | pmp | ±0.000 | 0.442 | +0.004 | 0.751 |
| flag | YelpChi | bwgnn | ±0.000 | 0.731 | +0.001 | 0.458 |
| flag_finetuned | YelpChi | bwgnn | +0.003 | 0.017 ✓ | +0.001 | 0.252 |
| flag | YelpChi | care_gnn | +0.001 | 0.367 | ±0.000 | 0.895 |
| flag_finetuned | YelpChi | care_gnn | +0.001 | 0.164 | +0.001 | 0.325 |
| flag | YelpChi | dga_gnn | +0.004 | 0.020 ✓ | +0.005 | 0.120 |
| flag_finetuned | YelpChi | dga_gnn | +0.001 | 0.300 | +0.001 | 0.263 |
| flag | YelpChi | gat | +0.004 | 0.006 ✓ | +0.006 | 0.001 ✓ |
| flag_finetuned | YelpChi | gat | +0.003 | 0.003 ✓ | +0.002 | 0.191 |
| flag | YelpChi | gcn | +0.001 | 0.711 | +0.001 | 0.442 |
| flag_finetuned | YelpChi | gcn | +0.002 | 0.148 | ±0.000 | 0.771 |
| flag | YelpChi | geniepath | −0.001 | 0.491 | −0.001 | 0.252 |
| flag_finetuned | YelpChi | geniepath | ±0.000 | 0.916 | −0.001 | 0.048 ✓ |
| flag | YelpChi | pmp | +0.002 | 0.263 | +0.001 | 0.491 |
| flag_finetuned | YelpChi | pmp | +0.003 | 0.003 ✓ | +0.002 | 0.026 ✓ |
<!-- AUTO:compare END -->

**Interpretation.**

- **Amazon is the only dataset where the sampler matters.** FLAG-MD improves F1-macro by about +0.015 on average, beyond the
  run-to-run spread for most backbones (largest for CARE-GNN, DGA-GNN and GAT at +0.02 to +0.03), and paired tests agree.
  Amazon is the densest graph (average degree 736) and its nodes are users with long multi-review texts, so *which* 3
  neighbours are kept out of hundreds matters most there.
- **Reddit, Instagram, YelpChi: no practical difference.** Every cell is within noise. A handful of cells are "significant"
  under the paired test in either direction (e.g. Reddit F1 has 3 small significant losses, Instagram 4 small wins), which is
  what one expects from 56 uncorrected tests per metric with very tight run-to-run spreads; none exceed the noise rule.
- **Sign.** FLAG-MD never loses beyond noise; its average effect is non-negative on every dataset and both variants.

---

## 10. FLAG vs baseline

*Auto-generated.*

<!-- AUTO:vs_baseline BEGIN -->
Average over the 7 backbones of (variant mean − baseline mean), complete cells only.

| Dataset | metric | flag · cosine | flag · FLAG-MD | flag_finetuned · cosine | flag_finetuned · FLAG-MD |
|---|---|---:|---:|---:|---:|
| Reddit | F1-macro | +0.034 (7/7) | +0.033 (7/7) | +0.030 (7/7) | +0.030 (7/7) |
| Reddit | AUC | +0.072 (7/7) | +0.070 (7/7) | +0.064 (7/7) | +0.065 (7/7) |
| Instagram | F1-macro | +0.024 (7/7) | +0.026 (7/7) | +0.025 (7/7) | +0.029 (7/7) |
| Instagram | AUC | +0.060 (7/7) | +0.061 (7/7) | +0.054 (7/7) | +0.053 (7/7) |
| Amazon | F1-macro | −0.055 (7/7) | −0.040 (7/7) | −0.059 (7/7) | −0.044 (7/7) |
| Amazon | AUC | +0.050 (7/7) | +0.053 (7/7) | +0.046 (7/7) | +0.052 (7/7) |
| YelpChi | F1-macro | +0.011 (7/7) | +0.013 (7/7) | +0.011 (7/7) | +0.013 (7/7) |
| YelpChi | AUC | +0.010 (7/7) | +0.012 (7/7) | +0.010 (7/7) | +0.011 (7/7) |
<!-- AUTO:vs_baseline END -->

**Interpretation.**

- **Reddit and Instagram (native text):** every FLAG variant beats the baseline on every backbone, by about +0.03 F1 and
  +0.06–0.07 AUC. This is the setting the paper studies, and the direction agrees with it.
- **Amazon and YelpChi are not like-for-like and should be reported separately.** The baseline there trains on the datasets'
  **engineered features** (Amazon: 25 user features; YelpChi: 32 review features), while the FLAG variants replace node
  features with **text embeddings** (raw text + LLM text). On Amazon the labels are derived from helpful votes (≥ 20 votes;
  > 0.8 benign / < 0.2 fraud), and CARE-GNN's own feature generator computes helpful-vote statistics as user features (the
  exact composition of the `.mat`'s 25 columns is unverified), so the baseline's features very likely correlate with the label;
  BWGNN, GCN, GeniePath and PMP reach F1 ≈ 0.91 there and the FLAG variants ≈ 0.78–0.80. AUC nevertheless rises with FLAG on
  Amazon (ranking improves while thresholded F1 drops). On YelpChi the weaker backbones (CARE-GNN, DGA-GNN, GAT) gain
  substantially (+0.04 to +0.12 F1) and the spectral/linear ones lose ~0.05. A fair text-vs-features comparison on these two
  datasets would need a variant that **concatenates** engineered and text features; that was out of scope.
- **flag_finetuned vs flag:** within ±0.01 F1 in 55 of 56 cells (largest gaps: Amazon CARE-GNN −0.013, Reddit CARE-GNN
  −0.010); on Reddit/Instagram slightly lower AUC (−0.005 to −0.008 averaged over backbones). Consistent with D-001: without LLM fine-tuning, the extra GNN epochs add compute but little signal.

---

## 11. Did the bigger LLM budget matter?

*Auto-generated.*

<!-- AUTO:before_after BEGIN -->
flag variant, F1-macro averaged over the 7 backbones. *Previous* = results/flag_md/flag (64-token / 300-char LLM budget, 0.6–12.9% node coverage, 4 seeds × 2 inits). *This run* = results/main (550 / 1200 budget, vLLM, 5 × 5). *Raw text only* = the previous run's `text` variant (Sentence-BERT of the raw node text, same sampler, no LLM text, 4 × 2), shown as a reference for how much the LLM branch adds.

| Dataset | Sampler | Raw text only (prev.) | flag, previous | flag, this run | Δ (this − previous) |
|---|---|---:|---:|---:|---:|
| Reddit | cosine | 0.532 | 0.535 | 0.537 | +0.001 |
| Reddit | FLAG-MD | 0.532 | 0.535 | 0.536 | +0.001 |
| Instagram | cosine | 0.540 | 0.537 | 0.534 | −0.003 |
| Instagram | FLAG-MD | 0.540 | 0.538 | 0.536 | −0.002 |
<!-- AUTO:before_after END -->

**Reading.** Raising the LLM decode budget lifted Reddit's discriminative-text coverage from 4.4% to 55.5% of nodes and
Instagram's from 0.6% to 9.1%, yet flag's average F1 moved by ≤ 0.003. Against the raw-text-only reference, flag is +0.005
on Reddit and −0.006 on Instagram. In this reproduction the attention-fused LLM branch adds little beyond the Sentence-BERT
embedding of the raw text; the large gain over the baseline (§10) comes mostly from using text at all. Caveat: the raw-text
reference is from the previous 4 × 2 run, not re-run at 5 × 5 (§17).

---

## 12. Compute, timing and scheduling

### 12.1 LLM generation

16 caches, 4 GPUs, one vLLM engine per GPU (data parallel, ~44 GB KV cache each): **≈ 4.5 h for the cosine stage, ≈ 3.5 h
for FLAG-MD**, ≈ 7.9 h in total including load and merge. Throughput per GPU ≈ 5–7k prompt tokens/s and ≈ 500–680 generated
tokens/s (prefill-bound: long prompts, short answers).

### 12.2 GNN training

*Auto-generated (median wall-clock minutes per run).*

<!-- AUTO:timing BEGIN -->
Median wall-clock minutes per run (result timestamp → file written), over all devices.

| Dataset | Backbone | baseline | flag (cos/MD) | flag_finetuned (cos/MD) |
|---|---|---:|---:|---:|
| Reddit | bwgnn | 1 | 1 / 1 | 4 / 4 |
| Reddit | care_gnn | 1 | 1 / 1 | 4 / 4 |
| Reddit | dga_gnn | 1 | 1 / 1 | 3 / 3 |
| Reddit | gat | 1 | 1 / 1 | 4 / 4 |
| Reddit | gcn | 1 | 1 / 1 | 3 / 3 |
| Reddit | geniepath | 5 | 9 / 9 | 40 / 45 |
| Reddit | pmp | 1 | 1 / 1 | 4 / 4 |
| Instagram | bwgnn | 1 | 0 / 0 | 2 / 2 |
| Instagram | care_gnn | 16 | 1 / 1 | 2 / 2 |
| Instagram | dga_gnn | 6 | 0 / 0 | 1 / 1 |
| Instagram | gat | 2 | 1 / 1 | 2 / 2 |
| Instagram | gcn | 1 | 0 / 0 | 1 / 1 |
| Instagram | geniepath | 5 | 5 / 5 | 15 / 15 |
| Instagram | pmp | 3 | 1 / 1 | 2 / 1 |
| Amazon | bwgnn | 0 | 1 / 1 | 5 / 5 |
| Amazon | care_gnn | 0 | 1 / 1 | 5 / 5 |
| Amazon | dga_gnn | 0 | 1 / 1 | 4 / 4 |
| Amazon | gat | 0 | 1 / 1 | 6 / 6 |
| Amazon | gcn | 0 | 1 / 1 | 4 / 4 |
| Amazon | geniepath | 4 | 9 / 9 | 70 / 121 |
| Amazon | pmp | 0 | 1 / 1 | 5 / 5 |
| YelpChi | bwgnn | 2 | 5 / 5 | 30 / 29 |
| YelpChi | care_gnn | 2 | 4 / 5 | 29 / 29 |
| YelpChi | dga_gnn | 2 | 4 / 3 | 23 / 23 |
| YelpChi | gat | 2 | 5 / 5 | 31 / 31 |
| YelpChi | gcn | 2 | 4 / 4 | 22 / 23 |
| YelpChi | geniepath | 20 | 57 / 57 | 572 / 524 |
| YelpChi | pmp | 2 | 5 / 5 | 28 / 29 |
<!-- AUTO:timing END -->

**Bottlenecks and what was done about them.**

1. **GeniePath** is 5–20× slower per run than any other backbone (its LSTM depth module over many small subgraphs), and
   **flag_finetuned** runs take 3–6× longer than flag (30 extra GNN epochs). Their product, YelpChi × GeniePath ×
   flag_finetuned, takes **~7–8.5 h per run** on a busy CPU (5.75 h on GPU).
2. **GPUs do not speed up a GeniePath run** (YelpChi flag: 46 min on GPU vs 45 min on CPU). The only lever is running more
   runs at once.
3. Sequence of scheduling changes:
   - 03:10–16:08: one process per (dataset, backbone, sampler), 25 runs sequential, 28–56 in parallel on CPU.
   - 16:08: YelpChi GeniePath cells moved to GPU, split per seed (20 processes).
   - 20:16: **every outstanding (seed, init) as its own process** (`logs/main/parallel/`), longest first, 52 CPU slots;
     expected tail cut from ~17 h to ~5–6 h.
   - 23:25: scheduler adds 16 GPU slots (4/GPU, 1 thread each) once the GPUs went idle.
4. **No run was duplicated or lost:** the scheduler skips any run already saved or running; results in `results/main_gpu/`
   supersede `results/main/` for the same (seed, init), so the 9 early CPU results for YelpChi GeniePath flag are superseded by
   their GPU twins in every analysis and that cell is single-device.

---

## 13. Progress dashboard

`tools/flag-dashboard/`: read-only loopback service (`nice 19`, 20 s snapshot cache, ~22 MB), run by supervisor as
`flag-dashboard`, exposed through the instance portal as **"FLAG Dashboard"** (token auth; external port 10100).

| Version | Change |
|---|---|
| 0.1.0 | Status, runs by variant, GPUs, unfinished cells with ETA, failures, LLM coverage. |
| 0.2.0 | Redesign: overall bar, critical path, "Xh YYm", colour roles, two-column layout, grouped/filterable table, coverage heatmap. |
| 0.3.0 | Pipeline view (5 stage cards, cosine/MD lanes, limiting stage); **Results: cosine vs MD** comparison; server adds per-cell metrics, sampling status, GPU history (sparklines). |
| 0.3.1 | **ETA fix:** wall-clock per run instead of `training_time`; provisional lower bound for unfinished cells; parallel-aware overall ETA; requeued jobs counted as superseded, not failed. |

Known limitation: for runs already in flight the ETA counts a full run rather than the remaining part, so it over-estimates
near the end.

---

## 14. Incidents and fixes

| Incident | Effect | Fix |
|---|---|---|
| vLLM refuses float16 for Gemma-2 | Smoke test failed | bf16 (D-005 §5). |
| vLLM engine fork after CUDA init | Smoke test failed | `VLLM_WORKER_MULTIPROC_METHOD=spawn`. |
| 27 CPU baseline processes each held a 416 MB CUDA context on GPU 0 | vLLM could not reserve 90% | Memory target 0.8; later CPU jobs run with `CUDA_VISIBLE_DEVICES=""`. |
| Repo's `datasets/` folder shadowed HF `datasets` | `sentence_transformers` import error | `.pth` path entries + `python -P`; installed HF `datasets`. |
| Full 2-hop sampling on Amazon/YelpChi | ~7.5 h and near-whole-graph subgraphs | Stopped; random top-3 baseline sampler (decision 7). |
| Auto-start chain waited on a process-name match that my own watcher commands also matched | FLAG training started ~40 min late | Started manually; later process control uses `/proc` argv matching (`procs.py`). |
| `pkill -f <pattern>` twice matched the issuing shell | Two commands aborted (no experiment impact) | Same as above. |
| ETA used `training_time`, which excludes flag_finetuned's fine-tuning epochs | Reported "2h 39m left" when ~17 h remained | Dashboard 0.3.1 measures wall-clock; parallel queue. |
| GPUs idle 22:00–23:25 after the GPU originals finished | ~1.5 h of unused GPU capacity | Scheduler with GPU slots. |
| Stopped sequential jobs left `FAIL` lines | Dashboard briefly showed false "failing" | Superseded logic aware of the parallel queue. |

No experiment run failed for a code or data reason: every `FAIL` in the logs is a deliberately stopped job whose runs were
requeued.

---

## 15. Caveats

1. **Not a reproduction of the paper's numbers.** Downsampling seed unpublished; bundled baselines are FLAG's rewrites
   (README §2); `flag_finetuned` does not fine-tune the LLM (D-001).
2. **Amazon and YelpChi are a text-augmented study** (`native_text: false`); the paper reports neither. Their prompts were
   drafted here; their sampler budget (top-3) and baseline sampler (random) differ from Reddit/Instagram; their baseline uses
   engineered features (on Amazon very likely correlated with the vote-based label rule). Report them separately from
   Reddit/Instagram.
3. **Instagram's LLM coverage is low (9–22%)**, so Instagram's flag results mostly reflect the raw-text branch.
4. **vLLM vs HF generation:** same prompt, greedy decoding and budget, but kernel numerics can flip near-tied tokens; bf16
   instead of float16. The engine is in the cache key.
5. **One data split.** The 25 runs vary batch order and weight initialisation, not the train/val/test split; stds therefore
   understate split-to-split variance.
6. **Multiple comparisons.** The paired p-values in §9 are uncorrected; the noise rule (|Δ| vs larger std) is the primary
   criterion.
7. **Mixed devices:** most runs on CPU; YelpChi GeniePath flag runs on GPU, and some flag_finetuned runs on GPU after 23:25.
   CPU/GPU numerics differ slightly; each run's device is recorded in its JSON.
8. **Raw-text reference in §11** comes from the previous 4 × 2 run.

---

## 16. Conclusions

*Final (3,500 / 3,500 runs).*

1. **Swapping cosine for the Markov-diffusion sampler is safe and sometimes helps.** It never hurts beyond noise, is neutral on
   Reddit, Instagram and YelpChi, and gives a consistent, statistically supported gain on Amazon (+0.015 F1-macro), the
   densest graph with the richest per-node text. The effect is the same with and without flag_finetuned's extra epochs.
2. **FLAG's benefit on the native-text datasets is real but mostly comes from using text, not from the LLM branch.** FLAG
   beats the shallow-feature baseline on Reddit and Instagram for every backbone, but raising LLM coverage 12× changed almost
   nothing and a raw-text-only model is about as good.
3. **On engineered-feature fraud benchmarks (Amazon, YelpChi), text-only FLAG does not dominate the feature baseline.** It
   improves Amazon AUC and the weaker backbones on YelpChi, but loses F1 to strong backbones that use the engineered
   (label-correlated) features.
4. **flag_finetuned, as implementable from the released code, is not worth its 3–6× cost** in this setup.

---

## 17. Next steps

1. ~~Refresh this report when the queue finishes~~: done 2026-10-02 06:15 UTC; hand-written numbers re-checked against the
   final tables.
2. **Re-run the `text` (raw-text-only) variant at 5 × 5** on all four datasets: about 700 cheap runs, to put §11's
   "LLM branch adds little" on equal footing.
3. **Feature + text variant for Amazon/YelpChi** (concatenate engineered features with text embeddings) for a fair comparison.
4. **Amazon Video** (approved, deferred): CARE-GNN's generator on McAuley's Amazon Instant Video reviews gives 3,032
   labelled users (1,821 benign, 1,211 fraud, 40% fraud); plan was to downsample to 1:10. New self-constructed dataset,
   no published reference.
5. **Housekeeping:** back up `cache/llm/` off-box (git-ignored, the only copy); commit the code changes on a branch;
   rotate the Hugging Face token (it was pasted in chat).
6. Optional dashboard fix: subtract elapsed time for in-flight runs in the ETA.

---

## 18. How to reproduce

```bash
# environments
VENV=.venv-gpu bash scripts/setup/install_gpu.sh
uv venv .venv-vllm --python 3.12 && VIRTUAL_ENV=.venv-vllm uv pip install vllm sentence-transformers datasets torch_geometric scikit-learn pandas
# (add src/ and the repo root to both venvs via a .pth file; run python with -P)

# data (Amazon/YelpChi)
python -P experiments/yelpchi_amazon/build_native_benchmark.py --dataset yelpchi   # and amazon
python -P -m scripts.verify_yelpchi_alignment && python -P -m scripts.verify_amazon_alignment
python -P -m scripts.prepare_dataset --dataset all
python -P -m scripts.preprocess.encode_text --dataset amazon_text   # and yelpchi_text
python -P -m scripts.preprocess.sample_subgraphs --dataset amazon_text --strategy semantic          # k from registry
python -P -m scripts.preprocess.sample_subgraphs --dataset amazon_text --strategy markov_diffusion --diffusion-steps 2 --md-selection matched_cosine
python -P -m scripts.preprocess.sample_subgraphs --dataset amazon_text --strategy none               # → random on dense graphs

# LLM text (needs HF_TOKEN in .env)
DATASETS="reddit instagram amazon_text yelpchi_text" bash scripts/setup/run_vllm_llm.sh

# training
VARIANT=baseline SAMPLERS=default DATASETS="reddit instagram amazon_text yelpchi_text" SEEDS=5 INITS=5 \
  OUT=results/main/baseline bash scripts/reproduce/run_flag_md_matrix.sh
bash scripts/reproduce/run_main_flag.sh
# or, fastest: per-(seed, init) parallel scheduling
python3 logs/main/parallel/plan.py && python3 logs/main/parallel/scheduler.py --cpu 52 --gpus 4 --gpu-per-device 4

# report
.venv-gpu/bin/python -P -m scripts.analyze.build_main_run_report
```

---

## 19. File inventory

| Path | Contents |
|---|---|
| `results/main/{baseline,flag,flag_finetuned}/{default,cosine,md_K2_matched}/` | One JSON per run (3,500 runs incl. `results/main_gpu/`). |
| `results/main_gpu/` | YelpChi GeniePath runs executed on GPU (supersede CPU duplicates). |
| `results/flag_md/` | Previous 4 × 2, 64-token run (unchanged; used in §11). |
| `cache/llm/*vllm*` keys | 16 LLM text caches + manifests (git-ignored). |
| `cache/embeddings/` | Sentence-BERT encodings of raw and LLM text (git-ignored). |
| `cache/sampling/` | Subgraph caches. |
| `logs/llm/`, `logs/main/`, `logs/main/parallel/` | Generation, matrix, queue and scheduler logs. |
| `tools/flag-dashboard/` | Dashboard source, backups, screenshots, changelog. |
| `research/decisions.md` | D-001 … D-005. |
| `results/2026-10-02-flag-cosine-vs-md-main-run-report.md` | This report. |

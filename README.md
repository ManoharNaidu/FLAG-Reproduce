# FLAG Reproduction Benchmark

A reproducible research benchmark for **FLAG: Fraud Detection with LLM-enhanced
Graph Neural Network** (KDD 2025), plus the seven GNN baselines it compares
against, on a shared dataset/split/metric protocol.

> ## Status: pipeline runs end to end; `flag` and `flag_finetuned` now have real GPU results
>
> The research audit is complete (**36/36 findings backed by passing tests**),
> both datasets are downloaded and verified, and `google/gemma-2-9b-it` has been
> run on a rented vast.ai GPU (decision D-003) to produce **real `+FLAG` and
> `+FLAG*` numbers** for 4 of 7 backbones (GAT, CARE-GNN, BWGNN, DGA-GNN) on
> both Reddit and Instagram — 208 `+FLAG` runs and 200 `+FLAG*` runs, no
> placeholders. GCN, GeniePath and PMP still only have `baseline`.
>
> **The `+text` variant is currently absent from the tracked results store.**
> `results/raw/` is git-ignored, so the `+text` numbers an earlier revision of
> this README quoted are not part of the committed record any more — only
> `baseline`, `flag` and `flag_finetuned` are in
> [`results/aggregated/results.csv`](results/aggregated/results.csv) today.
> Recorded as a gap to re-run, not silently dropped.
>
> **Nothing is claimed as "reproduced".** The paper's downsampling seed is
> unpublished, so exact agreement with Table 4 is not achievable in principle,
> and `+FLAG*` reproduces the *released code's* behaviour (GNN retrains, LLM
> frozen — decision D-001), not LLM fine-tuning.
> The ledger is [`research/reproduction_status.md`](research/reproduction_status.md).

---

## 1. Why this exists

The goal is not to run FLAG once. It is a unified benchmark that can:

- reproduce the published FLAG experiments as faithfully as the released artefacts allow,
- reproduce the seven baselines (GCN, GAT, GeniePath, CARE-GNN, BWGNN, DGA-GNN, PMP)
  **both** as FLAG ran them **and** as their own authors published them,
- run FLAG on multiple GNN backbones, zero-shot and fine-tuned,
- extend to YelpChi, Amazon, T-Finance, T-Social and Elliptic **without pretending
  they have text they do not have**,
- run on CPU-only and on GPU,
- and preserve provenance instead of silently rewriting research code.

## 2. What the audit found

These are the load-bearing findings. Each is backed by a test in
[`tests/integration/test_flag_upstream_claims.py`](tests/integration/test_flag_upstream_claims.py).

1. **The official FLAG repository does not run. 0 of 7 entrypoints execute.**
   Two fail on defects internal to the code, not on missing data:
   `test.py` and `test_dual.py` import `ECELoss` from `utils`, and **it does not
   exist**. Details: [`research/FLAG_ORIGINAL_STATUS.md`](research/FLAG_ORIGINAL_STATUS.md).

2. **The semantic-similarity sampler — a core contribution of the paper — has no
   source in the repository.** Every driver loads pre-built `*_sampler*.pt` files
   that nothing in the repo creates. It must be rebuilt from the paper's Eq. 3-4.

3. **LoRA fine-tuning is a no-op as shipped.** `model.generate()` is
   non-differentiable and the decode -> Sentence-BERT re-encode produces a fresh
   leaf tensor, so no gradient reaches the LoRA parameters. Yet the paper reports
   `+FLAG*` beating `+FLAG`. **We do not know what produced that column.**
   Decision [D-001](research/decisions.md) reproduces the released code as-is —
   LLM frozen, GNN inner loop retrains under the full three-term loss — and
   `flag_finetuned` results are now real GPU runs under that reading, each
   stamped `llm_finetuned: false`. Contacting the authors remains open.

4. **Four of the five bundled baselines are not their published algorithms.**
   FLAG ships its own PyG rewrites. CARE-GNN has no RL neighbour filtering and no
   multi-relation support; BWGNN evaluates its polynomial over **raw adjacency**
   instead of the normalised Laplacian, so it is not a beta wavelet; DGA-GNN has no
   dynamic grouping; GAT's second layer is a `SAGEConv`. This is why every result
   carries an `impl_source` of `flag_bundled` or `official`, never merged.

5. **The paper and the code disagree** on hidden size (64 vs 32), run count
   (25 vs 5), GeniePath depth (2 vs 4 layers), early stopping (described vs not
   implemented), the orthogonality loss (squared dot vs signed cosine — the code's
   form rewards *anti*-alignment, not orthogonality), and even the prompt text.
   Both sides of every disagreement are recorded and exposed as configuration.
   Nothing is silently chosen: [`research/paper_notes.md`](research/paper_notes.md).

6. **`dga.py` cannot aggregate sparse subgraphs.** Its `aggregate` override
   omits `dim_size`, so PyG cannot pass the node count and the result is
   truncated whenever a node has no incoming edge. Latent upstream, fatal here:
   20% of Reddit benchmark nodes are isolated after downsampling. Fixed at
   Level 3 in the adapter, with a test proving it changes nothing upstream could
   already compute. **Found by running the code, not reading it.**

7. **Several of the paper's baseline cells are the trivial classifier.**
   45.46 appears in four Reddit cells with std 0.01; the AUCs in those rows are
   at chance. Those numbers measure a class ratio, not a model
   ([`degenerate_baselines.md`](research/degenerate_baselines.md)).

8. **A torch build on this machine returns silently wrong numbers.**
   `torch==2.4.0+cpu` produced both crashes and an incorrect scatter result at 2
   threads. Pinned to `2.3.1+cpu` after bisection:
   [`research/compatibility_notes.md`](research/compatibility_notes.md).

## 3. Supported methods

| Model | As FLAG ran it | Official source | Fidelity of FLAG's version |
|---|---|---|---|
| GCN | `models.py:GCN` | PyG `GCNConv` | faithful |
| GAT | `models.py:GAT` | PyG `GATConv` | **2nd layer is SAGEConv** |
| GeniePath | `geniepath.py` | PyG example (**no author repo exists**) | closest to faithful |
| CARE-GNN | `caregnn.py` | `YingtongDou/CARE-GNN` | **not CARE-GNN** |
| BWGNN | `bwgnn.py` | `squareRoot3/Rethinking-Anomaly-Detection` | **wrong operator basis** |
| DGA-GNN | `dga.py` | `AtwoodDuan/DGA-GNN` | **not DGA-GNN** |
| PMP | `pmp.py:LASAGE_S` | `Xtra-Computing/PMP` | partial |

Variants: `baseline` (shallow features), `text` (raw-text embeddings),
`flag` (zero-shot), `flag_finetuned` (after LoRA).

## 4. Supported datasets

| Dataset | Source | Native text | Canonical FLAG |
|---|---|---|---|
| **Reddit** | GLBench (33,434 nodes / 198,448 edges) | yes | **yes** |
| **Instagram** | GLBench (11,339 / 144,010) | yes | **yes** |
| YelpChi, Amazon, T-Finance, T-Social, Elliptic | DGL / official Drives | **no** | **blocked** |
| Huabei (industrial) | proprietary Alipay | yes | **unobtainable** |

The text-free datasets are **hard-blocked** from the FLAG variants. The paper itself
says they *"lack textual information"*, so fabricating text and calling the result
FLAG would be a research-integrity failure. A text-augmented study is permitted
only as a separate experiment type stamped `native_text: false`, never merged with
the canonical reproduction.

Provenance chain (verified): **FLAG -> GLBench -> GraphAdapter (WWW'24) ->
ConvoKit / Kim et al. (WWW'20)**. Full detail, including the wrong sources to avoid
(`torch_geometric.datasets.Reddit` is a *different* dataset with the same name):
[`research/dataset_notes.md`](research/dataset_notes.md).

## 5. CPU setup

CPU support here is real, not a flag. All 7 backbones are verified to
forward **and** backward on CPU.

```bash
python -m venv .venv-cpu                  # do NOT use --system-site-packages
.venv-cpu/Scripts/python -m pip install -r environment/cpu.lock.txt   # Windows
# .venv-cpu/bin/python -m pip install -r environment/cpu.lock.txt     # POSIX

bash scripts/setup/fetch_methods.sh       # clone upstream at pinned SHAs
```

Verify:

```bash
.venv-cpu/Scripts/python tests/unit/test_compat_torch_scatter.py          # 9/9
.venv-cpu/Scripts/python tests/integration/test_flag_upstream_claims.py   # 36/36
```

**Pins that matter** — do not casually upgrade these:

| Package | Pin | Reason |
|---|---|---|
| `torch` | `2.3.1+cpu` | `2.4.0+cpu` crashes **and returns wrong numbers** here |
| `torch_geometric` | `2.3.1` | `>=2.4` crashes in `SAGEConv`, which FLAG's GAT needs |
| `torch_scatter` | **not installed** | the wheel destabilises PyG; replaced by an equivalence-tested shim in `src/flagbench/compat/` |
| `transformers` | `4.44.2` | `>=4.42` required for Gemma-2 |

## 6. GPU setup

**Used, on a rented instance.** This development machine still has no CUDA, but
`google/gemma-2-9b-it` has been run on a rented **vast.ai** GPU (decision D-003)
to generate the LLM text cache the `flag`/`flag_finetuned` variants consume. Full
workflow, instance sizing, and cache-integrity checks:
[`docs/vastai_gpu_workflow.md`](docs/vastai_gpu_workflow.md).

The production cache was generated at a **reduced decode budget**
(`max_new_tokens=64, truncate_chars=300` instead of the paper-faithful
550/1200) to control cost — see decision D-004 in
[`research/decisions.md`](research/decisions.md).
Node coverage from the generated text is low (0.6%-12.4% depending on
dataset/kind), so most `flag`/`flag_finetuned` results are, for the majority of
subgraphs, the documented raw-text fallback rather than genuine LLM-generated
text — a materially weaker test of the method than Table 4's.

## 7. Quick start

```bash
# one-time setup
python -m venv .venv-cpu                                  # NOT --system-site-packages
.venv-cpu/Scripts/python -m pip install -r environment/cpu.lock.txt
.venv-cpu/Scripts/python -m pip install -e .
bash scripts/setup/fetch_methods.sh                       # upstream at pinned SHAs

# data pipeline (CPU, ~40 min, mostly the 734 MB download)
python -m scripts.download.glbench            --dataset all
python -m scripts.preprocess.build_benchmark  --dataset all
python -m scripts.preprocess.encode_text      --dataset all
python -m scripts.preprocess.sample_subgraphs --dataset all
python -m scripts.preprocess.sample_subgraphs --dataset all --strategy none

# experiments
python -m scripts.train.run --dataset reddit --model gcn --variant baseline
python -m scripts.train.run --dataset reddit,instagram --models all        --variants baseline,text --threshold-policy argmax
python -m analysis.compare_reported --metric f1_macro

# verify the environment at any time
python -m scripts.smoke_test
```

`--dry-run` validates the whole experiment matrix without training and prints
every refused combination with its reason.

## 8-10. FLAG reproduction, baselines, fine-tuning

Implemented and run. The sampler was reimplemented from Eq. 3-4 (no upstream
source), the seven bundled baselines run as `baseline`/`text`, and `flag` /
`flag_finetuned` have real GPU-produced numbers for GAT, CARE-GNN, BWGNN and
DGA-GNN on both datasets (§11). GCN, GeniePath and PMP still only have
`baseline` — not yet run under `flag`/`flag_finetuned`. Status ledger:
[`research/reproduction_status.md`](research/reproduction_status.md).

## 11. Results

Live tables: [`results/tables/`](results/tables/). Reference targets:
[`research/reported_results.csv`](research/reported_results.csv) — 199 rows from
the paper, used for comparison only. No result is ever adjusted toward them.

**Figure 3(a)** (semantic sampling raises subgraph homophily) — the one paper
claim testable without a GPU:

| graph | verdict |
|---|---|
| original GLBench Instagram | **SUPPORTED** — SS > SS\* > RS ≈ FS' > NS, the paper's ordering |
| our 1:10 benchmark | **NOT SUPPORTED** — and diagnosed |

After downsampling only 1.2% of Reddit nodes have degree > 10, so top-10
selection is a no-op for 98.8% of them and every strategy picks the same
neighbours. Full analysis: [`research/figure3a_reproduction.md`](research/figure3a_reproduction.md).

**Current tables**, all `impl_source=flag_bundled`:
[`results/tables/comparison.md`](results/tables/comparison.md) (ours only) and
[`results/tables/comparison_vs_reported_f1_macro.md`](results/tables/comparison_vs_reported_f1_macro.md) /
[`comparison_vs_reported_auc.md`](results/tables/comparison_vs_reported_auc.md)
(ours vs. the paper's Table 4).

`+FLAG` and `+FLAG*` selected rows (paper vs. ours, AUC, mean over 25+ runs):

| dataset | model | variant | reported AUC | ours AUC | delta | status |
|---|---|---|---:|---:|---:|---|
| reddit | gat | flag | 60.61 | 64.33 | +3.72 | DEVIATION |
| reddit | gat | flag_finetuned | 60.57 | 63.94 | +3.37 | DEVIATION |
| reddit | dga_gnn | flag_finetuned | 61.61 | 62.07 | +0.46 | MATCH |
| reddit | care_gnn | flag | 58.43 | 62.86 | +4.43 | DEVIATION |
| instagram | bwgnn | flag_finetuned | 57.19 | 57.54 | +0.35 | MATCH |
| instagram | gat | flag | 54.97 | 60.49 | +5.52 | DEVIATION |

Full 56-row comparison (28 `baseline`/`flag`/`flag_finetuned` cells x 2 metrics):
F1-macro — 1 MATCH, 1 CLOSE, 28 DEVIATION, 26 UNAVAILABLE (`text` and the
3 not-yet-run models). AUC — 5 MATCH, 6 CLOSE, 19 DEVIATION, 26 UNAVAILABLE.

Four things to read carefully before drawing conclusions:

- **Every available `+FLAG`/`+FLAG*` AUC is above the paper's**, same direction
  and similar magnitude as the `baseline` gap below — this looks like the same
  protocol difference (split, downsampling draw, sampler reimplementation), not
  a `+FLAG`-specific effect.
- **`flag_finetuned` here is not LLM fine-tuning.** Per decision D-001 it
  reproduces the released code's actual behaviour — LLM frozen, GNN inner loop
  retrains under the residual + orthogonality losses — and every row carries
  `llm_finetuned: false`.
- **The LLM text cache has low node coverage** (0.6%-12.4%, decision D-004), so
  most subgraphs behind these numbers use the raw-text fallback, not
  LLM-generated text. A materially weaker test of the method than Table 4's.
- **47.62 is the trivial classifier** on a 10:1 split, not a model score, and
  shows up repeatedly in `baseline`/`text` rows. The paper's repeated 45.46
  (four Reddit cells, std 0.01) and 47.29 (five Instagram cells) are almost
  certainly the same thing on their split.
  [`research/degenerate_baselines.md`](research/degenerate_baselines.md).
- **The threshold policy is worth 2–4 F1 points** and the paper states none.
  Every result row records which was used.

## 12. Known limitations

- **No local GPU** -> the LLM stage only runs on a rented vast.ai instance
  (D-003). Working, but not free, and not repeatable at zero cost.
- **`flag`/`flag_finetuned` are only run for 4 of 7 backbones** (GAT, CARE-GNN,
  BWGNN, DGA-GNN). GCN, GeniePath and PMP have not been run under either variant.
- **The LLM text cache has low node coverage** (0.6%-12.4%) because it was
  generated at a reduced decode budget to control cost (D-004), so most
  `flag`/`flag_finetuned` results reflect the raw-text fallback for most
  subgraphs, not genuine LLM-generated text.
- **The `+text` variant has no results in the current tracked aggregation.**
  `results/raw/` is git-ignored; whatever produced the old `+text` numbers is
  not part of the committed record. Needs re-running.
- **`flag_finetuned` is not LLM fine-tuning as reproduced here** (D-001) — the
  released code's LoRA gradient path is severed, so this variant reproduces
  "GNN retrains, LLM frozen," not what the paper's `+FLAG*` label implies.
- **Huabei (Table 3) is permanently unreproducible** — proprietary Alipay data.
- **The 1:10 downsampling seed is unpublished and unrecoverable**, so exact
  agreement with Table 4 is impossible in principle. We will not claim it.
- **Split ratios are absent from the paper**; 10/10/80 is inherited from
  GLBench/GraphAdapter and labelled as such.
- **Reference numbers are `TRANSCRIBED`**, not yet re-checked by a second pass.
- The paper's stated ablation deltas do not recompute from its own Tables 4-5
  ([`paper_notes.md` §6](research/paper_notes.md)). Unresolved.

## 13. Provenance

| Doc | Contents |
|---|---|
| [`research/paper_notes.md`](research/paper_notes.md) | Paper extracted; every `[PAPER]` vs `[CODE]` conflict |
| [`research/flag_code_audit.md`](research/flag_code_audit.md) | 11 defects, 7 gaps, all test-backed |
| [`research/FLAG_ORIGINAL_STATUS.md`](research/FLAG_ORIGINAL_STATUS.md) | Every entrypoint run, verbatim errors |
| [`research/repository_provenance.md`](research/repository_provenance.md) | 9 repos, SHAs, licences, known bugs |
| [`research/dataset_notes.md`](research/dataset_notes.md) | Provenance chain; wrong sources to avoid |
| [`research/compatibility_notes.md`](research/compatibility_notes.md) | Dependency bisection |
| [`research/implementation_matrix.md`](research/implementation_matrix.md) | What exists vs what must be built |
| [`research/reproduction_status.md`](research/reproduction_status.md) | The ledger + open questions |
| [`methods/README.md`](methods/README.md) | Imported-source provenance table |

**Licensing.** FLAG, BWGNN, DGA-GNN and PMP ship **no LICENSE file**, so no
redistribution rights are granted. `methods/` is git-ignored and rebuilt from
upstream at pinned SHAs, so this repository re-hosts nobody's code. Permission
must be sought before publishing any artefact that vendors them.

## 14. Citation

```bibtex
@inproceedings{yang2025flag,
  title     = {FLAG: Fraud Detection with LLM-enhanced Graph Neural Network},
  author    = {Yang, Chengdong and Liu, Hongrui and Wang, Daixin and
               Zhang, Zhiqiang and Yang, Cheng and Shi, Chuan},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining},
  pages     = {5150--5160},
  year      = {2025},
  doi       = {10.1145/3711896.3737220}
}

@inproceedings{li2024glbench,
  title     = {GLBench: A Comprehensive Benchmark for Graph with Large Language Models},
  author    = {Li, Yuhan and Wang, Peisong and Zhu, Xiao and Chen, Aochuan and
               Jiang, Haiyun and Cai, Deng and Chan, Victor Wai Kin and Li, Jia},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2024}
}

@inproceedings{huang2024graphadapter,
  title     = {Can GNN be Good Adapter for LLMs?},
  author    = {Huang, Xuanwen and Han, Kaiqiao and Yang, Yang and Bao, Dezheng and
               Tao, Quanjin and Chai, Ziwei and Zhu, Qi},
  booktitle = {Proceedings of the ACM Web Conference 2024},
  pages     = {893--904},
  year      = {2024}
}
```

---

## Operating principles

1. Research fidelity over code beauty.
2. Official implementation over reimplementation.
3. Never label a re-implementation "official".
4. Never alter a baseline's mathematics to fit an interface.
5. Never fabricate text for a text-free dataset and call it FLAG.
6. Never hard-code or hand-edit a result.
7. Say **UNKNOWN** and investigate rather than guess.
8. **Never claim "reproduced" until it has run and been verified.**

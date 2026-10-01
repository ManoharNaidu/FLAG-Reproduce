# 01 - Repository map

[README](README.md) | [00 Master](00-master-workflow.md) | next: [02 Pipeline](02-flag-pipeline.md)

Categories are assigned from what the code does. `LLM` is an extra category (the
generic list has no slot for the Gemma text-generation stage).

Legend: **ENTRYPOINT**, **CORE PIPELINE**, **GRAPH**, **SUBGRAPH**,
**REPRESENTATION**, **SIMILARITY**, **FEATURES**, **DETECTION**, **EVALUATION**,
**CONFIG**, **UTILITY**, **TEST**, **LLM**, and
**NOT-IN-MAIN-PATH** = *Present but not observed in main execution path*.

## 1. Tree

```text
FLAG Reproduce/
|
|-- README.md                         project status + audit summary (prose)
|-- pyproject.toml                    CONFIG   package "flagbench", package-dir=src, deps
|-- run_all_at_once.sh                ENTRYPOINT (orchestration) 16-step GPU-box script: env -> data -> encode -> sample -> LLM
|-- train_cmds.txt                    (empty file)
|
|-- scripts/                          CLI entry points, one per stage
|   |-- download/glbench.py           ENTRYPOINT GRAPH        fetch + verify raw GLBench .pt
|   |-- preprocess/
|   |   |-- build_benchmark.py        ENTRYPOINT GRAPH        1:10 downsample + splits -> graph.pt
|   |   |-- encode_text.py            ENTRYPOINT REPRESENTATION  Sentence-BERT of raw text
|   |   |-- sample_subgraphs.py       ENTRYPOINT SUBGRAPH SIMILARITY  build subgraph cache (+ Fig 3a study)
|   |   |-- encode_llm_text.py        ENTRYPOINT REPRESENTATION  Sentence-BERT of LLM-generated text
|   |   `-- extract_prompts.py        UTILITY  AST-extract prompts from methods/flag/chat*.py -> prompts/
|   |-- llm/generate_text.py          ENTRYPOINT LLM (GPU)    Gemma text per subgraph
|   |-- train/run.py                  ENTRYPOINT CORE PIPELINE   <== START HERE (train + evaluate)
|   |-- analyze/                      EVALUATION  build_reported_results.py, compare_samplers.py, summarize_flag_md.py
|   |-- reproduce/run_flag_md_matrix.sh  ENTRYPOINT (orchestration) cosine vs Markov-diffusion matrix
|   |-- setup/                        UTILITY  fetch_methods.sh, install_gpu.sh, run_multi_gpu_llm.sh, run_md_llm.sh
|   |-- smoke_test.py                 TEST     environment + upstream-claims smoke test
|   |-- evaluate/                     (only __init__.py)  no evaluation script lives here
|   `-- build_flag_dataset.py, prepare_dataset.py, validate_dataset.py,
|       smoke_test_datasets.py, test_flag_dataset_compatibility.py,
|       verify_amazon_alignment.py, verify_yelpchi_alignment.py
|                                     NOT-IN-MAIN-PATH  text-augmented YelpChi/Amazon study (untracked in git)
|
|-- src/flagbench/                    everything written in this repo
|   |-- experiments/
|   |   |-- runner.py                 CORE PIPELINE FEATURES   run_single(), make_*_feature_fn(), loaders
|   |   `-- results.py                EVALUATION  RunResult, aggregate_to_files(), comparison table
|   |-- sampling/
|   |   |-- semantic.py               SUBGRAPH SIMILARITY  SamplingConfig, Subgraph, select_neighbors(), sample_all()
|   |   |-- markov_diffusion.py       SUBGRAPH  FLAG-MD ablation sampler (ranking by diffusion distance)
|   |   `-- cli.py                    CONFIG UTILITY  shared argparse -> SamplingConfig
|   |-- llm/enhance.py                LLM  prompts, Gemma generation, response parsing, cache key
|   |-- adapters/backbone.py          REPRESENTATION DETECTION  wraps methods/flag GNNs; DualBranchBackbone
|   |-- training/
|   |   |-- trainer.py                DETECTION  SubgraphTrainer (fit / predict / finetune_extra)
|   |   `-- flag_losses.py            DETECTION  non_causal_loss, orthogonal_loss (used only by finetune_extra)
|   |-- metrics/classification.py     EVALUATION  AUC, KS, ECE, F1, fit_threshold, evaluate_val_and_test
|   |-- datasets/
|   |   |-- glbench.py                GRAPH  download / load_raw / verify GLBench
|   |   `-- benchmark.py              GRAPH  1:10 downsampling, splits, induced subgraph
|   |-- registry/registry.py          CONFIG  MODEL/VARIANT/DATASET registries, validate()
|   |-- utils/{seeding,device}.py     UTILITY
|   |-- compat/torch_scatter.py       UTILITY  shim for a crashing torch_scatter wheel
|   |-- flag_adapter/                 NOT-IN-MAIN-PATH  FLAGDataset -> graph.pt payload (untracked)
|   |-- fraud_text/                   NOT-IN-MAIN-PATH  YelpChi/Amazon text datasets (untracked)
|   `-- {backbones,evaluation,flag}/  (only __init__.py, 1 line each)  empty packages
|
|-- methods/                          git-ignored, fetched at pinned SHAs
|   |-- flag/                         UPSTREAM REFERENCE  BUPT-GAMMA/FLAG @ cb83944
|   |   |-- models.py                 REPRESENTATION  GCN, GAT, DualGNN (imported by adapters/backbone.py)
|   |   |-- geniepath.py bwgnn.py caregnn.py dga.py pmp.py
|   |   |                             REPRESENTATION  backbones (imported by adapters/backbone.py)
|   |   |-- utils.py                  DETECTION (losses) - imported by tests/smoke only, NOT by src/flagbench
|   |   |-- chat.py chat1.py          LLM ENTRYPOINT (upstream)  Reddit / Instagram Gemma generation - not called
|   |   |-- encode.py                 REPRESENTATION (upstream) - not called
|   |   |-- train.py train1.py        DETECTION (upstream) LoRA + GNN alternating loop - not called
|   |   `-- test.py test_dual.py      EVALUATION (upstream) - not called, ImportError (ECELoss)
|   `-- bwgnn care_gnn dga_gnn geniepath glbench pmp
|                                     NOT-IN-MAIN-PATH  other repos' clones (README: "integrated: no")
|
|-- data/                             raw + built datasets
|   |-- raw/DATASET/DATASET.pt        GRAPH   GLBench original
|   `-- benchmark/
|       |-- flag_reddit/  flag_instagram/       GRAPH  graph.pt + dataset_manifest.json  (MAIN PATH)
|       `-- flag_yelpchi_text/ flag_amazon_text/ native_yelpchi/ native_amazon/
|                                     NOT-IN-MAIN-PATH  text-augmented study
|-- datasets/                         NOT-IN-MAIN-PATH  fraud_text raw/processed (untracked)
|
|-- cache/                            derived artefacts (git-ignored contents)
|   |-- embeddings/                   *__raw.pt (N x 384), and LLM-text embeddings (dict)
|   |-- sampling/                     DATASET__semantic_h2_k10_t0_perhop.pt, DATASET__none_h2_k10_t0_perhop.pt (+ .json)
|   `-- llm/                          Gemma text cache (.json + .manifest.json)   [EMPTY in this checkout]
|
|-- prompts/{reddit,instagram}/       CONFIG  system_instruction / global / discriminative / residual .txt
|                                     read by llm/enhance.py:PromptSet.load()
|-- configs/{datasets,models,experiments}/*.yaml
|                                     CONFIG  NOT-IN-MAIN-PATH: no Python file reads them
|-- environment/{cpu,gpu}.lock.txt    CONFIG  pinned requirements
|-- checkpoints/                      optional output of --save-checkpoint
|-- results/{raw,aggregated,tables,figures,flag_md,dataset_validation}/
|                                     EVALUATION outputs (raw = one JSON/run; aggregated = results.csv/json)
|-- logs/                             run logs
|-- analysis/compare_reported.py      EVALUATION  ours vs reported (Table 4)
|-- benchmark/                        (only __init__.py)  empty package
|-- experiments/yelpchi_amazon/       NOT-IN-MAIN-PATH  native YelpChi/Amazon benchmark builder + smoke test
|
|-- tests/
|   |-- unit/                         TEST  metrics, benchmark build, semantic + MD sampling, LLM prompts, DGA, layout
|   |-- integration/test_flag_upstream_claims.py   TEST  36 audit findings vs pristine upstream
|   |-- test_fraud_text_datasets.py   TEST  (untracked) text-augmented study
|   `-- smoke/, reproducibility/      (only __init__.py)
|
|-- research/                         prose audit + decisions (not code)
`-- docs/                             architecture.md, vastai_gpu_workflow.md, and THIS folder
```

## 2. Files on the traced execution path

| Category | File | Role in the path |
|---|---|---|
| ENTRYPOINT | [scripts/train/run.py](../../scripts/train/run.py) | `main()` |
| CORE PIPELINE | [runner.py](../../src/flagbench/experiments/runner.py) | `run_single()` orchestrates one run |
| CONFIG | [registry.py](../../src/flagbench/registry/registry.py) | `validate()`, `get_variant().default_sampling_strategy`, `get_model().module` |
| GRAPH | [scripts/preprocess/build_benchmark.py](../../scripts/preprocess/build_benchmark.py), [datasets/benchmark.py](../../src/flagbench/datasets/benchmark.py), [datasets/glbench.py](../../src/flagbench/datasets/glbench.py) | produce `graph.pt` |
| REPRESENTATION | [scripts/preprocess/encode_text.py](../../scripts/preprocess/encode_text.py), [encode_llm_text.py](../../scripts/preprocess/encode_llm_text.py) | Sentence-BERT vectors |
| SUBGRAPH + SIMILARITY | [scripts/preprocess/sample_subgraphs.py](../../scripts/preprocess/sample_subgraphs.py), [semantic.py](../../src/flagbench/sampling/semantic.py) | subgraph cache; cosine in `select_neighbors()` |
| LLM | [scripts/llm/generate_text.py](../../scripts/llm/generate_text.py), [enhance.py](../../src/flagbench/llm/enhance.py), [prompts/](../../prompts/) | discriminative / residual text |
| REPRESENTATION (GNN) | [adapters/backbone.py](../../src/flagbench/adapters/backbone.py) -> `methods/flag/{models,geniepath,bwgnn,caregnn,dga,pmp}.py` | node hidden state + logits |
| DETECTION | [trainer.py](../../src/flagbench/training/trainer.py), [flag_losses.py](../../src/flagbench/training/flag_losses.py) | training, scoring, `+FLAG*` phase |
| EVALUATION | [metrics/classification.py](../../src/flagbench/metrics/classification.py), [results.py](../../src/flagbench/experiments/results.py), [analysis/compare_reported.py](../../analysis/compare_reported.py) | thresholds, metrics, storage, comparison |
| UTILITY | [utils/seeding.py](../../src/flagbench/utils/seeding.py), [utils/device.py](../../src/flagbench/utils/device.py), [compat/torch_scatter.py](../../src/flagbench/compat/torch_scatter.py) | seeds, device, shim |

## 3. Present but not observed in main execution path

| Item | Why it looks relevant | Why it is not on the path |
|---|---|---|
| `SamplingConfig.include_center` ([semantic.py:91](../../src/flagbench/sampling/semantic.py#L91)) | sounds like it controls centre inclusion | defined, never read; `sample_subgraph` always includes the centre |
| `configs/*.yaml` | look like experiment config | no Python reads YAML |
| `methods/flag/{chat,chat1,encode,train,train1,test,test_dual}.py` | are FLAG's own drivers | never imported/called by `flagbench`; `prompts/` were *extracted* from `chat*.py` by AST, not by import |
| `methods/flag/utils.py` | defines the three FLAG losses | `flagbench` re-implements them in `training/flag_losses.py`; upstream `utils.py` is imported only by tests/smoke |
| `SubgraphTrainer.embeddings()` ([trainer.py:255](../../src/flagbench/training/trainer.py#L255)) | "the paper's t-SNE embeddings" | no caller found in `src/` or `scripts/` (grep) |
| `flagbench.sampling.markov_diffusion` | a sampler | used only when `strategy == "markov_diffusion"` (FLAG-MD ablation); default is cosine |
| `flag_adapter/`, `fraud_text/`, YelpChi/Amazon scripts | build a `graph.pt` for FLAG | separate text-augmented study; canonical `yelpchi`/`amazon` remain blocked for FLAG variants in `registry.validate()` |
| `DualGNN.linear1 = Linear(384, out)` ([models.py:29](../../methods/flag/models.py#L29)) | looks like a projection | assigned in `__init__`, never used in `forward` |
| `FlagBundledBackbone` skip term for GAT/CARE-GNN/DGA/PMP | Eq. 6 skip | `initial_x` is computed and discarded in those classes (their `forward` never adds it) |
| `GeniePath` (eager class in `geniepath.py`) | a GeniePath | registry points at `GeniePathLazy`; `GeniePath.forward` returns a single tensor and would raise in `FlagBundledBackbone.forward` |

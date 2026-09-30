# 08 - Paper -> code

[README](README.md) | [00 Master](00-master-workflow.md) | prev: [07 Data flow](07-data-flow.md)

**Source of "Paper says".** The FLAG paper PDF is not in this repository. "Paper says" below is taken from
[research/paper_notes.md](../../research/paper_notes.md), the project's own extraction (tagged `[PAPER]`
there); I did not re-read the PDF, so those statements are `UNVERIFIED against the PDF`. "Repository does" is read
from source. "Upstream" = `methods/flag/` (official code); "flagbench" = the reproduction in `src/` + `scripts/`.

## 1. Concept map

| FLAG concept | Repository implementation | File | Function | Notes |
|---|---|---|---|---|
| Fraud benchmark, minority class downsampled to about 1:10 (Sec 4.1.1) | flagbench: keep all majority, downsample minority to `\|majority\|/10` | [benchmark.py](../../src/flagbench/datasets/benchmark.py) | `select_minority_subset()`, `build()` | **Reimplemented.** Upstream ships no construction code; the paper's seed is unpublished, so exact numbers cannot match. |
| Train/val/test split | stratified 10/10/80 | [benchmark.py](../../src/flagbench/datasets/benchmark.py) | `make_splits()`, `BenchmarkConfig.split_ratios` | Paper states no ratio; 10/10/80 is inherited from GLBench/GraphAdapter (labelled as such in code). |
| 2-hop subgraph per node, GraphSAGE-like (Sec 4.1.2) | `hops=2`, per-hop top-N, frontier expansion, induced edges | [semantic.py](../../src/flagbench/sampling/semantic.py) | `sample_subgraph()`, `sample_all()` | Whether top-10 is per hop or total is `UNKNOWN` in the paper; code defaults to per-hop and exposes `per_hop`. |
| Eq. 3 cosine similarity of B(t_v), B(t_u) (Sec 3.2) | normalise rows, dot product | [semantic.py:208,251](../../src/flagbench/sampling/semantic.py#L208) | `normalize_embeddings()`, `select_neighbors()` | Upstream: **NOT FOUND IN REPOSITORY** (no sampler code; drivers load pre-built `*_sampler*.pt`). |
| Eq. 4 threshold delta then top-N | `sims >= 0` then top-10 | [semantic.py:255-265](../../src/flagbench/sampling/semantic.py#L255) | `select_neighbors()` | Order of threshold vs top-N ambiguous in paper; code filters first (`threshold_first=True`). |
| Frozen LM B(.) = Sentence-BERT | `all-MiniLM-L6-v2`, 384-d | [encode_text.py](../../scripts/preprocess/encode_text.py) | `encode()` | Checkpoint taken from upstream `encode.py:30`, not from the paper (which says only "Sentence-BERT"). |
| Eq. 5 subgraph edge homophily; Fig. 3(a) NS/RS/FS/SS*/SS study | implemented | [semantic.py:422](../../src/flagbench/sampling/semantic.py#L422), [sample_subgraphs.py:126](../../scripts/preprocess/sample_subgraphs.py) | `subgraph_homophily()`, `compare_strategies()` | FS uses the stored 4096-d `x`, not the paper's word2vec shallow features (code comment says so). Analysis only, not on the training path. |
| LLM = Gemma-9b-it; discriminative + residual text via two prompts (Sec 3.3, Table 1) | `google/gemma-2-9b-it`, prompts from `prompts/DATASET/*.txt` | [enhance.py](../../src/flagbench/llm/enhance.py), [generate_text.py](../../scripts/llm/generate_text.py) | `LLMEnhancer.enhance()`, `build_prompt()` | Upstream equivalent: `chat.py:generate_summary`. Prompt wording is upstream's ("causal / non-causal"), **not** the paper's Table 1 text. Production cache used a reduced decode budget (64 tokens / 300 chars vs upstream 550 / 1200). |
| Skip-GNN, Eq. 6: Z = GNN(X, A) + Linear(X) | `initial_x = linear1(x)` added to output | `methods/flag/models.py` `GCN`; `geniepath.py`; `bwgnn.py` | `forward()` | Present only in GCN, GeniePathLazy, BWGNN. GAT, CARE-GNN, DGA, PMP compute `initial_x` and drop it. |
| Two shared-parameter skip-GNNs + attention layer (Sec 3.4, inference) | one GNN applied to both branches, softmax over a learned `(1, 2)` weight | [backbone.py:284](../../src/flagbench/adapters/backbone.py#L284) -> `methods/flag/models.py:23` | `DualBranchBackbone`, `DualGNN.forward()` | Paper leaves the attention architecture unspecified. Code: one global weight pair, not per-node attention (see [05](05-feature-and-detection-pipeline.md)). |
| Eq. 7 L_Disc (BCE) | 2-logit cross-entropy | [trainer.py:151](../../src/flagbench/training/trainer.py#L151); upstream `utils.py:causal_loss` | `SubgraphTrainer.__init__` (`criterion`) | Equivalent for two classes. |
| Eq. 8 L_Res (KL to uniform) | `F.kl_div(log_softmax(logits), uniform, batchmean)` | [flag_losses.py:32](../../src/flagbench/training/flag_losses.py#L32); upstream `utils.py:86` | `non_causal_loss()` | Matches the shape actually used. |
| Eq. 9 L_Orthog = norm(Z_D . Z_R)^2 | **default**: `sum(a*b)**2`; option `signed_cosine` | [flag_losses.py:43](../../src/flagbench/training/flag_losses.py#L43) | `orthogonal_loss(mode=...)` | **Differs from upstream**: upstream `utils.py:94` returns a *signed cosine* (minimised at anti-alignment). flagbench defaults to the paper's form. |
| Eq. 10 total loss with lambda_1, lambda_2 | `alpha=0.1`, `beta=0.1` | [trainer.py:89-90](../../src/flagbench/training/trainer.py#L89) | `FinetuneConfig`, `finetune_extra()` | Paper: lambdas unspecified. Values from upstream `train.py --alpha/--beta`. |
| Two-stage alternating training; **LoRA fine-tuning** of the LLM (Sec 3.4) | **not reproduced**. `+FLAG*` = extra GNN epochs, LLM frozen | [trainer.py:335](../../src/flagbench/training/trainer.py#L335) | `finetune_extra()` | Upstream `train.py` cannot update LoRA weights (generate() is non-differentiable; text re-encoded into a new leaf tensor). Rows carry `llm_finetuned=False` ([runner.py:298](../../src/flagbench/experiments/runner.py#L298)). |
| Inference: fine-tuned LLM extracts discriminative text | zero-shot Gemma text, one prompt per subgraph | [enhance.py:293](../../src/flagbench/llm/enhance.py#L293) | `enhance()` | The `+FLAG` (zero-shot) variant matches this; `+FLAG*` in the paper implies a fine-tuned LLM, which is not run here. |
| Training: Adam lr 0.01, accumulate 10 subgraphs | `Adam(lr=0.01)`, sum of 10 losses per step | [trainer.py:44-53,194-231](../../src/flagbench/training/trainer.py#L44) | `TrainConfig`, `train_epoch()` | Matches paper and upstream; `_step` is keyed off the enumerate counter (upstream shadows `i`). |
| Early stopping "to prevent overfitting" | implemented (`patience=10`) | [trainer.py:317-326](../../src/flagbench/training/trainer.py#L317) | `fit()` | Upstream parses `--patience` but never reads it. In flagbench it **cannot fire at the default 5 epochs**. |
| Model selection by highest validation F1-macro | best-epoch `state_dict` by val F1-macro | [trainer.py:273-332](../../src/flagbench/training/trainer.py#L273) | `fit()` | Consistent. |
| Metrics: F1-macro, AUC (public), KS (industrial), "following BWGNN" | AUC, KS, ECE, F1-macro + fraud P/R/F1, accuracy | [classification.py](../../src/flagbench/metrics/classification.py) | `evaluate()`, `threshold_metrics()` | ECE is not a paper metric (upstream's `ECELoss` does not exist). |
| F1 decision threshold | default `validation_swept` over `linspace(0.05, 0.95, 19)`; also `argmax`, `fixed` | [classification.py:185](../../src/flagbench/metrics/classification.py#L185) | `fit_threshold()` | Paper states **no policy**. Upstream uses `argmax`. Choice is a documented decision (S-006) and recorded per row. |
| 25 runs = 5 seeds x 5 inits | grid via `--seeds`, `--inits` | [run.py:56-59](../../scripts/train/run.py#L56), [seeding.py](../../src/flagbench/utils/seeding.py) | `main()`, `RunIdentity` | **CLI defaults are 1 x 1**; the paper's protocol needs `--seeds 5 --inits 5`. Upstream does 5 x 1. |
| Hidden size 64, 2 layers | `--hidden-dim 32` default | [run.py:64](../../scripts/train/run.py#L64) | `TrainConfig.hidden_dim` | Conflict: paper 64, upstream 32 (`x32` guards break at other sizes; DGA/PMP raise). |
| Variants: baseline / +text / +FLAG / +FLAG* | four variants | [registry.py:248](../../src/flagbench/registry/registry.py#L248) | `VARIANT_REGISTRY`, `runner.make_feature_fn` | "Shallow embeddings" of `baseline` are the stored 4096-d features (dataset notes: Llama-2-derived), so the paper's label may not fit. |
| Huabei industrial dataset (Table 3) | registry entry `obtainable=False` | [registry.py:316](../../src/flagbench/registry/registry.py#L316) | `DATASET_REGISTRY["huabei"]` | Proprietary; not reproducible. |
| Backbones GCN, GAT, GeniePath, CARE-GNN, BWGNN, DGA-GNN, PMP | FLAG's own PyG rewrites, imported unmodified | [backbone.py:191](../../src/flagbench/adapters/backbone.py#L191) | `FlagBundledBackbone` | Registry records fidelity: GAT's 2nd layer is `SAGEConv`; CARE-GNN, BWGNN, DGA not the published algorithms. |
| **Not in the paper:** FLAG-MD (Markov-diffusion ranking) | alternative sampler | [markov_diffusion.py](../../src/flagbench/sampling/markov_diffusion.py) | `MarkovDiffusionNeighborSampler` | Reproduction-side ablation ("idea from DGP"); off by default. |
| **Not in the paper:** YelpChi/Amazon text-augmented study | `fraud_text/`, `flag_adapter/` | [fraud_text/](../../src/flagbench/fraud_text/), [flag_adapter/](../../src/flagbench/flag_adapter/) | `write_flag_payload()` | Separate experiment; not on the main path; not comparable to Table 4. |

## 2. Explicit differences: Paper says / Repository does

| # | Paper says | Repository does |
|---|---|---|
| 1 | Sampling (Eq. 3-4) is part of FLAG | Upstream ships **no** sampler. flagbench reimplements it in `semantic.py`; the sampler's exact top-N/threshold ordering is our reading of the prose. |
| 2 | The LLM is fine-tuned (LoRA) in stage 2, giving `+FLAG*` | Upstream `train.py` gradient path is severed. flagbench `+FLAG*` = frozen LLM + extra GNN epochs (`llm_finetuned=False`). |
| 3 | Orthogonality loss = squared dot product (Eq. 9) | Upstream = signed cosine. flagbench default = squared dot (paper), signed cosine selectable. |
| 4 | Discriminative/residual prompts as in Table 1 | Upstream (and flagbench, verbatim) uses "causal / non-causal" prompts with extra formatting instructions. |
| 5 | Hidden 64, two layers | 32; GeniePath has 4 layers and 256 dims by module globals. |
| 6 | 25 runs | Upstream 5; flagbench CLI default 1 (configurable). |
| 7 | Early stopping used | Upstream: not implemented. flagbench: implemented but unreachable at 5 epochs / patience 10. |
| 8 | Skip connection is part of Skip-GNN | Applied in 3 of 7 backbones. |
| 9 | F1 threshold: unstated | Upstream `argmax`; flagbench `validation_swept` by default. |
| 10 | AUC of the model's score | Upstream `test_dual.py` uses `sigmoid(logits)[:, 1]`; flagbench uses `softmax(logits)[1]`. They are different monotone maps of the two logits, so rankings can differ. |
| 11 | Fine-tuning LR (upstream `--lr 1e-4`) | flagbench's `FinetuneConfig.lr` looks ineffective: the update goes through `self.optimizer` built from `TrainConfig` (lr 0.01). By reading only, see [05](05-feature-and-detection-pipeline.md). |

## 3. Unresolved / `UNVERIFIED`

- Sentence-BERT tokenizer and pooling (inside the sentence-transformers library).
- Whether the stored GLBench edge lists are symmetric (asserted in `research/dataset_notes.md`, not checked in code).
- Which of the paper's variants produced its `+FLAG*` column (`research/reproduction_status.md` lists this as open).
- Behaviour of `flag` / `flag_finetuned` runs in this checkout: `cache/llm/` is empty, so they were not exercised.
- The `FinetuneConfig.lr` observation above was not confirmed by executing code.

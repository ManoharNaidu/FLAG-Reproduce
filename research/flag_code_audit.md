# FLAG Official Repository — Code Audit

**Status of this document:** every statement below is derived by *reading the cloned source*.
Nothing here is inferred from the paper. Claims sourced from the paper are marked
`[PAPER]` and are recorded separately in `paper_notes.md`.

| Field | Value |
|---|---|
| Repository | https://github.com/BUPT-GAMMA/FLAG |
| Clone path | `methods/flag/` |
| Default branch | `main` |
| Commit SHA | `cb83944ed8a8a9b070a3f5a167d363973369fc80` |
| Commit author | `guofeng97` |
| Commit date | 2025-06-04 10:10:22 +0800 |
| Commit message | `Add files via upload` (single commit; **no development history**) |
| Files | 14 Python files, 1992 insertions, flat layout (no packages) |
| README | **ABSENT** |
| LICENSE | **ABSENT** — see "Licensing" below |
| requirements.txt / environment.yml | **ABSENT** |
| Config files | **ABSENT** — all hyperparameters live in `argparse` defaults or module-level globals |
| Example commands / scripts | **ABSENT** |

## 1. Repository layout

```
methods/flag/
├── models.py      # GCN, DualGNN, GraphSAGE, GAT, CaGCN
├── geniepath.py   # GeniePath, GeniePathLazy    (baseline, re-implemented by FLAG authors)
├── bwgnn.py       # BWGNN, PolyConv             (baseline, re-implemented by FLAG authors)
├── caregnn.py     # CAREGNN, CAREGNNLayer       (baseline, re-implemented by FLAG authors)
├── dga.py         # DGA, IntraConv              (baseline, re-implemented by FLAG authors)
├── pmp.py         # LASAGE_S, LASAGESConv       (baseline, re-implemented by FLAG authors)
├── utils.py       # losses, t-SNE, homophily split
├── chat.py        # LLM text generation      — Reddit
├── chat1.py       # LLM text generation      — Instagram
├── encode.py      # Sentence-BERT encoding   — Instagram (hardcoded)
├── test.py        # GNN train/eval driver    — Reddit
├── test_dual.py   # GNN train/eval driver    — Instagram (dual-branch)
├── train.py       # LoRA fine-tuning driver  — Reddit
└── train1.py      # LoRA fine-tuning driver  — Instagram
```

### 1.1 File-naming convention (established by diffing the pairs)

The repository ships **one file per (stage, dataset)** pair. There is no `--dataset` flag.

| Stage | Reddit | Instagram |
|---|---|---|
| LLM text generation (zero-shot) | `chat.py` | `chat1.py` |
| Sentence-BERT encoding | *(not shipped separately)* | `encode.py` |
| GNN train / evaluate | `test.py` | `test_dual.py` |
| LoRA fine-tuning (FLAG*) | `train.py` | `train1.py` |

`diff chat.py chat1.py`, `diff train.py train1.py` and `diff test.py test_dual.py` differ **only**
in data paths, prompt wording, and (for `test_dual.py`) the use of `DualGNN`. This is verified,
not assumed.

## 2. Verified hyperparameters (from source)

These are read directly off `argparse` defaults and module globals.

| Parameter | Value | Source |
|---|---|---|
| Sentence-BERT encoder | `all-MiniLM-L6-v2` (384-dim) | `encode.py:30`, `train.py:56` |
| LLM | `gemma-2-9b-it` | `chat.py:52`, `train.py:44` |
| LLM dtype | `torch.float16` | `chat.py:56`, `train.py:46` |
| GNN hidden dim | **32** (`--hidden` default) | `train.py:24`, `test.py:27` |
| GNN output dim | 2 | `test.py:201`, `train.py:57` |
| GNN layers | 2 (GCN/GAT/CARE/DGA/PMP) | `models.py`, `dga.py`, `pmp.py` |
| GNN learning rate | 0.01 | `test.py:25` |
| LLM (LoRA) learning rate | 1e-4 | `train.py:22` |
| GNN optimizer | `Adam` | `test.py:203`, `train.py:62` |
| LLM optimizer | `AdamW` | `train.py:61` |
| weight decay | 0 (`test.py`), 5e-4 (`train.py`, unused) | `test.py:31`, `train.py:28` |
| dropout | 0.5 | `test.py:29` |
| gradient accumulation | **10** subgraphs | `test.py:47`, `train.py:63` |
| GNN epochs | 5 | `test.py:23` |
| LoRA outer epochs | 3 | `train.py:18` |
| LoRA inner (GNN) epochs | 10 | `train.py:20` |
| loss weight `alpha` (residual) | **0.1** | `train.py:31` |
| loss weight `beta` (orthogonality) | **0.1** | `train.py:32` |
| LoRA `r` | 8 | `train.py:38` |
| LoRA `lora_alpha` | 32 | `train.py:39` |
| LoRA `target_modules` | `["q_proj", "v_proj"]` | `train.py:40` |
| LoRA `lora_dropout` | 0.1 | `train.py:41` |
| LoRA `bias` | `"none"` | `train.py:42` |
| LLM `max_new_tokens` | 550 | `chat.py:71`, `train.py:123` |
| Per-node text truncation | 1200 characters | `chat.py:63`, `train.py:116` |
| Runs | `for i in range(5)` x `for j in range(1)` = **5 runs** | `test.py:195,198` |
| Seeding | `torch.manual_seed(i)` for `i` in 0..4 | `test.py:196` |
| GeniePath `dim` / `lstm_hidden` | 256 / 256 | `geniepath.py:6-7` |
| GeniePath heads / layers | 1 / **4** | `geniepath.py:8-9` |
| Baseline feature dim (`data.x`) | **4096** | `test.py:201` (`GeniePathLazy(4096, 2, ...)`) |

### 2.1 Discrepancies between the code and the paper

Recorded, **not resolved**. Both values are exposed through configuration in this repo.

| Item | `[PAPER]` (as stated in the project brief) | Official code | Note |
|---|---|---|---|
| hidden dimension | 64 | **32** | The code's `x32` variable names and the `if len(x[0]) == 32` guards in `dga.py:63` / `pmp.py:110` only work when hidden == 32. The code is self-consistent at 32. See section 5.3. |
| number of layers | 2 | 2 for GCN/GAT/CARE/DGA/PMP; **4** for GeniePath | `geniepath.py:9 layer_num = 4` |
| number of runs | 25 (5 seeds x 5 inits) | **5** (`range(5)` x `range(1)`) | `test.py:198` reads `for j in range(1)`. That it was reduced from 5 is inference; the shipped code runs 5. |
| optimizer LR | 0.01 (Adam) | 0.01 for the GNN; 1e-4 (AdamW) for LoRA | Consistent; the paper's 0.01 refers to the GNN. |
| early stopping | "enabled" | **not implemented** | `--patience 10` is parsed in `test.py:33` / `train.py:30` and then **never read**. Selection is "keep best-so-far checkpoint over a fixed epoch budget", not patience-based stopping. |

## 3. The FLAG pipeline as actually implemented

```
reddit1.pt / instagram.pt            (PyG Data: .x, .edge_index, .y, .raw_texts, masks)
        |
        |  [sampler construction — NOT SHIPPED, see section 6 GAP-1]
        v
{train,val,test}_sampler*.pt         (list of subgraph batches; each has
        |                             .subset, .central, .edge_index)
        |  chat.py / chat1.py         LLM = gemma-2-9b-it, zero-shot
        v                             prompt = unique_prompt (discriminative)
batch.unique  : List[str]
        |
        |  encode.py                  Sentence-BERT all-MiniLM-L6-v2
        v
batch.unique_embeddings : [N, 384]
        |
        |  test.py / test_dual.py     GNN training, Adam lr=0.01, accum=10
        v
metrics: AUC, F1-macro, ECE, accuracy
```

For the fine-tuned variant (`train.py` / `train1.py`) the LLM is wrapped with PEFT LoRA and
**both** prompts are used per batch (`unique_prompt` -> discriminative, `common_prompt` -> residual),
producing `batch.unique` and `batch.common`.

### 3.1 The four experimental variants are selected by *editing source*, not by flags

`test.py` computes a `text_embeddings` variable at lines 67-70:

```python
if hasattr(batch, "unique_embeddings"):
    text_embeddings = batch.unique_embeddings     # FLAG
else:
    text_embeddings = embeddings[batch.subset]    # +text (Sentence-BERT of raw text)
```

...and then **never uses it** — line 82 hardcodes the baseline path:

```python
emb32, output = model(data.x[batch.subset].cuda(), edge_index.cuda())   # baseline: data.x (4096-d)
```

So the variant is chosen by hand-swapping the tensor passed to `model(...)`, and the model
constructor's input dim must be edited to match (`4096` vs `384`). Likewise the backbone is
chosen by commenting/uncommenting `test.py:200-202`:

```python
#gnn_model = GCN(4096, args.hidden, 2).cuda()
gnn_model = GeniePathLazy(4096, 2, 'cuda').cuda()
#gnn_model = DualGNN(2, gnn).cuda()
```

**Implication for this project:** the variant/backbone matrix in `[PAPER]` is real, but the
repository has no mechanism to run it. Building that mechanism is genuine integration work,
not a rewrite of the research code.

### 3.2 Mapping: paper loss names -> code loss names

The code uses *causal / non-causal* vocabulary; the paper uses *discriminative / residual*.
Verified mapping (`utils.py:80-101`, applied at `train.py:155-157`):

| Paper name | Code symbol | Implementation |
|---|---|---|
| Discriminative Text Loss | `causal_loss` | `F.cross_entropy(causal_output, target_labels)` |
| Residual Text Loss | `non_causal_loss` | `F.kl_div(log_softmax(out, dim=0), uniform(1/2), reduction='batchmean')` |
| Orthogonality Loss | `orthogonal_loss` | `sum(normalize(c, dim=0) * normalize(nc, dim=0))` — a cosine-similarity term |

Total loss (`train.py:155-157`, `train.py:180-182`):

```
L = causal_loss(u, y) + alpha * non_causal_loss(c) + beta * orthogonal_loss(u, c)
    with alpha = beta = 0.1
```

This confirms `lambda_1 = lambda_2 = 0.1` **in the shipped code**. Whether the paper states the
same is tracked in `paper_notes.md`.

Note two implementation details worth flagging, because they affect meaning:

- `non_causal_loss` applies `log_softmax(..., dim=0)`. At the call site the tensor passed in is the
  *single central node's* logit vector of shape `[2]`, so `dim=0` is the class axis and the KL-to-uniform
  is well-formed for that call. But the function is written as if it accepted a batch, where `dim=0`
  would be the wrong axis. It is correct only because of how it happens to be called.
- `orthogonal_loss` likewise normalises over `dim=0` and returns a raw (signed) cosine similarity,
  not its absolute value or square. Minimising a signed cosine drives the two embeddings toward
  *anti*-alignment (cos = -1), not orthogonality (cos = 0). This is a real semantic gap between the
  function's name/`[PAPER]` intent and its behaviour. Recorded, not silently changed.

### 3.3 Skip-GNN

`[PAPER]` describes the final representation as GNN aggregation plus a direct linear projection
of the node features. In code this is `self.linear1 = Linear(in_channels, out_channels)` with
`return ..., x + initial_x`.

**It is applied inconsistently across backbones.** Verified per class:

| Backbone | class | `linear1` defined | `+ initial_x` returned | Skip active |
|---|---|---|---|---|
| GCN | `models.py:GCN` | yes | yes (`models.py:21`) | **YES** |
| GAT | `models.py:GAT` | yes | **no** (`models.py:72` returns `x32, x`) | **NO** |
| GraphSAGE | `models.py:GraphSAGE` | yes | **no** (returns a single tensor) | **NO** |
| GeniePath | `geniepath.py:GeniePath` | yes | yes (`:63`) | **YES** |
| GeniePathLazy | `geniepath.py:GeniePathLazy` | yes | yes (`:90`) | **YES** |
| BWGNN | `bwgnn.py:BWGNN` | yes | yes (`:73`) | **YES** |
| CARE-GNN | `caregnn.py:CAREGNN` | yes | **no** (`:51` returns `x32, x`) | **NO** |
| DGA | `dga.py:DGA` | yes | **no** (`:65` returns `x32, fc_out(x)`) | **NO** |
| PMP | `pmp.py:LASAGE_S` | yes | **no** (`:115` returns `x32, x`) | **NO** |

In every "NO" row `initial_x` is still *computed* and then discarded — dead code. This is
consistent with the skip being added to GCN/GeniePath/BWGNN and not propagated to the others.
Whether the paper's reported `+FLAG` numbers for GAT/CARE/DGA/PMP were produced with or without
the skip is **UNKNOWN** and cannot be determined from this repository.

**Decision for this project:** expose the skip as an explicit ablation flag (`SG` in the Phase-26
ablation set), defaulting per-backbone to the shipped behaviour, so both readings are runnable.

## 4. Prompts (verbatim, from source)

Extracted to `prompts/` under version control. Four prompts total; the Reddit and Instagram
`unique_prompt`s differ in more than dataset nouns:

- Reddit `unique_prompt` (`chat.py:19-34`): *"...generate a brief causal text for each user that
  **directly relates to classifying them** as either popular or normal..."*
- Instagram `unique_prompt` (`chat1.py:19-34`): *"...generate a brief causal text for each user that
  **highlights their unique characteristics without predicting their classification** as commercial
  or normal..."*

That is a substantive semantic difference between the two datasets' discriminative prompts, not a
noun swap. It is preserved verbatim rather than harmonised.

`common_prompt` (residual) is defined in `chat.py`/`chat1.py` but **never used there** — zero-shot
generation only produces discriminative text. `common_prompt` is used only in `train.py`/`train1.py`
(the fine-tuning path), which is internally consistent: the residual branch exists to supply the
residual/orthogonality losses during fine-tuning.

## 5. Defects found in the official code

Severity: **BLOCKER** = prevents execution; **SEMANTIC** = runs but changes research meaning;
**MINOR** = cosmetic/latent.

### 5.1 BLOCKER — `ECELoss` does not exist
`test.py:18` and `test_dual.py:17`:
```python
from utils import FocalLoss, visualization, ECELoss
```
`utils.py` defines `generate_homo`, `visualization`, `t_sne`, `causal_loss`, `non_causal_loss`,
`orthogonal_loss`, `FocalLoss`, `remove_empty_lines`. There is **no `ECELoss`**.
`test.py` and `test_dual.py` therefore fail at import time with `ImportError`. ECE is computed at
`test.py:108,170` and reported in the final summary, so the symbol is genuinely required.
=> Both evaluation drivers are non-executable as shipped.

### 5.2 BLOCKER — `train.py` unpacking mismatch
`models.py:GCN.forward` returns a 2-tuple `(x32, x + initial_x)`.
`train.py:150-152` does:
```python
unique_embeddings = gnn_model(unique_embeddings, batch.edge_index.cuda())   # -> tuple
unique_embeddings = unique_embeddings[batch.subset == batch.central][0]     # tuple[BoolTensor]
```
Indexing a `tuple` with a `torch.BoolTensor` raises `TypeError`. `test.py:82` unpacks correctly
(`emb32, output = model(...)`), `train.py` does not. The two drivers were evidently not in sync at
the time of upload.

### 5.3 BLOCKER (conditional) — `x32` is only defined when hidden == 32
`dga.py:63-64` and `pmp.py:110-111`:
```python
if len(x[0]) == 32:
    x32 = x
```
`x32` is returned unconditionally on the next line. With any `--hidden` other than 32 this raises
`UnboundLocalError`. This is the strongest internal evidence that **32 is the operative hidden
dimension** for the shipped code, and it is why `--hidden 64` cannot be run without a code change.

### 5.4 SEMANTIC — gradient accumulation counter is shadowed
`train.py:113-115` (and identically `train.py:200-202` in `val_model`):
```python
for i, batch in enumerate(train_loader):
    question = "..."
    for i in range(len(batch.subset)):      # <-- rebinds `i`
        ...
    ...
    if (i + 1) % accumulation_steps == 0:   # <-- uses the INNER i
```
The accumulation trigger therefore keys off `len(batch.subset) - 1`, i.e. the subgraph size, not the
batch counter. Accumulation fires whenever a subgraph happens to have a size congruent to 0 mod 10.
`test.py:65` does **not** have this bug (no inner rebinding), so the *baseline/variant* runs
accumulate correctly at 10; only the *fine-tuning* driver is affected.

### 5.5 SEMANTIC — the LoRA gradient path is severed
In `train.py:141-157` the LLM output is turned into features via:
```python
unique_result = [...]                                   # decoded strings
unique_embeddings = encoder.encode(unique_result)       # sentence-transformers -> numpy
unique_embeddings = torch.Tensor(unique_embeddings).cuda()   # NEW LEAF TENSOR
...
loss = causal_loss(...) + alpha * ... + beta * ...
batch_loss.backward()
optimizer.step()        # optimizer holds the LoRA parameters
```
`model.generate()` is non-differentiable and the decode -> re-encode round trip through numpy
produces a fresh leaf tensor. **No gradient can reach the LoRA parameters.** `backward()` still
succeeds (the GNN parameters require grad), but `optimizer.step()` on the PEFT parameters is a
no-op with `.grad is None`.
=> As shipped, `train.py` does not actually fine-tune the LLM.
This is the single most consequential finding in the audit. Reproducing "FLAG*" faithfully
requires knowing what the authors actually ran, which this repository does not show.
**Recorded as an open question — not silently repaired.** See `reproduction_status.md`.

### 5.6 SEMANTIC — `train_gnn` steps the wrong optimizer
`train.py:189-191`:
```python
if batch_loss != 0:
    batch_loss.backward()
    optimizer.step()      # should be gnn_optimizer.step()
```
The tail-flush of the GNN inner loop steps the **LLM** optimizer. Present in `train.py` and
`train1.py`.

### 5.7 MINOR — inconsistent prefix stripping in `encode.py`
`encode.py:38` strips `text[i][2:]` for the train split but `text[i][3:]` for val (`:51`) and test
(`:63`). These strip the LLM's `"1. "` list numbering; the off-by-one means train and val/test text
are trimmed differently. With `"1. "` (3 chars) the train split retains a leading space that
`.strip()` then removes, so the practical effect is small, but the asymmetry is real.

### 5.8 MINOR — `test_loader` is never evaluated in `train.py`
`main(...)` accepts `test_loader` (`train.py:282`) and never uses it. The fine-tuning driver
reports validation metrics only. No test-set number is produced by `train.py`/`train1.py`.

### 5.9 MINOR — `chat.py` retry loop cannot retry
`generate_summary(..., max_retries=1)`: on a format mismatch it builds `feedback_prompt` and
re-tokenises, but the `for attempt in range(1)` loop then terminates, so the corrected prompt is
never sent. Effectively zero retries.

### 5.10 MINOR — latent `NameError` in `pmp.py`
`pmp.LILinear.reset_parameters` calls `math.sqrt` but `pmp.py` never imports `math`. `LILinear` is
never instantiated by `LASAGE_S`, so this is dormant.

### 5.12 SEMANTIC — `dga.py` drops `dim_size`, so DGA cannot aggregate sparse subgraphs

**Found by running it, not by reading it** — this defect is not visible from a
forward pass on a dense graph.

`dga.py:42-43` overrides message passing as:

```python
def aggregate(self, inputs, index):
    return torch_scatter.scatter_mean(inputs, index, dim=0)
```

PyG's `MessagePassing` inspects the signature of `aggregate` and passes only the
parameters it declares. This one does not declare `dim_size`, so PyG cannot
supply the node count and `scatter_mean` infers the output length from
`max(index) + 1`.

**Consequence.** The aggregated tensor is shorter than the node count whenever
the highest-indexed node has no incoming edge, and the subsequent `out + x_self`
either raises or broadcasts wrongly. Measured on a 3-node graph:

| edges | result |
|---|---|
| into every node | 3 rows — correct |
| `0 -> 1` only | `RuntimeError`: size 2 vs 3 |
| none at all | `RuntimeError`: size 0 vs 3 |

**Why it did not surface upstream.** FLAG's sampled subgraphs are undirected and
centred, so every included node normally has at least one incoming edge and the
inferred length happens to be right. The bug only bites on subgraphs containing a
node with no incoming edge — in practice, **isolated nodes**.

**Why it bites here.** The 1:10 downsampling strands a large fraction of nodes:
**3,690 of 18,389 (20%) Reddit benchmark nodes are isolated**, and their 2-hop
subgraph is a single node with no edges. Every one crashes DGA. So this defect is
latent in the published pipeline but fatal in any reproduction that rebuilds the
benchmark — which is unavoidable, since the authors' seed is unpublished.

**Our fix: LEVEL 3** (Phase 36 — obvious repository bug, behaviour preserved),
applied in `flagbench/adapters/backbone.py:_patch_dga_class`, **not** in upstream
source. It re-declares `aggregate` to accept and forward `dim_size`. It must
patch the *class* rather than an instance, because `MessagePassing.__init__`
caches the inspected signature at construction time.

**Legitimacy is tested, not asserted.**
`tests/unit/test_dga_isolated_nodes.py::test_repair_is_identical_where_upstream_already_worked`
runs an unpatched and a patched model with identical weights on a fully connected
subgraph — the case upstream could already compute — and requires bit-comparable
output. `scatter_mean` with an explicit `dim_size` only ever *extends* the result
with empty groups, which are zero, so no previously-computable value changes.
10/10 tests pass, including two regression tests that will fail if upstream ever
fixes this itself.

### 5.11 MINOR — CPU is not supported anywhere
`.cuda()` is hardcoded at ~40 call sites across all 14 files, including
`criterion = torch.nn.CrossEntropyLoss().cuda()` at module scope and `GeniePathLazy(4096, 2, 'cuda')`.
There is no device argument. Running any script on a CUDA-free machine fails immediately.
(This machine has **no CUDA** — `torch 2.1.0+cpu`, `torch.cuda.is_available() == False`.)

## 6. Gaps — things the repository does not contain

| ID | Gap | Consequence |
|---|---|---|
| GAP-1 | **Subgraph sampler construction is not shipped.** Every driver *loads* `*_sampler*.pt` / `*_loader*.pt` but nothing *creates* them. The semantic-similarity neighbour sampling (cosine similarity, top-N, threshold) — a core FLAG contribution — has **no source in the repository**. | The `0_10_0` directory name is the only trace of the sampling configuration. Reproducing FLAG requires re-deriving this component. |
| GAP-2 | **Dataset files absent.** `reddit1.pt`, `reddit2.pt`, `instagram.pt`, `embeddings1.pt`, `embeddings.pt` are not in the repo and no download link is given. | Dataset provenance must be established externally. |
| GAP-3 | **`data.x` construction unknown.** Baseline features are 4096-dim. Nothing shows how they were produced. | 4096 is a common LLM hidden size; attributing it is speculation until the dataset source is confirmed. Marked **UNKNOWN**. |
| GAP-4 | **No pre-trained `gnn.pth`.** `train.py:58` loads `Reddit/model_lora2/gnn.pth` before fine-tuning starts. | Confirms a Stage-1 GNN pre-training run must precede fine-tuning, but the script that produces it into *that path* is not shipped (`test.py` writes to `args.path + 'gnn.pth'`). |
| GAP-5 | **No train/val/test split code**, no 1:10 downsampling code, no seed recording. | The `[PAPER]` 1:10 minority ratio construction must be re-derived. |
| GAP-6 | **No requirements/environment file.** | Dependency versions (PyG, PEFT, transformers, sentence-transformers) are **UNKNOWN**; must be inferred from API usage. |
| GAP-7 | **No KS metric, no threshold-selection policy.** Only AUC / F1-macro(argmax) / ECE / accuracy. | F1 here uses `argmax`, i.e. a fixed 0.5-equivalent threshold, not a validation-tuned one. |

### 6.1 What `0_10_0` most likely encodes

The sampler directory is `Reddit/0_10_0/` and `Instagram/0_10_0/`. `[PAPER]` specifies
similarity threshold 0 and top-10 sampling over 2 hops. The three fields are consistent with
`{threshold}_{top_k}_{?}`. The third field is **UNKNOWN**. Phase-27 sensitivity sweeps
(`top_k in {5,10,15,20}`, `threshold in {0.4,0.2,0,-0.2,-0.4}`) would produce sibling directories
such as `0.2_15_0`, which is consistent with this reading but does not prove it.

## 7. Baseline provenance — these are NOT the official baseline implementations

`geniepath.py`, `bwgnn.py`, `caregnn.py`, `dga.py`, `pmp.py` are **re-implementations written by
the FLAG authors in PyTorch Geometric**, not vendored copies of the baselines' official
repositories. Evidence: uniform PyG `MessagePassing` style, uniform `linear1` skip hook, uniform
`(x32, out)` return signature, no upstream file headers, no licence notices, all added in the same
single commit.

Assessment of fidelity to the original algorithms (per Phase 19). Classification per Phase 18.

| Baseline | FLAG's file | Fidelity assessment | Class |
|---|---|---|---|
| **GCN** | `models.py:GCN` | Canonical: two `GCNConv` layers. Faithful. | DIRECT |
| **GAT** | `models.py:GAT` | **Not canonical GAT.** `conv1 = GATConv(in, hidden, 8)` but `conv2 = SAGEConv(8*hidden, out)` — the *second layer is GraphSAGE, not attention*. A two-layer GAT should use `GATConv` twice. Also note `GATConv(in, hidden, 8)` passes `8` positionally to `heads`. | ADAPTER_REQUIRED |
| **GeniePath** | `geniepath.py` | Structurally matches the widely-used `shawnwang-tech/GeniePath-pytorch` layout (Breadth/Depth/GeniePathLayer/GeniePathLazy) plus the added `linear1` skip. Adaptive breadth (GAT) + depth (LSTM) preserved. Closest to faithful of the five. | ADAPTER_REQUIRED |
| **CARE-GNN** | `caregnn.py` | **Substantially not CARE-GNN.** CARE-GNN's defining components are (a) a label-aware similarity measure trained as an auxiliary classifier, (b) **reinforcement-learning-based adaptive neighbour filtering** with a learned per-relation threshold, and (c) **multi-relation** inter-relation aggregation. `caregnn.py` implements none of these — it is a similarity-gated mean aggregator (`sigmoid(MLP([x_i, x_j])) * x_j`). No RL, no relation types, no filtering. | REIMPLEMENT_REQUIRED |
| **BWGNN** | `bwgnn.py` | **Beta-wavelet basis is not correctly applied.** `calculate_theta2` correctly reproduces the official Beta-distribution polynomial coefficients (matching `(x/2)^i (1-x/2)^(d-i) / B(i+1, d+1-i)`). But `PolyConv.forward` applies the recursion via `self.propagate(...)` with `aggr='add'` and `message = x_j` — i.e. **raw adjacency `A x`**, with no normalisation and, critically, **no Laplacian** `L = I - D^{-1/2} A D^{-1/2}`. The polynomial is therefore evaluated in the wrong operator basis, so it is not a beta wavelet. Additionally every `PolyConv` is built with `lin=False`, leaving `self.linear`/`self.activation` dead. | REIMPLEMENT_REQUIRED |
| **DGA-GNN** | `dga.py` | **Not DGA-GNN.** DGA-GNN's contributions are decision-tree-based **dynamic grouping** of nodes by feature binning and **bidirectional** grouped aggregation. `dga.py` is a GraphSAGE-mean layer (`fc_self(x) + fc_neigh(mean(x_j))`). No grouping, no decision tree, no bidirectional pass. | REIMPLEMENT_REQUIRED |
| **PMP** | `pmp.py:LASAGE_S` | **Partial.** PMP's core idea — partitioning message passing so that heterophilic (fraud) and homophilic (benign) neighbours get *different* transformations — is present: `fc_neigh_fraud` and `fc_neigh_benign` combined by a learned per-node gate `sigmoid(fc_balance(x))`. However the partition is applied to the *aggregated* mean rather than per-neighbour before aggregation, which is weaker than true partitioning. Naming (`LASAGE_S`) suggests derivation from a different codebase. | ADAPTER_REQUIRED |

**Consequence.** Reproducing "the baselines used in the FLAG paper" and "the official baselines"
are two different experiments:

- **FLAG-as-published** used the implementations above. Reproducing the paper's *numbers* means
  running *these* files.
- **Fair baseline comparison** means running the baselines' *official* repositories.

This project must support both and must never label the first as the official baseline. Both are
carried as distinct `impl_source` values (`flag_bundled` vs `official`) in the experiment registry.

## 8. Licensing

The repository contains **no LICENSE file** and no per-file licence headers.
Under default copyright, absence of a licence means **no rights are granted** to copy, modify, or
redistribute. Practical consequences for this project:

- Keeping a local clone for research use is normal practice and is what we do.
- **Do not** redistribute `methods/flag/` as part of any published artefact of this project
  without first obtaining permission from the authors, or until a licence appears upstream.
- `methods/` is therefore **git-ignored** in this repository; `scripts/setup/fetch_methods.sh`
  re-creates it from upstream at the pinned SHA. Provenance is preserved without re-hosting.
- This is flagged for the maintainer as a genuine open item, not a resolved one.

## 9. Environment requirements inferred from API usage

No requirements file exists (GAP-6). Inferred **lower bounds** from the APIs actually called:

| Package | Evidence | Inferred constraint |
|---|---|---|
| `torch_geometric` | `GCNConv`, `GATConv`, `SAGEConv`, `MessagePassing`, `NeighborLoader`, `torch_geometric.typing.SparseTensor` | PyG >= 2.0 (`NeighborLoader` was added in 2.0) |
| `torch_scatter` | `dga.py:7` `scatter_mean` | separate compiled dependency; must match the torch build |
| `peft` | `LoraConfig`, `get_peft_model`, `PeftModel`, `TaskType` | any modern PEFT |
| `transformers` | `AutoModelForCausalLM` with `gemma-2` | **>= 4.42** (Gemma-2 support) |
| `sentence_transformers` | `SentenceTransformer("all-MiniLM-L6-v2")` | any 2.x |
| `sympy`, `scipy` | `bwgnn.py` `calculate_theta2` | any |
| `sklearn` | `f1_score`, `roc_auc_score`, `TSNE` | any |

Installed here: torch 2.1.0+cpu, transformers 4.57.6, sentence-transformers 2.6.1, sympy/scipy/sklearn
present. **Missing: `torch_geometric`, `torch_scatter`, `peft`.**

Note the tension: `torch 2.1.0` is old relative to `transformers 4.57.6`. Gemma-2 needs
transformers >= 4.42, which is satisfied. `torch_scatter` must be built against torch 2.1.0
specifically.

## 10. Memory requirements (for planning, not measured)

`gemma-2-9b-it` in `float16` is roughly 18.5 GB of weights alone. `train.py` additionally holds the
Sentence-BERT encoder and the GNN, and calls `model.generate(max_new_tokens=550)` with a KV cache.
**Full FLAG* fine-tuning is not feasible on this machine** (CPU-only, and fp16 is not meaningfully
supported on CPU). This is stated as a limitation, not worked around. See the Phase-15 split between
the "full reproduction environment" and the "CPU functional environment".

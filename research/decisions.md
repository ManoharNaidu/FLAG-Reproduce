# Research Decisions

Decisions that materially affect research validity, escalated to the maintainer
under Phase 44 and answered on **2026-09-09**. Each records the ambiguity, the
options, the decision, and what it commits the codebase to.

Engineering decisions that a careful colleague would just make are **not** here —
they live in the config files and in `research/implementation_matrix.md`.

---

## D-001 — How to reproduce the `+FLAG*` (fine-tuned) column

**Ambiguity.** The paper describes two-stage alternating fine-tuning with three
losses. The released `train.py` implements the loop structure, but its gradient
path to the LoRA parameters is **severed**: `model.generate()` is
non-differentiable, and the decode -> Sentence-BERT re-encode produces a fresh
leaf tensor, so `optimizer.step()` on the PEFT parameters is a no-op with
`.grad is None`. Verified by
`tests/integration/test_flag_upstream_claims.py::test_5_5_reencoded_embeddings_are_detached_leaves`.

Yet Table 4 reports `+FLAG*` beating `+FLAG` by an average of +0.84% F1-macro and
+0.42% AUC, so *something* differed between the two columns.

### DECISION: reproduce the released code as-is, and relabel honestly

Implement exactly what `train.py` does:

- the LLM is **effectively frozen** (LoRA parameters receive no gradient),
- the **GNN inner loop does train**, for `inner_epochs=10` per outer epoch, under
  the full three-term loss including the residual and orthogonality terms.

So `+FLAG*` is reproduced as *"additional GNN training epochs under the
residual + orthogonality regularisation, with the LLM frozen"* — **not** as LLM
fine-tuning.

**Consequences committed to.**

1. Every `flag_finetuned` result row carries
   `llm_finetuned: false` and
   `note: "upstream LoRA gradient path is severed; see research/decisions.md D-001"`.
2. Reported tables must **not** describe this variant as LLM fine-tuning. The
   variant key stays `flag_finetuned` for comparability with the paper's column,
   but its rendered label is qualified.
3. The LoRA machinery is still wired up faithfully (r=8, alpha=32,
   `["q_proj","v_proj"]`, dropout 0.1, AdamW lr 1e-4) so that a corrected gradient
   path can be switched on later without restructuring.
4. **Action for the maintainer:** contact the authors (Chuan Shi's group, BUPT
   GAMMA) to ask what produced the `+FLAG*` column. If they confirm a working
   gradient path existed, this decision is revisited and the alternative becomes a
   second, clearly-labelled variant. **Not yet done.**

**Rejected.** Designing our own differentiable LLM->GNN coupling was rejected for
now because it would substitute our design for the authors' and could not be
called a reproduction. It remains available as a labelled `REIMPLEMENTED` variant
if D-001's action item comes back positive.

---

## D-002 — What counts as "the baseline"

**Ambiguity.** FLAG's repository bundles its own PyG rewrites of five baselines,
and four of them are not the published algorithms:

| Baseline | What FLAG's version is missing |
|---|---|
| CARE-GNN | RL-based neighbour filtering, label-aware similarity, multi-relation aggregation — i.e. all three defining components |
| BWGNN | the normalised Laplacian; the polynomial is evaluated over **raw adjacency**, so it is not a beta wavelet |
| DGA-GNN | decision-tree dynamic grouping, bidirectional grouped aggregation |
| GAT | its second layer is a `SAGEConv`, not a `GATConv` |

Compounding this, **official CARE-GNN requires a multi-relation graph**, and
Reddit and Instagram are single-relation — which is presumably why FLAG's version
dropped those modules.

### DECISION: carry both lineages, never merged

| `impl_source` | What it reproduces | Where it runs |
|---|---|---|
| `flag_bundled` | **the FLAG paper's numbers** — Table 4 was produced by these files | Reddit, Instagram |
| `official` | **the baseline as its own authors published it** | the datasets it natively supports (Yelp, Amazon, T-Finance, T-Social, Elliptic) |

**Consequences committed to.**

1. `impl_source` is a **required** field on every result row. A run cannot be
   recorded without it.
2. The two lineages are **never averaged, merged, or placed in the same table
   column**. Comparison tables render them as separate rows.
3. A `flag_bundled` row is never described as CARE-GNN / BWGNN / DGA-GNN without
   qualification. Rendered labels are `CARE-GNN (FLAG's variant)` etc.
4. Where an official baseline cannot run on a dataset (official CARE-GNN on
   single-relation Reddit), the registry reports `NOT_COMPATIBLE` and **refuses**,
   rather than silently degrading the algorithm.
5. `research/reported_results.csv` comparisons are made against `flag_bundled`
   rows only, since those are what the paper's numbers came from.

**Rejected.** Reproducing only the paper's bundled versions would have left the
baseline comparison weak and slightly misleading. Faithfully reimplementing all
three for single-relation graphs was rejected as inventing variants the authors
never proposed — though it stays open as future work.

---

## D-003 — LLM stages target GPU only; GPUs rented on vast.ai

**Ambiguity.** `gemma-2-9b-it` needs ~18.5 GB in fp16 plus KV cache for
`max_new_tokens=550`. The paper reports an **A100 80 GB**. This development machine
has **no CUDA**, which blocks every `flag` and `flag_finetuned` variant.

### DECISION: implement the LLM stages for GPU only. No mock or substitute LLM.

The maintainer will rent GPUs on **vast.ai** for the LLM stages.

**Consequences committed to.**

1. **No stub LLM, and no small-LLM substitute.** The pipeline uses the real
   `google/gemma-2-9b-it`. We will not produce "FLAG-small" numbers, because they
   would not be comparable to Table 4 and would invite misreading.
2. **The GPU stage must be cleanly severable and its output portable.** This is
   the load-bearing architectural consequence of renting ephemeral instances. The
   pipeline splits at a hard boundary:

   ```
   [ CPU, local ]                  [ GPU, rented, ephemeral ]        [ CPU, local ]
   download -> benchmark build  ->  LLM text generation          ->  encode -> GNN
   -> semantic sampling             (discriminative + residual)       train -> eval
   -> export sampling cache         -> export LLM text cache
   ```

   The LLM stage consumes a self-contained input bundle and emits a
   self-contained, checksummed text cache. It never needs the graph, the labels,
   or the training loop. So a rented instance is used only for generation, and
   every downstream experiment re-runs locally from cache at no GPU cost.
   This also satisfies Phase 33 (LLM cost control) by construction.
3. **The LLM cache is a first-class, versioned artefact**, keyed by
   `{dataset_version, sampling_config, prompt_version, llm_model, decoding_params}`
   so it can be reused, audited and shipped between machines. Cache hits must
   never silently cross a prompt or model change.
4. **CPU support is retained for everything else** and is not weakened: dataset
   preprocessing, semantic sampling (Sentence-BERT MiniLM is small and CPU-fine),
   all `baseline` and `text` variants, all baseline GNN training, metrics and the
   full test suite. That capability is already verified — all 7 backbones
   forward+backward on CPU.
5. **Sentence-BERT encoding runs on either device.** It is not part of the GPU-only
   boundary; `all-MiniLM-L6-v2` is ~90 MB.
6. Deliverables this decision adds: `environment/gpu.yml`, `docker/gpu/`, a
   one-command remote bootstrap (`scripts/setup/install_gpu.sh`) that works on a
   bare vast.ai image, and `docs/vastai_gpu_workflow.md` covering instance
   selection, the gated-model login, cache export/import and checksum verification.
7. **Update, 2026-09-18: the GPU run has happened.** `google/gemma-2-9b-it` ran
   on a rented vast.ai instance; `flag` and `flag_finetuned` are `DONE` in
   `reproduction_status.md` for GAT, CARE-GNN, BWGNN and DGA-GNN on both
   Reddit and Instagram (208 `flag` runs, 200 `flag_finetuned` runs). GCN,
   GeniePath and PMP are not yet run under either variant. No placeholder
   numbers were ever generated for the unrun combinations.

**Note on the gated checkpoint.** `google/gemma-2-9b-it` requires accepting
Google's licence and an authenticated `HF_TOKEN`. On a rented instance the token
must be passed via environment, never committed. `.env` is git-ignored and
`.env.example` documents the variable.

---

## D-004 — Production LLM cache uses a reduced decode budget, not the paper-faithful one

**Ambiguity.** `enhance.py`'s `LLMConfig` defaults to the upstream-faithful
`max_new_tokens=550, truncate_chars=1200` (verified against `chat.py`). At those
settings, a full-corpus generation run (measured empirically on this vast.ai
instance, 2026-09-15, faithful pilot at `--limit 50`) costs roughly:

| dataset/kind | measured rate | full-corpus estimate |
|---|---|---|
| reddit residual | 10.2 s/subgraph | 18,389 subgraphs -> ~52h |
| reddit discriminative | ~similar (assumed) | ~51h |
| instagram discriminative | 50-66 s/subgraph | 7,946 subgraphs -> ~120h |
| instagram residual | ~similar (assumed) | ~120h |

-- roughly **10-14 GPU-days across the 4 (dataset x kind) combinations**, at
vast.ai A100 rates on the order of several hundred to ~$1,000. The maintainer
judged that cost disproportionate to a reproduction whose own protocol (D-003)
already tolerates partial LLM-text coverage (a subgraph whose generation fails
the format check falls back to raw-text embeddings, never a fabricated
substitute).

### DECISION: the production `cache/llm/` corpus was generated at `max_new_tokens=64, truncate_chars=300`

All four full-corpus caches actually committed to this repo used this reduced
budget, generated in ~9-20h per combination instead of the estimate above:

| dataset/kind | subgraphs | format success | **node coverage** |
|---|---:|---:|---:|
| reddit discriminative | 18,389 | 27.7% | 4.4% |
| reddit residual | 18,389 | 40.9% | 12.4% |
| instagram discriminative | 7,946 | 19.6% | 0.6% |
| instagram residual | 7,946 | 22.1% | 0.9% |

**Consequences committed to.**

1. `flagbench.experiments.runner._llm_embeddings_path` looks up the LLM cache
   using this reduced config (`PRODUCTION_LLM_CONFIG`), not `LLMConfig()`'s
   faithful default -- otherwise it would silently miss the corpus that was
   actually generated and refuse every `flag`/`flag_finetuned` run.
2. With coverage this low, `+FLAG` and `+FLAG*` results are, for the large
   majority of subgraphs, the documented raw-text fallback -- they are a much
   weaker test of the method than Table 4's, and this must be stated
   alongside any reported number, not left implicit in a coverage column
   nobody reads.
3. `RunResult.extra["feature_source"]` and the LLM cache's own
   `.manifest.json` (`stats.node_coverage`, `stats.subgraph_success_rate`)
   remain the source of truth per run; no result row claims full coverage.
4. **Reversible.** Regenerating at the faithful 550/1200 budget (decision
   D-003's original intent) only requires re-running
   `scripts.llm.generate_text` with `--force` (the cache key does not encode
   `--limit`, so a full run must force past any smaller cache at the same
   key) and re-pointing `PRODUCTION_LLM_CONFIG`. Nothing about the
   architecture changes.

**Rejected.** Keeping `+FLAG`/`+FLAG*` `BLOCKED` until a faithful-budget
corpus could be afforded was rejected: the reduced-budget corpus still
produces genuine (if low-coverage) `+FLAG` numbers under the same
never-substitute discipline as everything else in this project, and blocking
indefinitely was judged worse than reporting a clearly-labelled, weaker
result.

---

## Standing decisions (not escalated — recorded for traceability)

These were judgement calls made without escalation, because a defensible default
exists and both options remain runnable from config.

| ID | Item | Default | Alternative | Basis |
|---|---|---|---|---|
| S-001 | hidden dimension | **32** (`config: code`) | 64 (`config: paper`) | The code defaults to 32 and `dga.py`/`pmp.py` crash at any other value; the paper's own t-SNE section says embeddings are 32-dim. Both are run and reported. |
| S-002 | number of runs | **25** (`5 seeds x 5 inits`, per the paper) | 5 (as the code does) | The paper's protocol is the fairer one and costs only compute. |
| S-003 | split ratios | **10/10/80** | any | Inherited from GraphAdapter/GLBench; the FLAG paper states no split. Labelled as inherited, never as FLAG's. |
| S-004 | orthogonality loss | **paper Eq. 9** (squared dot) | signed cosine (as the code does) | The code's signed form is minimised by *anti*-alignment, not orthogonality — almost certainly not the intent. Both selectable. |
| S-005 | skip-GNN coverage | **per-backbone, as upstream** | uniform across all backbones | Upstream applies it to only 4 of 9 backbones; which produced Table 4 is UNKNOWN. Exposed as the `SG` ablation flag. |
| S-006 | F1 threshold policy | **validation-swept** (`linspace(0.05, 0.95, 19)`) | `argmax` (as the code does) | The paper says it follows BWGNN's setup, and BWGNN sweeps exactly this grid. The test set is never used to pick a threshold either way. |
| S-007 | GCN/GAT implementation | **PyG `GCNConv`/`GATConv`** | authors' TF1 repos | `tkipf/pygcn`'s README disclaims reproduction; GAT's author explicitly recommends PyG/DGL. FLAG also uses PyG. |
| S-008 | torch / PyG versions | **torch 2.3.1+cpu, PyG 2.3.1** | newer | torch 2.4.0+cpu returns **silently wrong** numbers on this machine; PyG >= 2.4 crashes in `SAGEConv`. See `compatibility_notes.md`. |

Any of S-001..S-008 becomes an escalated decision if evidence emerges that the
choice changes a conclusion rather than just a number.

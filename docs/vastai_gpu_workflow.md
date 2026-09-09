# GPU workflow (vast.ai)

Decision **D-003**: the LLM stage runs only on a GPU, on rented vast.ai
instances. No mock LLM and no smaller substitute — a "FLAG-small" number would
not be comparable to Table 4 and would invite misreading.

## The boundary

Because rented instances are ephemeral and billed by the hour, the pipeline
splits at one hard seam. Everything expensive happens **once**, on the GPU, and
produces a portable artefact.

```
 [ CPU, local, free ]                [ GPU, rented ]              [ CPU, local, free ]

 download GLBench                    LLM text generation           Sentence-BERT encode
 build 1:10 benchmark        --->    discriminative + residual --->  train every backbone
 semantic sampling (Eq. 3-4)         gemma-2-9b-it, fp16             every variant, every seed
 export input bundle                 export text cache               metrics, tables, figures
```

The GPU stage consumes node texts + subgraph definitions + prompts. It never
needs the graph structure, the labels, the splits or the training loop. So once
the text cache is back on your machine, **every downstream experiment replays
locally at no GPU cost** — including all 25 runs per cell, every ablation and
every hyper-parameter sweep.

This is also Phase 33 (LLM cost control) satisfied by construction rather than by
discipline.

## Before you rent anything

Do all of this locally. It costs nothing and catches every mistake that would
otherwise burn GPU-hours.

```bash
python -m scripts.download.glbench           --dataset all
python -m scripts.preprocess.build_benchmark --dataset all
python -m scripts.preprocess.encode_text     --dataset all
python -m scripts.preprocess.sample_subgraphs --dataset all          # SS cache

# Inspect the exact prompt that will be sent. Loads no model.
python -m scripts.llm.generate_text --dataset reddit --kind discriminative --dry-run
```

The dry run prints the assembled prompt and the cache key. Read it. A prompt bug
discovered after a full generation run is an expensive bug.

**Accept the Gemma licence now**, not on the instance:
<https://huggingface.co/google/gemma-2-9b-it> — the repo is gated. Create a read
token at <https://huggingface.co/settings/tokens>.

## Instance selection

| Requirement | Value |
|---|---|
| VRAM | **≥ 24 GB.** `gemma-2-9b-it` in fp16 is ~18.5 GB of weights, plus a KV cache for `max_new_tokens=550`. |
| Recommended | RTX 4090 (24 GB), A5000 (24 GB), A100 (40/80 GB) |
| Disk | ≥ 60 GB (weights ~18.5 GB + datasets ~750 MB + cache) |
| Image | any recent `pytorch/pytorch` CUDA image |

The paper reports deployment on an **A100 80 GB**, but that is for their online
serving setup; 24 GB is enough for batch generation at these settings.

Prefer an instance with good download bandwidth: pulling 18.5 GB of weights is
usually the largest single cost of a short run.

## On the instance

```bash
git clone <your fork of this repo> && cd "FLAG Reproduce"
bash scripts/setup/install_gpu.sh          # env + deps, verifies CUDA
bash scripts/setup/fetch_methods.sh        # upstream at pinned SHAs

export HF_TOKEN=hf_...
huggingface-cli login --token "$HF_TOKEN"
```

Ship the prepared inputs up rather than rebuilding them — the benchmark is built
from *your* seed, and rebuilding it on the instance with a different seed would
silently produce a different graph:

```bash
# from your machine
rsync -avz data/benchmark/ cache/sampling/ prompts/ \
      root@<host>:<port>/workspace/FLAG-Reproduce/
```

Smoke-test on a handful of subgraphs before committing to the full run:

```bash
python -m scripts.llm.generate_text --dataset reddit --kind discriminative --limit 20
```

Check the reported **node coverage**. Upstream discards any response whose line
count does not match the subgraph size, and so do we — repairing a malformed
response would invent text for a node. If coverage is low, the prompt or the
decoding settings need attention before you spend hours.

Then the real run:

```bash
python -m scripts.llm.generate_text --dataset all --kind both
```

`--kind both` produces discriminative **and** residual text. Residual text is
only needed for the fine-tuning losses, but generating both in one session is far
cheaper than renting a second instance later.

## Bring the cache home

```bash
# from your machine
rsync -avz root@<host>:<port>/workspace/FLAG-Reproduce/cache/llm/ cache/llm/
```

Verify integrity before you destroy the instance:

```bash
python - <<'PY'
import hashlib, json, pathlib
for m in sorted(pathlib.Path("cache/llm").glob("*.manifest.json")):
    meta = json.loads(m.read_text(encoding="utf-8"))
    blob = pathlib.Path(meta["output"])
    actual = hashlib.sha256(blob.read_bytes()).hexdigest()
    ok = actual == meta["output_sha256"]
    print(f"{'OK ' if ok else 'BAD'} {blob.name}  "
          f"coverage {meta['stats']['node_coverage']*100:.1f}%")
PY
```

Only then terminate the instance.

## Back on CPU

```bash
python -m scripts.train.run --dataset reddit,instagram --models all \
       --variants baseline,text,flag --seeds 5 --inits 5
python -m analysis.compare_reported --metric f1_macro
```

## Cost control

- **Generate once, train many.** The text cache is the expensive artefact; the
  GNN training is minutes on CPU. Never re-rent to re-train.
- **Cache the weights.** If you expect a second session, snapshot the instance or
  keep `HF_HOME` on a persistent volume; re-downloading 18.5 GB is pure cost.
- **`--limit` first, always.**
- The cache key covers dataset, sampling config, prompt hashes, model id and
  decoding parameters. Re-running with identical settings is a cache hit and
  costs nothing; changing a prompt correctly invalidates it.

## What is deliberately not provided

- **No mock LLM.** It would make the pipeline "run" while producing meaningless
  numbers.
- **No small-model substitute.** `Qwen2.5-0.5B` or `gemma-2-2b-it` would generate
  real text, but the resulting numbers would not be comparable to Table 4, and
  putting them beside the paper's would be misleading.
- **No CPU fallback for this stage.** It fails with an explanatory error instead.

Everything *else* runs on CPU, and does: dataset preparation, the 1:10
construction, semantic sampling, all `baseline` and `+text` experiments, every
metric, and the whole test suite.

## Known blocker beyond the GPU

Even with the text cache, `flag_finetuned` carries the unresolved question in
`research/decisions.md` **D-001**: upstream's LoRA gradient path is severed, so
the released code cannot fine-tune the LLM. Per your decision we reproduce the
code as-is and label the variant honestly (`llm_finetuned=false`) rather than
inventing a mechanism the authors may not have used. Contacting the authors
remains the open action item.

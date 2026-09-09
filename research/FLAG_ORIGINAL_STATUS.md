# FLAG Original Implementation — Execution Status

Phase 3 requires running the official implementation **as-is** before refactoring
anything. This file records what actually happened, command by command.

| Field | Value |
|---|---|
| Repository | https://github.com/BUPT-GAMMA/FLAG @ `cb83944ed8a8a9b070a3f5a167d363973369fc80` |
| Attempted | 2026-09-09 |
| Machine | Windows 11, AMD Ryzen 12 threads, **no CUDA** |
| Environment | `.venv-cpu` — python 3.11.9, torch 2.3.1+cpu, torch_geometric 2.3.1, transformers 4.44.2, sentence-transformers 3.0.1, peft 0.12.0 (`environment/cpu.lock.txt`) |

**There are no README commands to reproduce.** The repository ships no README, no
example invocations, no config files and no shell scripts. The commands below are
the only ones possible: run each of the seven executable scripts directly, which
is what their hardcoded module-scope paths imply.

---

## Summary

| # | Command | Level 1 result | Level 2 result | Blocking cause | Semantics changed? |
|---|---|---|---|---|---|
| 1 | `python test.py` | `ModuleNotFoundError: torch_scatter` | `ImportError: cannot import name 'ECELoss'` | **upstream defect** | no |
| 2 | `python test_dual.py` | `ModuleNotFoundError: torch_scatter` | `ImportError: cannot import name 'ECELoss'` | **upstream defect** | no |
| 3 | `python train.py` | `OSError: gemma-2-9b-it is not a local folder...` | — | missing model asset | no |
| 4 | `python train1.py` | `OSError: gemma-2-9b-it is not a local folder...` | — | missing model asset | no |
| 5 | `python chat.py` | `FileNotFoundError: 'Reddit/reddit1.pt'` | — | missing dataset | no |
| 6 | `python chat1.py` | `FileNotFoundError: 'Instagram/instagram.pt'` | — | missing dataset | no |
| 7 | `python encode.py` | `FileNotFoundError: 'Instagram/instagram.pt'` | — | missing dataset | no |

**Result: 0 of 7 entrypoints run.** Nothing in the official repository executes
end-to-end without work. Two fail for reasons that are **defects in the code
itself**, not missing assets.

---

## 1-2. `test.py` and `test_dual.py` — the GNN train/eval drivers

### Level 1 — unchanged

```
$ python test.py
Traceback (most recent call last):
  File ".../methods/flag/test.py", line 8, in <module>
    from dga import DGA
  File ".../methods/flag/dga.py", line 7, in <module>
    import torch_scatter
ModuleNotFoundError: No module named 'torch_scatter'
```

**Root cause.** `torch_scatter` is a compiled extension and is not installed. The
prebuilt wheel `torch_scatter==2.1.2+pt24cpu` *was* tried and **crashes the process**
(access violation) once PyTorch Geometric routes an aggregation through it — the
full bisection is in `compatibility_notes.md` section 5.

### Level 2 — environment fix, no source change

`src/compat/torch_scatter.py` provides a pure-PyTorch `scatter_mean` and is placed
on `PYTHONPATH`. `dga.py` remains byte-identical to upstream.

```
$ PYTHONPATH=src/compat python test.py
Traceback (most recent call last):
  File ".../methods/flag/test.py", line 18, in <module>
    from utils import FocalLoss, visualization, ECELoss
ImportError: cannot import name 'ECELoss' from 'utils' (.../methods/flag/utils.py)
```

**Root cause: a defect in the published code.** `utils.py` defines
`generate_homo`, `visualization`, `t_sne`, `causal_loss`, `non_causal_loss`,
`orthogonal_loss`, `FocalLoss` and `remove_empty_lines`. **`ECELoss` does not
exist anywhere in the repository.** It is not merely imported: `test.py:108` and
`:170` call it and print its value in the final summary, so the symbol is
genuinely required.

**Consequence.** Both evaluation drivers are **non-executable as published**. No
amount of environment setup or data provisioning fixes this. Whatever produced
the paper's Table 4 was not this exact file.

**Workaround status: NOT APPLIED.** Supplying an `ECELoss` would mean inventing a
binning scheme (number of bins, equal-width vs equal-mass) that the paper never
specifies. ECE is not a metric the paper reports, so it is not needed for the
reproduction target. We will implement ECE in our own metrics module with the
choice documented, and leave upstream untouched.

**Semantics changed by the Level-2 fix: NO.** The shim is equivalence-tested
against a dense reference (`tests/unit/test_compat_torch_scatter.py`, 9/9).

---

## 3-4. `train.py` and `train1.py` — the LoRA fine-tuning drivers

### Level 1 — unchanged

```
$ python train.py
OSError: gemma-2-9b-it is not a local folder and is not a valid model identifier
listed on 'https://huggingface.co/models'
```

**Root cause.** `train.py:44` hardcodes `model_name = "gemma-2-9b-it"`, which is a
**bare directory name, not a HuggingFace repo id**. The authors evidently had the
weights in a local folder of that name next to the scripts. The real id is
`google/gemma-2-9b-it`, which is additionally **gated** — it requires accepting
Google's licence and an authenticated `huggingface-cli login`.

**Not attempted, deliberately.** Downloading it would be ~18.5 GB for a model that
**cannot run on this machine**: no CUDA, and fp16 inference on CPU is not viable.
`train.py:46` also calls `.cuda()` on the model unconditionally, so it would fail
immediately after loading regardless.

**Blockers behind this one.** Even with the weights present, `train.py` would fail
next at line 50 (`torch.load('Reddit/reddit2.pt')` — dataset absent), then line 58
(`Reddit/model_lora2/gnn.pth` — a pre-trained GNN checkpoint that no shipped
script writes to that path), then line 152 with a `TypeError` from indexing a
tuple with a bool tensor (`flag_code_audit.md` 5.2).

**Most important:** even once running, `train.py` **cannot fine-tune the LLM**.
The decode -> Sentence-BERT re-encode round trip produces a fresh leaf tensor, so
no gradient reaches the LoRA parameters and `optimizer.step()` is a no-op. Proven
in `flag_code_audit.md` 5.5 with a passing test. This is a property of the code,
not of our environment.

---

## 5-7. `chat.py`, `chat1.py`, `encode.py` — data-preparation stages

### Level 1 — unchanged

```
$ python chat.py
FileNotFoundError: [Errno 2] No such file or directory: 'Reddit/reddit1.pt'

$ python chat1.py
FileNotFoundError: [Errno 2] No such file or directory: 'Instagram/instagram.pt'

$ python encode.py
FileNotFoundError: [Errno 2] No such file or directory: 'Instagram/instagram.pt'
```

**Root cause.** No dataset files are shipped and no download instructions exist
(`flag_code_audit.md` GAP-2). Provenance had to be established externally — see
`dataset_notes.md`.

**The deeper blocker is not the datasets.** All three scripts also load
`*_sampler*.pt` files, and **nothing in the repository creates them**
(`flag_code_audit.md` GAP-1). The semantic-similarity neighbour sampler — a core
contribution of the paper — has no source here. Obtaining `reddit.pt` from GLBench
therefore does **not** unblock these scripts; the sampler must be reimplemented
from the paper's Eq. 3-4 first.

### An incidental finding

`encode.py:24` is `criterion = torch.nn.CrossEntropyLoss().cuda()` at module
scope, before any `torch.load`. On this CUDA-free machine it **does not raise** —
verified:

```
CrossEntropyLoss().cuda() -> SUCCEEDED (no tensors to move, silent no-op)
torch.zeros(3).cuda()     -> raises AssertionError
```

`nn.Module.cuda()` only moves parameters and buffers, and `CrossEntropyLoss` has
neither. So this particular `.cuda()` is silently harmless, and the script gets
further than a naive reading of `flag_code_audit.md` 5.11 would suggest. The other
~40 `.cuda()` calls operate on real tensors and would fail. Recorded because it
changes *where* the CPU port has to intervene.

---

## Escalation ladder applied (Phase 36)

| Level | Description | Used? |
|---|---|---|
| 1 | Run original code unchanged | **yes** — all 7 scripts, results above |
| 2 | Fix environment/dependency compatibility without altering the algorithm | **yes** — venv pins + `torch_scatter` shim. Equivalence-tested. |
| 3 | Fix obvious repository bugs while preserving behaviour | **not yet** — `ECELoss`, the `train.py` tuple unpack, and the shadowed accumulation counter are candidates, each recorded with the semantic question it raises |
| 4 | Wrap/adapt the official implementation into the unified framework | not yet |
| 5 | Reimplement only the missing component | **unavoidable** for the sampler, the 1:10 construction, splitting, early stopping and metrics — these have no upstream source at all |

We did **not** jump to Level 5. The two components that require it do so because
they are absent from the repository, not because adapting them was inconvenient.

---

## What this means for the reproduction

1. **The published artefact does not run.** Two of seven entrypoints fail on
   defects internal to the code; the rest fail on assets that were never released.
   This is stated as an observation about the artefact, not a judgement of the work.
2. **The paper's numbers cannot be regenerated by executing this repository.** Any
   reproduction necessarily involves reconstructing missing components, so exact
   agreement with Table 4 should not be expected and will not be claimed.
3. **The repository is still highly valuable**, and remains the authority for
   every implementation detail the paper omits: the LoRA configuration,
   lambda_1 = lambda_2 = 0.1, the Sentence-BERT checkpoint, the concrete Gemma
   checkpoint, the prompts, the truncation limits, and the attention-fusion design.
   Those are recorded in `paper_notes.md` and are used as documented defaults.
4. **No run has produced a metric.** Nothing in this project is marked reproduced.
   `reproduction_status.md` is the ledger.

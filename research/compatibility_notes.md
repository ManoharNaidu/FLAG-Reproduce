# Compatibility Notes

Environment findings, the dependency bisection, and the resulting pins.

**Machine under test**

| Field | Value |
|---|---|
| OS | Windows 11 Pro, `Windows-10-10.0.26200-SP0` |
| CPU | AMD64 Family 23 Model 104 (AMD Ryzen, Zen 2), 12 logical cores |
| CUDA | **none** — `nvidia-smi` not present, `torch.cuda.is_available() == False` |
| Date | 2026-09-09 |

Everything below was established by running code, not by reading changelogs.

---

## 1. Headline findings

1. **`torch==2.4.0+cpu` is broken on this machine.** It produces hard native
   crashes (`0xC0000005` access violation, `0xC0000374` heap corruption) in
   ordinary operations, **and silently returns wrong numerical results** when
   more than one thread is used. `torch==2.3.1+cpu` and `2.2.2+cpu` are both
   clean at the full 6 threads. **Pinned: `torch==2.3.1+cpu`.**
2. **`torch_geometric>=2.4` crashes on this machine** (2.4.0, 2.5.3 and 2.6.1 all
   fail in `SAGEConv`). `2.3.1` is clean. **Pinned: `torch_geometric==2.3.1`.**
3. **The prebuilt `torch_scatter` wheel destabilises PyG.** Replaced with a
   native-torch shim (`src/flagbench/compat/torch_scatter.py`), equivalence-tested.
4. **The pre-existing global Python environment is internally inconsistent** and
   cannot run this project. Isolated venv required.

---

## 2. The pre-existing global environment is broken

The machine's global `python` (3.11.9) already had `torch 2.1.0+cpu` **and**
`transformers 4.57.6`. Those are mutually incompatible: transformers 4.57 calls
`torch.utils._pytree.register_pytree_node`, which only exists from torch 2.2
(torch 2.1 has `_register_pytree_node`).

The failure is non-obvious because it surfaces through an unrelated import chain:

```
import torch_geometric
  -> torch_geometric.isinstance -> torch._dynamo -> torch.onnx
  -> torch.onnx._internal.fx.patcher -> import transformers
  -> AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

So merely importing PyG blows up. Conclusion: **do not install into the global
environment.** This project uses an isolated venv, `.venv-cpu/`, created
*without* `--system-site-packages` (the first attempt used it and inherited the
broken pair).

---

## 3. The torch 2.4.0 bisection

### 3.1 Symptom

Running the backbone smoke checks produced Windows fatal exceptions with no
Python traceback. `faulthandler` located them in different primitives on
different runs:

| Run | Fault site |
|---|---|
| `SAGEConv` forward | `torch_geometric/nn/dense/linear.py:147` -> `F.linear` |
| `DGA` forward | `torch_geometric/nn/conv/message_passing.py:272` -> `_lift` -> `index_select` |

Crashing in *different* elementary operations is the signature of memory
corruption or a bad threading runtime, not of a logic bug in either library.

### 3.2 The decisive experiment

`tests/unit/test_compat_torch_scatter.py`, torch 2.4.0+cpu, PyG 2.3.1:

| Configuration | Result |
|---|---|
| no env var (6 threads), 6 consecutive runs | **0/6 pass** — all crash |
| `OMP_NUM_THREADS=1`, 6 consecutive runs | **6/6 pass** |
| `OMP_NUM_THREADS=2` | crash **and a wrong numeric result** before crashing |
| `torch.set_num_threads(1)` in-process, no env var, 4 runs | **0/4 pass** |

Two things matter here.

**(a) The wrong answer is worse than the crash.** At `OMP_NUM_THREADS=2` the shim
equivalence test reported `mismatch at groups=3 feat=16` — a scatter-mean result
that disagreed with a dense reference — *and then* corrupted the heap. A build
that silently computes wrong numbers would invalidate every experiment without
any visible failure. This is the reason the pin is treated as a correctness
requirement, not a performance tweak.

**(b) `torch.set_num_threads(1)` does not help.** Only the environment variable
works, because it must be read before the OpenMP runtime initialises — by the
time Python code runs, it is too late. That ruled out fixing this from inside the
framework and pushed us to look for a good torch build instead.

### 3.3 Resolution — a better build, not a crippled one

Rather than force single-threaded CPU everywhere (a ~6x throughput loss), we
tested other builds at **default threading, no env var**:

| torch | threads | runs | result |
|---|---|---|---|
| 2.4.0+cpu | 6 | 6 | **0/6** |
| **2.3.1+cpu** | 6 | 3 | **3/3 pass** |
| 2.2.2+cpu | 6 | 3 | 3/3 pass |

**`torch==2.3.1+cpu` is pinned** — the newest build that is correct here, keeping
full multi-threaded CPU throughput.

`torch.__config__.parallel_info()` on the broken build reported
`omp_get_max_threads() : 1` even while running 6 threads, so **that diagnostic is
not trustworthy on this machine** — do not use it to decide whether the guard is
active.

### 3.4 Scope of this finding

This is a **machine-and-build-specific** finding (AMD Zen 2 / Windows 11 /
torch 2.4.0+cpu wheel, which bundles MKL 2024.2). It is *not* a claim that
torch 2.4.0 is broken generally. On a different machine the pin may be
unnecessary. The reason it is documented at length is that the failure mode
includes **silent numerical error**, so any environment change must be re-validated
rather than assumed.

**Guard:** `scripts/smoke_test.py` re-runs the numerical equivalence check on
every invocation and exits non-zero on mismatch, so a bad environment is caught
before it can produce results.

---

## 4. The torch_geometric bisection

Independently of the torch issue, PyG versions were probed with the same three
convolutions (`GCNConv`, `GATConv`, `SAGEConv`) on a 12-node CPU graph:

| torch_geometric | GCNConv | GATConv | SAGEConv | Verdict |
|---|---|---|---|---|
| 2.6.1 | ok | ok | **crash** | rejected |
| 2.5.3 | ok | ok | **crash** (heap corruption) | rejected |
| 2.4.0 | ok | ok | **crash** (access violation) | rejected |
| **2.3.1** | ok | ok | **ok** | **PINNED** |

`SAGEConv` is not optional here: FLAG's `models.py:GAT` uses `SAGEConv` as its
**second layer** (see `flag_code_audit.md` section 7), and `models.py:GraphSAGE`
uses it throughout.

Corroboration: **PMP's official `requirements.txt` also pins
`torch_geometric==2.3.1`**, so this pin is consistent with at least one baseline's
published environment rather than being an isolated workaround.

---

## 5. `torch_scatter` — replaced with a shim

`methods/flag/dga.py:7` does `import torch_scatter` and calls `scatter_mean`.

The prebuilt wheel `torch_scatter==2.1.2+pt24cpu` installs and passes a
standalone `scatter_mean` call, **but crashes once PyG routes an aggregation
through it** (PyG <= 2.3 prefers `torch_scatter` when `WITH_TORCH_SCATTER` is true).
Uninstalling it and letting PyG fall back to `torch.Tensor.scatter_reduce_`
removes that failure mode.

Since `dga.py` imports it directly, the module name still has to resolve. Rather
than edit upstream source, `src/flagbench/compat/torch_scatter.py` provides a pure-PyTorch
implementation and `src/flagbench/compat/` is prepended to `sys.path`.

- **Classification: LEVEL 2** (Phase 36) — environment/dependency fix, no change
  to algorithm behaviour. `dga.py` is byte-identical to upstream.
- **Verified**, not asserted: `tests/unit/test_compat_torch_scatter.py` checks
  `scatter_mean`/`scatter_sum` against an independent dense group-by reference
  across several shapes, checks that empty groups yield 0 (torch_scatter's
  convention) rather than NaN, checks gradient values (`1/|group|` per row), and
  runs upstream `DGA` end-to-end through the shim. 9/9 pass.
- Only the functions FLAG actually uses are implemented. Any other attribute
  raises `AttributeError` with a pointer to this file, so the shim can never
  silently cover more surface than has been audited.

---

## 6. The pinned CPU environment

```
python                 3.11.9
torch                  2.3.1+cpu     # 2.4.0 is BROKEN here - see section 3
torch_geometric        2.3.1         # >=2.4 crashes in SAGEConv - see section 4
torch_scatter          NOT INSTALLED # shimmed - see section 5
numpy                  1.26.4        # <2 for compatibility with this torch
transformers           4.44.2        # >=4.42 required for Gemma-2; 4.57 needs torch>=2.2 but is untested here
sentence-transformers  3.0.1
peft                   0.12.0
scikit-learn, scipy, sympy, pandas, matplotlib, pyyaml, tqdm
```

Full freeze: **`environment/cpu.lock.txt`** (57 packages, exact versions).

Verified on this environment:

| Check | Result |
|---|---|
| `tests/integration/test_flag_upstream_claims.py` | **36/36** |
| `tests/unit/test_compat_torch_scatter.py` | **9/9** |
| All 7 backbones forward + backward on CPU at `hidden=32` | **pass** |
| `torch.cuda.is_available()` | `False` — CPU path is genuinely exercised, not merely offered |

---

## 7. FLAG's own dependency requirements

FLAG ships **no** requirements file (`flag_code_audit.md` GAP-6). Lower bounds
inferred from the APIs it actually calls:

| Package | Evidence | Constraint |
|---|---|---|
| `torch_geometric` | `GCNConv`, `GATConv`, `SAGEConv`, `MessagePassing`, `NeighborLoader`, `torch_geometric.typing.SparseTensor` | `>=2.0` (`NeighborLoader` added in 2.0) |
| `torch_scatter` | `dga.py` `scatter_mean` | must match the torch build |
| `transformers` | `AutoModelForCausalLM` with `gemma-2-9b-it` | **`>=4.42`** (Gemma-2 support) |
| `peft` | `LoraConfig`, `get_peft_model`, `PeftModel`, `TaskType` | any modern |
| `sentence_transformers` | `SentenceTransformer("all-MiniLM-L6-v2")` | any 2.x/3.x |
| `sympy`, `scipy` | `bwgnn.calculate_theta2` | any |

Our pins satisfy every one of these.

---

## 8. Per-baseline environment conflicts (Phase 17)

The baselines' published environments genuinely conflict with each other and with
ours. This is the evidence for **Option B — isolated per-method environments with
a unified orchestration layer**.

| Method | Published environment | Conflicts with our base |
|---|---|---|
| FLAG | unspecified | — |
| CARE-GNN | `torch>=1.4`, plain PyTorch, no DGL/PyG | compatible; needs no graph library |
| BWGNN | `pytorch 1.9.0`, **`dgl 0.8.1`** | **DGL not installed**; DGL 0.8 will not build against torch 2.3 |
| DGA-GNN | `torch==1.13.1`, `dgl==1.1.2`, `pytorch-lightning==1.9.4`, `hydra-core==1.3.2`, `wandb`, `toad` | **hard conflict** — torch 1.13 vs our 2.3.1; PL 1.9 is incompatible with modern torch |
| PMP | `torch==2.0.1`, `dgl==1.1.1+cu118`, `torch_geometric==2.3.1`, `+cu118` pins | **hard conflict** — CUDA-only pins; `pip install -r requirements.txt` fails outright without the DGL/PyG wheel indexes |
| GeniePath (community) | `torch==1.0.1`, `torch-geometric==1.1.2` | **hard conflict** — PyG 1.x API is incompatible with 2.x |
| GCN (tkipf) | `tensorflow>=1.15,<2.0` | **hard conflict** — TF1, dead on modern Python |
| GAT (PetarV-) | Python 3.5.2, `tensorflow-gpu==1.6.0`, CUDA 9 | **hard conflict** |

**Decision: Option B.** Forcing all of these into one environment would require
downgrading torch to 1.x, which would break FLAG itself and lose Gemma-2 support.
The top-level framework stays unified; individual methods get their own
environments and are invoked through a subprocess boundary.

`environment/` therefore holds one file per method, plus `base`/`cpu`/`gpu`.
Methods whose official environment cannot be built on this machine (all the
CUDA-pinned and TF1 ones) are marked **BLOCKED** with the reason recorded, rather
than being quietly replaced by a modern rewrite.

### 8.1 Which baselines can run on this CPU-only machine

| Method | Official code on CPU? | Basis |
|---|---|---|
| BWGNN | **likely yes** | `main.py` has no `.cuda()` calls at all; computes AUC on CPU |
| CARE-GNN | **partial** | hardcodes `os.environ["CUDA_VISIBLE_DEVICES"]="0"` at import; harmless without a GPU but needs checking |
| DGA-GNN | **no** | every config sets `usegpu: True, gpuid: 0` |
| PMP | **no** | `torch.cuda.set_device(args.gpu_id)` unconditionally at startup |
| GCN / GAT (TF1) | **no** | TensorFlow 1.x will not install on Python 3.11 |
| FLAG | **no, as shipped** | ~40 hardcoded `.cuda()` calls |

None of these has been *executed* yet — the column records what the source
implies. Each will be confirmed or corrected in `reproduction_status.md` as it is
attempted.

---

## 9. GPU status

**No GPU is present on this machine.** Every GPU claim in this project is
therefore **UNTESTED**, and is recorded as such rather than as "supported".

The GPU path will be written (device is threaded through configuration, never
hardcoded) and covered by tests that **skip** when CUDA is absent — a skipped test
is never reported as a pass.

The `gemma-2-9b-it` requirement is the binding constraint for full reproduction:
~18.5 GB of fp16 weights plus KV cache for `max_new_tokens=550`. The paper reports
deployment on an **A100 80 GB**. Full FLAG* fine-tuning is out of reach here, and
is documented as such rather than approximated.

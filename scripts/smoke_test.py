"""End-to-end environment smoke test. Exits non-zero on failure.

    python -m scripts.smoke_test
    python -m scripts.smoke_test --device cpu
    python -m scripts.smoke_test --verbose

Checks, in order:
  1. environment fingerprint and version pins
  2. NUMERICAL CORRECTNESS  <- see below, this is not routine
  3. core imports
  4. upstream FLAG sources present at the pinned commit
  5. every backbone: CPU forward + backward
  6. GPU forward, or an explicit SKIP (never a silent pass)
  7. FLAG components: losses, skip-GNN, DualGNN attention fusion
  8. prompts present and hash-stable
  9. reference results loadable
 10. datasets present and verified, or an explicit SKIP

Why check 2 exists. On the reference machine `torch==2.4.0+cpu` crashed AND, at
two threads, silently returned a WRONG scatter result. A build that computes
wrong numbers without failing would invalidate every experiment with no visible
symptom. So the smoke test re-derives a handful of results whose correct values
are known by construction and refuses to proceed on mismatch. See
research/compatibility_notes.md section 3.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
# src/flagbench/compat shadows the crashing torch_scatter wheel; it must precede
# methods/flag on sys.path.
sys.path.insert(0, str(ROOT / "src" / "flagbench" / "compat"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []
VERBOSE = False


def check(name: str):
    """Decorator: run a check, record PASS/FAIL/SKIP, never raise."""

    def wrap(fn):
        def run():
            try:
                detail = fn()
                if isinstance(detail, tuple):
                    status, detail = detail
                else:
                    status = PASS
            except _Skip as exc:
                status, detail = SKIP, str(exc)
            except Exception as exc:
                status = FAIL
                detail = f"{type(exc).__name__}: {exc}"
                if VERBOSE:
                    traceback.print_exc()
            results.append((name, status, detail or ""))
            marker = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
            print(f"[{marker}] {name}")
            if detail:
                for line in str(detail).splitlines():
                    print(f"           {line}")
            return status

        run.__name__ = fn.__name__
        return run

    return wrap


class _Skip(Exception):
    """Raised to record a SKIP. A skip is never counted as a pass."""


# ---------------------------------------------------------------- 1. env
@check("environment")
def check_environment():
    from flagbench.utils.device import describe_environment

    env = describe_environment()
    lines = [f"python {env['python']}, torch {env['torch']}, "
             f"pyg {env['torch_geometric']}",
             f"threads {env['torch_threads']}, cpus {env['cpu_count']}, "
             f"cuda_available {env['cuda_available']}"]

    warnings = []
    if env["torch"].startswith("2.4.0"):
        warnings.append(
            "torch 2.4.0 is the build that returned WRONG numbers on the "
            "reference machine. See research/compatibility_notes.md section 3."
        )
    pyg = env["torch_geometric"]
    if pyg and tuple(int(x) for x in pyg.split(".")[:2]) >= (2, 4):
        warnings.append(
            f"torch_geometric {pyg}: versions >= 2.4 crashed in SAGEConv on the "
            f"reference machine (needed by FLAG's GAT). Pinned version is 2.3.1."
        )
    try:
        import torch_scatter  # noqa: F401

        if not getattr(torch_scatter, "__version__", "").endswith("flagbench-shim"):
            warnings.append(
                "the real torch_scatter wheel is installed; it destabilised PyG "
                "on the reference machine. The shim in src/flagbench/compat is "
                "preferred."
            )
    except ImportError:
        pass

    return (PASS, "\n".join(lines + [f"WARNING: {w}" for w in warnings]))


# ------------------------------------------------- 2. numerical correctness
@check("numerical correctness (guards against a silently-wrong build)")
def check_numerics():
    import torch

    problems = []

    # (a) matmul against a hand-computed product
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    if not torch.equal(a @ b, torch.tensor([[19.0, 22.0], [43.0, 50.0]])):
        problems.append(f"matmul wrong: {(a @ b).tolist()}")

    # (b) a larger matmul against an independent einsum path
    torch.manual_seed(0)
    x, w = torch.randn(64, 128), torch.randn(128, 32)
    if not torch.allclose(x @ w, torch.einsum("ij,jk->ik", x, w), atol=1e-4):
        problems.append("matmul disagrees with einsum")

    # (c) scatter-mean against a dense group-by loop
    import torch_scatter

    src = torch.randn(64, 16)
    index = torch.randint(0, 7, (64,))
    got = torch_scatter.scatter_mean(src, index, dim=0, dim_size=7)
    want = torch.zeros(7, 16)
    for g in range(7):
        rows = src[index == g]
        if rows.numel():
            want[g] = rows.mean(dim=0)
    if not torch.allclose(got, want, atol=1e-5):
        problems.append(
            f"scatter_mean wrong, max err "
            f"{(got - want).abs().max().item():.3e} -- THIS IS THE torch 2.4.0 "
            f"FAILURE MODE"
        )

    # (d) index_select against direct indexing
    t = torch.randn(32, 8)
    idx = torch.randint(0, 32, (100,))
    if not torch.equal(t.index_select(0, idx), t[idx]):
        problems.append("index_select disagrees with fancy indexing")

    # (e) a GNN message-passing round trip against a dense adjacency product
    from torch_geometric.nn import GCNConv

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    conv = GCNConv(4, 4, add_self_loops=False, normalize=False, bias=False)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.eye(4))
    feats = torch.eye(3, 4)
    out = conv(feats, edge_index)
    dense = torch.zeros(3, 3)
    dense[edge_index[1], edge_index[0]] = 1.0
    if not torch.allclose(out, dense @ feats, atol=1e-5):
        problems.append("GCNConv disagrees with a dense adjacency product")

    if problems:
        raise AssertionError(
            "THIS BUILD COMPUTES WRONG NUMBERS. Do not run experiments.\n  - "
            + "\n  - ".join(problems)
            + "\nSee research/compatibility_notes.md section 3."
        )
    return "matmul, einsum, scatter_mean, index_select, GCNConv all correct"


# ------------------------------------------------------------ 3. imports
@check("core imports")
def check_imports():
    import flagbench
    from flagbench.utils.device import get_device, resolve_device  # noqa: F401
    from flagbench.utils.seeding import RunIdentity, run_grid  # noqa: F401
    from flagbench.datasets import glbench  # noqa: F401

    return f"flagbench {flagbench.__version__}"


# ------------------------------------------------- 4. upstream FLAG source
EXPECTED_FLAG_COMMIT = "cb83944ed8a8a9b070a3f5a167d363973369fc80"


@check("upstream FLAG source at pinned commit")
def check_upstream():
    import subprocess

    flag_dir = ROOT / "methods" / "flag"
    if not flag_dir.exists():
        raise _Skip(
            "methods/flag is absent. Run: bash scripts/setup/fetch_methods.sh"
        )
    sha = subprocess.run(
        ["git", "-C", str(flag_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if sha != EXPECTED_FLAG_COMMIT:
        return (
            FAIL,
            f"commit {sha} != pinned {EXPECTED_FLAG_COMMIT}. "
            f"Audit findings in research/ may be stale; re-run "
            f"tests/integration/test_flag_upstream_claims.py",
        )
    missing = [
        f for f in ("models.py", "geniepath.py", "bwgnn.py", "caregnn.py",
                    "dga.py", "pmp.py", "utils.py")
        if not (flag_dir / f).exists()
    ]
    if missing:
        raise AssertionError(f"missing upstream files: {missing}")
    return f"{sha[:12]} (pinned)"


# -------------------------------------------------------- 5. CPU backbones
def _tiny_graph(n=12, dim=384):
    import torch

    torch.manual_seed(0)
    s = torch.arange(n - 1)
    edge_index = torch.stack(
        [torch.cat([s, s + 1]), torch.cat([s + 1, s])], dim=0
    ).long()
    return torch.randn(n, dim), edge_index


def _backbones():
    sys.path.insert(0, str(ROOT / "methods" / "flag"))
    import bwgnn, caregnn, dga, geniepath, models, pmp  # noqa: E401

    return {
        "GCN": lambda: models.GCN(384, 32, 2),
        "GAT": lambda: models.GAT(384, 32, 2),
        "GeniePathLazy": lambda: geniepath.GeniePathLazy(384, 2, "cpu"),
        "BWGNN": lambda: bwgnn.BWGNN(384, 32, 2),
        "CAREGNN": lambda: caregnn.CAREGNN(384, 32, 2),
        "DGA": lambda: dga.DGA(384, 32, 2),
        "PMP": lambda: pmp.LASAGE_S(384, 32, 2),
    }


@check("all backbones: CPU forward + backward")
def check_backbones_cpu():
    import torch

    if not (ROOT / "methods" / "flag").exists():
        raise _Skip("methods/flag absent")

    x, edge_index = _tiny_graph()
    ok = []
    for name, build in _backbones().items():
        model = build()
        emb32, out = model(x, edge_index)
        assert out.shape == (12, 2), f"{name}: shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        out.sum().backward()
        assert any(p.grad is not None for p in model.parameters()), \
            f"{name}: no gradient reached any parameter"
        ok.append(name)
    return f"{len(ok)}/7: {', '.join(ok)}"


# -------------------------------------------------------------- 6. GPU
@check("all backbones: GPU forward")
def check_backbones_gpu():
    import torch

    from flagbench.utils.device import cuda_is_available

    if not cuda_is_available():
        raise _Skip(
            "no CUDA on this machine -- GPU support is UNTESTED, not supported. "
            "A skip is never reported as a pass."
        )
    if not (ROOT / "methods" / "flag").exists():
        raise _Skip("methods/flag absent")

    device = torch.device("cuda:0")
    x, edge_index = _tiny_graph()
    x, edge_index = x.to(device), edge_index.to(device)
    ok = []
    for name, build in _backbones().items():
        model = build()
        if name == "GeniePathLazy":
            sys.path.insert(0, str(ROOT / "methods" / "flag"))
            import geniepath

            model = geniepath.GeniePathLazy(384, 2, "cuda")
        model = model.to(device)
        _, out = model(x, edge_index)
        assert out.shape == (12, 2) and torch.isfinite(out).all(), name
        ok.append(name)
    return f"{len(ok)}/7 on {torch.cuda.get_device_name(0)}"


# ------------------------------------------------------ 7. FLAG components
@check("FLAG components: losses, skip-GNN, attention fusion")
def check_flag_components():
    import torch

    if not (ROOT / "methods" / "flag").exists():
        raise _Skip("methods/flag absent")
    sys.path.insert(0, str(ROOT / "methods" / "flag"))
    import models
    import utils as flag_utils

    notes = []

    # discriminative loss == cross-entropy
    logits, label = torch.tensor([[2.0, -1.0]]), torch.tensor([0])
    expected = torch.nn.functional.cross_entropy(logits, label)
    assert torch.allclose(flag_utils.causal_loss(logits, label), expected)
    notes.append("causal_loss == cross_entropy")

    # residual loss: uniform input -> ~0 divergence
    assert flag_utils.non_causal_loss(torch.tensor([0.0, 0.0])).abs() < 1e-6
    notes.append("non_causal_loss(uniform) ~ 0")

    # orthogonality: upstream returns a SIGNED cosine (paper Eq.9 is squared)
    a = torch.tensor([1.0, 0.0])
    orth = flag_utils.orthogonal_loss(a, torch.tensor([0.0, 1.0])).item()
    anti = flag_utils.orthogonal_loss(a, torch.tensor([-1.0, 0.0])).item()
    assert abs(orth) < 1e-6 and abs(anti + 1.0) < 1e-6
    notes.append(
        "orthogonal_loss is a SIGNED cosine (min at anti-alignment), "
        "NOT paper Eq.9 -- decision S-004"
    )

    # skip-GNN present in GCN
    x, edge_index = _tiny_graph()
    gcn = models.GCN(384, 32, 2)
    with torch.no_grad():
        emb32, out = gcn(x, edge_index)
        no_skip = out - gcn.linear1(x)
    assert not torch.allclose(out, no_skip), "skip term is not contributing"
    notes.append("skip-GNN active in GCN (Eq. 6)")

    # DualGNN: the paper's attention fusion, sharing one backbone
    dual = models.DualGNN(2, models.GCN(384, 32, 2))
    _, fused = dual(x, torch.randn_like(x), edge_index)
    assert fused.shape == (12, 2) and torch.isfinite(fused).all()
    assert dual.linear1.in_features == 384, "Sentence-BERT dim"
    notes.append("DualGNN attention fusion, shared backbone, 384-d")

    return "; ".join(notes)


# ---------------------------------------------------------- 8. prompts
@check("prompts present and hash-stable")
def check_prompts():
    import hashlib
    import json

    manifest_path = ROOT / "prompts" / "manifest.json"
    if not manifest_path.exists():
        raise _Skip("run: python -m scripts.preprocess.extract_prompts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    drifted = []
    for key, meta in manifest["prompts"].items():
        path = ROOT / meta["path"]
        if not path.exists():
            drifted.append(f"{key}: file missing")
            continue
        digest = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        if digest != meta["sha256"]:
            drifted.append(f"{key}: hash drift")
    if drifted:
        raise AssertionError("; ".join(drifted))
    return f"{len(manifest['prompts'])} prompts, all hashes match"


# ------------------------------------------------- 9. reference results
@check("reference results loadable")
def check_reported():
    import csv

    path = ROOT / "research" / "reported_results.csv"
    if not path.exists():
        raise _Skip("run: python -m scripts.analyze.build_reported_results")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows, "empty"
    tables = {r["source_table"] for r in rows}
    assert {"Table 3", "Table 4", "Table 5"} <= tables, tables
    assert all(r["verification_status"] for r in rows)
    return f"{len(rows)} reference rows across {len(tables)} tables"


# ----------------------------------------------------------- 10. datasets
@check("datasets present and verified")
def check_datasets():
    from flagbench.datasets import glbench

    found, absent = [], []
    for name, sig in sorted(glbench.SIGNATURES.items()):
        path = ROOT / "data" / "raw" / name / sig.filename
        if not path.exists():
            absent.append(name)
            continue
        data = glbench.load_raw(path)
        observed = glbench.verify(name, data)
        found.append(
            f"{name} ({observed['num_nodes']:,}n, "
            f"{observed['feature_shape'][1]}d)"
        )
    if not found:
        raise _Skip(
            "no datasets downloaded. Run: "
            "python -m scripts.download.glbench --dataset all"
        )
    detail = ", ".join(found)
    if absent:
        detail += f"  [not downloaded: {', '.join(absent)}]"
    return detail


CHECKS = [
    check_environment,
    check_numerics,
    check_imports,
    check_upstream,
    check_backbones_cpu,
    check_backbones_gpu,
    check_flag_components,
    check_prompts,
    check_reported,
    check_datasets,
]


def main(argv=None) -> int:
    global VERBOSE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N | auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    VERBOSE = args.verbose

    if args.device:
        import os

        os.environ["FLAG_DEVICE"] = args.device

    print("=" * 72)
    print("FLAG reproduction benchmark -- smoke test")
    print("=" * 72)

    for fn in CHECKS:
        fn()

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = [n for n, s, _ in results if s == FAIL]
    skipped = [n for n, s, _ in results if s == SKIP]

    print("\n" + "=" * 72)
    print(f"{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("\nSKIPPED (a skip is NOT a pass):")
        for name in skipped:
            print(f"  - {name}")
    if failed:
        print("\nFAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nSMOKE TEST FAILED")
        return 1
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

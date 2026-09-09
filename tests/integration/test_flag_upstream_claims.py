"""Executable verification of every claim in research/flag_code_audit.md.

This file does NOT test our own code. It probes the *unmodified upstream* FLAG
sources in methods/flag/ so that each audit claim is backed by a passing test
rather than by a reading of the source. If upstream is ever re-fetched at a
different commit, these tests tell us which findings changed.

Run:  .venv-cpu/Scripts/python.exe -m pytest tests/integration/test_flag_upstream_claims.py -v
      .venv-cpu/Scripts/python.exe tests/integration/test_flag_upstream_claims.py   (no pytest needed)

Upstream commit under test: cb83944ed8a8a9b070a3f5a167d363973369fc80
"""
from __future__ import annotations

import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLAG = ROOT / "methods" / "flag"
EXPECTED_COMMIT = "cb83944ed8a8a9b070a3f5a167d363973369fc80"

# Import upstream modules by putting methods/flag on sys.path. The model modules
# (models/geniepath/bwgnn/caregnn/dga/pmp) are import-safe: they only define
# classes. The driver scripts (test.py/train.py/chat.py/encode.py) are NOT
# import-safe (module-scope torch.load and .cuda()), so we inspect those by AST.
# src/compat/ shadows the crashing torch_scatter wheel with a native-torch shim
# (see research/compatibility_notes.md). It must precede methods/flag on the path.
sys.path.insert(0, str(ROOT / "src" / "compat"))
sys.path.insert(1, str(FLAG))

import models as flag_models  # noqa: E402
import geniepath as flag_geniepath  # noqa: E402
import bwgnn as flag_bwgnn  # noqa: E402
import caregnn as flag_caregnn  # noqa: E402
import dga as flag_dga  # noqa: E402
import pmp as flag_pmp  # noqa: E402
import utils as flag_utils  # noqa: E402


def tiny_graph(num_nodes: int = 12, in_dim: int = 384):
    """A small connected undirected graph, on CPU."""
    torch.manual_seed(0)
    src = torch.arange(num_nodes - 1)
    dst = src + 1
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    ).long()
    x = torch.randn(num_nodes, in_dim)
    return x, edge_index


# --------------------------------------------------------------------------
# Audit 5.1 -- BLOCKER: utils.ECELoss does not exist
# --------------------------------------------------------------------------
def test_5_1_ECELoss_missing_from_utils():
    assert not hasattr(flag_utils, "ECELoss"), (
        "utils.ECELoss now exists -- audit finding 5.1 is STALE, re-check upstream"
    )
    # ...and test.py genuinely imports it, so test.py cannot be imported.
    src = (FLAG / "test.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "utils"
        for alias in node.names
    }
    assert "ECELoss" in imported, "test.py no longer imports ECELoss"


def test_5_1_importing_test_py_raises_ImportError():
    """Prove the blocker end to end: `import test` fails on ECELoss."""
    import importlib

    try:
        importlib.import_module("test")
    except ImportError as exc:
        assert "ECELoss" in str(exc), f"failed for a different reason: {exc}"
    except Exception as exc:  # pragma: no cover - would mean a different blocker
        raise AssertionError(
            f"expected ImportError on ECELoss, got {type(exc).__name__}: {exc}"
        )
    else:
        raise AssertionError("test.py imported successfully -- 5.1 is STALE")


# --------------------------------------------------------------------------
# Audit 3.3 -- skip connection is applied inconsistently across backbones
# --------------------------------------------------------------------------
# Strategy: zero out every GNN-path parameter's contribution is fragile, so we
# instead compare forward output against an explicit recomputation of
# `linear1(x)`. If the skip is active, subtracting linear1(x) from the output
# must change it; more robustly, we detect the skip structurally from the AST of
# each class's forward() by checking whether `initial_x` appears in the return.
SKIP_EXPECTED = {
    ("models.py", "GCN"): True,
    ("models.py", "GAT"): False,
    ("models.py", "GraphSAGE"): False,
    ("geniepath.py", "GeniePath"): True,
    ("geniepath.py", "GeniePathLazy"): True,
    ("bwgnn.py", "BWGNN"): True,
    ("caregnn.py", "CAREGNN"): False,
    ("dga.py", "DGA"): False,
    ("pmp.py", "LASAGE_S"): False,
}


def _forward_returns_initial_x(filename: str, classname: str) -> bool:
    tree = ast.parse((FLAG / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "forward":
                    for ret in ast.walk(sub):
                        if isinstance(ret, ast.Return) and ret.value is not None:
                            names = {
                                n.id
                                for n in ast.walk(ret.value)
                                if isinstance(n, ast.Name)
                            }
                            if "initial_x" in names:
                                return True
                    return False
    raise AssertionError(f"class {classname} not found in {filename}")


def test_3_3_skip_connection_matrix():
    actual = {
        key: _forward_returns_initial_x(*key) for key in SKIP_EXPECTED
    }
    assert actual == SKIP_EXPECTED, (
        "skip-connection matrix changed upstream.\n"
        f"expected={SKIP_EXPECTED}\nactual  ={actual}"
    )


def test_3_3_dead_linear1_in_non_skip_backbones():
    """The 'NO' rows still define linear1 and compute initial_x -- dead code."""
    for (filename, classname), has_skip in SKIP_EXPECTED.items():
        if has_skip:
            continue
        src = (FLAG / filename).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == classname:
                body = ast.dump(node)
                assert "linear1" in body, f"{classname} lost linear1"
                if classname != "GraphSAGE":
                    assert "initial_x" in body, (
                        f"{classname} no longer computes the dead initial_x"
                    )


# --------------------------------------------------------------------------
# Audit 5.3 -- x32 is only bound when hidden == 32 (DGA, PMP)
# --------------------------------------------------------------------------
def test_5_3_dga_works_at_hidden_32():
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_dga.DGA(384, 32, 2)
    x32, out = model(x, edge_index)
    assert x32.shape == (12, 32)
    assert out.shape == (12, 2)


def test_5_3_dga_raises_at_hidden_64():
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_dga.DGA(384, 64, 2)
    try:
        model(x, edge_index)
    except UnboundLocalError:
        return  # audit claim confirmed
    raise AssertionError(
        "DGA(hidden=64) no longer raises UnboundLocalError -- 5.3 is STALE"
    )


def test_5_3_pmp_raises_at_hidden_64():
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_pmp.LASAGE_S(384, 64, 2)
    try:
        model(x, edge_index)
    except UnboundLocalError:
        return
    raise AssertionError(
        "LASAGE_S(hidden=64) no longer raises UnboundLocalError -- 5.3 is STALE"
    )


def test_5_3_pmp_works_at_hidden_32():
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_pmp.LASAGE_S(384, 32, 2)
    x32, out = model(x, edge_index)
    assert x32.shape == (12, 32)
    assert out.shape == (12, 2)


# --------------------------------------------------------------------------
# Audit 5.2 -- train.py's tuple-indexing blocker
# --------------------------------------------------------------------------
def test_5_2_gcn_returns_tuple_that_train_py_cannot_index():
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_models.GCN(384, 32, 2)
    result = model(x, edge_index)
    assert isinstance(result, tuple) and len(result) == 2

    # Reproduce train.py:152 exactly:  result[batch.subset == batch.central][0]
    subset = torch.arange(12)
    central = torch.tensor(3)
    mask = subset == central
    try:
        _ = result[mask]
    except (TypeError, IndexError):
        return  # audit claim confirmed
    raise AssertionError("tuple indexing by BoolTensor succeeded -- 5.2 is STALE")


# --------------------------------------------------------------------------
# Audit 5.4 -- loop variable `i` is shadowed in train.py's accumulation
# --------------------------------------------------------------------------
def test_5_4_accumulation_counter_shadowed_in_train_py():
    """The inner `for i in range(len(batch.subset))` rebinds the enumerate `i`."""
    tree = ast.parse((FLAG / "train.py").read_text(encoding="utf-8"))
    shadowed_funcs = []
    for fn in [n for n in ast.walk(tree) if isinstance(fn_t := n, ast.FunctionDef)]:
        for outer in ast.walk(fn):
            if not (isinstance(outer, ast.For) and isinstance(outer.target, ast.Tuple)):
                continue
            outer_names = [
                e.id for e in outer.target.elts if isinstance(e, ast.Name)
            ]
            if "i" not in outer_names:
                continue
            for inner in ast.walk(outer):
                if (
                    isinstance(inner, ast.For)
                    and inner is not outer
                    and isinstance(inner.target, ast.Name)
                    and inner.target.id == "i"
                ):
                    shadowed_funcs.append(fn.name)
    assert set(shadowed_funcs) == {"train_model", "val_model"}, (
        f"shadowing pattern changed: {sorted(set(shadowed_funcs))}"
    )


def test_5_4_test_py_accumulation_is_NOT_shadowed():
    """The baseline/variant driver is clean -- only fine-tuning is affected."""
    tree = ast.parse((FLAG / "test.py").read_text(encoding="utf-8"))
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for outer in ast.walk(fn):
            if not (isinstance(outer, ast.For) and isinstance(outer.target, ast.Tuple)):
                continue
            if "i" not in [
                e.id for e in outer.target.elts if isinstance(e, ast.Name)
            ]:
                continue
            for inner in ast.walk(outer):
                assert not (
                    isinstance(inner, ast.For)
                    and inner is not outer
                    and isinstance(inner.target, ast.Name)
                    and inner.target.id == "i"
                ), f"test.py:{fn.name} now shadows i -- 5.4 scope changed"


# --------------------------------------------------------------------------
# Audit 5.5 -- the LoRA gradient path is severed by the decode/re-encode
# --------------------------------------------------------------------------
def test_5_5_reencoded_embeddings_are_detached_leaves():
    """torch.Tensor(numpy_array) is a fresh leaf: no grad flows to the producer."""
    import numpy as np

    upstream = torch.randn(4, 8, requires_grad=True)
    as_numpy = (upstream * 2).detach().numpy()          # what encoder.encode gives
    reencoded = torch.Tensor(as_numpy)                   # train.py:142
    assert reencoded.is_leaf
    assert reencoded.requires_grad is False
    assert reencoded.grad_fn is None

    loss = reencoded.sum()
    try:
        loss.backward()
    except RuntimeError:
        pass  # no graph at all
    assert upstream.grad is None, (
        "gradient reached the producer -- 5.5 would be STALE"
    )


def test_5_5_train_py_optimizer_holds_only_llm_params():
    """Confirm structurally that `optimizer` is built over the PEFT model."""
    src = (FLAG / "train.py").read_text(encoding="utf-8")
    assert "optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)" in src
    assert "gnn_optimizer = torch.optim.Adam(gnn_model.parameters()" in src


# --------------------------------------------------------------------------
# Audit 5.6 -- train_gnn tail-flush steps the wrong optimizer
# --------------------------------------------------------------------------
def test_5_6_train_gnn_tail_flush_steps_llm_optimizer():
    tree = ast.parse((FLAG / "train.py").read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_gnn"
    )
    # collect every `<name>.step()` call inside train_gnn
    stepped = {
        node.func.value.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
    }
    assert "optimizer" in stepped, "train_gnn no longer steps `optimizer` -- 5.6 STALE"
    assert "gnn_optimizer" in stepped, "train_gnn should also step gnn_optimizer"


# --------------------------------------------------------------------------
# Audit 5.10 -- pmp.py uses math.sqrt without importing math
# --------------------------------------------------------------------------
def test_5_10_pmp_missing_math_import():
    src = (FLAG / "pmp.py").read_text(encoding="utf-8")
    assert "math.sqrt" in src
    tree = ast.parse(src)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "math" not in imported, "pmp.py now imports math -- 5.10 is STALE"

    x, edge_index = tiny_graph(in_dim=16)
    try:
        flag_pmp.LILinear(8, 4, origin_infeat=16)
    except NameError:
        return
    raise AssertionError("LILinear constructed without NameError -- 5.10 STALE")


# --------------------------------------------------------------------------
# Audit 5.11 -- no CPU support: .cuda() is hardcoded throughout
# --------------------------------------------------------------------------
def test_5_11_cuda_is_hardcoded():
    counts = {}
    for path in sorted(FLAG.glob("*.py")):
        counts[path.name] = path.read_text(encoding="utf-8").count(".cuda()")
    total = sum(counts.values())
    assert total > 30, f"expected many hardcoded .cuda() calls, found {total}: {counts}"
    # Every driver script must have at least one, i.e. none is CPU-clean.
    for driver in ["test.py", "test_dual.py", "train.py", "train1.py", "encode.py"]:
        assert counts[driver] > 0, f"{driver} is now CPU-clean -- 5.11 scope changed"


# --------------------------------------------------------------------------
# Audit 7 -- GAT's second layer is SAGEConv, not GATConv
# --------------------------------------------------------------------------
def test_7_gat_second_layer_is_sageconv():
    from torch_geometric.nn import GATConv, SAGEConv

    model = flag_models.GAT(384, 32, 2)
    assert isinstance(model.conv1, GATConv), "conv1 should be GATConv"
    assert isinstance(model.conv2, SAGEConv), (
        "conv2 is no longer SAGEConv -- the GAT finding is STALE"
    )
    assert model.conv1.heads == 8, "the positional 8 should land on heads"


# --------------------------------------------------------------------------
# Audit 7 -- BWGNN applies raw adjacency, not the normalised Laplacian
# --------------------------------------------------------------------------
def test_7_bwgnn_theta_coefficients_match_beta_formula():
    """calculate_theta2 IS faithful to the official Beta-wavelet coefficients."""
    import scipy.special
    import sympy

    d = 2
    thetas = flag_bwgnn.BWGNN.calculate_theta2(d=d)
    assert len(thetas) == d + 1
    # Recompute independently.
    x = sympy.symbols("x")
    expected = []
    for i in range(d + 1):
        f = sympy.poly(
            (x / 2) ** i * (1 - x / 2) ** (d - i)
            / scipy.special.beta(i + 1, d + 1 - i)
        )
        coeff = f.all_coeffs()
        expected.append([float(coeff[d - j]) for j in range(d + 1)])
    for got, want in zip(thetas, expected):
        assert all(abs(a - b) < 1e-9 for a, b in zip(got, want))


def test_7_bwgnn_polyconv_propagates_raw_adjacency():
    """PolyConv.propagate == A @ x  (sum aggregation, message = x_j).

    The official BWGNN evaluates the polynomial in the *Laplacian* basis
    L = I - D^-1/2 A D^-1/2. Here it is plain A, so the beta-wavelet basis is
    not reproduced. We prove propagate() equals a dense A @ x.
    """
    num_nodes, dim = 6, 3
    torch.manual_seed(0)
    x = torch.randn(num_nodes, dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    conv = flag_bwgnn.PolyConv(dim, dim, theta=[1.0, 0.0], lin=False)

    propagated = conv.propagate(edge_index, x=x)

    dense_A = torch.zeros(num_nodes, num_nodes)
    dense_A[edge_index[1], edge_index[0]] = 1.0   # PyG aggregates into edge_index[1]
    assert torch.allclose(propagated, dense_A @ x, atol=1e-6), (
        "PolyConv no longer computes raw A@x -- the BWGNN finding needs re-checking"
    )

    # And confirm it is NOT the normalised Laplacian.
    deg = dense_A.sum(dim=1)
    d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    norm_A = d_inv_sqrt.unsqueeze(1) * dense_A * d_inv_sqrt.unsqueeze(0)
    laplacian = torch.eye(num_nodes) - norm_A
    assert not torch.allclose(propagated, laplacian @ x, atol=1e-4)


def test_7_bwgnn_polyconv_linear_is_dead():
    """Every PolyConv in BWGNN is built with lin=False, so self.linear is unused."""
    model = flag_bwgnn.BWGNN(384, 32, 2, d=2)
    assert len(model.conv) == 3  # d+1 wavelets
    for conv in model.conv:
        assert conv.lin is False, "lin is now True -- the dead-linear finding is STALE"


# --------------------------------------------------------------------------
# Audit 7 -- CARE-GNN has no RL selector / no multi-relation support
# --------------------------------------------------------------------------
def test_7_caregnn_lacks_rl_and_multirelation():
    src = (FLAG / "caregnn.py").read_text(encoding="utf-8")
    lowered = src.lower()
    for absent in ["reinforce", "rl_", "reward", "relation", "threshold", "top_p"]:
        assert absent not in lowered, (
            f"caregnn.py now mentions '{absent}' -- re-audit CARE-GNN fidelity"
        )
    # forward() takes a single edge_index -- no per-relation edge lists.
    model = flag_caregnn.CAREGNN(384, 32, 2)
    import inspect

    params = list(inspect.signature(model.forward).parameters)
    assert params == ["x", "edge_index"], f"unexpected signature: {params}"


# --------------------------------------------------------------------------
# Audit 7 -- DGA has no dynamic grouping / decision tree
# --------------------------------------------------------------------------
def test_7_dga_lacks_grouping():
    src = (FLAG / "dga.py").read_text(encoding="utf-8").lower()
    for absent in ["group", "tree", "bin", "toad", "bidirectional"]:
        assert absent not in src, (
            f"dga.py now mentions '{absent}' -- re-audit DGA-GNN fidelity"
        )


# --------------------------------------------------------------------------
# Audit 3.2 -- loss semantics: orthogonal_loss is a SIGNED cosine
# --------------------------------------------------------------------------
def test_3_2_orthogonal_loss_is_signed_cosine_not_squared_dot():
    """Paper Eq.9 is ||Z_D . Z_R||^2 (min -> 0). Code returns a signed cosine
    (min -> -1). Anti-alignment, not orthogonality. Prove the gap numerically."""
    a = torch.tensor([1.0, 0.0])
    orthogonal = torch.tensor([0.0, 1.0])
    anti = torch.tensor([-1.0, 0.0])

    loss_orth = flag_utils.orthogonal_loss(a, orthogonal).item()
    loss_anti = flag_utils.orthogonal_loss(a, anti).item()

    assert abs(loss_orth - 0.0) < 1e-6, f"orthogonal pair -> {loss_orth}"
    assert abs(loss_anti - (-1.0)) < 1e-6, f"anti-aligned pair -> {loss_anti}"
    # The minimiser is anti-alignment, NOT orthogonality:
    assert loss_anti < loss_orth, (
        "signed-cosine behaviour changed -- re-check against paper Eq.9"
    )


def test_3_2_non_causal_loss_is_kl_to_uniform():
    """Well-formed for the shape actually used at the call site: a [2] logit vec."""
    logits = torch.tensor([0.3, -0.7])
    value = flag_utils.non_causal_loss(logits, num_classes=2)
    assert torch.isfinite(value)
    # A uniform input should give (near) zero divergence.
    uniform_logits = torch.tensor([0.0, 0.0])
    assert flag_utils.non_causal_loss(uniform_logits).abs().item() < 1e-6


def test_3_2_causal_loss_is_cross_entropy():
    logits = torch.tensor([[2.0, -1.0]])
    label = torch.tensor([0])
    expected = torch.nn.functional.cross_entropy(logits, label)
    assert torch.allclose(flag_utils.causal_loss(logits, label), expected)


# --------------------------------------------------------------------------
# Audit 3.1/paper -- DualGNN is the paper's attention-fusion inference model
# --------------------------------------------------------------------------
def test_dualgnn_shares_one_backbone_across_both_branches():
    """Paper: 'two shared-parameter skip-GNN modules' + 'an attention layer'."""
    backbone = flag_models.GCN(384, 32, 2)
    dual = flag_models.DualGNN(2, backbone)
    assert dual.gnn is backbone
    # Both branches call the SAME module instance -> parameters are shared.
    x_raw, edge_index = tiny_graph(in_dim=384)
    x_disc = torch.randn_like(x_raw)
    out32, out = dual(x_raw, x_disc, edge_index)
    assert out.shape == (12, 2)
    assert dual.attention_weights.shape == (1, 2)


def test_dualgnn_linear1_hardcodes_384_sentence_bert_dim():
    """Evidence for the Sentence-BERT dim: DualGNN hardcodes Linear(384, ...)."""
    dual = flag_models.DualGNN(2, flag_models.GCN(384, 32, 2))
    assert dual.linear1.in_features == 384


# --------------------------------------------------------------------------
# Verified-hyperparameter regression: argparse defaults + module globals
# --------------------------------------------------------------------------
def _argparse_defaults(filename: str) -> dict:
    tree = ast.parse((FLAG / filename).read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            flag = node.args[0].value if node.args else None
            default = None
            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        default = ast.literal_eval(kw.value)
                    except ValueError:
                        default = "<expr>"
            if isinstance(flag, str):
                out[flag.lstrip("-")] = default
    return out


def test_verified_hyperparameters_train_py():
    d = _argparse_defaults("train.py")
    assert d["hidden"] == 32
    assert d["lr"] == 1e-4
    assert d["alpha"] == 0.1
    assert d["beta"] == 0.1
    assert d["outer_epochs"] == 3
    assert d["inner_epochs"] == 10
    assert d["dropout"] == 0.5
    assert d["patience"] == 10


def test_verified_hyperparameters_test_py():
    d = _argparse_defaults("test.py")
    assert d["hidden"] == 32
    assert d["lr"] == 0.01
    assert d["epochs"] == 5
    assert d["weight_decay"] == 0
    assert d["path"] == "Reddit/0_10_0/"


def test_verified_lora_config_literals():
    src = (FLAG / "train.py").read_text(encoding="utf-8")
    for literal in [
        "r=8",
        "lora_alpha=32",
        'target_modules=["q_proj", "v_proj"]',
        "lora_dropout=0.1",
        'bias="none"',
        'model_name = "gemma-2-9b-it"',
        "accumulation_steps = 10",
    ]:
        assert literal in src, f"missing upstream literal: {literal}"


def test_verified_geniepath_globals():
    assert flag_geniepath.dim == 256
    assert flag_geniepath.lstm_hidden == 256
    assert flag_geniepath.heads == 1
    assert flag_geniepath.layer_num == 4, (
        "GeniePath uses 4 layers, contradicting the paper's 'two layers'"
    )


def test_verified_encoder_and_llm_ids():
    assert 'SentenceTransformer("all-MiniLM-L6-v2")' in (
        FLAG / "encode.py"
    ).read_text(encoding="utf-8")
    assert 'model_name = "gemma-2-9b-it"' in (FLAG / "chat.py").read_text(
        encoding="utf-8"
    )


def test_verified_run_count_is_5_not_25():
    src = (FLAG / "test.py").read_text(encoding="utf-8")
    assert "for i in range(5):" in src
    assert "for j in range(1):" in src, (
        "the inner init loop is no longer range(1) -- run count changed"
    )


# --------------------------------------------------------------------------
# Every backbone must run a CPU forward pass at hidden=32 (Phase 15 evidence)
# --------------------------------------------------------------------------
def test_all_backbones_cpu_forward_at_hidden_32():
    x, edge_index = tiny_graph(in_dim=384)
    builders = {
        "GCN": lambda: flag_models.GCN(384, 32, 2),
        "GAT": lambda: flag_models.GAT(384, 32, 2),
        "GeniePathLazy": lambda: flag_geniepath.GeniePathLazy(384, 2, "cpu"),
        "BWGNN": lambda: flag_bwgnn.BWGNN(384, 32, 2),
        "CAREGNN": lambda: flag_caregnn.CAREGNN(384, 32, 2),
        "DGA": lambda: flag_dga.DGA(384, 32, 2),
        "PMP": lambda: flag_pmp.LASAGE_S(384, 32, 2),
    }
    for name, build in builders.items():
        model = build()
        x32, out = model(x, edge_index)
        assert out.shape == (12, 2), f"{name}: bad output shape {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        out.sum().backward()  # gradients must flow on CPU
        assert any(
            p.grad is not None for p in model.parameters()
        ), f"{name}: no gradients"


def test_geniepath_eager_variant_returns_single_tensor():
    """GeniePath (non-Lazy) returns ONE tensor, so it breaks the 2-tuple unpack
    that every driver uses. Only GeniePathLazy is driver-compatible."""
    x, edge_index = tiny_graph(in_dim=384)
    model = flag_geniepath.GeniePath(384, 2, "cpu")
    result = model(x, edge_index)
    assert isinstance(result, torch.Tensor), "GeniePath now returns a tuple"
    assert result.shape == (12, 2)


def test_graphsage_returns_single_tensor():
    x, edge_index = tiny_graph(in_dim=384)
    result = flag_models.GraphSAGE(384, 32, 2)(x, edge_index)
    assert isinstance(result, torch.Tensor), "GraphSAGE now returns a tuple"


# --------------------------------------------------------------------------
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------
def _main() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed, failed = 0, []
    print(f"upstream commit expected: {EXPECTED_COMMIT}")
    print(f"probing: {FLAG}")
    print(f"device: cpu | torch {torch.__version__}\n")
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"ok    {name}")
    print(f"\n{passed}/{len(tests)} audit claims verified")
    if failed:
        print(f"\n{len(failed)} STALE or CHANGED finding(s):")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

"""Tests for the Level-3 repair of upstream DGA's dropped `dim_size`.

Two things must both hold, and the second is what makes the fix legitimate:

  1. the repaired model handles isolated nodes and sparse subgraphs;
  2. wherever upstream could already compute an answer, the repair produces
     the IDENTICAL answer.

If (2) failed, this would be a behaviour change dressed up as a bug fix.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "flagbench" / "compat"))
sys.path.insert(1, str(ROOT / "methods" / "flag"))

from flagbench.adapters.backbone import _repair_dga_aggregate, build_backbone  # noqa: E402


def fresh_upstream_dga(in_dim=8, hidden=32, out=2, seed=0):
    """An UNPATCHED upstream DGA with deterministic weights.

    The patch is global and sticky (it edits the class, because PyG caches
    aggregate's signature at construction). To get a genuinely pristine model we
    reload the module, which restores the original class object.
    """
    import importlib

    import flagbench.adapters.backbone as adapters

    import dga

    dga = importlib.reload(dga)
    adapters._DGA_PATCHED = False        # the reloaded class is unpatched again
    torch.manual_seed(seed)
    return dga.DGA(in_dim, hidden, out)


def repaired_dga(in_dim=8, hidden=32, out=2, seed=0):
    """A DGA built AFTER the class patch, which is the supported path."""
    import importlib

    import flagbench.adapters.backbone as adapters

    import dga

    dga = importlib.reload(dga)
    adapters._DGA_PATCHED = False
    adapters._patch_dga_class()
    torch.manual_seed(seed)
    model = dga.DGA(in_dim, hidden, out)
    _repair_dga_aggregate(model)          # verifies the patch took effect
    return model


# --------------------------------------------- the bug is real (regression)
def test_upstream_dga_crashes_on_an_isolated_node():
    """Documents the defect. If this ever passes, upstream changed."""
    model = fresh_upstream_dga()
    x = torch.randn(1, 8)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    try:
        model(x, edge_index)
    except (IndexError, RuntimeError):
        return
    raise AssertionError(
        "unpatched DGA handled a 0-edge subgraph -- the upstream bug may be "
        "fixed; re-check the adapter repair"
    )


def test_upstream_dga_crashes_when_the_last_node_has_no_incoming_edge():
    """The bug is broader than isolated nodes."""
    model = fresh_upstream_dga()
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0], [1]])      # nothing points at node 2
    try:
        model(x, edge_index)
    except (IndexError, RuntimeError):
        return
    raise AssertionError("unpatched DGA handled a truncating index")


# ------------------------------------------------------- the repair works
def test_repaired_dga_handles_an_isolated_node():
    model = repaired_dga()
    x = torch.randn(1, 8)
    out = model(x, torch.zeros((2, 0), dtype=torch.long))
    hidden, logits = out
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_repaired_dga_handles_a_sparse_subgraph():
    model = repaired_dga()
    x = torch.randn(3, 8)
    _, logits = model(x, torch.tensor([[0], [1]]))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_repaired_dga_gradients_flow_on_an_isolated_node():
    model = repaired_dga()
    x = torch.randn(1, 8)
    _, logits = model(x, torch.zeros((2, 0), dtype=torch.long))
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


# ------------------------------- the repair changes nothing it did not fix
def test_repair_is_identical_where_upstream_already_worked():
    """The legitimacy check: same weights, same input, same output.

    A fully connected subgraph gives every node an incoming edge, which is the
    case upstream could already compute. The repaired model must agree exactly.
    """
    num_nodes = 5
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst])
    torch.manual_seed(123)
    x = torch.randn(num_nodes, 8)

    upstream = fresh_upstream_dga(seed=7).eval()
    repaired = repaired_dga(seed=7).eval()

    with torch.no_grad():
        h_up, out_up = upstream(x, edge_index)
        h_rp, out_rp = repaired(x, edge_index)

    assert torch.allclose(out_up, out_rp, atol=1e-6), (
        "the repair changed a result upstream could already compute -- that "
        "would make it a behaviour change, not a bug fix"
    )
    assert torch.allclose(h_up, h_rp, atol=1e-6)


def test_repair_preserves_scatter_mean_semantics():
    """Empty groups must be 0, matching torch_scatter's convention."""
    import torch_scatter

    src = torch.ones(2, 3)
    index = torch.tensor([0, 0])
    out = torch_scatter.scatter_mean(src, index, dim=0, dim_size=4)
    assert out.shape == (4, 3)
    assert torch.allclose(out[0], torch.ones(3))
    assert torch.allclose(out[1], torch.zeros(3))
    assert torch.isfinite(out).all()


def test_repair_is_applied_by_the_factory():
    """build_backbone must apply it -- the runner never calls it directly."""
    model = build_backbone("dga_gnn", 8, 2, hidden_dim=32, device="cpu")
    x = torch.randn(1, 8)
    _, logits = model(x, torch.zeros((2, 0), dtype=torch.long))
    assert logits.shape == (1, 2)


def test_repair_reports_if_upstream_structure_changed():
    class Empty(torch.nn.Module):
        pass

    try:
        _repair_dga_aggregate(Empty())
    except RuntimeError as exc:
        assert "IntraConv" in str(exc)
        return
    raise AssertionError("a model with no IntraConv should be reported")


# ---------------------------------- the other backbones do not need repair
def test_other_backbones_already_handle_isolated_nodes():
    x = torch.randn(1, 8)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    for key in ("gcn", "gat", "care_gnn", "bwgnn", "pmp", "geniepath"):
        model = build_backbone(key, 8, 2, hidden_dim=32, device="cpu")
        _, logits = model(x, edge_index)
        assert logits.shape == (1, 2), key
        assert torch.isfinite(logits).all(), key


def _main() -> int:
    tests = [
        (n, o) for n, o in sorted(globals().items())
        if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

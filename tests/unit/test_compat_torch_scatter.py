"""Numerical equivalence tests for the torch_scatter compatibility shim.

The shim exists only because the prebuilt torch_scatter wheel crashes on this
platform (see research/compatibility_notes.md). Its right to exist depends on it
being numerically identical to the real thing, so we check it against an
independent dense reference rather than against itself.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "flagbench" / "compat"))

import torch_scatter  # noqa: E402  -- resolves to the shim


def dense_reference(src, index, num_groups, reduce):
    """Group-by-loop reference. Deliberately naive so it cannot share a bug."""
    out = torch.zeros(num_groups, *src.shape[1:], dtype=src.dtype)
    for g in range(num_groups):
        rows = src[index == g]
        if rows.numel() == 0:
            continue          # torch_scatter leaves empty groups at 0
        if reduce == "sum":
            out[g] = rows.sum(dim=0)
        elif reduce == "mean":
            out[g] = rows.mean(dim=0)
        else:
            raise ValueError(reduce)
    return out


def test_shim_is_the_shim_not_the_wheel():
    assert torch_scatter.__version__.endswith("flagbench-shim"), (
        "the real torch_scatter is shadowing the shim -- check sys.path order"
    )


def test_scatter_mean_matches_dense_reference():
    torch.manual_seed(0)
    for num_groups in (1, 3, 7):
        for feat in (1, 4, 16):
            src = torch.randn(20, feat)
            index = torch.randint(0, num_groups, (20,))
            got = torch_scatter.scatter_mean(src, index, dim=0, dim_size=num_groups)
            want = dense_reference(src, index, num_groups, "mean")
            assert got.shape == want.shape
            assert torch.allclose(got, want, atol=1e-6), (
                f"mismatch at groups={num_groups} feat={feat}"
            )


def test_scatter_sum_matches_dense_reference():
    torch.manual_seed(1)
    src = torch.randn(30, 5)
    index = torch.randint(0, 6, (30,))
    got = torch_scatter.scatter_sum(src, index, dim=0, dim_size=6)
    want = dense_reference(src, index, 6, "sum")
    assert torch.allclose(got, want, atol=1e-6)


def test_scatter_add_is_alias_of_sum():
    torch.manual_seed(2)
    src = torch.randn(12, 3)
    index = torch.randint(0, 4, (12,))
    a = torch_scatter.scatter_add(src, index, dim=0, dim_size=4)
    b = torch_scatter.scatter_sum(src, index, dim=0, dim_size=4)
    assert torch.equal(a, b)


def test_empty_groups_are_zero_not_nan():
    """torch_scatter yields 0 for groups with no members; reduce='mean' alone
    would yield the fill value, so this is the behaviour we must preserve."""
    src = torch.ones(3, 2)
    index = torch.tensor([0, 0, 0])          # groups 1 and 2 are empty
    out = torch_scatter.scatter_mean(src, index, dim=0, dim_size=3)
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.ones(2))
    assert torch.allclose(out[1], torch.zeros(2))
    assert torch.allclose(out[2], torch.zeros(2))
    assert torch.isfinite(out).all()


def test_dim_size_inferred_from_index_max():
    src = torch.randn(6, 2)
    index = torch.tensor([0, 0, 1, 1, 3, 3])
    out = torch_scatter.scatter_mean(src, index, dim=0)
    assert out.shape == (4, 2)               # max index 3 -> 4 groups
    assert torch.allclose(out[2], torch.zeros(2))


def test_gradients_flow_through_the_shim():
    src = torch.randn(8, 3, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2])
    torch_scatter.scatter_mean(src, index, dim=0, dim_size=3).sum().backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()
    # Each row's gradient is 1/|its group|.
    expected = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
    assert torch.allclose(src.grad[:, 0], expected, atol=1e-6)


def test_unaudited_attribute_raises_with_guidance():
    try:
        _ = torch_scatter.segment_coo
    except AttributeError as exc:
        assert "compatibility shim" in str(exc)
        return
    raise AssertionError("expected AttributeError for an unimplemented symbol")


def test_dga_forward_runs_through_the_shim():
    """End-to-end: upstream dga.py, unmodified, driven by the shim."""
    sys.path.insert(0, str(ROOT / "methods" / "flag"))
    import dga  # noqa: PLC0415

    assert dga.torch_scatter.__version__.endswith("flagbench-shim")

    torch.manual_seed(0)
    n = 12
    s = torch.arange(n - 1)
    edge_index = torch.stack(
        [torch.cat([s, s + 1]), torch.cat([s + 1, s])], dim=0
    ).long()
    x = torch.randn(n, 384)

    model = dga.DGA(384, 32, 2)
    x32, out = model(x, edge_index)
    assert out.shape == (n, 2)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


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

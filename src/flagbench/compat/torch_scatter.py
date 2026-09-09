"""Native-torch stand-in for the `torch_scatter` C++ extension.

WHY THIS EXISTS
---------------
`methods/flag/dga.py` does `import torch_scatter` and calls
`torch_scatter.scatter_mean(inputs, index, dim=0)`.

On this machine the prebuilt wheel `torch_scatter==2.1.2+pt24cpu` causes a hard
native crash (Windows access violation 0xC0000005 / heap corruption 0xC0000374)
as soon as PyTorch Geometric routes an aggregation through it. See
`research/compatibility_notes.md` for the full bisection. PyG >= 2.1 does not
need `torch_scatter` at all -- it falls back to `torch.Tensor.scatter_reduce_`.

Rather than edit upstream FLAG source, we shadow the module name with this pure
-PyTorch implementation and prepend `src/compat/` to `sys.path`. `dga.py` is
byte-identical to upstream; only the provider of `scatter_mean` changes.

CLASSIFICATION (Phase 18 / Phase 36): LEVEL 2 -- environment/dependency fix.
It does not alter algorithm behaviour. `tests/unit/test_compat_torch_scatter.py`
asserts numerical equivalence against `torch.scatter_reduce` and against a dense
reference implementation.

Only the functions FLAG actually needs are provided. Anything else raises, so we
can never silently paper over a wider dependency than we audited.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["scatter_mean", "scatter_add", "scatter_sum", "scatter_max", "scatter_min"]

__version__ = "0.0.0+flagbench-shim"


def _broadcast(index: Tensor, src: Tensor, dim: int) -> Tensor:
    """Expand a 1-D index to src's shape along `dim`, as torch_scatter does."""
    if dim < 0:
        dim = src.dim() + dim
    if index.dim() == 1:
        for _ in range(dim):
            index = index.unsqueeze(0)
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
    return index.expand_as(src)


def _out_shape(src: Tensor, dim: int, dim_size: Optional[int], index: Tensor):
    if dim < 0:
        dim = src.dim() + dim
    size = list(src.shape)
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size, dim


def _reduce(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_size: Optional[int],
    reduce: str,
    fill: float,
) -> Tensor:
    size, dim = _out_shape(src, dim, dim_size, index)
    idx = _broadcast(index, src, dim)
    out = src.new_full(size, fill)
    out.scatter_reduce_(dim, idx, src, reduce=reduce, include_self=False)
    return out


def scatter_sum(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    if out is not None:
        raise NotImplementedError(
            "the `out=` argument is not supported by the FLAG-bench shim"
        )
    return _reduce(src, index, dim, dim_size, "sum", 0.0)


def scatter_add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
    """Mean over each index group.

    Matches `torch_scatter.scatter_mean`, including its convention that groups
    with no contributing elements are 0 (not NaN).
    """
    if out is not None:
        raise NotImplementedError(
            "the `out=` argument is not supported by the FLAG-bench shim"
        )
    total = scatter_sum(src, index, dim, None, dim_size)

    # Count per group, then divide. Done explicitly (rather than via
    # reduce="mean") so that empty groups yield 0 rather than the fill value.
    if dim < 0:
        dim = src.dim() + dim
    ones = torch.ones_like(src)
    count = scatter_sum(ones, index, dim, None, total.shape[dim])
    return total / count.clamp(min=1)


def scatter_max(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
):
    if out is not None:
        raise NotImplementedError("the `out=` argument is not supported")
    return _reduce(src, index, dim, dim_size, "amax", 0.0), None


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
):
    if out is not None:
        raise NotImplementedError("the `out=` argument is not supported")
    return _reduce(src, index, dim, dim_size, "amin", 0.0), None


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> Tensor:
    dispatch = {
        "sum": scatter_sum,
        "add": scatter_add,
        "mean": scatter_mean,
    }
    if reduce not in dispatch:
        raise NotImplementedError(
            f"reduce={reduce!r} is outside the audited surface of this shim"
        )
    return dispatch[reduce](src, index, dim, out, dim_size)


def __getattr__(name: str):
    raise AttributeError(
        f"torch_scatter.{name} is not provided by the FLAG-bench compatibility "
        f"shim (src/compat/torch_scatter.py). Only {sorted(__all__)} are "
        f"implemented. If upstream code needs more, extend the shim and add an "
        f"equivalence test rather than installing the crashing wheel -- see "
        f"research/compatibility_notes.md."
    )

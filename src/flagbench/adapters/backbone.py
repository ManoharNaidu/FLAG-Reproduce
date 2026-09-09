"""Adapters wrapping FLAG's bundled backbones behind one interface.

Phase 4's rule: **do not alter a baseline's mathematics to fit an interface.**

These adapters therefore do exactly three things:
  1. construct the upstream class with the right positional signature,
  2. normalise its return shape,
  3. place it on a device.

They never reimplement a layer, never change an aggregation, and never add or
remove the Eq. 6 skip. Where upstream is inconsistent -- GAT computes
`initial_x` and discards it, DGA binds `x32` only at hidden=32 -- the adapter
*surfaces* that rather than papering over it, because the inconsistency is a
finding about the paper's released code (research/flag_code_audit.md section 3.3).

Upstream is imported from `methods/flag/`, which is git-ignored and fetched at a
pinned commit. `src/flagbench/compat` must precede it on sys.path so that
`import torch_scatter` resolves to our shim (compatibility_notes.md section 5).
"""
from __future__ import annotations

import importlib
import pathlib
import sys
from abc import ABC, abstractmethod

import torch

from flagbench.registry.registry import ModelSpec, get_model

ROOT = pathlib.Path(__file__).resolve().parents[3]
FLAG_DIR = ROOT / "methods" / "flag"
COMPAT_DIR = ROOT / "src" / "flagbench" / "compat"


def _ensure_upstream_importable() -> None:
    """Put compat before methods/flag on sys.path, exactly once."""
    if not FLAG_DIR.exists():
        raise FileNotFoundError(
            f"{FLAG_DIR} is missing. Upstream is git-ignored (it ships no "
            f"licence). Fetch it with:\n  bash scripts/setup/fetch_methods.sh"
        )
    for path in (str(FLAG_DIR), str(COMPAT_DIR)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(FLAG_DIR))
    sys.path.insert(0, str(COMPAT_DIR))   # compat must win


def _repair_dga_aggregate(net) -> None:
    """LEVEL 3 fix (Phase 36): restore `dim_size` in upstream DGA's aggregation.

    THE BUG. `methods/flag/dga.py` overrides message passing as

        def aggregate(self, inputs, index):
            return torch_scatter.scatter_mean(inputs, index, dim=0)

    PyG inspects the signature of `aggregate` and passes only the parameters it
    declares. This one does not declare `dim_size`, so PyG cannot supply the node
    count and `scatter_mean` infers the output length from `max(index) + 1`.

    THE CONSEQUENCE. The aggregated tensor is shorter than the node count
    whenever the highest-indexed node has no incoming edge, and `out + x_self`
    then either raises or broadcasts wrongly. Measured on a 3-node graph:

        edges into every node   -> 3 rows   OK
        edge 0->1 only          -> RuntimeError (2 vs 3)
        no edges at all         -> RuntimeError (0 vs 3)

    It matters here because FLAG's 1:10 downsampling strands many nodes: 3,690
    of 18,389 Reddit benchmark nodes (20%) are isolated, and their 2-hop
    subgraph is a single node with no edges. Every one of those crashes DGA.

    THE FIX. Re-bind `aggregate` to a version that accepts and forwards
    `dim_size`. This is what the upstream code clearly intended -- the comment
    says mean aggregation over neighbours -- and it changes no result that
    upstream could already compute: where the old code produced the right length
    it produced identical values, because `scatter_mean` with an explicit
    `dim_size` only ever *extends* the output with empty groups, which are 0.

    WHY IT PATCHES THE CLASS, NOT THE INSTANCE. `MessagePassing.__init__`
    inspects `aggregate`'s signature once and caches which arguments to collect.
    Rebinding the method on an already-constructed instance is therefore too
    late -- PyG still refuses to pass `dim_size`. The patch has to land on the
    class before any layer is built, which is what `_patch_dga_class` does; this
    function then verifies it actually took effect.

    Applied in the adapter, so `methods/flag/dga.py` stays byte-identical to
    upstream. Covered by tests in tests/unit/test_dga_isolated_nodes.py.
    """
    _patch_dga_class()
    found = [m for m in net.modules() if type(m).__name__ == "IntraConv"]
    if not found:
        raise RuntimeError(
            "expected at least one IntraConv layer in DGA; upstream may have "
            "changed. Re-check research/flag_code_audit.md."
        )
    for module in found:
        params = getattr(getattr(module, "inspector", None), "params", {})
        if "dim_size" not in params.get("aggregate", {}):
            # The layer predates the patch, so PyG cached the old signature.
            # Surfacing that beats returning a model that will crash later.
            raise RuntimeError(
                "this DGA layer was constructed before the dim_size patch was "
                "applied, so PyG will not pass dim_size and isolated nodes will "
                "crash. Build DGA through flagbench build_backbone()."
            )


_DGA_PATCHED = False


def _patch_dga_class() -> None:
    """Install the fixed `aggregate` on `dga.IntraConv`, once, before use.

    In-memory only. `methods/flag/dga.py` on disk is never modified, and
    `fetch_methods.sh --verify` still reports the clone as pristine.
    """
    global _DGA_PATCHED
    if _DGA_PATCHED:
        return

    _ensure_upstream_importable()
    import torch_scatter

    dga = importlib.import_module("dga")

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # The only change from upstream: dim_size is declared, so PyG supplies
        # the true node count, and forwarded, so empty groups become zero rows
        # instead of being dropped entirely.
        return torch_scatter.scatter_mean(inputs, index, dim=0, dim_size=dim_size)

    aggregate.__doc__ = (
        "FLAG-bench Level-3 repair of upstream's dropped dim_size. "
        "See flagbench.adapters.backbone._repair_dga_aggregate."
    )
    dga.IntraConv.aggregate = aggregate
    _DGA_PATCHED = True


class BaseBackbone(ABC, torch.nn.Module):
    """Common interface. Deliberately small -- every extra method is a chance to
    accidentally change a baseline's behaviour."""

    spec: ModelSpec

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Returns `(hidden_embedding, logits)`.

        `hidden_embedding` is upstream's `x32`, i.e. the hidden layer output.
        The paper's t-SNE study visualises exactly this tensor.
        """

    def get_embeddings(self, x, edge_index) -> torch.Tensor:
        with torch.no_grad():
            hidden, _ = self.forward(x, edge_index)
        return hidden

    def predict(self, x, edge_index) -> torch.Tensor:
        """Fraud-class probabilities."""
        with torch.no_grad():
            _, logits = self.forward(x, edge_index)
        return torch.softmax(logits, dim=-1)[:, 1]

    def save_checkpoint(self, path, **metadata) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_key": self.spec.key,
                "impl_source": self.spec.impl_source.value,
                **metadata,
            },
            path,
        )

    def load_checkpoint(self, path, strict: bool = True) -> dict:
        payload = torch.load(path, map_location="cpu")
        saved = payload.get("model_key")
        if saved and saved != self.spec.key:
            raise ValueError(
                f"checkpoint is for model {saved!r}, this is {self.spec.key!r}"
            )
        self.load_state_dict(payload["state_dict"], strict=strict)
        return payload


class FlagBundledBackbone(BaseBackbone):
    """Wraps a class from `methods/flag/` unmodified.

    The upstream constructors are not uniform, which is itself a finding:

        GCN / GAT / BWGNN / CAREGNN / DGA : (in_dim, hidden, out_dim)
        LASAGE_S (PMP)                    : (in_dim, hidden, out_dim)
        GeniePathLazy                     : (in_dim, out_dim, device_string)

    GeniePathLazy takes a device *string* and uses it to allocate its LSTM
    hidden state internally, so the device must be known at construction time.
    """

    def __init__(
        self,
        model_key: str,
        in_dim: int,
        out_dim: int = 2,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.spec = get_model(model_key)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device_str = str(device)

        if (
            self.spec.hidden_dim_locked is not None
            and hidden_dim != self.spec.hidden_dim_locked
        ):
            raise ValueError(
                f"{self.spec.display_name} only works at hidden_dim="
                f"{self.spec.hidden_dim_locked}, got {hidden_dim}. "
                f"Upstream binds `x32` behind `if len(x[0]) == 32` and raises "
                f"UnboundLocalError otherwise ({self.spec.notes}). "
                f"This is an upstream constraint we do not paper over."
            )

        _ensure_upstream_importable()
        if model_key == "dga_gnn":
            # Must precede construction: PyG caches aggregate's signature in
            # MessagePassing.__init__.
            _patch_dga_class()

        module_name, class_name = self.spec.module.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)

        if model_key == "geniepath":
            self.net = cls(in_dim, out_dim, self.device_str)
        elif model_key in ("gcn", "gat"):
            self.net = cls(in_dim, hidden_dim, out_dim, dropout)
        else:
            self.net = cls(in_dim, hidden_dim, out_dim)

        if model_key == "dga_gnn":
            _repair_dga_aggregate(self.net)

        self.to(device)

    def forward(self, x, edge_index):
        out = self.net(x, edge_index)
        if not isinstance(out, tuple):
            # GeniePath (eager) and GraphSAGE return a single tensor; every
            # upstream driver assumes a 2-tuple, so this would be a latent crash.
            raise TypeError(
                f"{self.spec.key} returned a single tensor, not (hidden, logits). "
                f"Upstream drivers unpack two values; see "
                f"research/implementation_matrix.md section 2.2."
            )
        hidden, logits = out
        return hidden, logits

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # GeniePathLazy allocates its LSTM state on self.device, a plain string.
        if hasattr(self.net, "device"):
            for arg in args:
                if isinstance(arg, (str, torch.device)):
                    self.net.device = str(arg)
                    self.device_str = str(arg)
        return result

    def extra_repr(self) -> str:
        return (
            f"{self.spec.key}, in={self.in_dim}, hidden={self.hidden_dim}, "
            f"out={self.out_dim}, impl={self.spec.impl_source.value}, "
            f"skip={self.spec.has_skip}"
        )


class DualBranchBackbone(BaseBackbone):
    """The paper's inference architecture: two shared-parameter skip-GNNs fused
    by an attention layer.

    Paper section 3.4: *"both the raw text and the discriminative text are
    encoded ... These encoded features are independently processed through two
    shared-parameter skip-GNN modules. The outputs of the two skip-GNN modules
    are fed into an attention layer."*

    Upstream implements this as `models.py:DualGNN`, and we reuse that class
    directly rather than reimplementing the fusion.
    """

    def __init__(self, backbone: FlagBundledBackbone, out_dim: int = 2,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.spec = backbone.spec
        _ensure_upstream_importable()
        models = importlib.import_module("models")
        # DualGNN shares ONE backbone instance across both branches -- that is
        # what makes the two skip-GNNs "shared-parameter".
        self.net = models.DualGNN(out_dim, backbone.net)
        self.backbone = backbone
        self.to(device)

    def forward(self, x_raw, x_disc, edge_index):  # type: ignore[override]
        return self.net(x_raw, x_disc, edge_index)


def build_backbone(
    model_key: str,
    in_dim: int,
    out_dim: int = 2,
    hidden_dim: int = 32,
    dropout: float = 0.5,
    device: torch.device | str = "cpu",
    dual_branch: bool = False,
) -> BaseBackbone:
    """Factory used by the training loop and the registry."""
    backbone = FlagBundledBackbone(
        model_key, in_dim, out_dim, hidden_dim, dropout, device
    )
    if dual_branch:
        return DualBranchBackbone(backbone, out_dim, device)
    return backbone

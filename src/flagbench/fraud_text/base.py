"""The unified text-augmented fraud-dataset schema.

One container for YelpChi and Amazon so downstream code never branches on the
dataset. The node *semantics* differ deliberately -- YelpChi nodes are reviews,
Amazon nodes are users -- and that is recorded in `node_type` rather than forced
into a common shape (brief section 42).

WHY NOT `src/datasets/`
-----------------------
The brief asks for a flat `src/datasets/`. This repo puts `methods/flag/` on
`sys.path` to import upstream FLAG modules, so a top-level `datasets` package
would collide with upstream module names -- pyproject.toml documents this. The
brief's module breakdown (base / yelpchi / amazon / loaders / validation /
serialization) is preserved verbatim inside `flagbench.fraud_text`.

PROVENANCE LABELS (brief section 40)
------------------------------------
Every field carries one, so nothing can be mistaken for canonical data it is not:

    CANONICAL   bit-for-bit from the upstream release
    DERIVED     computed from canonical data by a documented rule
    VERIFIED    reconstructed and then proven against canonical data
    UNRESOLVED  could not be established; deliberately left empty
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch

CANONICAL = "CANONICAL"
DERIVED = "DERIVED"
VERIFIED = "VERIFIED"
UNRESOLVED = "UNRESOLVED"


class DatasetIntegrityError(RuntimeError):
    """A structural invariant failed. Never downgraded to a warning."""


@dataclass
class FLAGDataset:
    """A fraud graph plus per-node text, with provenance attached.

    The canonical graph tensors (`node_features`, `labels`, `relation_edges`)
    are reproduced exactly as shipped upstream. `edge_index` is a DERIVED
    homogeneous view kept alongside -- never instead of -- `relation_edges`
    (brief section 13).
    """

    dataset_name: str
    node_type: str                      # "review" | "user"

    node_features: torch.Tensor         # [N, F]  CANONICAL
    labels: torch.Tensor                # [N]     CANONICAL, 1 = fraud
    raw_texts: list[str]                # len N   VERIFIED or UNRESOLVED per node

    edge_index: torch.Tensor            # [2, E]  DERIVED from relation_edges
    relation_edges: dict[str, torch.Tensor]   # CANONICAL
    relation_names: list[str]

    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor

    text_status: list[str] = field(default_factory=list)
    """Per-node provenance for `raw_texts`: VERIFIED or UNRESOLVED.

    A node whose text could not be established keeps an empty string AND an
    UNRESOLVED stamp, so "no text" is never silently indistinguishable from
    "empty review".
    """

    unlabeled_mask: torch.Tensor | None = None
    node_metadata: dict = field(default_factory=dict)
    mapping_metadata: dict = field(default_factory=dict)
    dataset_metadata: dict = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    # -- derived views ------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def text_coverage(self) -> dict:
        n = self.num_nodes
        verified = sum(1 for s in self.text_status if s == VERIFIED)
        nonempty = sum(1 for t in self.raw_texts if t)
        return {
            "num_nodes": n,
            "verified": verified,
            "unresolved": n - verified,
            "non_empty": nonempty,
            "verified_fraction": verified / n if n else 0.0,
        }

    def label_counts(self) -> dict[int, int]:
        counts = torch.bincount(self.labels.long(), minlength=2)
        return {i: int(c) for i, c in enumerate(counts)}

    def split_counts(self) -> dict:
        out = {}
        for name, idx in (("train", self.train_idx), ("val", self.val_idx),
                          ("test", self.test_idx)):
            lab = self.labels[idx].long()
            c = torch.bincount(lab, minlength=2)
            out[name] = {"n": int(idx.numel()),
                         "per_class": {i: int(v) for i, v in enumerate(c)}}
        return out

    # -- invariants ---------------------------------------------------------
    def validate(self) -> None:
        """Raise on any structural inconsistency. Called by every loader.

        These are the alignment invariants from brief sections 25 and 57; a
        violation means a node's text, features or label belong to a different
        node, which would silently corrupt every downstream result.
        """
        n = self.num_nodes
        if self.node_features.shape[0] != n:
            raise DatasetIntegrityError(
                f"node_features has {self.node_features.shape[0]} rows, labels {n}")
        if len(self.raw_texts) != n:
            raise DatasetIntegrityError(
                f"len(raw_texts)={len(self.raw_texts)} != num_nodes={n}")
        if self.text_status and len(self.text_status) != n:
            raise DatasetIntegrityError(
                f"len(text_status)={len(self.text_status)} != num_nodes={n}")

        if self.edge_index.numel() and int(self.edge_index.max()) >= n:
            raise DatasetIntegrityError("edge_index references a node >= num_nodes")
        if self.edge_index.numel() and int(self.edge_index.min()) < 0:
            raise DatasetIntegrityError("edge_index contains a negative index")
        for name, ei in self.relation_edges.items():
            if ei.numel() and (int(ei.max()) >= n or int(ei.min()) < 0):
                raise DatasetIntegrityError(
                    f"relation {name!r} has an out-of-bounds node index")

        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(getattr(self, f"{a}_idx").tolist()) & \
                      set(getattr(self, f"{b}_idx").tolist())
            if overlap:
                raise DatasetIntegrityError(
                    f"split leakage: {len(overlap)} nodes in both {a} and {b}")

        for split in ("train_idx", "val_idx", "test_idx"):
            idx = getattr(self, split)
            if idx.numel() and (int(idx.max()) >= n or int(idx.min()) < 0):
                raise DatasetIntegrityError(f"{split} out of bounds")

        if self.unlabeled_mask is not None:
            for split in ("train_idx", "val_idx", "test_idx"):
                idx = getattr(self, split)
                if idx.numel() and bool(self.unlabeled_mask[idx].any()):
                    raise DatasetIntegrityError(
                        f"{split} contains unlabeled nodes")

    def summary(self) -> str:
        cov = self.text_coverage()
        rel = ", ".join(f"{k}={v.shape[1]:,}" for k, v in self.relation_edges.items())
        sp = self.split_counts()
        return (
            f"{self.dataset_name}  node_type={self.node_type}\n"
            f"  nodes      {self.num_nodes:,}   features {self.num_features}\n"
            f"  labels     {self.label_counts()}  (1 = fraud)\n"
            f"  relations  {rel}\n"
            f"  edge_index {self.num_edges:,} (DERIVED union view)\n"
            f"  text       {cov['verified']:,}/{cov['num_nodes']:,} VERIFIED "
            f"({cov['verified_fraction'] * 100:.1f}%), "
            f"{cov['unresolved']:,} UNRESOLVED\n"
            f"  splits     train={sp['train']['n']:,} val={sp['val']['n']:,} "
            f"test={sp['test']['n']:,}"
        )

    def as_payload(self) -> dict:
        """Serializable dict. Tensors stay tensors; torch.save handles them."""
        d = dataclasses.asdict(self)
        d["relation_edges"] = dict(self.relation_edges)
        return d

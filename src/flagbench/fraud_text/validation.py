"""Dataset validator (brief section 18).

Fails loudly on any serious inconsistency. A check that cannot be evaluated is
reported as SKIP, never as a pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from flagbench.fraud_text.base import UNRESOLVED, VERIFIED, FLAGDataset


@dataclass
class Check:
    name: str
    ok: bool | None          # None = SKIP
    detail: str = ""

    @property
    def status(self) -> str:
        return "SKIP" if self.ok is None else ("PASS" if self.ok else "FAIL")


@dataclass
class ValidationReport:
    dataset: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name, ok, detail=""):
        self.checks.append(Check(name, ok, detail))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.ok is False]

    @property
    def ok(self) -> bool:
        return not self.failed

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) + 2
        lines = [f"validation: {self.dataset}", "-" * (width + 40)]
        for c in self.checks:
            lines.append(f"  {c.status:<5} {c.name:<{width}} {c.detail}")
        lines.append("-" * (width + 40))
        lines.append("RESULT: " + ("PASS" if self.ok else
                                   f"FAIL ({len(self.failed)} check(s))"))
        return "\n".join(lines)


def validate_dataset(ds: FLAGDataset) -> ValidationReport:
    r = ValidationReport(ds.dataset_name)
    n = ds.num_nodes

    r.add("node count > 0", n > 0, f"{n:,} nodes")
    r.add("feature rows == num_nodes", ds.node_features.shape[0] == n,
          f"{tuple(ds.node_features.shape)}")
    r.add("labels == num_nodes", ds.labels.shape[0] == n)
    r.add("len(raw_texts) == num_nodes", len(ds.raw_texts) == n,
          f"{len(ds.raw_texts):,}")
    r.add("len(text_status) == num_nodes", len(ds.text_status) == n)

    r.add("no NaN/Inf in features",
          bool(torch.isfinite(ds.node_features).all()))
    labels_ok = set(ds.labels.unique().tolist()) <= {0, 1}
    r.add("labels in {0,1}", labels_ok,
          f"values={sorted(set(ds.labels.unique().tolist()))}, 1=fraud")

    in_bounds = (ds.edge_index.numel() == 0 or
                 (int(ds.edge_index.max()) < n and int(ds.edge_index.min()) >= 0))
    r.add("edge_index in bounds", in_bounds, f"{ds.num_edges:,} edges")

    rel_ok = all(
        ei.numel() == 0 or (int(ei.max()) < n and int(ei.min()) >= 0)
        for ei in ds.relation_edges.values())
    r.add("relation indices in bounds", rel_ok,
          ", ".join(f"{k}={int(v.shape[1]):,}" for k, v in ds.relation_edges.items()))
    r.add("relation_names match relation_edges",
          set(ds.relation_names) == set(ds.relation_edges))

    # splits
    idx_all = torch.cat([ds.train_idx, ds.val_idx, ds.test_idx])
    r.add("no duplicate node in splits",
          int(idx_all.numel()) == len(set(idx_all.tolist())),
          f"{idx_all.numel():,} split assignments")
    leak = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leak += len(set(getattr(ds, f"{a}_idx").tolist()) &
                    set(getattr(ds, f"{b}_idx").tolist()))
    r.add("no train/val/test leakage", leak == 0, f"{leak} overlapping nodes")

    if ds.unlabeled_mask is not None:
        n_unlab = int(ds.unlabeled_mask.sum())
        bad = sum(int(ds.unlabeled_mask[getattr(ds, f"{s}_idx")].sum())
                  for s in ("train", "val", "test"))
        r.add("unlabeled nodes excluded from splits", bad == 0,
              f"{n_unlab:,} unlabeled, {bad} leaked into a split")
    else:
        r.add("unlabeled nodes excluded from splits", None, "no unlabeled_mask")

    # text
    cov = ds.text_coverage()
    r.add("text status values valid",
          set(ds.text_status) <= {VERIFIED, UNRESOLVED},
          f"{cov['verified']:,} VERIFIED / {cov['unresolved']:,} UNRESOLVED")
    mismatch = sum(1 for t, s in zip(ds.raw_texts, ds.text_status)
                   if s == VERIFIED and not t)
    r.add("no VERIFIED node with empty text", mismatch == 0,
          f"{mismatch} offending nodes")
    unresolved_with_text = sum(1 for t, s in zip(ds.raw_texts, ds.text_status)
                               if s == UNRESOLVED and t)
    r.add("no UNRESOLVED node carrying text", unresolved_with_text == 0,
          f"{unresolved_with_text} offending nodes")

    # every evaluated node must have text, or results are silently degraded
    evaluated = torch.cat([ds.train_idx, ds.val_idx, ds.test_idx])
    missing = sum(1 for i in evaluated.tolist() if ds.text_status[i] != VERIFIED)
    r.add("every train/val/test node has VERIFIED text", missing == 0,
          f"{missing:,} evaluated nodes lack text")

    dup = len(ds.raw_texts) - len({t for t in ds.raw_texts if t})
    r.add("duplicate-text report", True,
          f"{dup:,} nodes share a text with another (informational)")
    r.add("provenance recorded for every core field",
          {"node_features", "labels", "relation_edges", "edge_index",
           "raw_texts"} <= set(ds.provenance))
    return r

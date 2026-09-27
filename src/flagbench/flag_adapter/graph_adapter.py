"""Relation-aware graph -> the single homogeneous edge_index FLAG consumes.

WHY THIS IS LOSSY, AND WHY IT IS STILL THE HONEST CHOICE
--------------------------------------------------------
Nothing in this repository can consume a multi-relation graph. Every bundled
backbone's forward signature is `(x, edge_index)` -- a single tensor -- and there
is no relation-aware code path in flagbench, scripts/, or methods/flag/ (see
research/_evidence/loop1_d_flag_contract.md section 3).

So a FLAG run on YelpChi/Amazon MUST collapse three relations into one. This
module does that explicitly and records how, rather than letting it happen
implicitly somewhere downstream. `relation_edges` is never modified or
discarded -- it stays on the FLAGDataset for relation-aware consumers.
"""
from __future__ import annotations

import torch

from flagbench.fraud_text.base import FLAGDataset


def homogeneous_edge_index(ds: FLAGDataset, strategy: str = "canonical_homo"
                           ) -> tuple[torch.Tensor, dict]:
    """Return (edge_index, provenance_record).

    strategy:
        canonical_homo  use the .mat's own `homo` view, as shipped. For YelpChi
                        this was verified to equal the union of the three
                        relations exactly; for Amazon it does NOT (Jaccard
                        0.626) -- it is CARE-GNN's own fourth adjacency, and we
                        use it unchanged rather than substituting our own union.
        union           recompute the union of relation_edges ourselves.
    """
    if strategy == "canonical_homo":
        return ds.edge_index, {
            "strategy": "canonical_homo",
            "note": "the .mat's own 'homo' adjacency, used unchanged",
            "num_edges": int(ds.edge_index.shape[1]),
            "relations_collapsed": list(ds.relation_edges),
        }
    if strategy == "union":
        cat = torch.cat(list(ds.relation_edges.values()), dim=1)
        uniq = torch.unique(cat.t(), dim=0).t().contiguous()
        return uniq, {
            "strategy": "union",
            "note": "recomputed union of relation_edges; differs from the .mat "
                    "'homo' view for Amazon",
            "num_edges": int(uniq.shape[1]),
            "relations_collapsed": list(ds.relation_edges),
        }
    raise ValueError(f"unknown strategy {strategy!r}")

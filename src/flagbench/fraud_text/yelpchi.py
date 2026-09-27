"""Build the text-augmented YelpChi dataset.

Node = a review, so the text mapping is one-to-one: node i gets exactly the
review text of its own row. No aggregation, no truncation (brief sections 10,
27, 51).

ALIGNMENT PROVENANCE
--------------------
The node ordering used here is not assumed -- it was proven by rebuilding R-U-R
and R-S-R from the raw metadata and comparing sparsity patterns entry by entry
against `YelpChi.mat` (98,630 and 6,805,486 edges, zero mismatches), plus a
45,954/45,954 per-row label match. Run `python -m scripts.verify_yelpchi_alignment`
to reproduce. The text join itself is statistical evidence rather than proof --
see `research/evidence_matrix.md` section 3 -- and is stamped VERIFIED, not
CANONICAL.

The canonical graph tensors are taken from `data/benchmark/native_yelpchi/graph.pt`
(built by `experiments/yelpchi_amazon/build_native_benchmark.py`) and are NOT
recomputed here; this module only attaches text.
"""
from __future__ import annotations

import csv
import pathlib

import torch

from flagbench.fraud_text.base import (
    CANONICAL, DERIVED, UNRESOLVED, VERIFIED, FLAGDataset,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
NATIVE = ROOT / "data" / "benchmark" / "native_yelpchi" / "graph.pt"
MAPPING = ROOT / "datasets" / "processed" / "yelpchi" / "yelpchi_review_mapping.csv"
HOTEL = ROOT / "datasets" / "raw" / "yelpchi" / "raw_text.txt"
REST = ROOT / "datasets" / "raw" / "yelpchi" / "output_review_yelpResData_NRYRcleaned.txt"


def _load_texts() -> list[str]:
    """Hotel block then restaurant block, the order proven in claim 2."""
    hotel = HOTEL.read_text(encoding="utf-8", errors="replace").splitlines()
    rest = REST.read_text(encoding="utf-8", errors="replace").splitlines()
    return hotel + rest


def _mask_to_idx(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def build() -> FLAGDataset:
    for p in (NATIVE, MAPPING, HOTEL, REST):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing.\n"
                f"  canonical graph: python -m experiments.yelpchi_amazon."
                f"build_native_benchmark --dataset yelpchi\n"
                f"  mapping:         python -m scripts.verify_yelpchi_alignment"
            )

    payload = torch.load(NATIVE, map_location="cpu")
    texts = _load_texts()

    rows = list(csv.DictReader(MAPPING.open(encoding="utf-8")))
    n = int(payload["y"].shape[0])
    if len(rows) != n:
        raise RuntimeError(f"mapping has {len(rows)} rows, graph has {n} nodes")

    raw_texts: list[str] = [""] * n
    status: list[str] = [UNRESOLVED] * n
    meta_user, meta_prod, meta_rating, meta_date = [], [], [], []
    for r in rows:
        i = int(r["graph_node_id"])
        src = int(r["source_row"])
        if r["confidence"] == "EXACT" and 0 <= src < len(texts):
            raw_texts[i] = texts[src]
            status[i] = VERIFIED
        meta_user.append(int(r["user_id"]))
        meta_prod.append(int(r["product_id"]))
        meta_rating.append(float(r["rating"]))
        meta_date.append(r["date"])

    relation_edges = {k: v for k, v in payload["edge_index_relations"].items()}

    ds = FLAGDataset(
        dataset_name="yelpchi",
        node_type="review",
        node_features=payload["x"],
        labels=payload["y"],
        raw_texts=raw_texts,
        edge_index=payload["edge_index_homo"],
        relation_edges=relation_edges,
        relation_names=list(payload["relation_names"]),
        train_idx=_mask_to_idx(payload["train_mask"]),
        val_idx=_mask_to_idx(payload["val_mask"]),
        test_idx=_mask_to_idx(payload["test_mask"]),
        text_status=status,
        unlabeled_mask=payload.get("unlabeled_mask"),
        node_metadata={"user_id": meta_user, "product_id": meta_prod,
                       "rating": meta_rating, "date": meta_date},
        mapping_metadata={
            "method": "adjacency_proof",
            "node_ordering": "PROVEN — R-U-R (98,630) and R-S-R (6,805,486) "
                             "sparsity patterns matched entry-for-entry; labels "
                             "45,954/45,954",
            "text_join": "VERIFIED (statistical) — within-product TF-IDF cosine "
                         "0.0747 vs 0.0433 same-block/different-product control "
                         "(1.72x), decaying to control under shift",
            "rtr_reproduced": False,
            "rtr_note": "date-dependent relation not reproducible from this "
                        "metadata snapshot; does not affect the ordering proof",
            "verify_command": "python -m scripts.verify_yelpchi_alignment",
        },
        dataset_metadata={
            "graph_source": "CARE-GNN YelpChi.mat",
            "text_source": "Mukherjee et al. ICWSM 2013 corpus (mirror)",
            "canonical_node_count": n,
            "edge_index_note": "DERIVED union view ('homo' in the .mat, verified "
                               "to equal the union of the three relations for "
                               "YelpChi). relation_edges keeps all three.",
            "not_a_flag_reproduction": "The FLAG paper reports no YelpChi "
                                       "results and states the dataset lacks "
                                       "text. Any run here is a novel experiment.",
        },
        provenance={
            "node_features": CANONICAL, "labels": CANONICAL,
            "relation_edges": CANONICAL, "edge_index": DERIVED,
            "raw_texts": VERIFIED, "splits": DERIVED,
        },
    )
    ds.validate()
    return ds

"""Build the text-augmented Amazon dataset.

Node = a USER, so node text must be aggregated from that user's reviews
(brief sections 7-8). Two views are produced and both are kept:

    full_user_text   every review, losslessly serialised   <- canonical store
    flag_text        a controlled view for prompt budgets  <- experiment knob

Nothing is silently truncated: `flag_text` is built by an explicit, recorded
config, and `full_user_text` always retains everything (brief sections 26, 50).

ALIGNMENT PROVENANCE
--------------------
Nodes [3305, 11944) are PROVEN: the labelled block follows reviewerID
first-appearance order in `reviews_Musical_Instruments.json.gz`, confirmed by a
8,639/8,639 label-sequence match AND an exact induced U-P-U comparison
(294,764 edges, zero mismatches). Reproduce with
`python -m scripts.verify_amazon_alignment`.

Nodes [0, 3305) are UNRESOLVED and carry empty text. They are the unlabelled 1%
sample drawn by `rd.sample` over a set-difference list, whose order depends on
PYTHONHASHSEED. CARE-GNN's own protocol excludes them from train/val/test, so no
evaluated node lacks text. They are NOT guessed (brief section 56).
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass

import torch

from flagbench.fraud_text.base import (
    CANONICAL, DERIVED, UNRESOLVED, VERIFIED, FLAGDataset,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
NATIVE = ROOT / "data" / "benchmark" / "native_amazon" / "graph.pt"
MAPPING = ROOT / "datasets" / "processed" / "amazon" / "amazon_user_mapping.csv"
RAW = ROOT / "datasets" / "raw" / "amazon_musical" / "reviews_Musical_Instruments.json.gz"


@dataclass(frozen=True)
class TextConfig:
    """How to turn a user's reviews into one string.

    Defaults are lossless. Any lossy setting is recorded in dataset_metadata so
    a result can never be read without knowing which view produced it.
    """

    aggregation: str = "all_reviews"     # all_reviews | most_recent_n
    max_reviews: int | None = None
    max_chars: int | None = None

    def as_record(self) -> dict:
        return {"aggregation": self.aggregation, "max_reviews": self.max_reviews,
                "max_chars": self.max_chars}


def serialize_reviews(reviews: list[dict], config: TextConfig) -> str:
    """Deterministic user-level serialisation. Reviews stay in corpus order."""
    items = reviews
    if config.aggregation == "most_recent_n" and config.max_reviews:
        items = sorted(reviews, key=lambda r: r.get("unixReviewTime", 0),
                       reverse=True)[:config.max_reviews]
    elif config.max_reviews:
        items = reviews[:config.max_reviews]

    blocks = []
    for k, r in enumerate(items, 1):
        blocks.append(
            f"Review {k}:\n"
            f"Rating: {r.get('overall', '')}\n"
            f"Date: {r.get('reviewTime', '')}\n"
            f"Product: {r.get('asin', '')}\n"
            f"Summary: {r.get('summary', '') or ''}\n"
            f"Text: {r.get('reviewText', '') or ''}"
        )
    out = "\n\n".join(blocks)
    if config.max_chars is not None and len(out) > config.max_chars:
        out = out[:config.max_chars]
    return out


def _collect_reviews(wanted: set[str]) -> dict[str, list[dict]]:
    by_user: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(RAW, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rid = d["reviewerID"]
            if rid in wanted:
                by_user[rid].append(d)
    return by_user


def _mask_to_idx(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def build(config: TextConfig | None = None) -> FLAGDataset:
    config = config or TextConfig()
    for p in (NATIVE, MAPPING, RAW):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing.\n"
                f"  canonical graph: python -m experiments.yelpchi_amazon."
                f"build_native_benchmark --dataset amazon\n"
                f"  mapping:         python -m scripts.verify_amazon_alignment"
            )

    payload = torch.load(NATIVE, map_location="cpu")
    n = int(payload["y"].shape[0])

    rows = list(csv.DictReader(MAPPING.open(encoding="utf-8")))
    if len(rows) != n:
        raise RuntimeError(f"mapping has {len(rows)} rows, graph has {n} nodes")

    node_rid: list[str] = [""] * n
    for r in rows:
        if r["confidence"] == "EXACT" and r["reviewer_id"]:
            node_rid[int(r["graph_node_id"])] = r["reviewer_id"]

    by_user = _collect_reviews({r for r in node_rid if r})

    raw_texts: list[str] = [""] * n
    full_texts: list[str] = [""] * n
    status: list[str] = [UNRESOLVED] * n
    n_reviews: list[int] = [0] * n
    lossless = TextConfig()
    for i, rid in enumerate(node_rid):
        if not rid:
            continue
        reviews = by_user.get(rid, [])
        full_texts[i] = serialize_reviews(reviews, lossless)
        raw_texts[i] = (full_texts[i] if config == lossless
                        else serialize_reviews(reviews, config))
        status[i] = VERIFIED
        n_reviews[i] = len(reviews)

    relation_edges = {k: v for k, v in payload["edge_index_relations"].items()}

    ds = FLAGDataset(
        dataset_name="amazon",
        node_type="user",
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
        node_metadata={"reviewer_id": node_rid, "n_reviews": n_reviews,
                       "full_user_text": full_texts},
        mapping_metadata={
            "method": "generator_order + upu_adjacency_proof",
            "labeled_block": "PROVEN — nodes [3305, 11944) follow reviewerID "
                             "first-appearance order; labels 8,639/8,639 and "
                             "induced U-P-U 294,764 edges, zero mismatches",
            "unlabeled_prefix": "UNRESOLVED — nodes [0, 3305) come from "
                                "rd.sample over a set-difference list "
                                "(PYTHONHASHSEED-dependent). 16-column "
                                "fingerprint recovery gave only 17% unique, "
                                "so identities are left empty, not guessed",
            "verify_command": "python -m scripts.verify_amazon_alignment",
        },
        dataset_metadata={
            "graph_source": "CARE-GNN Amazon.mat",
            "text_source": "McAuley 2014 reviews_Musical_Instruments.json.gz "
                           "(full, NOT 5-core)",
            "text_config": config.as_record(),
            "edge_index_note": "DERIVED. NOTE: for Amazon the .mat 'homo' is NOT "
                               "the union of the three relations (Jaccard 0.626) "
                               "— it is CARE-GNN's own fourth view, used as-is.",
            "not_a_flag_reproduction": "The FLAG paper reports no Amazon results "
                                       "and states the dataset lacks text. Any "
                                       "run here is a novel experiment.",
        },
        provenance={
            "node_features": CANONICAL, "labels": CANONICAL,
            "relation_edges": CANONICAL, "edge_index": CANONICAL,
            "raw_texts": DERIVED, "splits": DERIVED,
        },
    )
    ds.validate()
    return ds

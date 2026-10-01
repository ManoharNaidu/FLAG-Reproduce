"""FLAGDataset -> the exact payload `flagbench.experiments.runner` already reads.

    FLAGDataset  ->  data/benchmark/flag_<key>/graph.pt  ->  existing FLAG code

No FLAG core file is modified. The runner, trainer, sampler, metrics and every
bundled backbone consume this payload unchanged; the adapter's whole job is to
satisfy the contract documented in research/dataset_integration_audit.md
section 2.

WHAT GETS STAMPED, AND WHY
--------------------------
The manifest written alongside the payload records that this is a
text-augmented STUDY, the mapping evidence, the text coverage, and the fact that
three relations were collapsed. Those facts decide whether a number is readable,
so they travel with the data rather than living only in a document.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

import torch

from flagbench.flag_adapter.graph_adapter import homogeneous_edge_index
from flagbench.flag_adapter.text_adapter import texts_for_flag
from flagbench.fraud_text.base import FLAGDataset

ROOT = pathlib.Path(__file__).resolve().parents[3]

REGISTRY_KEY = {"yelpchi": "yelpchi_text", "amazon": "amazon_text"}
"""FLAGDataset name -> the registry key its runs are recorded under.

Deliberately NOT the canonical `yelpchi` / `amazon` keys: those stay blocked for
every text variant, because the FLAG paper reports no results for them.
"""


def _idx_to_mask(idx: torch.Tensor, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return mask


def to_flag_payload(ds: FLAGDataset, max_chars: int | None = None,
                    edge_strategy: str = "canonical_homo") -> tuple[dict, dict]:
    """Build the runner-compatible payload plus its manifest."""
    n = ds.num_nodes
    edge_index, edge_prov = homogeneous_edge_index(ds, edge_strategy)
    texts, text_prov = texts_for_flag(ds, max_chars=max_chars)

    payload = {
        "x": ds.node_features.float(),
        "edge_index": edge_index,
        "y": ds.labels.long(),
        "raw_texts": texts,
        "train_mask": _idx_to_mask(ds.train_idx, n),
        "val_mask": _idx_to_mask(ds.val_idx, n),
        "test_mask": _idx_to_mask(ds.test_idx, n),
        "original_node_ids": torch.arange(n),
        "label_names": ["benign", "fraud"],
    }

    manifest = {
        "dataset": REGISTRY_KEY.get(ds.dataset_name, ds.dataset_name),
        "source_dataset": ds.dataset_name,
        "node_type": ds.node_type,
        "experiment_type": "text_augmented_study",
        "native_text": False,
        "is_flag_reproduction": False,
        "not_comparable_to": "FLAG Table 4 -- the paper reports no YelpChi or "
                             "Amazon results and states both lack textual "
                             "information",
        "constructed": {
            "num_nodes": n,
            "num_features": ds.num_features,
            "num_edges": int(edge_index.shape[1]),
            "label_distribution": ds.label_counts(),
            "splits": ds.split_counts(),
        },
        "edge_index_provenance": edge_prov,
        "text_provenance": text_prov,
        "mapping": ds.mapping_metadata,
        "field_provenance": ds.provenance,
        "relations_preserved_in": "datasets/processed/<name>/flag_dataset.pt "
                                  "(relation_edges); the payload here is the "
                                  "collapsed view FLAG requires",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    return payload, manifest


def write_flag_payload(ds: FLAGDataset, max_chars: int | None = None,
                       edge_strategy: str = "canonical_homo") -> pathlib.Path:
    key = REGISTRY_KEY.get(ds.dataset_name, ds.dataset_name)
    payload, manifest = to_flag_payload(ds, max_chars, edge_strategy)
    out = ROOT / "data" / "benchmark" / f"flag_{key}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "graph.pt"
    torch.save(payload, path)
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path

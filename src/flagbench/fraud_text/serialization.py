"""Persist a FLAGDataset, with checksums and a reproduction manifest.

Layout per dataset (brief sections 21, 29, 43):

    datasets/processed/<name>/
        flag_dataset.pt      the payload
        metadata.json        schema-level description
        manifest.json        sources, checksums, mapping status
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib

import torch

from flagbench.fraud_text.base import FLAGDataset

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "datasets" / "processed"


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def out_dir(name: str) -> pathlib.Path:
    return PROCESSED / name


def _display_path(path: pathlib.Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    `save()` must work for an arbitrary directory (tests write to a tmpdir,
    which on Windows can even be on another drive), so this never raises.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def save(ds: FLAGDataset, directory: pathlib.Path | None = None) -> pathlib.Path:
    ds.validate()
    directory = directory or out_dir(ds.dataset_name)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_name": ds.dataset_name,
        "node_type": ds.node_type,
        "x": ds.node_features,
        "y": ds.labels,
        "raw_texts": ds.raw_texts,
        "text_status": ds.text_status,
        "edge_index": ds.edge_index,
        "relation_edges": ds.relation_edges,
        "relation_names": ds.relation_names,
        "train_idx": ds.train_idx,
        "val_idx": ds.val_idx,
        "test_idx": ds.test_idx,
        "unlabeled_mask": ds.unlabeled_mask,
        "node_metadata": ds.node_metadata,
        "mapping_metadata": ds.mapping_metadata,
        "dataset_metadata": ds.dataset_metadata,
        "provenance": ds.provenance,
    }
    path = directory / "flag_dataset.pt"
    torch.save(payload, path)

    cov = ds.text_coverage()
    metadata = {
        "dataset": ds.dataset_name,
        "node_type": ds.node_type,
        "num_nodes": ds.num_nodes,
        "num_features": ds.num_features,
        "num_edges_derived": ds.num_edges,
        "relations": {k: int(v.shape[1]) for k, v in ds.relation_edges.items()},
        "label_counts": ds.label_counts(),
        "label_semantics": "1 = fraud, 0 = benign (canonical, unchanged)",
        "splits": ds.split_counts(),
        "text_coverage": cov,
        "provenance": ds.provenance,
        "mapping": ds.mapping_metadata,
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "dataset": ds.dataset_name,
        "node_type": ds.node_type,
        "node_count": ds.num_nodes,
        "feature_dim": ds.num_features,
        "relations": list(ds.relation_edges),
        "text_alignment": ("EXACT" if cov["verified"] == cov["num_nodes"]
                           else "PARTIAL"),
        "mapping_status": ("EXACT" if cov["verified"] == cov["num_nodes"]
                           else "PARTIAL — see mapping_metadata"),
        "verified_nodes": cov["verified"],
        "unresolved_nodes": cov["unresolved"],
        **{k: v for k, v in ds.dataset_metadata.items() if isinstance(v, (str, int))},
        "output_file": _display_path(path),
        "output_sha256": sha256_file(path),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load(name: str, directory: pathlib.Path | None = None) -> FLAGDataset:
    directory = directory or out_dir(name)
    path = directory / "flag_dataset.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python -m scripts.build_flag_dataset "
            f"--dataset {name}")
    p = torch.load(path, map_location="cpu")
    ds = FLAGDataset(
        dataset_name=p["dataset_name"], node_type=p["node_type"],
        node_features=p["x"], labels=p["y"], raw_texts=p["raw_texts"],
        edge_index=p["edge_index"], relation_edges=p["relation_edges"],
        relation_names=p["relation_names"],
        train_idx=p["train_idx"], val_idx=p["val_idx"], test_idx=p["test_idx"],
        text_status=p.get("text_status", []),
        unlabeled_mask=p.get("unlabeled_mask"),
        node_metadata=p.get("node_metadata", {}),
        mapping_metadata=p.get("mapping_metadata", {}),
        dataset_metadata=p.get("dataset_metadata", {}),
        provenance=p.get("provenance", {}),
    )
    ds.validate()
    return ds

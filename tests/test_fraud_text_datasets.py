"""Tests for the text-augmented YelpChi/Amazon datasets (brief section 35).

These are alignment tests, not smoke tests. The failure mode they exist to catch
is silent: a node carrying another node's text, features or label still trains
and still produces plausible numbers.

Run:
    .venv-fraudtext/Scripts/python -m pytest tests/test_fraud_text_datasets.py -v
"""
from __future__ import annotations

import pathlib

import pytest
import torch

pytest.importorskip("flagbench.fraud_text", reason="flagbench not importable")

from flagbench.flag_adapter import REGISTRY_KEY, to_flag_payload  # noqa: E402
from flagbench.fraud_text import (  # noqa: E402
    UNRESOLVED, VERIFIED, load_dataset, validate_dataset,
)
from flagbench.fraud_text.base import DatasetIntegrityError, FLAGDataset  # noqa: E402
from flagbench.fraud_text.serialization import out_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASETS = ["yelpchi", "amazon"]

EXPECTED = {
    # Canonical statistics, verified against the .mat bytes. Hard-coded on
    # purpose: if a rebuild changes any of these, the canonical data has been
    # altered and the test must fail.
    "yelpchi": {"nodes": 45_954, "features": 32, "node_type": "review",
                "relations": {"rur": 98_630, "rtr": 1_147_232, "rsr": 6_805_486},
                "labels": {0: 39_277, 1: 6_677}},
    "amazon": {"nodes": 11_944, "features": 25, "node_type": "user",
               "relations": {"upu": 351_216, "usu": 7_132_958, "uvu": 2_073_474},
               "labels": {0: 11_123, 1: 821}},
}


def _available(name: str) -> bool:
    return (out_dir(name) / "flag_dataset.pt").exists()


@pytest.fixture(scope="module", params=DATASETS)
def ds(request) -> FLAGDataset:
    name = request.param
    if not _available(name):
        pytest.skip(f"{name} not built; run scripts.prepare_dataset --dataset {name}")
    return load_dataset(name)


# --- canonical data must be untouched -------------------------------------
def test_node_and_feature_counts(ds):
    e = EXPECTED[ds.dataset_name]
    assert ds.num_nodes == e["nodes"]
    assert ds.num_features == e["features"]
    assert ds.node_type == e["node_type"]


def test_labels_unchanged(ds):
    assert ds.label_counts() == EXPECTED[ds.dataset_name]["labels"]
    assert set(ds.labels.unique().tolist()) <= {0, 1}


def test_relations_preserved_and_not_collapsed(ds):
    e = EXPECTED[ds.dataset_name]["relations"]
    assert set(ds.relation_edges) == set(e)
    for name, count in e.items():
        assert int(ds.relation_edges[name].shape[1]) == count, name
    assert set(ds.relation_names) == set(e)


# --- alignment -------------------------------------------------------------
def test_text_aligned_to_nodes(ds):
    assert len(ds.raw_texts) == ds.num_nodes
    assert len(ds.text_status) == ds.num_nodes


def test_features_aligned_to_nodes(ds):
    assert ds.node_features.shape[0] == ds.num_nodes
    assert ds.labels.shape[0] == ds.num_nodes


def test_text_status_is_consistent_with_text(ds):
    for t, s in zip(ds.raw_texts, ds.text_status):
        assert s in (VERIFIED, UNRESOLVED)
        if s == VERIFIED:
            assert t, "a VERIFIED node must carry text"
        else:
            assert not t, "an UNRESOLVED node must carry no text"


def test_every_evaluated_node_has_text(ds):
    """The only coverage guarantee that affects results."""
    evaluated = torch.cat([ds.train_idx, ds.val_idx, ds.test_idx])
    missing = [i for i in evaluated.tolist() if ds.text_status[i] != VERIFIED]
    assert not missing, f"{len(missing)} evaluated nodes lack text"


# --- splits ----------------------------------------------------------------
def test_no_split_leakage(ds):
    tr = set(ds.train_idx.tolist())
    va = set(ds.val_idx.tolist())
    te = set(ds.test_idx.tolist())
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_splits_exclude_unlabeled(ds):
    if ds.unlabeled_mask is None:
        pytest.skip("no unlabeled_mask")
    for idx in (ds.train_idx, ds.val_idx, ds.test_idx):
        assert not bool(ds.unlabeled_mask[idx].any())


def test_indices_in_bounds(ds):
    for idx in (ds.train_idx, ds.val_idx, ds.test_idx):
        assert int(idx.max()) < ds.num_nodes and int(idx.min()) >= 0
    assert int(ds.edge_index.max()) < ds.num_nodes
    for ei in ds.relation_edges.values():
        assert int(ei.max()) < ds.num_nodes


# --- provenance ------------------------------------------------------------
def test_provenance_recorded(ds):
    for field in ("node_features", "labels", "relation_edges", "edge_index",
                  "raw_texts"):
        assert field in ds.provenance
    assert ds.mapping_metadata.get("method")


def test_never_claims_to_be_a_flag_reproduction(ds):
    assert "not_a_flag_reproduction" in ds.dataset_metadata


# --- validator -------------------------------------------------------------
def test_validator_passes(ds):
    report = validate_dataset(ds)
    assert report.ok, report.render()


def test_validator_catches_text_misalignment(ds):
    import copy
    broken = copy.copy(ds)
    broken.raw_texts = ds.raw_texts[:-1]
    with pytest.raises(DatasetIntegrityError):
        broken.validate()


def test_validator_catches_split_leakage(ds):
    import copy
    broken = copy.copy(ds)
    broken.val_idx = ds.train_idx[:5].clone()
    with pytest.raises(DatasetIntegrityError):
        broken.validate()


# --- serialization ---------------------------------------------------------
def test_roundtrip(tmp_path, ds):
    from flagbench.fraud_text.serialization import load, save
    save(ds, tmp_path)
    again = load(ds.dataset_name, tmp_path)
    assert again.num_nodes == ds.num_nodes
    assert again.label_counts() == ds.label_counts()
    assert again.raw_texts[:20] == ds.raw_texts[:20]
    assert torch.equal(again.node_features, ds.node_features)
    assert set(again.relation_edges) == set(ds.relation_edges)


# --- FLAG adapter ----------------------------------------------------------
def test_flag_payload_has_the_keys_the_runner_reads(ds):
    payload, manifest = to_flag_payload(ds)
    for key in ("x", "edge_index", "y", "raw_texts", "train_mask", "val_mask",
                "test_mask", "original_node_ids", "label_names"):
        assert key in payload, key
    n = ds.num_nodes
    assert payload["x"].shape[0] == n
    assert payload["y"].shape[0] == n
    assert len(payload["raw_texts"]) == n
    for m in ("train_mask", "val_mask", "test_mask"):
        assert payload[m].dtype == torch.bool and payload[m].shape[0] == n


def test_flag_payload_masks_match_the_index_splits(ds):
    payload, _ = to_flag_payload(ds)
    assert int(payload["train_mask"].sum()) == int(ds.train_idx.numel())
    assert int(payload["val_mask"].sum()) == int(ds.val_idx.numel())
    assert int(payload["test_mask"].sum()) == int(ds.test_idx.numel())


def test_flag_manifest_is_stamped_as_a_study_not_a_reproduction(ds):
    _, manifest = to_flag_payload(ds)
    assert manifest["experiment_type"] == "text_augmented_study"
    assert manifest["native_text"] is False
    assert manifest["is_flag_reproduction"] is False
    assert manifest["dataset"] == REGISTRY_KEY[ds.dataset_name]
    assert manifest["dataset"].endswith("_text")


def test_relations_survive_the_adapter(ds):
    """The adapter collapses relations for FLAG but must not destroy them."""
    to_flag_payload(ds)
    assert len(ds.relation_edges) == 3


# --- registry --------------------------------------------------------------
def test_canonical_keys_still_refuse_text_variants():
    from flagbench.registry.registry import validate
    for key in ("yelpchi", "amazon"):
        for variant in ("text", "flag", "flag_finetuned"):
            assert not validate(key, "gcn", variant).ok, (key, variant)


def test_derived_keys_permit_text_variants():
    from flagbench.registry.registry import validate
    for key in ("yelpchi_text", "amazon_text"):
        for variant in ("text", "flag"):
            assert validate(key, "gcn", variant).ok, (key, variant)


def test_derived_keys_are_not_marked_native():
    from flagbench.registry.registry import DATASET_REGISTRY
    for key in ("yelpchi_text", "amazon_text"):
        spec = DATASET_REGISTRY[key]
        assert spec.text_source == "derived"
        assert spec.has_native_text is False

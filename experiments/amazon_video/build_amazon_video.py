"""Build the Amazon Video fraud graph (text-augmented study) the way CARE-GNN built Amazon.

    python -P experiments/amazon_video/build_amazon_video.py

Input:  datasets/raw/amazon_video/reviews_Amazon_Instant_Video.json.gz
        (McAuley 2014 full category dump, snap.stanford.edu; sha256 recorded in the manifest)
Output: data/benchmark/flag_amazon_video_text/{graph.pt, dataset_manifest.json}

Construction, following methods/care_gnn/amazon_preprocess.py except where noted:

* Users = reviewers, in first-appearance order; each user's reviews in corpus order.
* Labels (CARE-GNN rule): users with >= 20 helpful-vote ratings; helpful ratio > 0.8 -> benign (0),
  < 0.2 -> fraud (1); everyone else unlabelled.
* OUR ADDITION: the labelled set is downsampled to a 1:10 fraud:benign ratio (all benign kept,
  fraud sampled with a seeded RNG), matching the Reddit/Instagram benchmark protocol. Amazon Video's
  raw labelled set is 40% fraud.
* Context nodes (CARE-GNN): 1% of the unlabelled users, sampled with a seeded RNG (CARE-GNN's
  rd.sample over a set difference is not reproducible), placed first; excluded from every split.
* Graph: user-product-user (CARE-GNN's only implemented relation): two users are connected if they
  reviewed at least one common product. Computed with a sparse incidence product instead of the
  O(n^2) Python loop; identical result.
* Features: CARE-GNN's own `build_features` (the generator module is imported as-is; its two
  training-only imports, xgboost and run_ours, are stubbed because no function used here touches
  them), then CARE-GNN's "filter polluted features" step: keep columns 0-18 and 30-35 (drops the
  helpful-vote aggregates the labels are derived from) -> 25 columns.
* Text: each user's reviews serialised exactly like the Amazon (Musical Instruments) dataset
  (flagbench.fraud_text.amazon.serialize_reviews, lossless). Every node has text.
* Split (CARE-GNN convention): stratified train_test_split(test_size=0.60, random_state=2) over the
  labelled nodes, then 25% of train carved out as validation (random_state=0), as for Amazon/YelpChi.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import types
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw" / "amazon_video" / "reviews_Amazon_Instant_Video.json.gz"
GENERATOR = ROOT / "methods" / "care_gnn" / "amazon_preprocess.py"
OUT = ROOT / "data" / "benchmark" / "flag_amazon_video_text"
SEED = 0
RATIO = 10                    # benign : fraud after downsampling
CONTEXT_FRACTION = 0.01       # CARE-GNN's share of unlabelled users kept as graph context
KEEP_COLUMNS = list(range(0, 19)) + list(range(30, 36))   # CARE-GNN "filter polluted features"

sys.path.insert(0, str(ROOT / "src"))
from flagbench.fraud_text.amazon import TextConfig, serialize_reviews  # noqa: E402


def load_care_gnn_generator():
    """Import methods/care_gnn/amazon_preprocess.py unchanged, as a module."""
    for name, attrs in (("xgboost", {}), ("run_ours", {"pos_neg_split": None, "undersample": None})):
        if name not in sys.modules:
            stub = types.ModuleType(name); stub.__dict__.update(attrs); sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location("care_gnn_amazon_preprocess", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # defines functions only; its pipeline is under __main__
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rng = np.random.default_rng(SEED)
    users: "OrderedDict[str, list]" = OrderedDict()
    n_reviews = 0
    with gzip.open(RAW, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            r.setdefault("reviewText", ""); r.setdefault("summary", "")
            users.setdefault(r["reviewerID"], []).append(r)
            n_reviews += 1

    benign, fraud, unlabelled = [], [], []
    for u, revs in users.items():
        helpful = sum(x["helpful"][0] for x in revs); votes = sum(x["helpful"][1] for x in revs)
        if votes >= 20 and helpful / votes > 0.8:
            benign.append(u)
        elif votes >= 20 and helpful / votes < 0.2:
            fraud.append(u)
        else:
            unlabelled.append(u)
    n_fraud_keep = round(len(benign) / RATIO)
    fraud_kept = {fraud[i] for i in rng.choice(len(fraud), size=n_fraud_keep, replace=False)}
    context = [unlabelled[i] for i in sorted(rng.choice(len(unlabelled), size=int(len(unlabelled) * CONTEXT_FRACTION),
                                                        replace=False).tolist())]
    benign_set = set(benign)
    labelled = [u for u in users if u in benign_set or u in fraud_kept]      # first-appearance order
    nodes = context + labelled
    y = np.array([-1] * len(context) + [0 if u in benign_set else 1 for u in labelled])
    node_reviews = OrderedDict((u, users[u]) for u in nodes)
    print(f"users {len(users):,} reviews {n_reviews:,} | benign {len(benign)} fraud {len(fraud)} "
          f"(kept {len(fraud_kept)}) | context {len(context)} | nodes {len(nodes)}", flush=True)

    # user-product-user via sparse incidence (same edges as CARE-GNN's pairwise loop)
    prods = {p: j for j, p in enumerate(sorted({r["asin"] for u in nodes for r in users[u]}))}
    rows, cols = [], []
    for i, u in enumerate(nodes):
        for p in {r["asin"] for r in users[u]}:
            rows.append(i); cols.append(prods[p])
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(prods)))
    A = (B @ B.T).tocoo()
    keep = A.row != A.col
    edge_index = torch.tensor(np.vstack([A.row[keep], A.col[keep]]), dtype=torch.long)
    print(f"products {len(prods):,} | edges {edge_index.shape[1]:,} | avg degree {edge_index.shape[1] / len(nodes):.1f}",
          flush=True)

    gen = load_care_gnn_generator()
    with contextlib.redirect_stdout(io.StringIO()):          # CARE-GNN prints every user index
        feats36 = gen.build_features(node_reviews).toarray()
    x = torch.tensor(feats36[:, KEEP_COLUMNS], dtype=torch.float32)

    texts = [serialize_reviews(users[u], TextConfig()) for u in nodes]

    lab_idx = np.arange(len(context), len(nodes)); lab_y = y[lab_idx]
    idx_train, idx_test = train_test_split(lab_idx, stratify=lab_y, test_size=0.60, random_state=2)
    idx_train, idx_val = train_test_split(idx_train, stratify=y[idx_train], test_size=0.25, random_state=0)
    n = len(nodes)
    mask = lambda idx: torch.zeros(n, dtype=torch.bool).index_fill_(0, torch.tensor(idx), True)
    yt = torch.tensor(np.where(y < 0, 0, y), dtype=torch.long)   # unlabelled get 0 but are in no split

    payload = {"x": x, "edge_index": edge_index, "y": yt, "raw_texts": texts,
               "train_mask": mask(idx_train), "val_mask": mask(idx_val), "test_mask": mask(idx_test),
               "original_node_ids": torch.arange(n), "label_names": ["benign", "fraud"]}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT / "graph.pt")

    per = lambda idx: {"n": int(len(idx)), "per_class": {str(c): int((y[idx] == c).sum()) for c in (0, 1)}}
    manifest = {
        "dataset": "amazon_video_text", "source_dataset": "amazon_instant_video", "node_type": "user",
        "experiment_type": "text_augmented_study", "self_constructed": True, "native_text": False,
        "is_flag_reproduction": False,
        "not_comparable_to": "FLAG Table 4 and any published Amazon result: graph built here with CARE-GNN's recipe",
        "source": {"file": str(RAW.relative_to(ROOT)), "sha256": sha256(RAW),
                   "url": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video.json.gz",
                   "reviewers": len(users), "reviews": n_reviews},
        "labels": {"rule": "votes>=20; helpful ratio >0.8 benign, <0.2 fraud (CARE-GNN)",
                   "raw_benign": len(benign), "raw_fraud": len(fraud),
                   "downsampling": f"all benign kept; fraud sampled to 1:{RATIO} (seed {SEED})",
                   "kept_fraud": len(fraud_kept)},
        "context_nodes": {"count": len(context), "fraction_of_unlabelled": CONTEXT_FRACTION, "seed": SEED,
                          "note": "graph context only, excluded from every split; have text"},
        "constructed": {"num_nodes": n, "num_features": int(x.shape[1]), "num_edges": int(edge_index.shape[1]),
                        "avg_degree": round(edge_index.shape[1] / n, 1), "num_products": len(prods),
                        "label_distribution": {"0": int((y == 0).sum()), "1": int((y == 1).sum()),
                                               "unlabelled": len(context)},
                        "splits": {"train": per(idx_train), "val": per(idx_val), "test": per(idx_test)}},
        "graph": "user-product-user (share >= 1 product), CARE-GNN's implemented relation; sparse B·Bᵀ",
        "features": {"generator": str(GENERATOR.relative_to(ROOT)), "columns_kept": KEEP_COLUMNS,
                     "note": "CARE-GNN build_features imported and run unchanged; 'filter polluted features' applied"},
        "text": {"serializer": "flagbench.fraud_text.amazon.serialize_reviews", "config": TextConfig().as_record(),
                 "empty_texts": int(sum(1 for t in texts if not t))},
        "split_protocol": "stratified train_test_split(test_size=0.60, random_state=2) over labelled nodes; "
                          "val = 25% of train (random_state=0)",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["constructed"], indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}/graph.pt")


if __name__ == "__main__":
    main()

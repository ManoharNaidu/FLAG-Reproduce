"""Turn YelpChi.mat / Amazon.mat into a trainable graph payload.

These two datasets are registered in `flagbench.registry` with
`has_native_text=False` (registry.py:294-303), which already hard-blocks the
`flag` / `+text` variants -- correctly, since the FLAG paper itself says they
"lack textual information". So this script only ever prepares them for
*native* baseline evaluation (CARE-GNN, BWGNN, GCN, GAT, GeniePath, ... on
their own published protocol), never for FLAG.

It also deliberately does NOT go through `flagbench.datasets.benchmark`'s
1:10 downsampling construction. That construction is FLAG's own invented
protocol for artificially imbalancing Reddit/Instagram (which are naturally
near-balanced). YelpChi and Amazon are already naturally imbalanced fraud
graphs with an established split convention from their own literature, so
downsampling them again would not reproduce anything -- it would invent a
third, non-standard dataset variant.

Split protocol -- reproduced exactly from the official CARE-GNN code
(methods/care_gnn/train.py:50-61, methods/care_gnn/utils.py:16-46), because that
is the one actually cited in the provenance table (methods/README.md) as the
source for these two datasets:

  - .mat keys verified by direct inspection (not assumed):
      YelpChi: net_rur, net_rtr, net_rsr, homo, features, label
      Amazon:  net_upu, net_usu, net_uvu, homo, features, label
  - YelpChi: stratified train_test_split(test_size=0.60, random_state=2) over
    all 45,954 nodes.
  - Amazon: nodes 0..3304 are excluded entirely (CARE-GNN's own comment: "0-3304
    are unlabeled nodes"); the same split is applied to nodes 3305..11943 only.

Everything past that point -- carving a validation set out of CARE-GNN's train
split -- is OUR addition, not part of the official protocol, and is labelled
as such in the manifest so nobody mistakes it for upstream's convention.

Usage:
    python -m experiments.yelpchi_amazon.build_native_benchmark --dataset yelpchi
    python -m experiments.yelpchi_amazon.build_native_benchmark --dataset amazon
    python -m experiments.yelpchi_amazon.build_native_benchmark --dataset all
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "benchmark"

CARE_GNN_REPO = "https://github.com/YingtongDou/CARE-GNN"
CARE_GNN_PAPER = (
    "Dou et al., Enhancing Graph Neural Network-based Fraud Detectors "
    "against Camouflaged Fraudsters, CIKM 2020 (arXiv:2008.08692)"
)


@dataclass(frozen=True)
class MatSpec:
    dataset: str
    filename: str
    relation_keys: dict[str, str]  # our name -> .mat key
    unlabeled_prefix: int  # nodes [0, unlabeled_prefix) excluded (0 = none)
    feature_key: str = "features"
    label_key: str = "label"
    homo_key: str = "homo"


SPECS: dict[str, MatSpec] = {
    "yelpchi": MatSpec(
        dataset="yelpchi",
        filename="YelpChi.mat",
        relation_keys={"rur": "net_rur", "rtr": "net_rtr", "rsr": "net_rsr"},
        unlabeled_prefix=0,
    ),
    "amazon": MatSpec(
        dataset="amazon",
        filename="Amazon.mat",
        relation_keys={"upu": "net_upu", "usu": "net_usu", "uvu": "net_uvu"},
        unlabeled_prefix=3305,
    ),
}

# Our addition, not CARE-GNN's. Carve a validation fold out of their train
# split so early stopping is possible. Kept small so the official train/test
# sizes are barely perturbed.
VAL_FRACTION_OF_TRAIN = 0.25
VAL_SPLIT_SEED = 0


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_to_edge_index(mat: csc_matrix) -> torch.Tensor:
    """Exact nonzero pattern as a [2, E] edge_index. No symmetrisation added."""
    coo = mat.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)


def is_symmetric(mat: csc_matrix) -> bool:
    return (mat != mat.T).nnz == 0


def build_masks(
    num_nodes: int, labels: np.ndarray, spec: MatSpec
) -> dict[str, torch.Tensor]:
    unlabeled_mask = np.zeros(num_nodes, dtype=bool)
    unlabeled_mask[: spec.unlabeled_prefix] = True

    labeled_index = np.arange(spec.unlabeled_prefix, num_nodes)
    labeled_y = labels[labeled_index]

    # Official CARE-GNN split: stratified, test_size=0.60, random_state=2.
    idx_train, idx_test = train_test_split(
        labeled_index,
        stratify=labeled_y,
        test_size=0.60,
        random_state=2,
        shuffle=True,
    )

    # Our addition: carve val out of the official train split.
    idx_train, idx_val = train_test_split(
        idx_train,
        stratify=labels[idx_train],
        test_size=VAL_FRACTION_OF_TRAIN,
        random_state=VAL_SPLIT_SEED,
        shuffle=True,
    )

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return {
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
        "test_mask": torch.from_numpy(test_mask),
        "unlabeled_mask": torch.from_numpy(unlabeled_mask),
    }


def build(dataset: str) -> tuple[dict, dict]:
    spec = SPECS[dataset]
    raw_path = RAW_DIR / dataset / spec.filename
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Expected the official CARE-GNN .mat file "
            f"({CARE_GNN_REPO})."
        )

    mat = loadmat(raw_path)
    missing = [
        k
        for k in [spec.feature_key, spec.label_key, spec.homo_key,
                  *spec.relation_keys.values()]
        if k not in mat
    ]
    if missing:
        raise KeyError(
            f"{raw_path.name} is missing expected key(s) {missing}. "
            f"This does not look like the official CARE-GNN {dataset} file."
        )

    labels = mat[spec.label_key].flatten().astype(np.int64)
    features = mat[spec.feature_key]
    x = torch.tensor(np.asarray(features.todense()), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    num_nodes = int(y.shape[0])

    homo = mat[spec.homo_key]
    relations = {name: mat[key] for name, key in spec.relation_keys.items()}

    edge_index_relations = {
        name: sparse_to_edge_index(m) for name, m in relations.items()
    }
    edge_index_homo = sparse_to_edge_index(homo)

    masks = build_masks(num_nodes, labels, spec)

    payload = {
        "x": x,
        "y": y,
        "edge_index_homo": edge_index_homo,
        "edge_index_relations": edge_index_relations,
        "relation_names": list(spec.relation_keys.keys()),
        **masks,
    }

    labeled_mask = ~masks["unlabeled_mask"].numpy()
    observed = {
        "num_nodes": num_nodes,
        "num_features": int(x.shape[1]),
        "num_edges_homo": int(edge_index_homo.shape[1]),
        "homo_is_symmetric": is_symmetric(homo),
        "num_edges_per_relation": {
            name: int(ei.shape[1]) for name, ei in edge_index_relations.items()
        },
        "relation_is_symmetric": {
            name: is_symmetric(m) for name, m in relations.items()
        },
        "num_unlabeled_excluded": int(spec.unlabeled_prefix),
        "label_distribution_all": {
            str(k): int(v)
            for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "label_distribution_labeled_only": {
            str(k): int(v)
            for k, v in zip(*np.unique(labels[labeled_mask], return_counts=True))
        },
        "num_train": int(masks["train_mask"].sum()),
        "num_val": int(masks["val_mask"].sum()),
        "num_test": int(masks["test_mask"].sum()),
    }

    manifest = {
        "dataset_name": dataset,
        "stage": "native_benchmark",
        "source": "CARE-GNN official release",
        "source_repo": CARE_GNN_REPO,
        "source_paper": CARE_GNN_PAPER,
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_file_sha256": sha256_file(raw_path),
        "flag_variant_status": (
            "BLOCKED by design: has_native_text=False in "
            "flagbench.registry.DATASET_REGISTRY (registry.py:294-303). "
            "This payload supports baseline / native-GNN variants only, "
            "never `flag` or `+text`."
        ),
        "downsampling": (
            "NONE. This dataset is natively imbalanced already; FLAG's 1:10 "
            "construction (flagbench.datasets.benchmark) was not applied -- "
            "doing so would invent a non-standard variant, not reproduce one."
        ),
        "split_protocol": {
            "train_test": (
                "Exact reproduction of methods/care_gnn/train.py:50-61: "
                "stratified train_test_split(test_size=0.60, random_state=2). "
                "Amazon additionally excludes node indices "
                f"[0, {spec.unlabeled_prefix}) as unlabeled, per CARE-GNN's "
                "own comment."
            ),
            "val": (
                f"NOT part of the official CARE-GNN protocol. Carved out of "
                f"the official train split via stratified "
                f"train_test_split(test_size={VAL_FRACTION_OF_TRAIN}, "
                f"random_state={VAL_SPLIT_SEED}) for early stopping / model "
                f"selection. If exact CARE-GNN train-set size is required, "
                f"merge train_mask and val_mask back together."
            ),
        },
        "graph_structure": (
            "Three relation-specific edge_index tensors plus one 'homo' "
            "(union) edge_index, matching CARE-GNN's own four adjacency "
            "views. No relation was collapsed or invented."
        ),
        "observed": observed,
        "preprocessing_version": "native-benchmark-v1",
    }

    return payload, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["yelpchi", "amazon", "all"], default="all"
    )
    args = parser.parse_args()

    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        print(f"=== {dataset} ===")
        payload, manifest = build(dataset)

        out_dir = OUT_DIR / f"native_{dataset}"
        out_dir.mkdir(parents=True, exist_ok=True)

        graph_path = out_dir / "graph.pt"
        torch.save(payload, graph_path)

        manifest_path = out_dir / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        print(json.dumps(manifest["observed"], indent=2))
        print(f"wrote {graph_path}")
        print(f"wrote {manifest_path}")
        print()


if __name__ == "__main__":
    main()

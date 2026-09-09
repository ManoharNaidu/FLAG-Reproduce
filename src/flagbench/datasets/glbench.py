"""GLBench Reddit / Instagram: download, verification and inspection.

Provenance (verified end to end -- see research/dataset_notes.md):

    FLAG (KDD'25) cites [25] = GLBench (NeurIPS'24 D&B, arXiv:2407.07457)
      -> GLBench repackages GraphAdapter's data (WWW'24, arXiv:2402.12984)
        -> Instagram: Kim et al. WWW'20 influencer dataset + Instagram public API
        -> Reddit:    ConvoKit Subreddit Corpus (Cornell)

The verification in this module is not ceremonial. There is a real and easy
failure mode: `torch_geometric.datasets.Reddit` is a COMPLETELY DIFFERENT graph
(Hamilton et al.'s GraphSAGE Reddit -- 232,965 nodes, 41 communities, no text)
that downloads without error and would silently produce meaningless results.
`verify()` refuses anything that does not match the published signature.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Public Google Drive folder linked from the GLBench README.
GLBENCH_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1WfBIPA3dMd8qQZ6QlQRg9MIFGMwnPdFj"
)
GLBENCH_REPO = "https://github.com/NineAbyss/GLBench"
GLBENCH_PAPER = "https://arxiv.org/abs/2407.07457"


@dataclass(frozen=True)
class DatasetSignature:
    """Published statistics, used to prove we downloaded the right graph.

    Sources: GLBench Table 3, corroborated independently by the GraphAdapter
    paper, which reports identical node and edge counts.
    """

    name: str
    filename: str
    num_nodes: int
    num_edges: int
    num_classes: int
    approx_size_mb: float
    label_names: tuple[str, ...]
    minority_class_name: str
    text_description: str
    edge_description: str

    @property
    def minority_class_index(self) -> int:
        return self.label_names.index(self.minority_class_name)


SIGNATURES: dict[str, DatasetSignature] = {
    "reddit": DatasetSignature(
        name="reddit",
        filename="reddit.pt",
        num_nodes=33_434,
        num_edges=198_448,
        num_classes=2,
        approx_size_mb=552.5,
        # GLBench stores `label_name`; exact strings are confirmed on download
        # and written into the manifest. Order here is our expectation, and
        # verify() reports rather than assumes it.
        label_names=("normal", "popular"),
        minority_class_name="popular",
        text_description="content of the user's last three posts, ';'-separated",
        edge_description="reply between users",
    ),
    "instagram": DatasetSignature(
        name="instagram",
        filename="instagram.pt",
        num_nodes=11_339,
        num_edges=144_010,
        num_classes=2,
        approx_size_mb=181.3,
        label_names=("normal", "commercial"),
        minority_class_name="commercial",
        text_description="the user's personal introduction",
        edge_description="following relationship",
    ),
}


class DatasetVerificationError(RuntimeError):
    """The downloaded file is not the dataset FLAG used."""


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    dataset: str,
    dest_dir: pathlib.Path,
    force: bool = False,
) -> pathlib.Path:
    """Fetch `<dataset>.pt` from the GLBench Drive folder into `dest_dir`.

    Requires `gdown` (``pip install -e ".[download]"``). Google Drive is the only
    distribution channel -- GLBench is not on HuggingFace, verified via the hub
    API -- so this is a genuine supply-chain dependency. Checksum the result and
    mirror it.
    """
    if dataset not in SIGNATURES:
        raise KeyError(
            f"unknown dataset {dataset!r}; expected one of {sorted(SIGNATURES)}"
        )
    sig = SIGNATURES[dataset]
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / sig.filename

    if target.exists() and not force:
        logger.info("%s already present at %s; skipping download", dataset, target)
        return target

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download from Google Drive.\n"
            '  pip install -e ".[download]"\n'
            f"Alternatively download {sig.filename} manually from\n"
            f"  {GLBENCH_DRIVE_FOLDER}\n"
            f"and place it at {target}"
        ) from exc

    logger.info(
        "downloading %s (~%.0f MB) from the GLBench Drive folder",
        sig.filename,
        sig.approx_size_mb,
    )
    # Fetch the folder listing, then pull only the file we want. gdown handles
    # Drive's large-file confirmation interstitial.
    files = gdown.download_folder(
        url=GLBENCH_DRIVE_FOLDER,
        output=str(dest_dir),
        quiet=False,
        skip_download=True,
    )
    match = next(
        (f for f in (files or []) if pathlib.Path(f.path).name == sig.filename),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"{sig.filename} was not found in the GLBench Drive folder listing. "
            f"The folder may have changed. Check {GLBENCH_DRIVE_FOLDER} and "
            f"download manually to {target}."
        )
    gdown.download(id=match.id, output=str(target), quiet=False)
    if not target.exists():
        raise RuntimeError(f"download reported success but {target} is missing")
    return target


def load_raw(path: pathlib.Path):
    """`torch.load` a GLBench `.pt`, normalising the mask-list quirk.

    GLBench's own loader does ``data.test_mask = data.test_mask[0]`` because the
    masks may be a *list* of per-split-seed masks rather than a tensor. FLAG's
    code indexes them directly and would break on the list form, so we normalise
    here and record which split index was taken.
    """
    import torch

    data = torch.load(path, map_location="cpu")
    chosen_split = None
    for field in ("train_mask", "val_mask", "test_mask"):
        value = getattr(data, field, None)
        if isinstance(value, (list, tuple)):
            chosen_split = 0
            setattr(data, field, value[0])
            logger.info(
                "%s was a list of %d masks; took index 0", field, len(value)
            )
    if chosen_split is not None:
        data.glbench_split_index = chosen_split
    return data


def verify(dataset: str, data) -> dict:
    """Check a loaded graph against the published signature.

    Raises DatasetVerificationError on mismatch. Returns an observation dict for
    the manifest -- including the answer to the open question in
    research/dataset_notes.md section 7 about whether `x` is really "shallow".
    """
    import torch

    sig = SIGNATURES[dataset]
    problems: list[str] = []

    # -- structure -------------------------------------------------------
    for field in ("x", "edge_index", "y", "raw_texts"):
        if not hasattr(data, field):
            problems.append(f"missing required field `{field}`")
    if problems:
        raise DatasetVerificationError(
            f"{dataset}: not a GLBench text-attributed graph.\n  - "
            + "\n  - ".join(problems)
            + "\n\nIf you downloaded torch_geometric.datasets.Reddit, that is a "
            "DIFFERENT dataset with the same name and no text. See "
            "research/dataset_notes.md section 9."
        )

    num_nodes = int(data.y.shape[0])
    num_edges = int(data.edge_index.shape[1])

    # -- node count (exact) ----------------------------------------------
    if num_nodes != sig.num_nodes:
        problems.append(
            f"node count {num_nodes:,} != published {sig.num_nodes:,}"
        )

    # -- edge count ------------------------------------------------------
    # MEASURED structure of the shipped files (dataset_notes.md section 4.1):
    # GraphAdapter applies `to_undirected` then `add_self_loops`, so the stored
    # edge_index is symmetric and carries at least one self-loop per node.
    #
    # The published "#Edges" is the ORIGINAL edge-list length, and the two
    # datasets relate to it differently:
    #   instagram: published 144,010 -> stored 144,010 (already symmetric)
    #   reddit:    published 198,448 -> stored 268,472 (symmetrisation added edges)
    # Symmetrising an edge list of length P yields between P (already fully
    # reciprocal) and 2P (no reciprocal pairs) directed edges. So the correct,
    # convention-independent check is that the stored count falls in [P, 2P]
    # and that the graph is actually symmetric.
    num_self_loops = int((data.edge_index[0] == data.edge_index[1]).sum())
    num_edges_no_loops = num_edges - num_self_loops
    if not (sig.num_edges <= num_edges_no_loops <= 2 * sig.num_edges):
        problems.append(
            f"edge count excluding self-loops is {num_edges_no_loops:,} "
            f"({num_edges:,} stored - {num_self_loops:,} self-loops), outside "
            f"the range [{sig.num_edges:,}, {2 * sig.num_edges:,}] implied by "
            f"symmetrising the published {sig.num_edges:,}-edge list"
        )

    loopless = data.edge_index[:, data.edge_index[0] != data.edge_index[1]]
    pairs = set(map(tuple, loopless.t().tolist()))
    is_symmetric = all((b, a) in pairs for a, b in pairs)
    if not is_symmetric:
        problems.append(
            "edge_index is not symmetric; GLBench ships `to_undirected` graphs"
        )

    if num_self_loops < num_nodes:
        problems.append(
            f"only {num_self_loops:,} self-loops for {num_nodes:,} nodes; "
            f"GLBench ships `add_self_loops` graphs"
        )

    # -- text ------------------------------------------------------------
    if len(data.raw_texts) != num_nodes:
        problems.append(
            f"len(raw_texts)={len(data.raw_texts):,} != num_nodes={num_nodes:,}"
        )
    elif not isinstance(data.raw_texts[0], str):
        problems.append(
            f"raw_texts[0] is {type(data.raw_texts[0]).__name__}, expected str"
        )

    # -- labels ----------------------------------------------------------
    classes = torch.unique(data.y)
    if len(classes) != sig.num_classes:
        problems.append(
            f"{len(classes)} distinct labels, expected {sig.num_classes}"
        )

    counts = torch.bincount(data.y.long().flatten()).tolist()
    if len(counts) == 2 and min(counts) > 0:
        ratio = max(counts) / min(counts)
        # The ORIGINAL GLBench graph is roughly balanced. FLAG's 1:10 imbalance
        # is something we construct. An already-imbalanced download means we
        # have the wrong file, or someone's pre-processed derivative.
        if ratio > 3.0:
            problems.append(
                f"class ratio {ratio:.1f}:1 -- the original GLBench graph is "
                f"approximately balanced. This looks like a pre-processed or "
                f"wrong file."
            )

    if problems:
        raise DatasetVerificationError(
            f"{dataset}: does not match the published GLBench signature.\n  - "
            + "\n  - ".join(problems)
            + f"\n\nExpected: {sig.num_nodes:,} nodes / {sig.num_edges:,} edges, "
            f"labels {sig.label_names}, node text = {sig.text_description}.\n"
            f"Correct source: {GLBENCH_DRIVE_FOLDER}\n"
            f"See research/dataset_notes.md section 9 for wrong sources."
        )

    label_names = getattr(data, "label_name", None)
    text_lengths = [len(t) for t in data.raw_texts]

    observed = {
        "num_nodes": num_nodes,
        "num_edges_stored": num_edges,
        "num_self_loops": num_self_loops,
        "num_self_loops_pre_existing": num_self_loops - num_nodes,
        "num_edges_excluding_self_loops": num_edges_no_loops,
        "num_edges_undirected": num_edges_no_loops // 2,
        "edge_index_is_symmetric": is_symmetric,
        "published_edge_count": sig.num_edges,
        "symmetrisation_expansion": round(num_edges_no_loops / sig.num_edges, 4),
        "num_classes": int(len(classes)),
        "label_distribution": {str(i): c for i, c in enumerate(counts)},
        "label_names_in_file": list(label_names)
        if label_names is not None
        else None,
        "feature_shape": list(data.x.shape),
        "feature_dtype": str(data.x.dtype),
        "text_len_min": min(text_lengths),
        "text_len_max": max(text_lengths),
        "text_len_mean": round(sum(text_lengths) / len(text_lengths), 1),
        "has_train_mask": hasattr(data, "train_mask"),
        "has_val_mask": hasattr(data, "val_mask"),
        "has_test_mask": hasattr(data, "test_mask"),
    }
    observed["baseline_feature_provenance"] = _classify_features(data.x.shape[1])
    return observed


def _classify_features(dim: int) -> dict:
    """Record what the `x` dimensionality implies, without overclaiming.

    Open question (research/dataset_notes.md section 7): the FLAG paper calls the
    `baseline` variant's inputs "shallow embeddings" and elsewhere identifies
    shallow features as word2vec, but the released code builds the baseline model
    with an input dimension of 4096 -- not a word2vec size, and exactly
    Llama-2-7B's hidden size. GraphAdapter, the dataset's origin, produces its
    sentence embeddings with Llama 2.

    We record the observation and the implication. We do not assert an answer.
    """
    if dim == 4096:
        return {
            "observed_dim": dim,
            "assessment": "LIKELY_LLM_EMBEDDING",
            "note": (
                "4096 matches Llama-2-7B's hidden size and the input dim "
                "hardcoded in FLAG's test.py. GraphAdapter generates its "
                "sentence embeddings with Llama 2. If so, the paper's "
                "'baseline / shallow embeddings' are NOT shallow, and the "
                "baseline-vs-+text comparison does not mean what the labels "
                "suggest. See research/dataset_notes.md section 7."
            ),
        }
    if dim in (100, 128, 200, 300):
        return {
            "observed_dim": dim,
            "assessment": "LIKELY_SHALLOW",
            "note": (
                "Consistent with word2vec/GloVe-scale shallow features, "
                "matching the paper's description."
            ),
        }
    return {
        "observed_dim": dim,
        "assessment": "UNKNOWN",
        "note": (
            "Dimensionality matches neither a typical shallow embedding nor a "
            "known LLM hidden size. Investigate before using the baseline variant."
        ),
    }


def build_manifest(
    dataset: str,
    raw_path: pathlib.Path,
    observed: dict,
    download_date: str,
) -> dict:
    sig = SIGNATURES[dataset]
    return {
        "dataset_name": dataset,
        "stage": "raw",
        "source": "GLBench",
        "source_repo": GLBENCH_REPO,
        "source_paper": GLBENCH_PAPER,
        "source_url": GLBENCH_DRIVE_FOLDER,
        "origin_paper": (
            "Huang et al., Can GNN be Good Adapter for LLMs?, WWW 2024 "
            "(arXiv:2402.12984)"
        ),
        "upstream_raw_data": (
            "ConvoKit Subreddit Corpus (Cornell)"
            if dataset == "reddit"
            else "Kim et al. WWW'20 influencer dataset + Instagram public API"
        ),
        "cited_by": "FLAG (KDD 2025, DOI 10.1145/3711896.3737220) reference [25]",
        "license": "GLBench packaging is MIT; upstream platform terms still apply",
        "download_date": download_date,
        "file": str(raw_path.name),
        "file_size_bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
        "published_signature": dataclasses.asdict(sig),
        "observed": observed,
        "preprocessing_version": "raw-v1",
        "notes": [
            "data/raw is written once and never modified.",
            "Label semantics: minority (fraud) class is "
            f"'{sig.minority_class_name}'.",
            f"Edges represent: {sig.edge_description}.",
            f"Node text is: {sig.text_description}.",
        ],
    }


def write_manifest(manifest: dict, dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path

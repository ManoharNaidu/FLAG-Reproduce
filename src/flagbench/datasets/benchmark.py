"""FLAG's fraud-detection benchmark construction: 1:10 downsampling + splits.

REIMPLEMENTED from the paper. The official repository ships no code for this
(flag_code_audit.md GAP-5), and the paper gives one sentence:

    "we treat the popular category in Reddit and the commercial category in
     Instagram as the minority class. The minority class nodes are randomly
     selected so that the final ratio between the minority and majority classes
     is about 1:10."

Read precisely: the **minority class is downsampled**; the majority is untouched.

Everything stochastic here is seeded and written into the manifest, because the
paper's own seed is unpublished and unrecoverable. Our numbers therefore cannot
be expected to match Table 4 exactly, and we say so rather than implying
otherwise -- see research/reproduction_status.md section 6.

The original graph in data/raw is never modified.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from flagbench.utils.seeding import numpy_generator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """How to turn a balanced source graph into a fraud-detection benchmark."""

    minority_class: int = 1
    """Label index of the fraud/minority class.

    Reddit: 1 = 'Popular Users'. Instagram: 1 = 'Commercial Users'.
    Verified from `data.label_name` in the shipped files.
    """

    imbalance_ratio: float = 10.0
    """Target majority:minority ratio. The paper says "about 1:10"."""

    downsample_seed: int = 0
    """Seed for choosing which minority nodes to keep."""

    split_ratios: tuple[float, float, float] = (0.10, 0.10, 0.80)
    """train / val / test.

    NOT stated in the FLAG paper. Inherited from GraphAdapter/GLBench, which use
    10/10/80. Labelled as inherited, never as FLAG's. See decision S-003.
    """

    split_seed: int = 0
    stratified: bool = True
    """Preserve the class ratio within each split. With ~1:10 imbalance an
    unstratified split can leave a validation fold with very few positives,
    making model selection noise-dominated."""

    drop_self_loops: bool = True
    """GLBench ships `add_self_loops` graphs (and Reddit has 970 pre-existing
    self-loops). Leaving them in gives every node a spurious same-label
    neighbour, inflating homophily and corrupting semantic sampling.
    See research/dataset_notes.md section 4.1."""

    preprocessing_version: str = "benchmark-v1"

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class BenchmarkGraph:
    """The constructed benchmark. Node ids are re-indexed to be contiguous."""

    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    raw_texts: list[str]
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    original_node_ids: torch.Tensor
    """Maps new index -> index in the original graph. Keeps provenance so any
    result can be traced back to the source graph."""
    label_names: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.y.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def class_counts(self) -> dict[int, int]:
        counts = torch.bincount(self.y.long(), minlength=2)
        return {i: int(c) for i, c in enumerate(counts)}

    def split_counts(self) -> dict:
        out = {}
        for name, mask in (
            ("train", self.train_mask),
            ("val", self.val_mask),
            ("test", self.test_mask),
        ):
            labels = self.y[mask].long()
            counts = torch.bincount(labels, minlength=2)
            out[name] = {
                "n": int(mask.sum()),
                "per_class": {i: int(c) for i, c in enumerate(counts)},
            }
        return out


def select_minority_subset(
    y: torch.Tensor,
    config: BenchmarkConfig,
) -> tuple[torch.Tensor, dict]:
    """Choose which nodes survive the 1:10 downsampling.

    Returns the kept node ids (sorted) and a record of what happened.

    The majority class is kept whole; the minority is subsampled to
    ``|majority| / imbalance_ratio``. If the minority is already smaller than
    that target we keep all of it and report the achieved ratio honestly rather
    than silently failing the spec.
    """
    y = y.long().flatten()
    minority = config.minority_class
    minority_ids = torch.nonzero(y == minority, as_tuple=False).flatten()
    majority_ids = torch.nonzero(y != minority, as_tuple=False).flatten()

    n_minority, n_majority = len(minority_ids), len(majority_ids)
    target = int(round(n_majority / config.imbalance_ratio))

    rng = numpy_generator(config.downsample_seed)
    if target >= n_minority:
        kept_minority = minority_ids
        note = (
            f"minority class already has {n_minority:,} nodes, at or below the "
            f"target {target:,}; kept all. Achieved ratio is weaker than "
            f"1:{config.imbalance_ratio:g}."
        )
        logger.warning(note)
    else:
        chosen = rng.choice(n_minority, size=target, replace=False)
        kept_minority = minority_ids[torch.from_numpy(np.sort(chosen))]
        note = ""

    kept = torch.cat([majority_ids, kept_minority])
    kept, _ = torch.sort(kept)

    achieved = len(majority_ids) / max(len(kept_minority), 1)
    record = {
        "minority_class": minority,
        "original_minority": n_minority,
        "original_majority": n_majority,
        "original_ratio_majority_to_minority": round(n_majority / n_minority, 4),
        "target_minority": target,
        "kept_minority": int(len(kept_minority)),
        "kept_majority": int(n_majority),
        "achieved_ratio_majority_to_minority": round(achieved, 4),
        "requested_ratio": config.imbalance_ratio,
        "downsample_seed": config.downsample_seed,
        "minority_discarded": int(n_minority - len(kept_minority)),
        "minority_discarded_fraction": round(
            1 - len(kept_minority) / n_minority, 4
        ),
        "note": note,
    }
    return kept, record


def make_splits(
    y: torch.Tensor,
    config: BenchmarkConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Stratified (or plain) random train/val/test masks."""
    train_r, val_r, test_r = config.split_ratios
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {total}")

    n = int(y.shape[0])
    rng = numpy_generator(config.split_seed)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    if config.stratified:
        groups = [
            torch.nonzero(y.long() == c, as_tuple=False).flatten()
            for c in torch.unique(y.long()).tolist()
        ]
    else:
        groups = [torch.arange(n)]

    for ids in groups:
        perm = rng.permutation(len(ids))
        ids = ids[torch.from_numpy(perm)]
        n_train = int(round(len(ids) * train_r))
        n_val = int(round(len(ids) * val_r))
        train_mask[ids[:n_train]] = True
        val_mask[ids[n_train:n_train + n_val]] = True
        test_mask[ids[n_train + n_val:]] = True

    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask & test_mask).any()
    assert int((train_mask | val_mask | test_mask).sum()) == n

    record = {
        "split_ratios": list(config.split_ratios),
        "split_seed": config.split_seed,
        "stratified": config.stratified,
        "split_source": (
            "10/10/80 inherited from GraphAdapter/GLBench; the FLAG paper "
            "states no split ratio (decision S-003)"
        ),
    }
    return train_mask, val_mask, test_mask, record


def induce_subgraph(
    edge_index: torch.Tensor,
    kept: torch.Tensor,
    num_original_nodes: int,
    drop_self_loops: bool,
) -> tuple[torch.Tensor, dict]:
    """Induced subgraph on `kept`, re-indexed to 0..len(kept)-1."""
    keep_mask = torch.zeros(num_original_nodes, dtype=torch.bool)
    keep_mask[kept] = True

    src, dst = edge_index[0], edge_index[1]
    edge_kept = keep_mask[src] & keep_mask[dst]

    remap = torch.full((num_original_nodes,), -1, dtype=torch.long)
    remap[kept] = torch.arange(len(kept), dtype=torch.long)

    new_edges = torch.stack([remap[src[edge_kept]], remap[dst[edge_kept]]])

    n_before_loop_drop = int(new_edges.shape[1])
    n_self_loops = int((new_edges[0] == new_edges[1]).sum())
    if drop_self_loops:
        new_edges = new_edges[:, new_edges[0] != new_edges[1]]

    record = {
        "edges_in_original": int(edge_index.shape[1]),
        "edges_after_induction": n_before_loop_drop,
        "self_loops_found": n_self_loops,
        "self_loops_dropped": drop_self_loops,
        "edges_final": int(new_edges.shape[1]),
    }
    return new_edges, record


def build(
    data,
    config: BenchmarkConfig,
    dataset_name: str = "",
) -> tuple[BenchmarkGraph, dict]:
    """Construct the FLAG fraud-detection benchmark from a source graph.

    `data` is a loaded GLBench PyG `Data` (see `flagbench.datasets.glbench`).
    Returns the benchmark and a manifest fragment recording every choice.
    """
    y_all = data.y.long().flatten()
    n_original = int(y_all.shape[0])

    kept, downsample_record = select_minority_subset(y_all, config)

    edge_index, edge_record = induce_subgraph(
        data.edge_index, kept, n_original, config.drop_self_loops
    )

    y = y_all[kept]
    x = data.x[kept]
    raw_texts = [data.raw_texts[i] for i in kept.tolist()]

    train_mask, val_mask, test_mask, split_record = make_splits(y, config)

    graph = BenchmarkGraph(
        x=x,
        edge_index=edge_index,
        y=y,
        raw_texts=raw_texts,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        original_node_ids=kept,
        label_names=list(getattr(data, "label_name", []) or []) or None,
    )

    isolated = _count_isolated(edge_index, graph.num_nodes)
    manifest = {
        "dataset_name": dataset_name,
        "stage": "benchmark",
        "preprocessing_version": config.preprocessing_version,
        "config": config.as_record(),
        "original": {
            "num_nodes": n_original,
            "num_edges": int(data.edge_index.shape[1]),
            "label_distribution": {
                str(i): int(c)
                for i, c in enumerate(torch.bincount(y_all, minlength=2))
            },
            "feature_shape": list(data.x.shape),
        },
        "downsampling": downsample_record,
        "edges": edge_record,
        "constructed": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "label_distribution": {
                str(k): v for k, v in graph.class_counts().items()
            },
            "feature_shape": list(x.shape),
            "isolated_nodes": isolated,
            "splits": graph.split_counts(),
        },
        "splits": split_record,
        "provenance": (
            "REIMPLEMENTED from the FLAG paper section 4.1.1. The official "
            "repository ships no construction code and the paper's RNG seed is "
            "unpublished, so exact agreement with Table 4 is not achievable. "
            "See research/reproduction_status.md section 6."
        ),
    }
    graph.metadata = manifest
    return graph, manifest


def _count_isolated(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Nodes with no edges after induction.

    Downsampling removes ~82% of the minority class on Instagram, so isolated
    nodes are expected and are worth reporting: a 2-hop subgraph around an
    isolated node contains only itself, which makes the GNN's aggregation a
    no-op there and pushes the prediction onto the skip connection alone.
    """
    if edge_index.numel() == 0:
        return num_nodes
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(
        0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long)
    )
    return int((degree == 0).sum())

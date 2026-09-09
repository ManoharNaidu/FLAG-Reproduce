"""Unit tests for the 1:10 benchmark construction.

REIMPLEMENTED from one sentence of the paper, so these tests are the
specification. Cases use synthetic graphs whose correct answer is countable.
"""
from __future__ import annotations

import torch

from flagbench.datasets.benchmark import (
    BenchmarkConfig,
    build,
    induce_subgraph,
    make_splits,
    select_minority_subset,
)


class FakeData:
    """Minimal stand-in for a GLBench PyG Data."""

    def __init__(self, y, edge_index, x=None, raw_texts=None, label_name=None):
        self.y = y
        self.edge_index = edge_index
        n = int(y.shape[0])
        self.x = x if x is not None else torch.arange(n * 4, dtype=torch.float).reshape(n, 4)
        self.raw_texts = raw_texts or [f"text of node {i}" for i in range(n)]
        self.label_name = label_name or ["Normal Users", "Fraud Users"]


def balanced_labels(n_per_class: int = 100) -> torch.Tensor:
    return torch.cat([
        torch.zeros(n_per_class, dtype=torch.long),
        torch.ones(n_per_class, dtype=torch.long),
    ])


# ------------------------------------------------------------ downsampling
def test_downsampling_hits_the_target_ratio():
    """100 majority / 100 minority at 1:10 -> keep 100 majority, 10 minority."""
    y = balanced_labels(100)
    kept, record = select_minority_subset(y, BenchmarkConfig(minority_class=1))
    assert record["kept_majority"] == 100
    assert record["kept_minority"] == 10
    assert abs(record["achieved_ratio_majority_to_minority"] - 10.0) < 0.01
    assert len(kept) == 110


def test_majority_class_is_never_downsampled():
    """The paper downsamples the minority only."""
    y = balanced_labels(50)
    kept, record = select_minority_subset(y, BenchmarkConfig(minority_class=1))
    kept_labels = y[kept]
    assert int((kept_labels == 0).sum()) == 50, "majority must be untouched"
    assert record["original_majority"] == record["kept_majority"]


def test_downsampling_is_seed_reproducible():
    y = balanced_labels(200)
    a, _ = select_minority_subset(y, BenchmarkConfig(downsample_seed=3))
    b, _ = select_minority_subset(y, BenchmarkConfig(downsample_seed=3))
    c, _ = select_minority_subset(y, BenchmarkConfig(downsample_seed=4))
    assert torch.equal(a, b), "same seed must give the same node set"
    assert not torch.equal(a, c), "different seed must give a different set"


def test_minority_class_index_is_honoured():
    """Class 0 as the minority must downsample class 0, not class 1."""
    y = balanced_labels(100)
    kept, record = select_minority_subset(y, BenchmarkConfig(minority_class=0))
    kept_labels = y[kept]
    assert int((kept_labels == 0).sum()) == 10
    assert int((kept_labels == 1).sum()) == 100


def test_already_imbalanced_input_keeps_all_minority_and_reports_it():
    """Guard against silently claiming 1:10 when the source cannot supply it."""
    y = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(3, dtype=torch.long)])
    kept, record = select_minority_subset(y, BenchmarkConfig(minority_class=1))
    assert record["kept_minority"] == 3, "must keep all available minority nodes"
    assert record["note"], "an unreachable target must be reported, not hidden"
    assert record["achieved_ratio_majority_to_minority"] > 10.0


def test_downsample_record_reports_discarded_fraction():
    y = balanced_labels(100)
    _, record = select_minority_subset(y, BenchmarkConfig(minority_class=1))
    assert record["minority_discarded"] == 90
    assert abs(record["minority_discarded_fraction"] - 0.9) < 1e-9


# ------------------------------------------------------------------ splits
def test_splits_are_disjoint_and_cover_every_node():
    y = balanced_labels(100)
    train, val, test, _ = make_splits(y, BenchmarkConfig())
    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()
    assert int((train | val | test).sum()) == len(y)


def test_split_ratios_are_approximately_respected():
    y = balanced_labels(500)
    train, val, test, _ = make_splits(
        y, BenchmarkConfig(split_ratios=(0.10, 0.10, 0.80))
    )
    n = len(y)
    assert abs(int(train.sum()) / n - 0.10) < 0.01
    assert abs(int(val.sum()) / n - 0.10) < 0.01
    assert abs(int(test.sum()) / n - 0.80) < 0.01


def test_stratified_split_preserves_class_ratio():
    """With ~1:10 imbalance an unstratified split can starve val of positives."""
    y = torch.cat([torch.zeros(1000, dtype=torch.long), torch.ones(100, dtype=torch.long)])
    train, val, test, _ = make_splits(y, BenchmarkConfig(stratified=True))
    for mask in (train, val, test):
        labels = y[mask]
        if len(labels) == 0:
            continue
        ratio = int((labels == 0).sum()) / max(int((labels == 1).sum()), 1)
        assert 8.0 < ratio < 12.5, f"class ratio {ratio} drifted from 10:1"


def test_stratified_split_gives_every_split_some_positives():
    y = torch.cat([torch.zeros(1000, dtype=torch.long), torch.ones(100, dtype=torch.long)])
    train, val, test, _ = make_splits(y, BenchmarkConfig(stratified=True))
    for name, mask in (("train", train), ("val", val), ("test", test)):
        assert int((y[mask] == 1).sum()) > 0, f"{name} has no positive examples"


def test_splits_are_seed_reproducible():
    y = balanced_labels(200)
    a = make_splits(y, BenchmarkConfig(split_seed=1))[0]
    b = make_splits(y, BenchmarkConfig(split_seed=1))[0]
    c = make_splits(y, BenchmarkConfig(split_seed=2))[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_bad_split_ratios_are_rejected():
    y = balanced_labels(10)
    try:
        make_splits(y, BenchmarkConfig(split_ratios=(0.5, 0.3, 0.3)))
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
        return
    raise AssertionError("split ratios summing to 1.1 should be rejected")


# --------------------------------------------------------------- subgraph
def test_induced_subgraph_keeps_only_internal_edges():
    #  0-1, 1-2, 2-3 ; keep {0,1,2} -> only 0-1 and 1-2 survive
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    kept = torch.tensor([0, 1, 2])
    new_edges, record = induce_subgraph(edge_index, kept, 4, drop_self_loops=True)
    pairs = set(map(tuple, new_edges.t().tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert record["edges_final"] == 4


def test_induced_subgraph_reindexes_contiguously():
    edge_index = torch.tensor([[0, 5], [5, 0]])
    kept = torch.tensor([0, 5])
    new_edges, _ = induce_subgraph(edge_index, kept, 6, drop_self_loops=True)
    assert new_edges.max() < 2, "node ids must be remapped into 0..len(kept)-1"
    assert set(map(tuple, new_edges.t().tolist())) == {(0, 1), (1, 0)}


def test_self_loops_are_dropped_by_default():
    """GLBench ships add_self_loops graphs; keeping them inflates homophily."""
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])
    kept = torch.tensor([0, 1])
    dropped, record = induce_subgraph(edge_index, kept, 2, drop_self_loops=True)
    assert record["self_loops_found"] == 1
    assert record["edges_final"] == 2
    assert not (dropped[0] == dropped[1]).any()

    keptloops, record2 = induce_subgraph(edge_index, kept, 2, drop_self_loops=False)
    assert record2["edges_final"] == 3


# ------------------------------------------------------------------ build
def test_build_end_to_end_is_consistent():
    y = balanced_labels(100)
    n = len(y)
    src = torch.arange(n - 1)
    edge_index = torch.stack([
        torch.cat([src, src + 1]), torch.cat([src + 1, src])
    ])
    data = FakeData(y, edge_index)

    graph, manifest = build(data, BenchmarkConfig(minority_class=1), "synthetic")

    assert graph.num_nodes == 110
    assert graph.class_counts() == {0: 100, 1: 10}
    assert len(graph.raw_texts) == graph.num_nodes
    assert graph.x.shape[0] == graph.num_nodes
    assert graph.original_node_ids.shape[0] == graph.num_nodes
    if graph.num_edges:
        assert int(graph.edge_index.max()) < graph.num_nodes


def test_build_preserves_node_identity_across_reindexing():
    """original_node_ids must let any result be traced back to the source."""
    y = balanced_labels(50)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = FakeData(y, edge_index)
    graph, _ = build(data, BenchmarkConfig(minority_class=1), "synthetic")

    for new_idx in range(graph.num_nodes):
        original = int(graph.original_node_ids[new_idx])
        assert graph.y[new_idx] == data.y[original], "label followed the wrong node"
        assert graph.raw_texts[new_idx] == data.raw_texts[original], "text mismatch"
        assert torch.equal(graph.x[new_idx], data.x[original]), "features mismatch"


def test_manifest_records_every_stochastic_choice():
    """A build that cannot be reproduced from its manifest is a bug."""
    y = balanced_labels(100)
    data = FakeData(y, torch.zeros((2, 0), dtype=torch.long))
    config = BenchmarkConfig(minority_class=1, downsample_seed=7, split_seed=9)
    _, manifest = build(data, config, "synthetic")

    assert manifest["config"]["downsample_seed"] == 7
    assert manifest["config"]["split_seed"] == 9
    assert manifest["config"]["imbalance_ratio"] == 10.0
    assert manifest["config"]["split_ratios"] == (0.10, 0.10, 0.80)
    assert manifest["downsampling"]["downsample_seed"] == 7
    assert "REIMPLEMENTED" in manifest["provenance"]
    assert "not achievable" in manifest["provenance"]


def test_build_is_deterministic_for_a_fixed_config():
    y = balanced_labels(100)
    data = FakeData(y, torch.zeros((2, 0), dtype=torch.long))
    config = BenchmarkConfig(minority_class=1, downsample_seed=5, split_seed=5)
    a, _ = build(data, config, "s")
    b, _ = build(data, config, "s")
    assert torch.equal(a.original_node_ids, b.original_node_ids)
    assert torch.equal(a.train_mask, b.train_mask)
    assert torch.equal(a.y, b.y)


def test_isolated_nodes_are_counted_not_hidden():
    """Downsampling strands many minority nodes; that must be visible."""
    y = balanced_labels(100)
    data = FakeData(y, torch.zeros((2, 0), dtype=torch.long))
    _, manifest = build(data, BenchmarkConfig(minority_class=1), "s")
    assert manifest["constructed"]["isolated_nodes"] == manifest["constructed"]["num_nodes"]


def _main() -> int:
    tests = [
        (n, o) for n, o in sorted(globals().items())
        if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

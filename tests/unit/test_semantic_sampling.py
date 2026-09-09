"""Unit tests for semantic similarity sampling (FLAG Eq. 3-4).

Every case uses a tiny synthetic graph whose correct answer is known by
construction -- hand-placed embeddings with pre-computed cosine similarities, or
a star graph where the right top-k is obvious. That matters more than usual
here: this component is REIMPLEMENTED from the paper with no upstream source to
diff against, so the tests are the only specification.
"""
from __future__ import annotations

import numpy as np
import torch

from flagbench.sampling.semantic import (
    SamplingConfig,
    build_adjacency,
    normalize_embeddings,
    sample_all,
    sample_subgraph,
    sampling_stats,
    select_neighbors,
    subgraph_homophily,
)


# ---------------------------------------------------------------- fixtures
def star_graph(num_leaves: int = 6):
    """Node 0 at the centre, `num_leaves` leaves, undirected."""
    centre = np.zeros(num_leaves, dtype=np.int64)
    leaves = np.arange(1, num_leaves + 1, dtype=np.int64)
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([centre, leaves]),
            np.concatenate([leaves, centre]),
        ]),
        dtype=torch.long,
    )
    return edge_index, num_leaves + 1


def graded_embeddings(num_leaves: int = 6, dim: int = 4) -> torch.Tensor:
    """Leaf i has cosine similarity to the centre that DECREASES with i.

    Centre is the unit vector e0. Leaf i is a unit vector at angle theta_i from
    e0 with theta increasing, so cos = 1, 0.8, 0.6, ... exactly.
    """
    emb = torch.zeros(num_leaves + 1, dim)
    emb[0, 0] = 1.0
    for i in range(1, num_leaves + 1):
        cos = 1.0 - 0.2 * (i - 1)          # 1.0, 0.8, 0.6, 0.4, 0.2, 0.0
        sin = (1 - cos**2) ** 0.5
        emb[i, 0] = cos
        emb[i, 1] = sin
    return emb


# ------------------------------------------------------- normalisation
def test_normalize_makes_cosine_a_dot_product():
    emb = torch.randn(10, 8) * 5.0
    normed = normalize_embeddings(emb)
    assert torch.allclose(normed.norm(dim=1), torch.ones(10), atol=1e-6)
    expected = torch.nn.functional.cosine_similarity(emb[0], emb[3], dim=0)
    assert torch.allclose(normed[0] @ normed[3], expected, atol=1e-6)


def test_zero_embedding_rows_survive_normalisation():
    """Instagram has nodes whose raw text is the empty string."""
    emb = torch.zeros(3, 4)
    emb[1, 0] = 2.0
    normed = normalize_embeddings(emb)
    assert torch.isfinite(normed).all(), "zero row produced NaN/inf"
    assert torch.allclose(normed[0], torch.zeros(4))
    # A zero row has similarity exactly 0 to everything.
    assert abs(float(normed[0] @ normed[1])) < 1e-9


# ------------------------------------------------------------- adjacency
def test_build_adjacency_matches_edge_list():
    edge_index, n = star_graph(4)
    adj = build_adjacency(edge_index, n)
    assert sorted(adj[0].tolist()) == [1, 2, 3, 4]
    for leaf in range(1, 5):
        assert adj[leaf].tolist() == [0]


def test_build_adjacency_drops_self_loops():
    """GLBench ships add_self_loops graphs; a self-loop would waste a top-k slot."""
    edge_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    adj = build_adjacency(edge_index, 2, drop_self_loops=True)
    assert adj[0].tolist() == [1]
    assert adj[1].tolist() == [0]
    kept = build_adjacency(edge_index, 2, drop_self_loops=False)
    assert sorted(kept[0].tolist()) == [0, 1]


# ------------------------------------------------- Eq. 3 / Eq. 4 selection
def test_top_k_picks_the_most_similar_neighbours():
    """Leaves are graded 1.0, 0.8, 0.6, 0.4, 0.2, 0.0 -- top-3 must be 1, 2, 3."""
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(6))
    config = SamplingConfig(top_k=3, similarity_threshold=-1.0)

    picked = select_neighbors(0, adj[0], normed, config)
    assert sorted(picked.tolist()) == [1, 2, 3]


def test_threshold_filters_before_top_k():
    """delta = 0.5 admits only leaves 1 (1.0), 2 (0.8), 3 (0.6)."""
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(6))
    config = SamplingConfig(top_k=10, similarity_threshold=0.5)

    picked = select_neighbors(0, adj[0], normed, config)
    assert sorted(picked.tolist()) == [1, 2, 3]


def test_threshold_can_empty_the_neighbourhood():
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(6))
    config = SamplingConfig(top_k=10, similarity_threshold=1.5)
    assert select_neighbors(0, adj[0], normed, config).size == 0


def test_paper_default_threshold_zero_admits_nonnegative_only():
    """delta = 0 is the paper's setting. Leaf 6 has cosine exactly 0.0."""
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(6))
    config = SamplingConfig(top_k=10, similarity_threshold=0.0)
    picked = select_neighbors(0, adj[0], normed, config)
    assert sorted(picked.tolist()) == [1, 2, 3, 4, 5, 6]  # 0.0 >= 0.0 is kept


def test_negative_threshold_admits_dissimilar_neighbours():
    """The sensitivity sweep uses delta = -0.2 and -0.4."""
    emb = torch.zeros(3, 2)
    emb[0, 0] = 1.0        # centre
    emb[1, 0] = 1.0        # cos = +1
    emb[2, 0] = -1.0       # cos = -1
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)
    adj = build_adjacency(edge_index, 3)
    normed = normalize_embeddings(emb)

    strict = select_neighbors(0, adj[0], normed, SamplingConfig(top_k=10, similarity_threshold=0.0))
    assert strict.tolist() == [1]
    loose = select_neighbors(0, adj[0], normed, SamplingConfig(top_k=10, similarity_threshold=-1.0))
    assert sorted(loose.tolist()) == [1, 2]


def test_centre_is_never_its_own_neighbour():
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.long)
    adj = build_adjacency(edge_index, 2, drop_self_loops=False)
    normed = normalize_embeddings(torch.eye(2))
    picked = select_neighbors(0, adj[0], normed, SamplingConfig(top_k=10, similarity_threshold=-1.0))
    assert 0 not in picked.tolist()


# ------------------------------------------------------------ strategies
def test_strategy_none_keeps_every_neighbour():
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    config = SamplingConfig(strategy="none", top_k=2)
    picked = select_neighbors(0, adj[0], None, config)
    assert sorted(picked.tolist()) == [1, 2, 3, 4, 5, 6]


def test_strategy_random_is_seeded_and_respects_top_k():
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    config = SamplingConfig(strategy="random", top_k=3, seed=7)
    a = select_neighbors(0, adj[0], None, config, np.random.default_rng(7))
    b = select_neighbors(0, adj[0], None, config, np.random.default_rng(7))
    assert a.size == 3
    assert a.tolist() == b.tolist(), "same seed must give the same sample"
    c = select_neighbors(0, adj[0], None, config, np.random.default_rng(8))
    assert a.size == c.size


def test_semantic_nothreshold_ignores_delta():
    """SS* in the paper's Figure 3(a): top-N with no threshold."""
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(6))
    config = SamplingConfig(
        strategy="semantic_nothreshold", top_k=6, similarity_threshold=0.9
    )
    picked = select_neighbors(0, adj[0], normed, config)
    assert len(picked) == 6, "threshold should have been ignored"


# ------------------------------------------------------------- subgraphs
def test_subgraph_centre_is_first_and_findable():
    edge_index, n = star_graph(4)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(4))
    sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=1, top_k=2, similarity_threshold=-1.0))
    assert int(sg.subset[0]) == 0
    assert sg.center_position() == 0
    assert int(sg.hop[0]) == 0


def test_two_hop_expansion_reaches_second_ring():
    """Path 0-1-2-3: 2 hops from node 0 must reach node 2 but not node 3."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    adj = build_adjacency(edge_index, 4)
    normed = normalize_embeddings(torch.ones(4, 2))   # all identical -> cos 1
    sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=2, top_k=10, similarity_threshold=-1.0))
    got = sorted(sg.subset.tolist())
    assert got == [0, 1, 2], f"2-hop from 0 on a path should be {{0,1,2}}, got {got}"
    hop_of = dict(zip(sg.subset.tolist(), sg.hop.tolist()))
    assert hop_of == {0: 0, 1: 1, 2: 2}


def test_hop_count_is_respected():
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    adj = build_adjacency(edge_index, 4)
    normed = normalize_embeddings(torch.ones(4, 2))
    for hops, expected in [(1, {0, 1}), (2, {0, 1, 2}), (3, {0, 1, 2, 3})]:
        sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=hops, top_k=10, similarity_threshold=-1.0))
        assert set(sg.subset.tolist()) == expected, f"hops={hops}"


def test_isolated_node_yields_a_singleton_subgraph():
    """After 1:10 downsampling many minority nodes lose all their neighbours."""
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    adj = build_adjacency(edge_index, 3)
    normed = normalize_embeddings(torch.eye(3))
    sg = sample_subgraph(1, adj, normed, SamplingConfig())
    assert sg.num_nodes == 1
    assert sg.subset.tolist() == [1]
    assert sg.edge_index.shape == (2, 0)
    assert sg.center_position() == 0


def test_subgraph_edges_are_reindexed_within_subset():
    edge_index, n = star_graph(4)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(graded_embeddings(4))
    sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=1, top_k=2, similarity_threshold=-1.0))
    assert sg.edge_index.max() < sg.num_nodes, "edge index escapes the subset"
    assert sg.edge_index.min() >= 0


def test_per_hop_vs_total_budget_differ():
    """Documented ambiguity: is top_k per hop or a total budget?"""
    edge_index, n = star_graph(8)
    # Attach a second ring so hop 2 has something to expand into.
    extra = torch.tensor([[1, 9], [9, 1]], dtype=torch.long)
    edge_index = torch.cat([edge_index, extra], dim=1)
    adj = build_adjacency(edge_index, 10)
    normed = normalize_embeddings(torch.ones(10, 2))

    per_hop = sample_subgraph(
        0, adj, normed, SamplingConfig(hops=2, top_k=3, similarity_threshold=-1.0, per_hop=True)
    )
    total = sample_subgraph(
        0, adj, normed, SamplingConfig(hops=2, top_k=3, similarity_threshold=-1.0, per_hop=False)
    )
    assert total.num_nodes <= per_hop.num_nodes
    assert total.num_nodes <= 1 + 3, "total budget must cap added nodes at top_k"


# --------------------------------------------------------------- metrics
def test_homophily_matches_hand_computation():
    """Eq. 5 on a graph where the answer is countable by hand."""
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    adj = build_adjacency(edge_index, 3)
    normed = normalize_embeddings(torch.ones(3, 2))
    y = torch.tensor([0, 0, 1])          # 0-1 same, 0-2 different
    sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=1, top_k=10, similarity_threshold=-1.0))
    # Subgraph has directed edges 0<->1 and 0<->2: 2 same-label, 2 different.
    assert abs(subgraph_homophily([sg], y) - 0.5) < 1e-9


def test_homophily_of_a_perfectly_homophilous_graph_is_one():
    edge_index, n = star_graph(4)
    adj = build_adjacency(edge_index, n)
    normed = normalize_embeddings(torch.ones(n, 2))
    y = torch.zeros(n, dtype=torch.long)
    sg = sample_subgraph(0, adj, normed, SamplingConfig(hops=1, top_k=10, similarity_threshold=-1.0))
    assert abs(subgraph_homophily([sg], y) - 1.0) < 1e-9


def test_semantic_sampling_raises_homophily_over_random():
    """The paper's Figure 3(a) motivation, on a graph built to exhibit it.

    Centre is class 0. Same-class neighbours get near-identical embeddings;
    other-class neighbours get orthogonal ones. Semantic sampling should
    therefore pick same-class neighbours and beat random selection.
    """
    num_same, num_diff = 5, 15
    n = 1 + num_same + num_diff
    centre = np.zeros(n - 1, dtype=np.int64)
    others = np.arange(1, n, dtype=np.int64)
    edge_index = torch.tensor(
        np.stack([
            np.concatenate([centre, others]),
            np.concatenate([others, centre]),
        ]),
        dtype=torch.long,
    )
    adj = build_adjacency(edge_index, n)

    emb = torch.zeros(n, 2)
    emb[0] = torch.tensor([1.0, 0.0])
    y = torch.zeros(n, dtype=torch.long)
    for i in range(1, 1 + num_same):
        emb[i] = torch.tensor([1.0, 0.05])       # near-parallel -> cos ~ 1
        y[i] = 0
    for i in range(1 + num_same, n):
        emb[i] = torch.tensor([0.0, 1.0])        # orthogonal -> cos 0
        y[i] = 1
    normed = normalize_embeddings(emb)

    semantic = sample_subgraph(
        0, adj, normed, SamplingConfig(strategy="semantic", hops=1, top_k=5)
    )
    random_cfg = SamplingConfig(strategy="random", hops=1, top_k=5, seed=0)
    random_sg = sample_subgraph(
        0, adj, None, random_cfg, np.random.default_rng(0)
    )

    h_semantic = subgraph_homophily([semantic], y)
    h_random = subgraph_homophily([random_sg], y)
    assert h_semantic == 1.0, f"semantic should pick only same-class, got {h_semantic}"
    assert h_semantic > h_random, (
        f"semantic {h_semantic} should beat random {h_random} -- this is the "
        f"paper's Figure 3(a) claim"
    )


def test_sample_all_and_stats():
    edge_index, n = star_graph(6)
    adj = build_adjacency(edge_index, n)
    emb = graded_embeddings(6)
    subgraphs = sample_all(range(n), adj, emb, SamplingConfig(hops=2, top_k=3))
    assert len(subgraphs) == n
    stats = sampling_stats(subgraphs)
    assert stats["num_subgraphs"] == n
    assert stats["nodes_per_subgraph"]["min"] >= 1


def test_cache_key_distinguishes_configs():
    """A cache hit across a config change would silently mix experiments."""
    base = SamplingConfig()
    assert base.cache_key() != SamplingConfig(top_k=5).cache_key()
    assert base.cache_key() != SamplingConfig(similarity_threshold=0.2).cache_key()
    assert base.cache_key() != SamplingConfig(hops=3).cache_key()
    assert base.cache_key() != SamplingConfig(per_hop=False).cache_key()
    assert base.cache_key() != SamplingConfig(strategy="random").cache_key()
    assert base.cache_key() == SamplingConfig().cache_key()


def test_paper_defaults_are_the_published_values():
    """top-10, threshold 0, 2 hops -- and the `0_10_0` directory upstream."""
    config = SamplingConfig()
    assert config.hops == 2
    assert config.top_k == 10
    assert config.similarity_threshold == 0.0
    assert config.strategy == "semantic"


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

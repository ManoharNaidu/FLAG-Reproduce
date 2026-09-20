"""Unit tests for Markov-diffusion neighbour sampling (FLAG-MD).

Small synthetic graphs whose correct answer is known by construction. The
cosine sampler's own behaviour is covered by test_semantic_sampling.py; the
tests here also pin that adding MD did not change it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flagbench.sampling import markov_diffusion as MD
from flagbench.sampling import semantic as S
from flagbench.sampling.semantic import SamplingConfig, build_adjacency


def adj_from_edges(edges, n):
    ei = torch.tensor(edges, dtype=torch.long).T
    ei = torch.cat([ei, ei.flip(0)], dim=1)               # undirected
    return build_adjacency(ei, n, drop_self_loops=True)


def cfg(**kw):
    return SamplingConfig(strategy="markov_diffusion", **kw)


# ----------------------------------------------------------------- operator
def test_transition_matrix_is_row_stochastic_and_zero_for_isolated_nodes():
    adj = adj_from_edges([(0, 1), (0, 2), (1, 2)], 4)      # node 3 isolated
    T = MD.transition_matrix(MD.adjacency_to_csr(adj, 4))
    sums = np.asarray(T.sum(axis=1)).ravel()
    assert np.allclose(sums[:3], 1.0)
    assert sums[3] == 0.0                                  # no division by zero
    assert np.isfinite(T.toarray()).all()


def test_duplicate_edges_and_self_loops_collapse():
    adj = [np.array([1, 1, 1, 0]), np.array([0, 0])]       # dup edges + a self-loop on 0
    A = MD.adjacency_to_csr(adj, 2)
    assert A.toarray().tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_diffusion_matches_dense_formula_for_every_operator():
    rng = np.random.default_rng(0)
    n, d = 12, 5
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.3]
    A = MD.adjacency_to_csr(adj_from_edges(edges, n), n)
    T = MD.transition_matrix(A)
    X = rng.normal(size=(n, d))
    Td = T.toarray()
    for K in (1, 2, 3, 5):
        for op, (first, scale) in {
            "paper_eq6": (0, 1 / K), "mean_k0": (0, 1 / (K + 1)), "walk_only": (1, 1 / K),
        }.items():
            Z = scale * sum(np.linalg.matrix_power(Td, k) for k in range(first, K + 1))
            assert np.allclose(MD.diffusion_embeddings(T, X, K, op), Z @ X)


def test_operator_scale_does_not_change_the_ranking():
    """paper_eq6 (1/K) and mean_k0 (1/(K+1)) differ by a positive constant."""
    rng = np.random.default_rng(1)
    n = 30
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
    adj = adj_from_edges(edges, n)
    emb = torch.randn(n, 8)
    a = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(diffusion_steps=3, md_selection="top_n", top_k=2))
    b = MD.MarkovDiffusionNeighborSampler(
        adj, emb, cfg(diffusion_steps=3, md_selection="top_n", top_k=2, diffusion_operator="mean_k0"))
    for v in range(n):
        assert a.select(v, adj[v]).tolist() == b.select(v, adj[v]).tolist()


def test_walk_only_differs_from_paper_eq6():
    """Star, K=1. Every leaf's only neighbour is the centre, so walk_only gives all
    leaves the SAME H (= x_centre) and cannot tell them apart (tie -> lowest id 1).
    paper_eq6 keeps each leaf's own feature (k=0) and ranks leaf 2, the one
    closest to the leaves' mean, first. Hand-computed: eq6 d(0,1)=0.943, d(0,2)=0.471."""
    adj = adj_from_edges([(0, 1), (0, 2), (0, 3)], 4)
    emb = torch.tensor([[1.0, 0], [0, 1.0], [1.0, 0], [1.0, 0]])
    eq6 = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(diffusion_steps=1, md_selection="top_n", top_k=1))
    walk = MD.MarkovDiffusionNeighborSampler(
        adj, emb, cfg(diffusion_steps=1, md_selection="top_n", top_k=1, diffusion_operator="walk_only"))
    assert np.allclose(eq6.distances(0, np.array([1, 2])), [0.9428, 0.4714], atol=1e-3)
    assert eq6.select(0, adj[0]).tolist() == [2]
    assert np.allclose(walk.distances(0, np.array([1, 2, 3])), walk.distances(0, np.array([1, 2, 3]))[0])
    assert walk.select(0, adj[0]).tolist() == [1]


# ---------------------------------------------------------------- selection
def test_smallest_distance_is_selected_not_largest():
    """Star: centre 0; leaves 1,2 share the centre's direction, 3,4 are orthogonal."""
    adj = adj_from_edges([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    emb = torch.tensor([[1.0, 0], [1.0, 0.05], [1.0, 0.1], [0, 1.0], [0, 1.0]])
    s = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(diffusion_steps=2, md_selection="top_n", top_k=2))
    sel = s.select(0, adj[0]).tolist()
    d = dict(zip([1, 2, 3, 4], s.distances(0, np.array([1, 2, 3, 4]))))
    assert sorted(sel) == sorted(sorted(d, key=d.get)[:2])
    assert max(d[u] for u in sel) <= min(d[u] for u in d if u not in sel)


def test_top_n_is_respected_and_centre_excluded():
    adj = adj_from_edges([(0, i) for i in range(1, 9)], 9)
    emb = torch.randn(9, 6)
    s = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(md_selection="top_n", top_k=3))
    sel = s.select(0, adj[0])
    assert sel.size == 3 and 0 not in sel.tolist()
    assert 0 not in s.select(0, np.array([0, 1, 2, 2, 2])).tolist()   # self + duplicates in candidates


def test_matched_cosine_keeps_exactly_the_cosine_count():
    rng = np.random.default_rng(2)
    n = 40
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.25]
    adj = adj_from_edges(edges, n)
    emb = torch.randn(n, 16)
    md = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(diffusion_steps=2))
    cos = S.CosineNeighborSampler(S.normalize_embeddings(emb), dataclasses.replace(cfg(), strategy="semantic"))
    for v in range(n):
        assert md.select(v, adj[v]).size == cos.select(v, adj[v]).size


def test_isolated_and_empty_candidates_are_safe():
    adj = adj_from_edges([(0, 1)], 3)                        # node 2 isolated
    s = MD.MarkovDiffusionNeighborSampler(adj, torch.randn(3, 4), cfg())
    assert s.select(2, adj[2]).size == 0
    assert s.select(2, np.array([], dtype=np.int64)).size == 0


def test_zero_embedding_rows_do_not_produce_nans():
    adj = adj_from_edges([(0, 1), (0, 2), (1, 2)], 3)
    emb = torch.tensor([[0.0, 0], [1.0, 0], [0, 1.0]])       # empty-text node
    s = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(md_selection="top_n", top_k=2))
    assert np.isfinite(s.H).all()
    assert np.isfinite(s.distances(0, np.array([1, 2]))).all()


def test_optional_threshold_filters_far_candidates():
    """Distances from the centre are [~0, 0.707, 0.707] (hand-checked), so tau=0.1
    keeps only leaf 1, and a huge tau keeps everyone up to the N budget."""
    adj = adj_from_edges([(0, 1), (0, 2), (0, 3)], 4)
    emb = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
    tight = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(md_selection="top_n", top_k=3, diffusion_threshold=0.1))
    assert np.allclose(tight.distances(0, np.array([1, 2, 3])), [0.0, 0.7071, 0.7071], atol=1e-3)
    assert tight.select(0, adj[0]).tolist() == [1]
    loose = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(md_selection="top_n", top_k=3, diffusion_threshold=1e9))
    assert loose.select(0, adj[0]).size == 3


def test_ties_are_broken_by_node_id_deterministically():
    adj = adj_from_edges([(0, 1), (0, 2), (0, 3)], 4)
    emb = torch.tensor([[1.0, 0], [0, 1.0], [0, 1.0], [0, 1.0]])   # 1,2,3 identical
    s = MD.MarkovDiffusionNeighborSampler(adj, emb, cfg(md_selection="top_n", top_k=2))
    assert s.select(0, adj[0]).tolist() == [1, 2]


# ------------------------------------------------------- integration / keys
def test_sampler_never_sees_labels():
    import inspect

    params = inspect.signature(MD.MarkovDiffusionNeighborSampler.__init__).parameters
    assert set(params) == {"self", "adjacency", "embeddings", "config"}


def test_md_subgraph_is_within_the_hop_ball_and_starts_at_centre():
    adj = adj_from_edges([(0, 1), (1, 2), (2, 3), (3, 4)], 5)
    emb = torch.randn(5, 4)
    sgs = S.sample_all(range(5), adj, emb, cfg(hops=2, md_selection="top_n", top_k=10))
    sg = sgs[0]
    assert sg.subset[0].item() == 0 and set(sg.subset.tolist()) == {0, 1, 2}
    assert sg.hop.tolist() == [0, 1, 2]


def test_cosine_cache_keys_and_records_are_unchanged():
    assert SamplingConfig().cache_key() == "semantic_h2_k10_t0_perhop"
    assert SamplingConfig(strategy="none").cache_key() == "none_h2_k10_t0_perhop"
    assert "diffusion_steps" not in SamplingConfig().as_record()


def test_md_cache_keys_distinguish_every_md_option():
    keys = {
        cfg().cache_key(),
        cfg(diffusion_steps=3).cache_key(),
        cfg(md_selection="top_n").cache_key(),
        cfg(diffusion_operator="walk_only").cache_key(),
        cfg(diffusion_threshold=0.5).cache_key(),
        SamplingConfig().cache_key(),
    }
    assert len(keys) == 6
    assert cfg().cache_key() == "markov_diffusion_h2_k10_t0_perhop_K2_matched_cosine"


def test_invalid_options_raise():
    adj = adj_from_edges([(0, 1)], 2)
    with pytest.raises(ValueError):
        MD.MarkovDiffusionNeighborSampler(adj, torch.randn(2, 3), cfg(md_selection="bogus"))
    with pytest.raises(ValueError):
        MD.diffusion_embeddings(sp.csr_matrix((2, 2)), np.zeros((2, 3)), 0)
    with pytest.raises(ValueError):
        MD.diffusion_embeddings(sp.csr_matrix((2, 2)), np.zeros((2, 3)), 1, "bogus")

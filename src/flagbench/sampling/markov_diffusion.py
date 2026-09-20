"""Markov-diffusion neighbour sampling (FLAG-MD).

An isolated ablation of ONE component of FLAG: the criterion that ranks a node's
candidate neighbours. Everything else -- the candidate set, the frontier
expansion, the per-hop budget, the induced-subgraph construction, the LLM stage,
the GNN -- is `flagbench.sampling.semantic`, unchanged.

    FLAG (cosine):    rank candidates u of node w by  cos(B(t_w), B(t_u))   high = relevant
    FLAG-MD:          rank candidates u of node w by  ||H_w - H_u||_2        low  = relevant

The operator is the Markov Diffusion Kernel of DGP (AAAI'26, Eq. 6-8), taken as
the *source of the idea only*; nothing else of DGP is used.

    A       binary adjacency of the graph FLAG samples on (self-loops dropped,
            duplicate edges collapsed)
    D       diag(A 1)
    T       D^-1 A                               row-stochastic transition matrix
    Z(K)    (1/K) * sum_{k=0..K} T^k             DGP Eq. 6, literal
    X       L2-normalised Sentence-BERT embeddings of the RAW node text -- the SAME
            matrix the cosine sampler compares (Eq. 3 of FLAG)
    H       Z(K) X                               DGP Eq. 7
    delta(w,u) = ||H_w - H_u||_2                 DGP Eq. 8   (small = relevant)

Why this is exact and cheap. Row w of H depends only on the K-hop ball around w,
so computing H once for the whole graph is *identical* to running the diffusion
locally per target node -- but costs K sparse-times-dense products,
O(K * nnz(A) * d) time and O(N * d) memory. Z(K) and T^k are never formed, so no
N x N matrix exists at any point.

Choices the papers leave open, all exposed as config and recorded in the cache key:

* `operator="paper_eq6"` keeps the k=0 (identity) term, as DGP Eq. 6 prints it.
  `"mean_k0"` is 1/(K+1) instead of 1/K; that is a positive constant, so it must
  give the identical ranking (tested). `"walk_only"` drops k=0 and DOES change the
  ranking: with no self-loops, K=1 walk-only compares mean(neighbours of w) with
  mean(neighbours of u) and never looks at x_w or x_u themselves. Ablation only.
* `md_selection` decides how many neighbours are kept:
    "matched_cosine" -- keep exactly as many as the cosine sampler keeps for the
        same node (threshold, then top-N). Only the RANKING differs; the
        neighbourhood SIZE is identical by construction. Primary comparison.
    "top_n" -- keep the N smallest-distance candidates, no threshold. On the
        benchmark graphs most nodes have <= N neighbours, so this keeps nearly
        everyone; it differs from cosine mostly by NOT applying delta=0.
* `diffusion_threshold` (optional) drops candidates with delta > tau before the
  budget is applied. The numeric value of FLAG's cosine delta is NOT reusable:
  cosine similarity and an L2 diffusion distance live on different scales.

Labels are never read in this module.
"""
from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import scipy.sparse as sp
import torch

from flagbench.sampling.semantic import (
    SamplingConfig,
    normalize_embeddings,
    select_neighbors,
)

logger = logging.getLogger(__name__)

OPERATORS = ("paper_eq6", "mean_k0", "walk_only")
MD_SELECTIONS = ("matched_cosine", "top_n")


def _operator_terms(operator: str, steps: int) -> tuple[int, float]:
    """(first power included, scale). See the module docstring."""
    if steps < 1:
        raise ValueError(f"diffusion_steps must be >= 1, got {steps}")
    if operator == "paper_eq6":
        return 0, 1.0 / steps
    if operator == "mean_k0":
        return 0, 1.0 / (steps + 1)
    if operator == "walk_only":
        return 1, 1.0 / steps
    raise ValueError(f"unknown diffusion_operator {operator!r}; choose from {OPERATORS}")


def adjacency_to_csr(adjacency: list[np.ndarray], num_nodes: int) -> sp.csr_matrix:
    """Binary CSR adjacency from FLAG's list-of-neighbour-arrays.

    Built from the *same* `adjacency` the cosine sampler expands over, so both
    samplers see one graph. Duplicate edges collapse to weight 1 (a repeated
    edge must not double a neighbour's transition probability); self-loops are
    already gone (`build_adjacency(drop_self_loops=True)`) and are removed again
    defensively.
    """
    lengths = np.fromiter((len(a) for a in adjacency), dtype=np.int64, count=num_nodes)
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    indices = (
        np.concatenate(adjacency).astype(np.int64)
        if lengths.sum() else np.zeros(0, dtype=np.int64)
    )
    rows = np.repeat(np.arange(num_nodes), lengths)
    keep = rows != indices                      # defensive self-loop removal
    A = sp.csr_matrix(
        (np.ones(int(keep.sum())), (rows[keep], indices[keep])),
        shape=(num_nodes, num_nodes),
    )
    A.sum_duplicates()
    A.data[:] = 1.0                             # collapse duplicate edges
    return A


def transition_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    """T = D^-1 A. Zero-degree rows stay all-zero instead of dividing by zero."""
    degree = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return (sp.diags(inv) @ A).tocsr()


def diffusion_embeddings(
    T: sp.csr_matrix, X: np.ndarray, steps: int, operator: str = "paper_eq6"
) -> np.ndarray:
    """H = Z(K) X by K sparse-dense products; Z(K) and T^k are never formed."""
    first, scale = _operator_terms(operator, steps)
    if X.ndim != 2 or X.shape[0] != T.shape[0]:
        raise ValueError(f"X must be (N={T.shape[0]}, d), got {X.shape}")
    current = np.asarray(X, dtype=np.float64)
    total = current.copy() if first == 0 else np.zeros_like(current)
    for _ in range(steps):
        current = T @ current
        total += current
    return scale * total


def diffusion_distances(H: np.ndarray, target: int, candidates: np.ndarray) -> np.ndarray:
    """delta(target, u) = ||H_target - H_u||_2 for each candidate u."""
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float64)
    diff = H[candidates] - H[target]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


class MarkovDiffusionNeighborSampler:
    """Ranks candidates by diffusion distance; smaller = more relevant.

    Implements the same `select(center, candidates)` interface as
    `semantic.CosineNeighborSampler`, so `sample_subgraph` cannot tell them apart.
    """

    def __init__(
        self,
        adjacency: list[np.ndarray],
        embeddings: torch.Tensor,
        config: SamplingConfig,
    ):
        if config.md_selection not in MD_SELECTIONS:
            raise ValueError(
                f"md_selection {config.md_selection!r}; choose from {MD_SELECTIONS}"
            )
        self.config = config
        num_nodes = len(adjacency)
        start = time.time()

        # X: identical to what the cosine sampler compares (L2-normalised SBERT).
        self.normalized = normalize_embeddings(embeddings)
        X = self.normalized.numpy().astype(np.float64)

        self.A = adjacency_to_csr(adjacency, num_nodes)
        self.T = transition_matrix(self.A)
        self.H = diffusion_embeddings(
            self.T, X, config.diffusion_steps, config.diffusion_operator
        )
        if not np.isfinite(self.H).all():
            bad = int((~np.isfinite(self.H).all(axis=1)).sum())
            raise FloatingPointError(f"{bad} non-finite rows in H = Z(K)X")
        self.precompute_seconds = time.time() - start

        # Cosine config used ONLY to size the budget in "matched_cosine" mode.
        self._cosine_config = dataclasses.replace(config, strategy="semantic")
        self.stats = {"calls": 0, "empty": 0, "nonfinite_dropped": 0}

    # --------------------------------------------------------------- scoring
    def clean_candidates(self, center: int, candidates: np.ndarray) -> np.ndarray:
        """Unique candidates in ascending id order, centre excluded."""
        candidates = np.unique(np.asarray(candidates, dtype=np.int64))
        return candidates[candidates != center]

    def distances(self, center: int, candidates: np.ndarray) -> np.ndarray:
        return diffusion_distances(self.H, center, candidates)

    def budget(self, center: int, candidates: np.ndarray) -> int:
        if self.config.md_selection == "top_n":
            return self.config.top_k
        return int(
            select_neighbors(center, candidates, self.normalized, self._cosine_config).size
        )

    # ------------------------------------------------------------- selection
    def select(self, center: int, candidates: np.ndarray) -> np.ndarray:
        """Selected neighbour ids (ascending), never containing `center`."""
        self.stats["calls"] += 1
        candidates = self.clean_candidates(center, candidates)
        if candidates.size == 0:
            self.stats["empty"] += 1
            return candidates

        k = self.budget(center, candidates)
        dist = self.distances(center, candidates)

        finite = np.isfinite(dist)
        if not finite.all():
            self.stats["nonfinite_dropped"] += int((~finite).sum())
            candidates, dist = candidates[finite], dist[finite]

        tau = self.config.diffusion_threshold
        if tau is not None:
            keep = dist <= tau
            candidates, dist = candidates[keep], dist[keep]

        if k <= 0 or candidates.size == 0:
            return candidates[:0]
        if candidates.size > k:
            # SMALLEST distance first; ties broken by lower node id (deterministic).
            order = np.lexsort((candidates, dist))[:k]
            candidates = candidates[order]
        return np.sort(candidates)

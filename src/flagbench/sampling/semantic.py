"""Semantic similarity neighbour sampling (FLAG paper, Eq. 3-4).

REIMPLEMENTED. This is a **core contribution of the paper with no source in the
official repository** -- every FLAG driver loads pre-built `*_sampler*.pt` files
that nothing in the repo creates (flag_code_audit.md GAP-1). The only trace of
the configuration upstream is the directory name `Reddit/0_10_0/`.

Paper, section 3.2:

    Eq. 3   sim(v, u) = B(t_v) . B(t_u) / (||B(t_v)|| ||B(t_u)||)
    Eq. 4   N_selected(v) = { u in N(v) : sim(v, u) >= delta }

    "we rank the neighboring nodes based on their similarity to the center node
     v's text t_v. We then select the top-N neighbors N_selected(v) with the
     highest similarity that exceed a predefined threshold delta ... The set of
     selected neighbors N_selected(v) is finally used to form a simplified
     subgraph G_v centered around node v."

    section 4.1.2: "we evaluate the models on 2-hop subgraphs of each node,
     employing a training strategy similar to GraphSAGE. ... we set the
     similarity threshold to 0 and select the top-10 most similar nodes"

B(.) is the frozen LM -- Sentence-BERT. Similarity is computed on **raw text**
embeddings, so the sampler runs before any LLM enhancement and its output is
reusable across variants.

Two points the paper leaves ambiguous, resolved explicitly and configurably:

1. **Order of threshold and top-N.** Eq. 4 defines the threshold filter, and the
   prose says "select the top-N ... that exceed a predefined threshold". We
   therefore filter by delta first, then take the top-N of what survives. With
   the paper's delta = 0 the two orders coincide for any node with >= N
   non-negative neighbours, so this choice only bites in the sensitivity sweep
   (delta = 0.4 / -0.4). Exposed as `threshold_first`.

2. **How top-N applies across 2 hops.** The paper says "2-hop subgraphs" and
   "top-10", without saying whether 10 is per-hop or total. We default to
   per-hop expansion (`per_hop=True`), i.e. GraphSAGE-style: select top-N
   neighbours of the centre, then top-N neighbours of each of those. That
   matches the paper's "training strategy similar to GraphSAGE" and is the only
   reading under which `hops` and `top_k` are independent knobs. The
   total-budget reading is available as `per_hop=False`. **UNKNOWN which the
   authors used** -- recorded, not guessed.

Self-loops are always excluded from candidate neighbours: GLBench ships
`add_self_loops` graphs and a node is trivially similarity-1 to itself, so
leaving them in would consume a top-N slot with no information and inflate
homophily (research/dataset_notes.md section 4.1).
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingConfig:
    """Paper defaults. `Reddit/0_10_0/` is consistent with threshold 0, top 10."""

    hops: int = 2
    top_k: int = 10
    similarity_threshold: float = 0.0

    strategy: str = "semantic"
    """One of the strategies the paper compares in Figure 3(a):
        semantic  -- SS,  the proposed method (threshold + top-N by cosine)
        semantic_nothreshold -- SS*, top-N by cosine, no threshold
        random    -- RS,  uniform random neighbours
        none      -- NS,  keep all neighbours (no sampling)
        feature   -- FS,  similarity on the graph's own node features
    """

    per_hop: bool = True
    """top_k applies per hop (GraphSAGE-style). See module docstring point 2."""

    threshold_first: bool = True
    """Filter by threshold, then take top-N. See module docstring point 1."""

    seed: int = 0
    """Only used by strategy='random'."""

    include_center: bool = True
    """Keep the centre node in its own subgraph. Required -- the centre is the
    node being classified."""

    def cache_key(self) -> str:
        """Stable identifier for the sampling cache.

        Mirrors upstream's `0_10_0` convention (threshold_topk_?) but is
        unambiguous, because guessing the third field would be a fabrication.
        """
        parts = [
            f"{self.strategy}",
            f"h{self.hops}",
            f"k{self.top_k}",
            f"t{self.similarity_threshold:g}",
            "perhop" if self.per_hop else "total",
        ]
        if self.strategy == "random":
            parts.append(f"s{self.seed}")
        return "_".join(parts)

    def as_record(self) -> dict:
        return {**dataclasses.asdict(self), "cache_key": self.cache_key()}


@dataclass
class Subgraph:
    """A sampled subgraph centred on one node.

    Field names match what upstream FLAG's drivers expect (`subset`, `central`,
    `edge_index`) so an adapter can feed them without translation.
    """

    central: int
    """Node id of the centre, in ORIGINAL graph indexing."""

    subset: torch.Tensor
    """Node ids in the subgraph, original indexing. `subset[0] == central`."""

    edge_index: torch.Tensor
    """Edges re-indexed to positions within `subset`."""

    hop: torch.Tensor
    """Hop distance of each node in `subset` from the centre (0 for the centre)."""

    @property
    def num_nodes(self) -> int:
        return int(self.subset.shape[0])

    def center_position(self) -> int:
        """Index of the centre within `subset`.

        Upstream computes this as `subset == central`, which is why `subset`
        must contain the centre exactly once.
        """
        matches = torch.nonzero(self.subset == self.central, as_tuple=False)
        if matches.numel() != 1:
            raise ValueError(
                f"centre {self.central} appears {matches.numel()} times in "
                f"subset; expected exactly once"
            )
        return int(matches[0, 0])


def build_adjacency(
    edge_index: torch.Tensor, num_nodes: int, drop_self_loops: bool = True
) -> list[np.ndarray]:
    """CSR-ish adjacency as a list of neighbour arrays.

    A plain list of arrays beats repeated boolean masking over `edge_index`:
    sampling touches every node, and masking an E-length tensor per node is
    O(N*E). Memory is O(E), which is fine at Reddit's scale.
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    if drop_self_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]

    order = np.argsort(src, kind="stable")
    src_sorted, dst_sorted = src[order], dst[order]
    boundaries = np.searchsorted(src_sorted, np.arange(num_nodes + 1))
    return [
        dst_sorted[boundaries[i]:boundaries[i + 1]] for i in range(num_nodes)
    ]


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalise rows so cosine similarity becomes a dot product.

    Zero rows (a node whose text is empty -- Instagram has texts of length 0)
    are left as zeros, giving them similarity 0 to everything. That is the
    honest behaviour: cosine is undefined for a zero vector, and 0 places such
    a node exactly at the paper's default threshold rather than silently
    ranking it first or last.
    """
    norms = embeddings.norm(dim=1, keepdim=True)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    return embeddings / safe


def select_neighbors(
    center: int,
    candidates: np.ndarray,
    normalized: torch.Tensor | None,
    config: SamplingConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply Eq. 3-4 to one node's candidate neighbours.

    Returns the selected neighbour ids. Never includes `center`.
    """
    candidates = candidates[candidates != center]
    if candidates.size == 0:
        return candidates

    if config.strategy == "none":
        return candidates

    if config.strategy == "random":
        assert rng is not None, "strategy='random' needs an rng"
        if candidates.size <= config.top_k:
            return candidates
        picked = rng.choice(candidates.size, size=config.top_k, replace=False)
        return candidates[np.sort(picked)]

    # -- similarity-based strategies (semantic, semantic_nothreshold, feature)
    assert normalized is not None, f"strategy={config.strategy!r} needs embeddings"
    center_vec = normalized[center]
    neighbor_vecs = normalized[torch.from_numpy(candidates.astype(np.int64))]
    sims = (neighbor_vecs @ center_vec).numpy()          # Eq. 3

    use_threshold = config.strategy != "semantic_nothreshold"

    if use_threshold and config.threshold_first:
        keep = sims >= config.similarity_threshold        # Eq. 4
        candidates, sims = candidates[keep], sims[keep]
        if candidates.size == 0:
            return candidates

    if candidates.size > config.top_k:
        # argpartition then sort: O(n) selection instead of a full O(n log n).
        top = np.argpartition(-sims, config.top_k - 1)[:config.top_k]
        top = top[np.argsort(-sims[top], kind="stable")]
        candidates, sims = candidates[top], sims[top]

    if use_threshold and not config.threshold_first:
        candidates = candidates[sims >= config.similarity_threshold]

    return np.sort(candidates)


def sample_subgraph(
    center: int,
    adjacency: list[np.ndarray],
    normalized: torch.Tensor | None,
    config: SamplingConfig,
    rng: np.random.Generator | None = None,
) -> Subgraph:
    """Build the sampled k-hop subgraph around `center`.

    Frontier expansion: at each hop, expand only the nodes newly added by the
    previous hop, applying `select_neighbors` to each. Already-visited nodes are
    not re-expanded, which keeps the subgraph a tree-like neighbourhood rather
    than the full induced subgraph.
    """
    visited: dict[int, int] = {center: 0}      # node -> hop at which it entered
    order: list[int] = [center]                # preserves centre-first ordering
    frontier = [center]

    for hop in range(1, config.hops + 1):
        next_frontier: list[int] = []
        budget = None
        if not config.per_hop:
            # Total-budget reading: share top_k across the whole expansion.
            budget = max(config.top_k - (len(order) - 1), 0)
            if budget == 0:
                break

        for node in frontier:
            selected = select_neighbors(
                node, adjacency[node], normalized, config, rng
            )
            for neighbor in selected.tolist():
                if neighbor in visited:
                    continue
                if budget is not None and budget <= 0:
                    break
                visited[neighbor] = hop
                order.append(neighbor)
                next_frontier.append(neighbor)
                if budget is not None:
                    budget -= 1
            if budget is not None and budget <= 0:
                break
        frontier = next_frontier
        if not frontier:
            break

    subset = torch.tensor(order, dtype=torch.long)
    hops = torch.tensor([visited[n] for n in order], dtype=torch.long)

    # Induced edges among the selected nodes, re-indexed to subset positions.
    position = {node: i for i, node in enumerate(order)}
    selected_set = set(order)
    rows, cols = [], []
    for node in order:
        i = position[node]
        for neighbor in adjacency[node]:
            neighbor = int(neighbor)
            if neighbor in selected_set:
                rows.append(i)
                cols.append(position[neighbor])
    edge_index = (
        torch.tensor([rows, cols], dtype=torch.long)
        if rows
        else torch.zeros((2, 0), dtype=torch.long)
    )

    return Subgraph(
        central=center, subset=subset, edge_index=edge_index, hop=hops
    )


def sample_all(
    node_ids,
    adjacency: list[np.ndarray],
    embeddings: torch.Tensor | None,
    config: SamplingConfig,
    progress: bool = False,
) -> list[Subgraph]:
    """Sample a subgraph for each node in `node_ids`."""
    normalized = (
        normalize_embeddings(embeddings) if embeddings is not None else None
    )
    rng = np.random.default_rng(config.seed) if config.strategy == "random" else None

    node_ids = list(node_ids)
    iterator = node_ids
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(node_ids, desc=f"sampling [{config.cache_key()}]")
        except ImportError:
            pass

    return [
        sample_subgraph(int(n), adjacency, normalized, config, rng)
        for n in iterator
    ]


def subgraph_homophily(subgraphs: list[Subgraph], y: torch.Tensor) -> float:
    """Average edge homophily over sampled subgraphs -- the paper's Eq. 5.

        h = sum_{(u,v) in E} I(y(u) = y(v)) / |E|

    This is the quantity Figure 3(a) uses to argue semantic sampling beats
    random and no-sampling, so it is the natural check that our reimplementation
    reproduces the paper's *motivation*, independently of any model.
    """
    same, total = 0, 0
    for sg in subgraphs:
        if sg.edge_index.numel() == 0:
            continue
        labels = y[sg.subset]
        src_labels = labels[sg.edge_index[0]]
        dst_labels = labels[sg.edge_index[1]]
        same += int((src_labels == dst_labels).sum())
        total += int(sg.edge_index.shape[1])
    return same / total if total else float("nan")


def sampling_stats(subgraphs: list[Subgraph]) -> dict:
    sizes = np.array([sg.num_nodes for sg in subgraphs])
    edges = np.array([sg.edge_index.shape[1] for sg in subgraphs])
    return {
        "num_subgraphs": len(subgraphs),
        "nodes_per_subgraph": {
            "min": int(sizes.min()), "max": int(sizes.max()),
            "mean": round(float(sizes.mean()), 3),
            "median": int(np.median(sizes)),
        },
        "edges_per_subgraph": {
            "min": int(edges.min()), "max": int(edges.max()),
            "mean": round(float(edges.mean()), 3),
        },
        "isolated_centers": int((sizes == 1).sum()),
    }

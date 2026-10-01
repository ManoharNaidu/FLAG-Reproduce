"""Debug, sanity-check and compare the cosine (FLAG) and Markov-diffusion (FLAG-MD) samplers.

    python -m scripts.analyze.compare_samplers debug   --dataset reddit --nodes 6
    python -m scripts.analyze.compare_samplers sanity  --dataset reddit
    python -m scripts.analyze.compare_samplers analyze --dataset reddit --diffusion-steps 2

`debug`   prints, for several target nodes, what each sampler selects and why.
`sanity`  runs the ten pre-training checks; exits non-zero if any fails.
`analyze` compares the NEIGHBOURHOODS the two samplers produce (overlap, graph
          distance, score distributions, homophily, fraud/normal composition).

LABELS. `y` is used ONLY inside `analyze` -- after both samplers have already
produced their subgraphs -- to describe them. The samplers themselves never
receive labels (they are built from `adjacency` and `embeddings` alone); check 10
of `sanity` verifies that by construction.
"""
from __future__ import annotations

import argparse
import inspect
import json
import pathlib
import re
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]

from flagbench.sampling import markov_diffusion as MD  # noqa: E402
from flagbench.sampling import semantic as S  # noqa: E402
from flagbench.sampling.cli import add_md_args, config_from_args  # noqa: E402
from scripts.preprocess.sample_subgraphs import load_inputs  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------- setup
def setup(dataset: str, args):
    payload, emb = load_inputs(dataset, args.model)
    n = int(payload["y"].shape[0])
    adjacency = S.build_adjacency(payload["edge_index"], n, drop_self_loops=True)
    cos_cfg = config_from_args(args, strategy="semantic")
    md_cfg = config_from_args(args, strategy="markov_diffusion")
    cos = S.make_sampler(cos_cfg, adjacency, emb)
    md = S.make_sampler(md_cfg, adjacency, emb)
    return payload, emb, adjacency, cos_cfg, md_cfg, cos, md


def cosine_scores(cos, center, cands):
    return (cos.normalized[torch.from_numpy(cands)] @ cos.normalized[center]).numpy()


# --------------------------------------------------------------------- debug
def cmd_debug(args) -> int:
    payload, emb, adj, cos_cfg, md_cfg, cos, md = setup(args.dataset, args)
    deg = np.array([len(a) for a in adj])
    rng = np.random.default_rng(args.seed)
    big = np.flatnonzero(deg > cos_cfg.top_k)
    mid = np.flatnonzero((deg >= 3) & (deg <= cos_cfg.top_k))
    pick = list(rng.choice(big, min(args.nodes // 2 + args.nodes % 2, len(big)), replace=False)) \
        + list(rng.choice(mid, min(args.nodes // 2, len(mid)), replace=False))

    print(f"dataset={args.dataset}  diffusion_steps={md_cfg.diffusion_steps}  "
          f"operator={md_cfg.diffusion_operator}  md_selection={md_cfg.md_selection}  "
          f"top_n={md_cfg.top_k}  cosine_threshold={cos_cfg.similarity_threshold}")
    for v in map(int, pick):
        cands = md.clean_candidates(v, adj[v])
        c_sel = cos.select(v, adj[v])
        m_sel = md.select(v, adj[v])
        c_scores = dict(zip(cands.tolist(), cosine_scores(cos, v, cands)))
        d = dict(zip(cands.tolist(), md.distances(v, cands)))
        print(f"\nTarget node: {v}")
        print(f"  candidate neighbours: {cands.size}   "
              f"cosine selected: {c_sel.size}   MD selected: {m_sel.size}   "
              f"diffusion steps: {md_cfg.diffusion_steps}")
        print("  Cosine selected:            neighbor_id    cosine_score")
        for u in sorted(c_sel.tolist(), key=lambda u: -c_scores[u]):
            print(f"                              {u:>10d}    {c_scores[u]:+.4f}")
        print("  MD selected:                neighbor_id    diffusion_distance   (cosine)")
        for u in sorted(m_sel.tolist(), key=lambda u: d[u]):
            print(f"                              {u:>10d}    {d[u]:.4f}             ({c_scores[u]:+.3f})")
        both = len(set(c_sel.tolist()) & set(m_sel.tolist()))
        print(f"  overlap: {both}/{max(len(c_sel), len(m_sel))}")
    return 0


# -------------------------------------------------------------------- sanity
def cmd_sanity(args) -> int:
    payload, emb, adj, cos_cfg, md_cfg, cos, md = setup(args.dataset, args)
    n, d = emb.shape
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1. dimensions
    check("1 matrix dimensions",
          md.A.shape == (n, n) and md.T.shape == (n, n) and md.H.shape == (n, d),
          f"A,T={md.T.shape}  X={tuple(emb.shape)}  H={md.H.shape}")

    # 2. transition matrix valid: non-negative, rows sum to 1 (deg>0) or 0 (deg==0)
    rowsum = np.asarray(md.T.sum(axis=1)).ravel()
    deg = np.asarray(md.A.sum(axis=1)).ravel()
    ok_rows = np.allclose(rowsum[deg > 0], 1.0) and np.all(rowsum[deg == 0] == 0)
    check("2 transition matrix valid",
          ok_rows and md.T.data.min() >= 0,
          f"row sums (deg>0) in [{rowsum[deg>0].min():.6f}, {rowsum[deg>0].max():.6f}], "
          f"{int((deg==0).sum())} zero-degree rows == 0, min entry {md.T.data.min():.3g}")

    # 3. diffusion operator valid: sparse H equals dense Z(K)X on a small induced graph
    rng = np.random.default_rng(0)
    nodes = np.sort(rng.choice(np.flatnonzero(deg > 0), size=min(400, int((deg > 0).sum())),
                               replace=False))
    sub_adj = [np.searchsorted(nodes, a[np.isin(a, nodes)]) for a in (adj[i] for i in nodes)]
    A_small = MD.adjacency_to_csr(sub_adj, len(nodes))
    T_small = MD.transition_matrix(A_small)
    X_small = md.normalized.numpy().astype(np.float64)[nodes]
    worst = 0.0
    for K in (1, 2, 3, 5):
        for op in MD.OPERATORS:
            first, scale = MD._operator_terms(op, K)
            Td = T_small.toarray()
            power, total = np.eye(len(nodes)), np.zeros((len(nodes),) * 2)
            for k in range(K + 1):
                if k >= first:
                    total += power
                power = power @ Td
            dense = scale * total @ X_small
            sparse_ = MD.diffusion_embeddings(T_small, X_small, K, op)
            worst = max(worst, float(np.abs(dense - sparse_).max()))
    check("3 diffusion operator == dense Z(K)X (K=1,2,3,5; 3 operators)", worst < 1e-10,
          f"max abs error {worst:.2e} on a {len(nodes)}-node induced graph")

    # 4. no NaN / inf
    all_d = np.concatenate([md.distances(v, np.asarray(a, dtype=np.int64)) for v, a in enumerate(adj) if len(a)])
    check("4 no NaN / inf", np.isfinite(md.H).all() and np.isfinite(all_d).all(),
          f"{all_d.size:,} distances, min {all_d.min():.4f} max {all_d.max():.4f}")

    # 5. no division by zero (zero-degree rows are zero, and no warnings raised)
    with np.errstate(all="raise"):
        try:
            MD.transition_matrix(md.A)
            MD.diffusion_embeddings(md.T, md.normalized.numpy(), md_cfg.diffusion_steps)
            ok = True
        except FloatingPointError as exc:
            ok = False
            print("   ", exc)
    check("5 no division by zero", ok, f"{int((deg==0).sum())} isolated nodes handled")

    # 6-9. per-call invariants over EVERY node
    bad_subset = bad_budget = bad_self = bad_order = 0
    differs = contested = 0
    for v, a in enumerate(adj):
        cands = md.clean_candidates(v, a)
        sel = md.select(v, a)
        csel = cos.select(v, a)
        if not set(sel.tolist()) <= set(cands.tolist()):
            bad_subset += 1
        k = md.budget(v, cands)
        if sel.size > max(k, 0) or (md_cfg.md_selection == "matched_cosine" and sel.size != csel.size):
            bad_budget += 1
        if v in sel.tolist():
            bad_self += 1
        if cands.size > sel.size > 0:                    # ranking actually mattered
            contested += 1
            dist = dict(zip(cands.tolist(), md.distances(v, cands)))
            chosen = max(dist[u] for u in sel.tolist())
            rest = [dist[u] for u in cands.tolist() if u not in set(sel.tolist())]
            if rest and chosen > min(rest) + 1e-12:      # a nearer node was passed over
                bad_order += 1
        if set(sel.tolist()) != set(csel.tolist()):
            differs += 1
    check("6 selected nodes are in the candidate neighbourhood", bad_subset == 0, f"violations {bad_subset}")
    check("7 top-N / budget respected", bad_budget == 0,
          f"violations {bad_budget}  (mode={md_cfg.md_selection}, N={md_cfg.top_k})")
    check("8 target never selected as its own neighbour", bad_self == 0, f"violations {bad_self}")
    check("8b smallest distances chosen (not largest)", bad_order == 0,
          f"{contested:,} contested nodes, violations {bad_order}")
    check("9 MD ranking differs from cosine for some nodes", differs > 0,
          f"{differs:,} of {n:,} nodes select a different neighbour set")

    # 10. no labels in selection
    src = inspect.getsource(MD)
    params = list(inspect.signature(MD.MarkovDiffusionNeighborSampler.__init__).parameters)
    sig = ", ".join(params)
    uses_labels = bool(re.search(r"\b(labels?|y_true|payload\[.y.\])\b", re.sub(r'""".*?"""', "", src, flags=re.S)))
    label_like = [p for p in params if re.fullmatch(r"y|labels?|targets?|y_true", p)]
    check("10 no labels used during sampling", not label_like and not uses_labels,
          f"sampler parameters ({sig}) include no label tensor; no label reference in code")

    print(f"\nSANITY [{args.dataset}] K={md_cfg.diffusion_steps} "
          f"selection={md_cfg.md_selection} N={md_cfg.top_k}\n{'-' * 76}")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}\n        {detail}")
    failed = [r for r in results if not r[1]]
    print(f"\n  {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


# ------------------------------------------------------------------- analyze
def _stat(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {"mean": float(x.mean()), "p05": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "n": int(x.size)}


def cmd_analyze(args) -> int:
    payload, emb, adj, cos_cfg, md_cfg, cos, md = setup(args.dataset, args)
    y = payload["y"].numpy()
    n = len(y)

    # subgraphs for every node from both samplers (with the labels NOT passed in)
    cos_sg = S.sample_all(range(n), adj, emb, cos_cfg, sampler=cos)
    md_sg = S.sample_all(range(n), adj, emb, md_cfg, sampler=md)

    # ---- 1-hop selection (the decision the samplers actually differ on)
    one_c, one_m, contested = [], [], []
    for v in range(n):
        c, m = set(cos.select(v, adj[v]).tolist()), set(md.select(v, adj[v]).tolist())
        one_c.append(c), one_m.append(m)
        contested.append(md.clean_candidates(v, adj[v]).size > len(c) > 0)
    contested = np.array(contested)
    has_nbr = np.array([len(a) > 0 for a in adj])

    def jacc(a, b):
        return len(a & b) / len(a | b) if (a | b) else 1.0

    j1 = np.array([jacc(a, b) for a, b in zip(one_c, one_m)])
    sets_c = [set(s.subset.tolist()) - {s.central} for s in cos_sg]
    sets_m = [set(s.subset.tolist()) - {s.central} for s in md_sg]
    js = np.array([jacc(a, b) for a, b in zip(sets_c, sets_m)])
    differ = np.array([a != b for a, b in zip(sets_c, sets_m)])

    # ---- score distributions over selected neighbours (1-hop)
    def scores_of(sel_sets, fn):
        out = []
        for v, s in enumerate(sel_sets):
            if s:
                u = np.array(sorted(s), dtype=np.int64)
                out.append(fn(v, u))
        return np.concatenate(out) if out else np.zeros(0)

    cos_fn = lambda v, u: cosine_scores(cos, v, u)          # noqa: E731
    md_fn = lambda v, u: md.distances(v, u)                 # noqa: E731

    # ---- graph distance = hop at which the node entered the subgraph
    def hops(sgs):
        h = np.concatenate([s.hop.numpy()[1:] for s in sgs if s.num_nodes > 1])
        return h

    # ---- labels: analysis only, after selection
    def composition(sgs):
        fraud_c, norm_c = [], []
        for s in sgs:
            if s.num_nodes <= 1:
                continue
            nb = s.subset.numpy()[1:]
            frac_fraud = float((y[nb] == 1).mean())
            (fraud_c if y[s.central] == 1 else norm_c).append(frac_fraud)
        return {"fraud_neighbour_frac_of_FRAUD_centres": float(np.mean(fraud_c)) if fraud_c else float("nan"),
                "fraud_neighbour_frac_of_NORMAL_centres": float(np.mean(norm_c)) if norm_c else float("nan"),
                "n_fraud_centres_with_neighbours": len(fraud_c),
                "n_normal_centres_with_neighbours": len(norm_c)}

    all_nodes = np.arange(n)
    report = {
        "dataset": args.dataset,
        "cosine_config": cos_cfg.as_record(),
        "md_config": md_cfg.as_record(),
        "nodes": n,
        "nodes_with_neighbours": int(has_nbr.sum()),
        "nodes_where_top_n_or_threshold_bites (candidates > kept)": int(contested.sum()),
        "overlap": {
            "1hop_jaccard_all_nodes_with_neighbours": float(j1[has_nbr].mean()),
            "1hop_jaccard_contested_nodes": float(j1[contested].mean()) if contested.any() else None,
            "subgraph_jaccard_all": float(js[has_nbr].mean()),
            "centres_whose_subgraph_node_set_differs": int(differ.sum()),
            "fraction_of_centres_differing": float(differ.mean()),
        },
        "unique_neighbours_per_centre (subgraph, mean)": {
            "cosine_only": float(np.mean([len(a - b) for a, b in zip(sets_c, sets_m)])),
            "md_only": float(np.mean([len(b - a) for a, b in zip(sets_c, sets_m)])),
            "both": float(np.mean([len(a & b) for a, b in zip(sets_c, sets_m)])),
        },
        "nodes_per_subgraph_mean": {"cosine": float(np.mean([s.num_nodes for s in cos_sg])),
                                    "md": float(np.mean([s.num_nodes for s in md_sg]))},
        "graph_distance_hop_mean": {"cosine": float(hops(cos_sg).mean()), "md": float(hops(md_sg).mean())},
        "cosine_similarity_of_selected_1hop": {"selected_by_cosine": _stat(scores_of(one_c, cos_fn)),
                                               "selected_by_md": _stat(scores_of(one_m, cos_fn))},
        "diffusion_distance_of_selected_1hop": {"selected_by_cosine": _stat(scores_of(one_c, md_fn)),
                                                "selected_by_md": _stat(scores_of(one_m, md_fn))},
        "subgraph_edge_homophily": {
            "cosine": S.subgraph_homophily(cos_sg, payload["y"]),
            "md": S.subgraph_homophily(md_sg, payload["y"]),
            "no_sampling(reference)": S.subgraph_homophily(
                S.sample_all(all_nodes.tolist()[: min(n, 4000)], adj, None,
                             S.SamplingConfig(strategy="none")), payload["y"]),
        },
        "composition (ANALYSIS ONLY; labels never used to select)": {
            "cosine": composition(cos_sg), "md": composition(md_sg)},
        "md_sampler_stats": md.stats,
    }
    # restrict the same metrics to contested centres, where the ranking matters
    idx = np.flatnonzero(contested)
    if idx.size:
        report["contested_only"] = {
            "n": int(idx.size),
            "subgraph_edge_homophily": {
                "cosine": S.subgraph_homophily([cos_sg[i] for i in idx], payload["y"]),
                "md": S.subgraph_homophily([md_sg[i] for i in idx], payload["y"]),
            },
        }

    out = ROOT / "results" / "tables" / f"sampler_analysis__{args.dataset}__{md_cfg.cache_key()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["debug", "sanity", "analyze"])
    parser.add_argument("--dataset", default="reddit", choices=["reddit", "instagram"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=6, help="debug: number of target nodes")
    add_md_args(parser)
    args = parser.parse_args(argv)
    return {"debug": cmd_debug, "sanity": cmd_sanity, "analyze": cmd_analyze}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

"""Build sampled subgraphs (FLAG Eq. 3-4), and reproduce the Figure 3(a) study.

    python -m scripts.preprocess.sample_subgraphs --dataset all
    python -m scripts.preprocess.sample_subgraphs --dataset reddit --compare-strategies
    python -m scripts.preprocess.sample_subgraphs --dataset reddit --top-k 5 --threshold 0.2

Writes cache/sampling/<dataset>__<cache_key>.pt  (+ .json stats)

`--compare-strategies` reproduces the paper's **Figure 3(a) motivation study**,
which compares average subgraph edge homophily (Eq. 5) across:

    NS  no sampling            (keep every neighbour)
    RS  random sampling
    FS  shallow-feature similarity   (uses the graph's own `x`)
    SS* semantic similarity, no threshold
    SS  semantic similarity + threshold   <- the proposed method

The paper's claim is that SS yields the highest homophily. That claim is
checkable **without training any model**, which makes it the cheapest genuine
reproduction target in the whole paper -- and the first one we attempt.

The upstream repository ships no sampler at all (flag_code_audit.md GAP-1), so
this is REIMPLEMENTED from Eq. 3-4. Its unit tests are in
tests/unit/test_semantic_sampling.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys
import time

import torch

from flagbench.sampling import semantic

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")


def load_inputs(dataset: str, model: str):
    graph_path = ROOT / "data" / "benchmark" / f"flag_{dataset}" / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"{graph_path} missing. Run:\n"
            f"  python -m scripts.preprocess.build_benchmark --dataset {dataset}"
        )
    payload = torch.load(graph_path, map_location="cpu")

    safe_model = model.replace("/", "_")
    emb_path = ROOT / "cache" / "embeddings" / f"{dataset}__{safe_model}__raw.pt"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} missing. Run:\n"
            f"  python -m scripts.preprocess.encode_text --dataset {dataset}"
        )
    embeddings = torch.load(emb_path, map_location="cpu")

    if embeddings.shape[0] != payload["y"].shape[0]:
        raise RuntimeError(
            f"embedding count {embeddings.shape[0]:,} != node count "
            f"{payload['y'].shape[0]:,}. The cache is stale for this benchmark "
            f"build; re-run encode_text with --force."
        )
    return payload, embeddings


STRATEGY_LABELS = {
    "none": "NS  (no sampling)",
    "random": "RS  (random)",
    "feature": "FS  (shallow-feature similarity)",
    "semantic_nothreshold": "SS* (semantic, no threshold)",
    "semantic": "SS  (semantic + threshold)  <- proposed",
}


def compare_strategies(dataset: str, args) -> dict:
    """Reproduce the Figure 3(a) homophily comparison."""
    payload, embeddings = load_inputs(dataset, args.model)
    y = payload["y"]
    adjacency = semantic.build_adjacency(
        payload["edge_index"], int(y.shape[0]), drop_self_loops=True
    )

    centers = torch.nonzero(payload["test_mask"], as_tuple=False).flatten()
    if args.max_centers and len(centers) > args.max_centers:
        centers = centers[: args.max_centers]
    center_list = centers.tolist()

    print(f"\n{'=' * 74}")
    print(f"{dataset.upper()} -- Figure 3(a) reproduction: subgraph homophily")
    print(f"{'=' * 74}")
    print(f"  centres: {len(center_list):,} test nodes")
    print(f"  hops={args.hops}  top_k={args.top_k}  "
          f"threshold={args.threshold}\n")
    print(f"  {'strategy':<38} {'homophily':>10} {'nodes/sg':>10} {'edges/sg':>10}")
    print(f"  {'-' * 38} {'-' * 10} {'-' * 10} {'-' * 10}")

    rows = {}
    for strategy in ["none", "random", "feature", "semantic_nothreshold", "semantic"]:
        config = semantic.SamplingConfig(
            hops=args.hops,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            strategy=strategy,
            seed=args.seed,
        )
        # FS uses the graph's own node features; the rest use the LM embeddings.
        source = payload["x"] if strategy == "feature" else embeddings
        emb = None if strategy in ("none", "random") else source

        subgraphs = semantic.sample_all(center_list, adjacency, emb, config)
        homophily = semantic.subgraph_homophily(subgraphs, y)
        stats = semantic.sampling_stats(subgraphs)
        rows[strategy] = {
            "homophily": homophily,
            "stats": stats,
            "config": config.as_record(),
        }
        print(f"  {STRATEGY_LABELS[strategy]:<38} {homophily:>10.4f} "
              f"{stats['nodes_per_subgraph']['mean']:>10.2f} "
              f"{stats['edges_per_subgraph']['mean']:>10.2f}")

    ss = rows["semantic"]["homophily"]
    verdict = {
        name: ss - rows[name]["homophily"]
        for name in ["none", "random", "feature", "semantic_nothreshold"]
    }
    print(f"\n  SS advantage over:")
    for name, delta in verdict.items():
        marker = "OK  " if delta > 0 else "MISS"
        print(f"    {marker} {STRATEGY_LABELS[name]:<38} {delta:+.4f}")

    claim_holds = all(d > 0 for d in verdict.values())
    print(f"\n  Paper's Figure 3(a) claim (SS highest homophily): "
          f"{'SUPPORTED' if claim_holds else 'NOT SUPPORTED'} on {dataset}")
    if not claim_holds:
        print("  Recorded as-is. Results are never adjusted toward the paper.")

    return {
        "dataset": dataset,
        "num_centers": len(center_list),
        "hops": args.hops,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "strategies": rows,
        "ss_advantage": verdict,
        "figure_3a_claim_supported": claim_holds,
        "note": (
            "Homophily is FLAG Eq. 5 over sampled subgraphs of test nodes. "
            "Sampler REIMPLEMENTED from Eq. 3-4; no upstream source exists. "
            "Benchmark built with our downsampling seed, not the authors'."
        ),
    }


def build_cache(dataset: str, args) -> dict:
    payload, embeddings = load_inputs(dataset, args.model)
    y = payload["y"]
    num_nodes = int(y.shape[0])
    adjacency = semantic.build_adjacency(
        payload["edge_index"], num_nodes, drop_self_loops=True
    )
    config = semantic.SamplingConfig(
        hops=args.hops,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        strategy=args.strategy,
        seed=args.seed,
    )
    out = ROOT / "cache" / "sampling" / f"{dataset}__{config.cache_key()}.pt"

    print(f"\n{'=' * 74}\n{dataset.upper()} -- sampling\n{'=' * 74}")
    print(f"  config: {config.cache_key()}")
    if out.exists() and not args.force:
        print(f"  cache hit: {out.relative_to(ROOT)}")
        return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))

    start = time.time()
    subgraphs = semantic.sample_all(
        range(num_nodes), adjacency, embeddings, config, progress=True
    )
    elapsed = time.time() - start

    stats = semantic.sampling_stats(subgraphs)
    homophily = semantic.subgraph_homophily(subgraphs, y)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            {
                "central": sg.central,
                "subset": sg.subset,
                "edge_index": sg.edge_index,
                "hop": sg.hop,
            }
            for sg in subgraphs
        ],
        out,
    )

    meta = {
        "dataset": dataset,
        "config": config.as_record(),
        "num_nodes": num_nodes,
        "stats": stats,
        "subgraph_homophily": homophily,
        "seconds": round(elapsed, 1),
        "embedding_model": args.model,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
        "provenance": (
            "REIMPLEMENTED from FLAG Eq. 3-4; the official repository ships no "
            "sampler (flag_code_audit.md GAP-1)."
        ),
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"  sampled {num_nodes:,} subgraphs in {elapsed:.1f}s "
          f"({num_nodes / elapsed:.0f}/s)")
    print(f"  nodes/subgraph  min {stats['nodes_per_subgraph']['min']} "
          f"mean {stats['nodes_per_subgraph']['mean']} "
          f"max {stats['nodes_per_subgraph']['max']}")
    print(f"  isolated centres {stats['isolated_centers']:,}")
    print(f"  subgraph homophily {homophily:.4f}")
    print(f"  wrote {out.relative_to(ROOT)}")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all",
                        choices=["reddit", "instagram", "all"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--strategy", default="semantic",
                        choices=["semantic", "semantic_nothreshold", "random",
                                 "none", "feature"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compare-strategies", action="store_true",
                        help="reproduce the paper's Figure 3(a) homophily study")
    parser.add_argument("--max-centers", type=int, default=2000,
                        help="cap centres in --compare-strategies (0 = all)")
    args = parser.parse_args(argv)

    datasets = ["instagram", "reddit"] if args.dataset == "all" else [args.dataset]
    out_records = {}
    failed = []
    for dataset in datasets:
        try:
            if args.compare_strategies:
                out_records[dataset] = compare_strategies(dataset, args)
            else:
                out_records[dataset] = build_cache(dataset, args)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n  {dataset}: {exc}")
            failed.append(dataset)

    if args.compare_strategies and out_records:
        dest = ROOT / "results" / "tables" / "figure3a_homophily.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(out_records, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n  wrote {dest.relative_to(ROOT)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

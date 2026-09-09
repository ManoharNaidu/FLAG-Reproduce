"""Construct FLAG's 1:10 fraud-detection benchmark from the raw GLBench graphs.

    python -m scripts.preprocess.build_benchmark --dataset all
    python -m scripts.preprocess.build_benchmark --dataset reddit --downsample-seed 1

Reads   data/raw/<dataset>/<dataset>.pt        (never modified)
Writes  data/benchmark/flag_<dataset>/graph.pt
        data/benchmark/flag_<dataset>/dataset_manifest.json

REIMPLEMENTED from the paper's one-sentence description; the official repo ships
no construction code and its RNG seed is unpublished. Every stochastic choice
here is seeded and recorded, so our build is reproducible even though it cannot
be identical to the authors'.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys

import torch

from flagbench.datasets import benchmark as bench
from flagbench.datasets import glbench

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def minority_class_for(dataset: str, data) -> int:
    """Resolve the fraud class index from the file's own label names.

    Reads `label_name` rather than trusting a hardcoded index, so a change in
    GLBench's label ordering surfaces as an error instead of silently inverting
    the task.
    """
    sig = glbench.SIGNATURES[dataset]
    names = list(getattr(data, "label_name", []) or [])
    if not names:
        raise RuntimeError(
            f"{dataset}: no `label_name` in the file; cannot resolve which class "
            f"is the minority. Expected {sig.minority_class_name!r}."
        )
    wanted = sig.minority_class_name.lower()
    matches = [i for i, n in enumerate(names) if wanted in n.lower()]
    if len(matches) != 1:
        raise RuntimeError(
            f"{dataset}: expected exactly one label containing {wanted!r} in "
            f"{names}, found {matches}. Refusing to guess the fraud class."
        )
    return matches[0]


def process(dataset: str, args) -> dict | None:
    sig = glbench.SIGNATURES[dataset]
    raw_path = ROOT / "data" / "raw" / dataset / sig.filename
    out_dir = ROOT / "data" / "benchmark" / f"flag_{dataset}"

    print(f"\n{'=' * 74}\n{dataset.upper()}\n{'=' * 74}")
    if not raw_path.exists():
        print(f"  MISSING {raw_path}")
        print("  Run: python -m scripts.download.glbench --dataset all")
        return None

    print("  loading and re-verifying the raw graph...")
    data = glbench.load_raw(raw_path)
    glbench.verify(dataset, data)          # never build on an unverified graph

    minority = minority_class_for(dataset, data)
    names = list(getattr(data, "label_name", []))
    print(f"  minority (fraud) class = {minority} ({names[minority]!r})")

    config = bench.BenchmarkConfig(
        minority_class=minority,
        imbalance_ratio=args.imbalance_ratio,
        downsample_seed=args.downsample_seed,
        split_ratios=tuple(args.split_ratios),
        split_seed=args.split_seed,
        stratified=not args.no_stratify,
        drop_self_loops=not args.keep_self_loops,
    )

    print(f"  building 1:{config.imbalance_ratio:g} benchmark "
          f"(downsample_seed={config.downsample_seed}, "
          f"split_seed={config.split_seed})...")
    graph, manifest = bench.build(data, config, dataset_name=dataset)

    ds = manifest["downsampling"]
    print(f"\n  downsampling")
    print(f"    original      {ds['original_majority']:,} majority / "
          f"{ds['original_minority']:,} minority "
          f"({ds['original_ratio_majority_to_minority']}:1)")
    print(f"    target        {ds['target_minority']:,} minority")
    print(f"    kept          {ds['kept_majority']:,} majority / "
          f"{ds['kept_minority']:,} minority "
          f"({ds['achieved_ratio_majority_to_minority']}:1)")
    print(f"    discarded     {ds['minority_discarded']:,} minority "
          f"({ds['minority_discarded_fraction'] * 100:.1f}% of the class)")
    if ds["note"]:
        print(f"    NOTE: {ds['note']}")

    ed = manifest["edges"]
    print(f"\n  edges")
    print(f"    original      {ed['edges_in_original']:,}")
    print(f"    after induce  {ed['edges_after_induction']:,}")
    print(f"    self-loops    {ed['self_loops_found']:,} "
          f"({'dropped' if ed['self_loops_dropped'] else 'kept'})")
    print(f"    final         {ed['edges_final']:,}")

    con = manifest["constructed"]
    print(f"\n  constructed benchmark")
    print(f"    nodes         {con['num_nodes']:,}")
    print(f"    edges         {con['num_edges']:,}")
    print(f"    labels        {con['label_distribution']}")
    print(f"    isolated      {con['isolated_nodes']:,} "
          f"({con['isolated_nodes'] / con['num_nodes'] * 100:.1f}%)")
    print(f"    splits        ", end="")
    for name, info in con["splits"].items():
        print(f"{name}={info['n']:,}{info['per_class']} ", end="")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "y": graph.y,
        "raw_texts": graph.raw_texts,
        "train_mask": graph.train_mask,
        "val_mask": graph.val_mask,
        "test_mask": graph.test_mask,
        "original_node_ids": graph.original_node_ids,
        "label_names": graph.label_names,
    }
    graph_path = out_dir / "graph.pt"
    torch.save(payload, graph_path)

    manifest["source"] = {
        "raw_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": glbench.sha256_file(raw_path),
    }
    manifest["built_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    )
    manifest["output_file"] = str(graph_path.relative_to(ROOT)).replace("\\", "/")
    manifest["output_sha256"] = glbench.sha256_file(graph_path)

    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n    wrote {graph_path.relative_to(ROOT)} "
          f"({graph_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"    wrote {(out_dir / 'dataset_manifest.json').relative_to(ROOT)}")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="all",
        choices=[*sorted(glbench.SIGNATURES), "all"],
    )
    parser.add_argument("--imbalance-ratio", type=float, default=10.0)
    parser.add_argument("--downsample-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, default=[0.10, 0.10, 0.80],
        metavar=("TRAIN", "VAL", "TEST"),
        help="default 10/10/80, inherited from GraphAdapter/GLBench; "
             "the FLAG paper states no split",
    )
    parser.add_argument("--no-stratify", action="store_true")
    parser.add_argument(
        "--keep-self-loops", action="store_true",
        help="NOT recommended: self-loops inflate homophily and waste a top-k "
             "slot in semantic sampling",
    )
    args = parser.parse_args(argv)

    datasets = (
        sorted(glbench.SIGNATURES) if args.dataset == "all" else [args.dataset]
    )
    results = {d: process(d, args) for d in datasets}

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    ok = 0
    for name, manifest in results.items():
        if manifest is None:
            print(f"  {name:12s} FAILED")
            continue
        ok += 1
        con = manifest["constructed"]
        ds = manifest["downsampling"]
        print(f"  {name:12s} {con['num_nodes']:>7,} nodes  "
              f"{con['num_edges']:>9,} edges  "
              f"ratio {ds['achieved_ratio_majority_to_minority']:>6}:1  "
              f"isolated {con['isolated_nodes']:>6,}")

    print(f"\n  {ok}/{len(results)} built")
    print("\n  REMINDER: the paper's downsampling seed is unpublished, so these")
    print("  splits are ours, not the authors'. Exact agreement with Table 4 is")
    print("  not achievable. See research/reproduction_status.md section 6.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

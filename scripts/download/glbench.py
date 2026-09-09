"""Download and verify the GLBench Reddit / Instagram graphs.

    python -m scripts.download.glbench --dataset instagram
    python -m scripts.download.glbench --dataset all
    python -m scripts.download.glbench --dataset reddit --verify-only

Writes data/raw/<dataset>/<dataset>.pt plus a dataset_manifest.json recording
checksum, published signature, and observed statistics. data/raw is written once
and never modified afterwards.

Nothing is trusted on download: the graph is checked against GLBench's published
node/edge counts, text presence and class balance, and the script exits non-zero
on mismatch. `torch_geometric.datasets.Reddit` is a different dataset with the
same name and would otherwise pass unnoticed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import sys

from flagbench.datasets import glbench

# Node text contains emoji; a cp1252 Windows console would raise on print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download.glbench")


def process(dataset: str, raw_root: pathlib.Path, args) -> dict | None:
    sig = glbench.SIGNATURES[dataset]
    dest = raw_root / dataset
    target = dest / sig.filename

    print(f"\n{'=' * 72}\n{dataset.upper()}\n{'=' * 72}")

    if not target.exists():
        if args.verify_only:
            print(f"  MISSING: {target}")
            print(f"  Download from {glbench.GLBENCH_DRIVE_FOLDER}")
            return None
        print(f"  downloading ~{sig.approx_size_mb:.0f} MB -> {target}")
        try:
            target = glbench.download(dataset, dest, force=args.force)
        except Exception as exc:
            print(f"  DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
            print(f"\n  Manual fallback:")
            print(f"    1. open {glbench.GLBENCH_DRIVE_FOLDER}")
            print(f"    2. download {sig.filename}")
            print(f"    3. place it at {target}")
            print(f"    4. re-run with --verify-only")
            return None
    else:
        print(f"  present: {target} ({target.stat().st_size / 1024**2:.1f} MB)")

    print("  loading...")
    try:
        data = glbench.load_raw(target)
    except Exception as exc:
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
        return None

    print("  verifying against the published GLBench signature...")
    try:
        observed = glbench.verify(dataset, data)
    except glbench.DatasetVerificationError as exc:
        print(f"\n  VERIFICATION FAILED\n{exc}")
        return None

    print("  OK")
    print(f"    nodes            {observed['num_nodes']:,} "
          f"(published {sig.num_nodes:,})")
    print(f"    edges (stored)   {observed['num_edges_stored']:,} = "
          f"{observed['num_edges_excluding_self_loops']:,} directed + "
          f"{observed['num_self_loops']:,} self-loops"
          f" ({observed['num_self_loops_pre_existing']:,} pre-existing)")
    print(f"                     symmetric={observed['edge_index_is_symmetric']}, "
          f"{observed['num_edges_undirected']:,} undirected; published "
          f"{sig.num_edges:,} x{observed['symmetrisation_expansion']:.3f} "
          f"after symmetrisation")
    print(f"    classes          {observed['num_classes']}  "
          f"distribution {observed['label_distribution']}")
    print(f"    label names      {observed['label_names_in_file']}")
    print(f"    node features    {observed['feature_shape']} "
          f"{observed['feature_dtype']}")
    print(f"    raw_texts        len {observed['num_nodes']:,}, "
          f"chars min/mean/max = {observed['text_len_min']}/"
          f"{observed['text_len_mean']}/{observed['text_len_max']}")

    prov = observed["baseline_feature_provenance"]
    print(f"\n    OPEN QUESTION - baseline feature provenance")
    print(f"      dim={prov['observed_dim']}  assessment={prov['assessment']}")
    for line in _wrap(prov["note"], 66):
        print(f"      {line}")

    manifest = glbench.build_manifest(
        dataset, target, observed,
        download_date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    path = glbench.write_manifest(manifest, dest)
    print(f"\n    sha256   {manifest['sha256']}")
    print(f"    manifest {path.relative_to(ROOT)}")
    return manifest


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="all",
        choices=[*sorted(glbench.SIGNATURES), "all"],
    )
    parser.add_argument("--raw-dir", type=pathlib.Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="do not download; only verify what is already on disk",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if present",
    )
    args = parser.parse_args(argv)

    datasets = (
        sorted(glbench.SIGNATURES) if args.dataset == "all" else [args.dataset]
    )
    results = {d: process(d, args.raw_dir, args) for d in datasets}

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    ok = 0
    for name, manifest in results.items():
        if manifest is None:
            print(f"  {name:12s} FAILED")
        else:
            ok += 1
            obs = manifest["observed"]
            print(f"  {name:12s} OK   {obs['num_nodes']:>7,} nodes  "
                  f"feat {obs['feature_shape'][1]:>5}d  "
                  f"{obs['baseline_feature_provenance']['assessment']}")

    if ok and ok == len(results):
        dims = {
            m["observed"]["feature_shape"][1] for m in results.values() if m
        }
        print(f"\n  Feature dimensionalities observed: {sorted(dims)}")
        print("  -> update research/dataset_notes.md section 7 with this finding.")

    print(f"\n  {ok}/{len(results)} dataset(s) verified")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

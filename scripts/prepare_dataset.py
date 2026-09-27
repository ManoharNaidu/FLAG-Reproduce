"""End-to-end preparation: build the dataset, validate it, emit FLAG's payload.

    python -m scripts.prepare_dataset --dataset yelpchi
    python -m scripts.prepare_dataset --dataset all

Equivalent to build_flag_dataset + validate_dataset + the FLAG adapter. Existing
artifacts are reused unless --rebuild is passed (brief section 45).
"""
from __future__ import annotations

import argparse
import sys

from flagbench.flag_adapter import write_flag_payload
from flagbench.fraud_text import load_dataset, save, validate_dataset
from flagbench.fraud_text.loaders import BUILDERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=[*sorted(BUILDERS), "all"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="truncate the FLAG text view (recorded in the manifest)")
    args = ap.parse_args(argv)

    names = sorted(BUILDERS) if args.dataset == "all" else [args.dataset]
    rc = 0
    for name in names:
        print("=" * 74)
        print(name.upper())
        print("=" * 74)
        ds = load_dataset(name, rebuild=args.rebuild)
        print(ds.summary())

        report = validate_dataset(ds)
        if not report.ok:
            print(report.render())
            rc = 1
            continue
        print(f"\n  validation: PASS ({len(report.checks)} checks)")

        save(ds)
        path = write_flag_payload(ds, max_chars=args.max_chars)
        print(f"  FLAG payload -> {path.relative_to(path.parents[3])}")
        print(f"  registry key -> {'yelpchi_text' if name == 'yelpchi' else 'amazon_text'}"
              f"  (NOT the canonical '{name}' key)")
    return rc


if __name__ == "__main__":
    sys.exit(main())

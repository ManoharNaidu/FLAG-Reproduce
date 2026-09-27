"""Build and serialize a text-augmented fraud dataset.

    python -m scripts.build_flag_dataset --dataset yelpchi
    python -m scripts.build_flag_dataset --dataset all

Prerequisites (each prints its own instruction if missing):
    python -m experiments.yelpchi_amazon.build_native_benchmark --dataset all
    python -m scripts.verify_yelpchi_alignment
    python -m scripts.verify_amazon_alignment
"""
from __future__ import annotations

import argparse
import sys

from flagbench.fraud_text import save, validate_dataset
from flagbench.fraud_text.loaders import BUILDERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all",
                    choices=[*sorted(BUILDERS), "all"])
    args = ap.parse_args(argv)

    names = sorted(BUILDERS) if args.dataset == "all" else [args.dataset]
    failed = False
    for name in names:
        print("=" * 74)
        print(name.upper())
        print("=" * 74)
        ds = BUILDERS[name]()
        print(ds.summary())
        report = validate_dataset(ds)
        if not report.ok:
            print(report.render())
            failed = True
            continue
        path = save(ds)
        print(f"\n  wrote {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

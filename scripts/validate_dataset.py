"""Validate a serialized text-augmented fraud dataset.

    python -m scripts.validate_dataset --dataset yelpchi
    python -m scripts.validate_dataset --dataset all

Exits non-zero on any failed check, so it is usable as a gate.
"""
from __future__ import annotations

import argparse
import sys

from flagbench.fraud_text import load_dataset, validate_dataset
from flagbench.fraud_text.loaders import BUILDERS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all",
                    choices=[*sorted(BUILDERS), "all"])
    args = ap.parse_args(argv)

    names = sorted(BUILDERS) if args.dataset == "all" else [args.dataset]
    rc = 0
    for name in names:
        ds = load_dataset(name)
        report = validate_dataset(ds)
        print(report.render())
        print()
        if not report.ok:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

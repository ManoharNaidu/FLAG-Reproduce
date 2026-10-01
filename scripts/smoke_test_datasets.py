"""One-screen status of both text-augmented datasets (brief section 36).

    python -m scripts.smoke_test_datasets

Exit code 0 only if BOTH datasets load and pass every validation check.
"""
from __future__ import annotations

import sys

from flagbench.fraud_text import load_dataset, validate_dataset
from flagbench.fraud_text.base import VERIFIED
from flagbench.fraud_text.loaders import BUILDERS
from flagbench.fraud_text.serialization import out_dir


def main(argv=None) -> int:
    rc = 0
    for name in sorted(BUILDERS):
        print("=" * 66)
        print(f"Dataset:\n{name}\n")
        if not (out_dir(name) / "flag_dataset.pt").exists():
            print(f"NOT BUILT — run: python -m scripts.prepare_dataset "
                  f"--dataset {name}\n")
            rc = 1
            continue

        ds = load_dataset(name)
        cov = ds.text_coverage()
        sp = ds.split_counts()
        rel = ", ".join(f"{k}={int(v.shape[1]):,}"
                        for k, v in ds.relation_edges.items())
        print(f"Node type:\n{ds.node_type}\n")
        print(f"Nodes:\n{ds.num_nodes:,}\n")
        print(f"Features:\n{ds.num_features}\n")
        print(f"Relations:\n{rel}\n")
        print(f"Texts:\n{cov['non_empty']:,}\n")
        print(f"Matched text:\n{cov['verified']:,}/{cov['num_nodes']:,} "
              f"({cov['verified_fraction'] * 100:.1f}%)\n")
        print(f"Labels:\n{ds.label_counts()}  (1 = fraud)\n")
        print(f"Train:\n{sp['train']['n']:,}\n")
        print(f"Validation:\n{sp['val']['n']:,}\n")
        print(f"Test:\n{sp['test']['n']:,}\n")

        evaluated = cov['num_nodes']
        missing = sum(1 for i in
                      list(ds.train_idx.tolist()) + list(ds.val_idx.tolist())
                      + list(ds.test_idx.tolist())
                      if ds.text_status[i] != VERIFIED)
        print(f"Evaluated nodes lacking text:\n{missing:,}\n")

        report = validate_dataset(ds)
        print(f"Validation:\n{'PASS' if report.ok else 'FAIL'} "
              f"({len(report.checks)} checks)\n")
        if not report.ok:
            print(report.render())
            rc = 1
    print("=" * 66)
    print("NOTE: these are text-augmented STUDY datasets. The FLAG paper")
    print("reports no YelpChi/Amazon results and states both lack text, so")
    print("nothing here is comparable to a published FLAG number.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

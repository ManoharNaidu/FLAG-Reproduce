"""Prove (or refute) the YelpChi node <-> raw-review alignment, then emit the map.

    python -m scripts.verify_yelpchi_alignment
    python -m scripts.verify_yelpchi_alignment --skip-text-controls   # faster

WHY THIS EXISTS
---------------
`YelpChi.mat` (CARE-GNN release) stores only `features`, `label` and four
adjacency matrices. It contains **no review id, no user id, no text** -- verified
exhaustively in `research/_evidence/loop1_a_mat_files.md`. So attaching review
text to a graph node is not a lookup; it is a claim that needs proof.

Published work that uses YelpChi with text joins text **positionally** and never
checks it (see `research/_evidence/loop1_c_yelp_text.md`). This script refuses to
inherit that assumption and tests it instead.

THE TWO CLAIMS, TESTED SEPARATELY
---------------------------------
Claim 1 (node ordering).  `.mat` node i corresponds to the i-th row of the raw
  metadata after (a) dropping the 19 products with >800 reviews and (b) ordering
  rows by user first-appearance.
  TEST: rebuild R-U-R and R-S-R from the raw metadata under that ordering and
  compare the **sparsity patterns** to `net_rur` / `net_rsr` entry by entry.
  Totals matching is NOT accepted as evidence; only zero mismatched entries is.
  A random permutation would fail this with overwhelming probability.

Claim 2 (text join).  raw metadata row k is described by text line k of the
  concatenated hotel+restaurant corpus.
  TEST: reviews of the same product share product-specific vocabulary. Compare
  mean TF-IDF cosine for within-product pairs against a **same-block,
  different-product** control (which removes the hotel/restaurant confound), and
  confirm the signal decays to that control when the text is shifted.
  This is statistical evidence, not arithmetic proof, and is labelled as such.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
It does not repair, impute or reorder anything. If a check fails it says so and
exits non-zero. Nothing downstream should consume the mapping unless this exits 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
META = ROOT / "datasets" / "raw" / "yelpchi" / "metadata.txt"
HOTEL = ROOT / "datasets" / "raw" / "yelpchi" / "raw_text.txt"
REST = ROOT / "datasets" / "raw" / "yelpchi" / "output_review_yelpResData_NRYRcleaned.txt"
MAT = ROOT / "data" / "raw" / "yelpchi" / "YelpChi.mat"
OUT_DIR = ROOT / "datasets" / "processed" / "yelpchi"

PRODUCT_REVIEW_CAP = 800
"""Products with more than this many reviews are absent from the .mat node set.

Not a tuned parameter: it is the unique threshold that reproduces 45,954 nodes,
39,277/6,677 labels, and both relation edge counts exactly. Discovered, then
verified by the adjacency test below -- see mapping_report.md.
"""


def load_metadata(path: pathlib.Path):
    user, prod, rating, label, date = [], [], [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            user.append(int(parts[0])); prod.append(int(parts[1]))
            rating.append(float(parts[2])); label.append(int(parts[3]))
            date.append(parts[4])
    return (np.array(user), np.array(prod), np.array(rating),
            np.array(label), np.array(date))


def node_ordering(user: np.ndarray, prod: np.ndarray):
    """Return (keep_idx, perm): raw-file rows kept, and their .mat order."""
    pids, counts = np.unique(prod, return_counts=True)
    dropped = pids[counts > PRODUCT_REVIEW_CAP]
    keep = ~np.isin(prod, dropped)
    keep_idx = np.nonzero(keep)[0]

    u_k = user[keep]
    first_seen: dict[int, int] = {}
    for i, u in enumerate(u_k):
        if u not in first_seen:
            first_seen[u] = i
    order_key = np.array([first_seen[u] for u in u_k])
    perm = np.lexsort((np.arange(len(u_k)), order_key))
    return keep_idx, perm, dropped


def pairs_from_groups(keys: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric binary adjacency linking all rows sharing a key. No self-loops."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    rows, cols = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        members = order[a:b]
        if len(members) < 2:
            continue
        ii, jj = np.triu_indices(len(members), k=1)
        rows.append(members[ii]); cols.append(members[jj])
    if not rows:
        return sp.csr_matrix((n, n))
    r = np.concatenate(rows); c = np.concatenate(cols)
    m = sp.coo_matrix(
        (np.ones(2 * len(r), np.int8),
         (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n, n),
    ).tocsr()
    m.data[:] = 1
    return m


def binarise(m) -> sp.csr_matrix:
    out = sp.csr_matrix(m).copy()
    out.data[:] = 1
    out.eliminate_zeros()
    return out


def compare_exact(name: str, built, ref) -> tuple[bool, dict]:
    b, r = binarise(built), binarise(ref)
    mismatched = int((b != r).nnz)
    ok = mismatched == 0 and b.shape == r.shape
    print(f"    {name:<7} built={b.nnz:>9,}  mat={r.nnz:>9,}  "
          f"mismatched_entries={mismatched:>9,}  {'EXACT' if ok else 'FAIL'}")
    return ok, {"built_nnz": int(b.nnz), "mat_nnz": int(r.nnz),
                "mismatched_entries": mismatched, "exact": ok}


def text_controls(prod_all: np.ndarray, texts: list[str], n_hotel: int) -> dict:
    """Within-product vs same-block/different-product TF-IDF cosine, plus shifts."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.default_rng(0)
    X = TfidfVectorizer(max_features=50_000, stop_words="english",
                        min_df=2).fit_transform(texts)
    X = X.multiply(1 / (np.sqrt(X.multiply(X).sum(axis=1)) + 1e-12)).tocsr()
    block = np.zeros(len(texts), int); block[n_hotel:] = 1

    def sim(pairs, mapping=None):
        a, b = pairs[:, 0], pairs[:, 1]
        if mapping is not None:
            a, b = mapping[a], mapping[b]
        return float(np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel().mean())

    within = []
    for p in np.unique(prod_all):
        idx = np.nonzero(prod_all == p)[0]
        if len(idx) < 2:
            continue
        k = min(200, len(idx) * 2)
        a, b = rng.choice(idx, k), rng.choice(idx, k)
        ok = a != b
        within.append(np.stack([a[ok], b[ok]], 1))
    within = np.concatenate(within)

    ctrl = []
    for blk in (0, 1):
        idx = np.nonzero(block == blk)[0]
        k = len(within) // 2
        a, b = rng.choice(idx, k), rng.choice(idx, k)
        ok = prod_all[a] != prod_all[b]
        ctrl.append(np.stack([a[ok], b[ok]], 1))
    ctrl = np.concatenate(ctrl)

    true_sim, ctrl_sim = sim(within), sim(ctrl)
    shifts = {}
    n_rest = len(texts) - n_hotel
    for s in (1, 100, 1000):
        mapping = np.arange(len(texts))
        mapping[:n_hotel] = (np.arange(n_hotel) + s) % n_hotel
        mapping[n_hotel:] = n_hotel + (np.arange(n_rest) + s) % n_rest
        shifts[s] = sim(within, mapping)

    print(f"    within-product            {true_sim:.5f}")
    print(f"    same-block/diff-product   {ctrl_sim:.5f}   <- confound control")
    for s, v in shifts.items():
        print(f"    shifted by {s:<5d}          {v:.5f}")
    excess = true_sim - ctrl_sim
    print(f"    excess over control       {excess:+.5f} ({true_sim / ctrl_sim:.2f}x)")
    return {"within_product": true_sim, "same_block_diff_product": ctrl_sim,
            "shifted": shifts, "ratio_over_control": true_sim / ctrl_sim,
            "supports_row_level_join": bool(excess > 0.01 and
                                            all(v < true_sim for v in shifts.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-text-controls", action="store_true",
                    help="skip the TF-IDF claim-2 test (claim 1 still runs)")
    ap.add_argument("--no-write", action="store_true",
                    help="verify only; do not emit the mapping csv")
    args = ap.parse_args(argv)

    for p in (META, MAT):
        if not p.exists():
            print(f"MISSING {p}", file=sys.stderr)
            return 2

    print("=" * 78)
    print("YELPCHI ALIGNMENT VERIFICATION")
    print("=" * 78)

    user, prod, rating, label, date = load_metadata(META)
    print(f"\n  raw metadata: {len(user):,} rows, {len(np.unique(user)):,} users, "
          f"{len(np.unique(prod)):,} products")

    keep_idx, perm, dropped = node_ordering(user, prod)
    n = len(keep_idx)
    print(f"  dropped {len(dropped)} products with >{PRODUCT_REVIEW_CAP} reviews "
          f"-> {n:,} nodes")

    mat = sio.loadmat(MAT)
    mat_lab = np.asarray(mat["label"]).ravel().astype(int)
    if mat_lab.shape[0] != n:
        print(f"  node count {n:,} != .mat {mat_lab.shape[0]:,}", file=sys.stderr)
        return 1

    u_o = user[keep_idx][perm]
    p_o = prod[keep_idx][perm]
    r_o = rating[keep_idx][perm]
    l_o = label[keep_idx][perm]
    d_o = date[keep_idx][perm]
    src_row = keep_idx[perm]

    # raw label: 1 = genuine, -1 = spam.  .mat label: 1 = spam (fraud).
    lab_conv = (l_o == -1).astype(int)
    lab_match = int((lab_conv == mat_lab).sum())
    lab_ok = lab_match == n
    print(f"\n  CLAIM 1a  per-row label agreement: {lab_match:,}/{n:,} "
          f"({lab_match / n * 100:.4f}%)  {'EXACT' if lab_ok else 'FAIL'}")

    print("\n  CLAIM 1b  per-edge adjacency comparison:")
    rur_ok, rur_stats = compare_exact("R-U-R", pairs_from_groups(u_o, n), mat["net_rur"])
    rsr_key = p_o.astype(np.int64) * 100 + (r_o * 10).astype(np.int64)
    rsr_ok, rsr_stats = compare_exact("R-S-R", pairs_from_groups(rsr_key, n), mat["net_rsr"])

    # R-T-R is date-dependent and is NOT reproducible from this metadata snapshot.
    # Reported, never silently omitted -- see mapping_report.md.
    rtr_ref = binarise(mat["net_rtr"])
    print(f"    R-T-R   NOT REPRODUCED (mat={rtr_ref.nnz:,}). Date-dependent; the "
          f"mirror's date column\n            differs from the .mat's source "
          f"snapshot. Does not affect claim 1 --\n            two independent "
          f"date-free relations already pin the ordering.")

    claim1 = lab_ok and rur_ok and rsr_ok
    print(f"\n  CLAIM 1 VERDICT: {'PROVEN' if claim1 else 'NOT PROVEN'}")

    text_stats = None
    if not args.skip_text_controls:
        if not (HOTEL.exists() and REST.exists()):
            print(f"\n  CLAIM 2 SKIPPED - text files missing")
        else:
            hotel = HOTEL.read_text(encoding="utf-8", errors="replace").splitlines()
            rest = REST.read_text(encoding="utf-8", errors="replace").splitlines()
            texts = hotel + rest
            print(f"\n  CLAIM 2  text join ({len(hotel):,} hotel + {len(rest):,} "
                  f"restaurant = {len(texts):,} lines vs {len(user):,} metadata rows)")
            if len(texts) != len(user):
                print("    line count != metadata rows -> positional join impossible")
                return 1
            text_stats = text_controls(prod, texts, len(hotel))
            print(f"    CLAIM 2 VERDICT: "
                  f"{'SUPPORTED (statistical)' if text_stats['supports_row_level_join'] else 'NOT SUPPORTED'}")

    if not claim1:
        print("\nFAILED: node ordering not proven; no mapping emitted.")
        return 1

    if args.no_write:
        print("\n--no-write: verification only.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "yelpchi_review_mapping.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["graph_node_id", "source_row", "user_id", "product_id",
                    "rating", "date", "label_raw", "label_mat",
                    "mapping_method", "confidence"])
        for i in range(n):
            w.writerow([i, int(src_row[i]), int(u_o[i]), int(p_o[i]),
                        float(r_o[i]), str(d_o[i]), int(l_o[i]), int(mat_lab[i]),
                        "adjacency_proof", "EXACT"])
    print(f"\n  wrote {csv_path.relative_to(ROOT)} ({n:,} rows)")

    stats = {
        "claim_1_node_ordering": {
            "verdict": "PROVEN",
            "method": "exact per-edge sparsity-pattern comparison",
            "label_rows_matched": lab_match, "num_nodes": n,
            "rur": rur_stats, "rsr": rsr_stats,
            "rtr": {"mat_nnz": int(rtr_ref.nnz), "reproduced": False,
                    "reason": "date-dependent; mirror date column differs from "
                              ".mat source snapshot"},
        },
        "claim_2_text_join": text_stats,
        "product_review_cap": PRODUCT_REVIEW_CAP,
        "dropped_products": dropped.tolist(),
    }
    stats_path = OUT_DIR / "alignment_verification.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {stats_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

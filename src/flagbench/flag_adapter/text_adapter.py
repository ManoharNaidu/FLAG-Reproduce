"""Select which text view feeds FLAG, without mutating the canonical store.

The dataset keeps the lossless text. An experiment may need a smaller view for
prompt budgets (brief sections 26, 50); that choice is made here and recorded,
never applied silently during dataset construction.
"""
from __future__ import annotations

from flagbench.fraud_text.base import UNRESOLVED, VERIFIED, FLAGDataset


def texts_for_flag(ds: FLAGDataset, max_chars: int | None = None
                   ) -> tuple[list[str], dict]:
    """Return (texts, provenance_record).

    A node whose text is UNRESOLVED keeps an empty string. That is deliberate:
    an empty string is what an honest "we do not know" looks like, and the
    accompanying record reports exactly how many there are so it cannot be
    mistaken for coverage.
    """
    texts = list(ds.raw_texts)
    truncated = 0
    if max_chars is not None:
        out = []
        for t in texts:
            if len(t) > max_chars:
                truncated += 1
                t = t[:max_chars]
            out.append(t)
        texts = out

    cov = ds.text_coverage()
    return texts, {
        "max_chars": max_chars,
        "truncated_nodes": truncated,
        "verified_nodes": cov["verified"],
        "unresolved_nodes": cov["unresolved"],
        "verified_fraction": cov["verified_fraction"],
        "status_values": sorted({VERIFIED, UNRESOLVED} & set(ds.text_status)),
    }

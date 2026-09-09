"""Encode benchmark node text with the frozen LM (Sentence-BERT).

    python -m scripts.preprocess.encode_text --dataset all
    python -m scripts.preprocess.encode_text --dataset reddit --device cuda:0

Writes cache/embeddings/<dataset>__<model>__<textkind>.pt

This is B(.) in the paper's Eq. 3 -- the frozen LM whose embeddings drive both
semantic similarity sampling and the `+text` variant's node features.

Model: `all-MiniLM-L6-v2` (384-d). The paper says only "Sentence-BERT"; the
official code pins this exact checkpoint, corroborated by DualGNN hardcoding
`Linear(384, ...)`. See research/paper_notes.md section 3.1.

Runs fine on CPU -- MiniLM is ~90 MB. This stage is NOT part of the GPU-only
boundary (decision D-003); only the 9B LLM stage is.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("encode")


def cache_path(dataset: str, model: str, text_kind: str) -> pathlib.Path:
    safe_model = model.replace("/", "_")
    return ROOT / "cache" / "embeddings" / f"{dataset}__{safe_model}__{text_kind}.pt"


def load_texts(dataset: str, source: str):
    """Texts for either the constructed benchmark or the original raw graph.

    The `original` source exists to test whether a paper claim measured on the
    full graph survives our 1:10 downsampling -- see
    research/figure3a_reproduction.md.
    """
    if source == "benchmark":
        path = ROOT / "data" / "benchmark" / f"flag_{dataset}" / "graph.pt"
        if not path.exists():
            return None, path
        return torch.load(path, map_location="cpu")["raw_texts"], path

    from flagbench.datasets import glbench

    sig = glbench.SIGNATURES[dataset]
    path = ROOT / "data" / "raw" / dataset / sig.filename
    if not path.exists():
        return None, path
    return glbench.load_raw(path).raw_texts, path


def texts_fingerprint(texts: list[str]) -> str:
    """Hash the exact text corpus, so a cache entry can never be reused across a
    different benchmark build (a different downsampling seed changes the node
    set, and therefore the texts)."""
    digest = hashlib.sha256()
    digest.update(str(len(texts)).encode())
    for t in texts:
        digest.update(t.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def encode(dataset: str, args) -> dict | None:
    banner = "=" * 74
    print(f"\n{banner}\n{dataset.upper()} [{args.source}]\n{banner}")
    texts, graph_path = load_texts(dataset, args.source)
    if texts is None:
        print(f"  MISSING {graph_path}")
        return None
    fingerprint = texts_fingerprint(texts)
    text_kind = "raw" if args.source == "benchmark" else "raw_original"
    out = cache_path(dataset, args.model, text_kind)

    if out.exists() and not args.force:
        meta_path = out.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("texts_sha256") == fingerprint:
                print(f"  cache hit: {out.relative_to(ROOT)}")
                print(f"    {meta['num_texts']:,} x {meta['dim']}d, "
                      f"encoded {meta['encoded_at']}")
                return meta
            print("  cache present but the text fingerprint changed "
                  "(different benchmark build); re-encoding")

    from flagbench.utils.device import resolve_device

    device_info = resolve_device(args.device)
    print(f"  device: {device_info}")
    print(f"  model:  {args.model}")
    print(f"  texts:  {len(texts):,}")

    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"  NOTE: {empty:,} texts are empty; their embeddings will be "
              f"near-zero and they sit exactly at the paper's threshold of 0")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.model, device=str(device_info.device))
    start = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # sampling normalises explicitly
    )
    elapsed = time.time() - start
    embeddings = torch.from_numpy(embeddings).float()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out)

    meta = {
        "dataset": dataset,
        "text_kind": text_kind,
        "model": args.model,
        "dim": int(embeddings.shape[1]),
        "num_texts": len(texts),
        "empty_texts": empty,
        "texts_sha256": fingerprint,
        "source_graph": str(graph_path.relative_to(ROOT)).replace("\\", "/"),
        "device": str(device_info.device),
        "batch_size": args.batch_size,
        "normalize_at_encode": False,
        "seconds": round(elapsed, 1),
        "encoded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
        "provenance": (
            "frozen LM B(.) of FLAG Eq. 3. Checkpoint taken from the official "
            "code (encode.py), not from the paper, which says only "
            "'Sentence-BERT'."
        ),
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n  encoded {len(texts):,} texts -> {tuple(embeddings.shape)} "
          f"in {elapsed:.1f}s ({len(texts) / elapsed:.0f} texts/s)")
    print(f"  norms: min {embeddings.norm(dim=1).min():.3f} "
          f"max {embeddings.norm(dim=1).max():.3f}")
    print(f"  wrote {out.relative_to(ROOT)}")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all",
                        choices=["reddit", "instagram", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | cuda:N | auto (default: FLAG_DEVICE or cpu)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source", default="benchmark", choices=["benchmark", "original"],
        help="benchmark = the 1:10 graph; original = the raw GLBench graph",
    )
    args = parser.parse_args(argv)

    datasets = ["instagram", "reddit"] if args.dataset == "all" else [args.dataset]
    results = {d: encode(d, args) for d in datasets}

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    ok = sum(1 for m in results.values() if m)
    for name, meta in results.items():
        if meta:
            print(f"  {name:12s} {meta['num_texts']:>7,} x {meta['dim']}d  "
                  f"({meta['empty_texts']:,} empty)")
        else:
            print(f"  {name:12s} FAILED")
    print(f"\n  {ok}/{len(results)} encoded")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

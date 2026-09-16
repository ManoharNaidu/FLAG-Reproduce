"""Encode the GPU-generated LLM text (decision D-003) into Sentence-BERT vectors.

    python -m scripts.preprocess.encode_llm_text --dataset reddit --kind discriminative
    python -m scripts.preprocess.encode_llm_text --dataset all --kind both

Reads cache/llm/<cache_key>.json (written by `scripts.llm.generate_text`, which
must be run first, on a GPU). Writes
cache/embeddings/<cache_key>__all-MiniLM-L6-v2.pt : dict[int, Tensor[k, 384]]
one entry per subgraph that has text, keyed by the subgraph's central node id,
`k` nodes in `subgraph.subset` order -- exactly aligned with how
`load_subgraphs` returns subgraphs, so a feature_fn can index straight off
`subgraph.subset`'s length with no re-alignment.

This is the missing half of the D-003 boundary: the GPU stage produces text,
this stage (CPU, cheap, the same encoder as the `+text` variant) turns it into
the vectors the dual-branch model actually consumes. See
`flagbench.experiments.runner.load_llm_embeddings`.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("encode_llm")


def _llm_cache_key(dataset: str, kind: str, args) -> tuple[str, pathlib.Path]:
    from flagbench.llm.enhance import LLMConfig, PromptSet, cache_key
    from flagbench.sampling.semantic import SamplingConfig
    from flagbench.registry.registry import get_variant

    strategy = get_variant("flag").default_sampling_strategy
    sampling = SamplingConfig(
        hops=args.hops, top_k=args.top_k,
        similarity_threshold=args.threshold, strategy=strategy,
    )
    prompts = PromptSet.load(dataset)
    config = LLMConfig(
        model_id=args.llm_model,
        max_new_tokens=args.max_new_tokens, truncate_chars=args.truncate_chars,
    )
    key = cache_key(dataset, sampling.cache_key(), prompts, config, kind)
    return key, ROOT / "cache" / "llm" / f"{key}.json"


def encode(dataset: str, kind: str, args) -> dict | None:
    banner = "=" * 74
    print(f"\n{banner}\n{dataset.upper()} [{kind}]\n{banner}")

    llm_key, llm_path = _llm_cache_key(dataset, kind, args)
    if not llm_path.exists():
        print(f"  MISSING {llm_path}")
        print("  Run on a GPU instance first:")
        print(f"    python -m scripts.llm.generate_text --dataset {dataset} "
              f"--kind {kind}")
        return None

    manifest = json.loads(llm_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    texts_by_central: dict[str, list[str]] = json.loads(llm_path.read_text(encoding="utf-8"))

    out = ROOT / "cache" / "embeddings" / f"{llm_key}__{args.model}.pt"
    if out.exists() and not args.force:
        meta_path = out.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"  cache hit: {out.relative_to(ROOT)}")
            print(f"    {meta['num_subgraphs']:,} subgraphs encoded "
                  f"{meta['encoded_at']}")
            return meta

    from flagbench.utils.device import resolve_device

    device_info = resolve_device(args.device)
    print(f"  device:       {device_info}")
    print(f"  llm cache:    {llm_path.relative_to(ROOT)}")
    print(f"  llm coverage: {manifest['stats']['node_coverage'] * 100:.1f}% of nodes, "
          f"{manifest['stats']['subgraph_success_rate'] * 100:.1f}% of subgraphs")
    print(f"  subgraphs:    {len(texts_by_central):,}")

    if not texts_by_central:
        print("  nothing to encode (zero subgraphs succeeded); writing an empty cache")

    centrals = sorted(texts_by_central, key=int)
    flat_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for central in centrals:
        lines = texts_by_central[central]
        start = len(flat_texts)
        flat_texts.extend(lines)
        spans.append((start, len(flat_texts)))

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.model, device=str(device_info.device))
    start_time = time.time()
    if flat_texts:
        flat_embeddings = encoder.encode(
            flat_texts, batch_size=args.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        flat_embeddings = torch.from_numpy(flat_embeddings).float()
    else:
        dim = encoder.get_sentence_embedding_dimension()
        flat_embeddings = torch.empty((0, dim), dtype=torch.float32)
    elapsed = time.time() - start_time

    embeddings: dict[int, torch.Tensor] = {
        int(central): flat_embeddings[start:end]
        for central, (start, end) in zip(centrals, spans)
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out)

    meta = {
        "dataset": dataset,
        "kind": kind,
        "llm_cache_key": llm_key,
        "llm_cache_sha256": manifest.get("output_sha256", ""),
        "llm_coverage": manifest["stats"],
        "model": args.model,
        "dim": int(flat_embeddings.shape[1]) if flat_embeddings.numel() else 0,
        "num_subgraphs": len(embeddings),
        "device": str(device_info.device),
        "batch_size": args.batch_size,
        "seconds": round(elapsed, 1),
        "encoded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\n  encoded {len(flat_texts):,} lines across {len(embeddings):,} "
          f"subgraphs in {elapsed:.1f}s")
    print(f"  wrote {out.relative_to(ROOT)}")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=["reddit", "instagram", "all"])
    parser.add_argument("--kind", default="discriminative",
                        choices=["discriminative", "residual", "both"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-model", default="google/gemma-2-9b-it",
                        help="must match the model used by scripts.llm.generate_text")
    from flagbench.experiments.runner import PRODUCTION_LLM_CONFIG
    parser.add_argument("--max-new-tokens", type=int,
                        default=PRODUCTION_LLM_CONFIG["max_new_tokens"],
                        help="must match the generate_text run being encoded "
                             "(decision D-004 default: the production cache's budget)")
    parser.add_argument("--truncate-chars", type=int,
                        default=PRODUCTION_LLM_CONFIG["truncate_chars"],
                        help="must match the generate_text run being encoded")
    parser.add_argument("--device", default=None,
                        help="cpu | cuda | cuda:N | auto (default: FLAG_DEVICE or cpu)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    datasets = ["instagram", "reddit"] if args.dataset == "all" else [args.dataset]
    kinds = ["discriminative", "residual"] if args.kind == "both" else [args.kind]

    results = {}
    for dataset in datasets:
        for kind in kinds:
            results[(dataset, kind)] = encode(dataset, kind, args)

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    ok = sum(1 for m in results.values() if m)
    for (dataset, kind), meta in results.items():
        if meta:
            print(f"  {dataset:10s} {kind:14s} {meta['num_subgraphs']:>6,} subgraphs")
        else:
            print(f"  {dataset:10s} {kind:14s} FAILED")
    print(f"\n  {ok}/{len(results)} encoded")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

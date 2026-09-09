"""GPU-only stage: generate FLAG's discriminative / residual text.

Run this on a rented GPU instance (decision D-003), then copy the emitted cache
back and replay every downstream experiment locally on CPU.

    # on the GPU box
    python -m scripts.llm.generate_text --dataset reddit --kind discriminative
    python -m scripts.llm.generate_text --dataset reddit --kind both

    # inspect what WOULD be sent, on CPU, without loading 18.5 GB of weights
    python -m scripts.llm.generate_text --dataset reddit --dry-run

Writes cache/llm/<cache_key>.json  (+ .manifest.json)

The cache key covers dataset, sampling config, prompt hashes, model id and
decoding parameters, so a cache hit can never silently cross a prompt or model
change. See docs/vastai_gpu_workflow.md for the full remote workflow.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import pathlib
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

from flagbench.experiments.runner import load_benchmark, load_subgraphs  # noqa: E402
from flagbench.llm.enhance import (  # noqa: E402
    LLMConfig,
    LLMEnhancer,
    PromptSet,
    build_prompt,
    cache_key,
)
from flagbench.sampling.semantic import SamplingConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("llm")

# Upstream uses dataset-specific nouns in the question block:
#   chat.py  -> "The posts of these users are as follows:"
#   chat1.py -> "The introductions of these users are as follows:"
DATASET_NOUN = {"reddit": "posts", "instagram": "introductions"}


def run(dataset: str, kind: str, args) -> dict | None:
    payload = load_benchmark(dataset)
    raw_texts = payload["raw_texts"]

    sampling = SamplingConfig(
        hops=args.hops, top_k=args.top_k,
        similarity_threshold=args.threshold, strategy="semantic",
    )
    subgraphs = load_subgraphs(dataset, sampling)
    if args.limit:
        subgraphs = subgraphs[: args.limit]

    prompts = PromptSet.load(dataset)
    config = LLMConfig(
        model_id=args.model, max_new_tokens=args.max_new_tokens,
        truncate_chars=args.truncate_chars, dtype=args.dtype,
    )
    noun = DATASET_NOUN.get(dataset, "posts")
    key = cache_key(dataset, sampling.cache_key(), prompts, config, kind)
    out = ROOT / "cache" / "llm" / f"{key}.json"

    print(f"\n{'=' * 76}")
    print(f"{dataset.upper()}  kind={kind}")
    print(f"{'=' * 76}")
    print(f"  model        {config.model_id} ({config.dtype})")
    print(f"  sampling     {sampling.cache_key()}")
    print(f"  prompts      {prompts.version}")
    for role, digest in prompts.hashes.items():
        print(f"                 {role:20s} {digest[:16]}")
    print(f"  subgraphs    {len(subgraphs):,}")
    print(f"  cache key    {key}")

    if out.exists() and not args.force:
        meta = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        print(f"  CACHE HIT -> {out.relative_to(ROOT)}")
        print(f"    coverage {meta['stats']['node_coverage'] * 100:.1f}% of nodes")
        return meta

    if args.dry_run:
        sample = subgraphs[0]
        node_ids = sample.subset.tolist()
        prompt = build_prompt(
            prompts, kind, [raw_texts[i] for i in node_ids], config, noun
        )
        print(f"\n  DRY RUN -- example prompt for subgraph centred on node "
              f"{sample.central} ({len(node_ids)} nodes)\n")
        print("  " + "-" * 72)
        for line in prompt.splitlines():
            print(f"  | {line[:110]}")
        print("  " + "-" * 72)
        print(f"\n  prompt characters: {len(prompt):,}")
        print(f"  would generate {len(subgraphs):,} responses at "
              f"max_new_tokens={config.max_new_tokens}")
        print("  no model was loaded; nothing was written")
        return None

    enhancer = LLMEnhancer(config, device=args.device)
    results, stats = enhancer.enhance(
        subgraphs, raw_texts, prompts, kind, noun=noun
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    serialised = {str(k): v for k, v in results.items()}
    out.write_text(
        json.dumps(serialised, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "cache_key": key,
        "dataset": dataset,
        "kind": kind,
        "llm_config": config.as_record(),
        "sampling_config": sampling.as_record(),
        "prompt_version": prompts.version,
        "prompt_hashes": prompts.hashes,
        "dataset_version": payload.get("_manifest", {}).get(
            "preprocessing_version", ""
        ),
        "dataset_sha256": payload.get("_manifest", {}).get("output_sha256", ""),
        "stats": stats.as_record(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "device": args.device,
        "output": str(out.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "deviations": [
            "greedy decoding is pinned explicitly; upstream calls generate() "
            "with model defaults, which would not be reproducible",
            "list numbering is stripped by pattern; upstream uses fixed slices "
            "that differ between train (2 chars) and val/test (3 chars)",
        ],
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n  generated    {stats.succeeded:,}/{stats.total_subgraphs:,} "
          f"subgraphs in {stats.seconds:.0f}s")
    print(f"  format fails {stats.format_failures:,} "
          f"({stats.format_failures / max(stats.total_subgraphs, 1) * 100:.1f}%)")
    print(f"  node coverage {stats.as_record()['node_coverage'] * 100:.1f}%")
    print(f"  wrote {out.relative_to(ROOT)}")
    print(f"  sha256 {manifest['output_sha256'][:32]}...")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="reddit",
                        choices=["reddit", "instagram", "all"])
    parser.add_argument("--kind", default="discriminative",
                        choices=["discriminative", "residual", "both"])
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=550)
    parser.add_argument("--truncate-chars", type=int, default=1200)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N subgraphs (for a smoke run)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print an example prompt and exit; loads no model")
    args = parser.parse_args(argv)

    if not args.dry_run and not torch.cuda.is_available():
        print("ERROR: no CUDA device.")
        print("  This stage is GPU-only by decision D-003: gemma-2-9b-it is")
        print("  ~18.5 GB in fp16 and no smaller substitute is provided,")
        print("  because its numbers would not be comparable to the paper's.")
        print("  See docs/vastai_gpu_workflow.md.")
        print("  Use --dry-run to inspect prompts on CPU.")
        return 2

    datasets = ["reddit", "instagram"] if args.dataset == "all" else [args.dataset]
    kinds = (
        ["discriminative", "residual"] if args.kind == "both" else [args.kind]
    )

    for dataset in datasets:
        for kind in kinds:
            try:
                run(dataset, kind, args)
            except FileNotFoundError as exc:
                print(f"\n  MISSING INPUT: {exc}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

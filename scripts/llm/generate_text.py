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
import dataclasses
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
    GenerationStats,
    LLMConfig,
    PromptSet,
    build_prompt,
    cache_key,
    make_enhancer,
)
from flagbench.sampling.cli import STRATEGIES, add_md_args, config_from_args  # noqa: E402
from flagbench.sampling.semantic import SamplingConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("llm")

# Upstream uses dataset-specific nouns in the question block:
#   chat.py  -> "The posts of these users are as follows:"
#   chat1.py -> "The introductions of these users are as follows:"
DATASET_NOUN = {"reddit": "posts", "instagram": "introductions",
                "amazon_text": "reviews", "yelpchi_text": "reviews"}


def combine_stats(subgraphs, results: dict, failed: set, seconds: float) -> GenerationStats:
    """Corpus-level stats over ALL subgraphs (reused + regenerated)."""
    stats = GenerationStats(total_subgraphs=len(subgraphs), seconds=seconds)
    for sg in subgraphs:
        stats.total_nodes += sg.num_nodes
        if sg.central in results:
            stats.succeeded += 1
            stats.nodes_with_text += sg.num_nodes
        else:
            stats.format_failures += 1
    return stats


def reuse_split(dataset, kind, subgraphs, sampling, prompts, config):
    """Split `subgraphs` into (reusable text, known failures, still-to-generate).

    Gemma's input is fully determined by (prompt, ordered node texts, decoding
    config), and decoding is greedy at batch size 1. So if an MD subgraph has
    EXACTLY the same `subset` (same nodes, same order) as the cosine subgraph for
    the same centre, its prompt is byte-identical and the text the cosine run
    produced -- or its format failure, which is not stored but is by
    construction every attempted subgraph missing from the cache -- is what
    generation would return again. Only subgraphs whose node list differs are
    regenerated. Nothing else is reused.
    """
    import dataclasses as _dc

    cosine = _dc.replace(sampling, strategy="semantic")
    cos_key = cache_key(dataset, cosine.cache_key(), prompts, config, kind)
    cos_path = ROOT / "cache" / "llm" / f"{cos_key}.json"
    if not cos_path.exists():
        raise FileNotFoundError(f"cosine LLM cache {cos_path} missing; cannot reuse")
    manifest = json.loads(cos_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["stats"]["total_subgraphs"] != len(load_subgraphs(dataset, cosine)):
        raise RuntimeError("cosine LLM cache does not cover all subgraphs; refusing to reuse")
    cos_texts = json.loads(cos_path.read_text(encoding="utf-8"))
    cos_by_center = {sg.central: sg for sg in load_subgraphs(dataset, cosine)}

    reuse, failed, todo = {}, set(), []
    for sg in subgraphs:
        ref = cos_by_center.get(sg.central)
        if ref is not None and torch.equal(ref.subset, sg.subset):
            if str(sg.central) in cos_texts:
                reuse[sg.central] = cos_texts[str(sg.central)]
            else:
                failed.add(sg.central)
        else:
            todo.append(sg)
    meta = {
        "source_cache_key": cos_key,
        "source_output_sha256": manifest.get("output_sha256"),
        "reused_texts": len(reuse),
        "reused_known_failures": len(failed),
        "regenerated_subgraphs": len(todo),
        "rule": "reused iff subset (node ids AND order) is identical to the cosine subgraph "
                "for the same centre; prompt and greedy decoding are then identical",
    }
    return reuse, failed, todo, meta


def merge_shards(out, key, dataset, kind, args, subgraphs, sampling, prompts, config,
                 payload, reuse, reuse_failed, reuse_meta):
    parts = sorted(out.parent.glob(f"{out.stem}.part*of{args.num_shards}.json"))
    if len(parts) != args.num_shards:
        raise RuntimeError(f"expected {args.num_shards} shard files, found {len(parts)}")
    results = dict(reuse)
    seconds = 0.0
    overflows = 0
    for part in parts:
        blob = json.loads(part.read_text(encoding="utf-8"))
        results.update({int(k): v for k, v in blob["results"].items()})
        seconds = max(seconds, blob["stats"]["seconds"])       # shards ran in parallel
        overflows += blob["stats"].get("context_overflows", 0)
    stats = combine_stats(subgraphs, results, reuse_failed, seconds)
    stats.context_overflows = overflows
    return write_cache(out, key, dataset, kind, args, sampling, prompts, config,
                       payload, results, stats, reuse_meta)


def write_cache(out, key, dataset, kind, args, sampling, prompts, config, payload,
                results, stats, reuse_meta):
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
        "reuse_from_cosine": reuse_meta or None,
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


def run(dataset: str, kind: str, args) -> dict | None:
    payload = load_benchmark(dataset)
    raw_texts = payload["raw_texts"]

    sampling = config_from_args(args, dataset=dataset)
    subgraphs = load_subgraphs(dataset, sampling)
    if args.limit:
        subgraphs = subgraphs[: args.limit]

    prompts = PromptSet.load(dataset)
    config = LLMConfig(
        model_id=args.model, max_new_tokens=args.max_new_tokens,
        truncate_chars=args.truncate_chars, dtype=args.dtype, engine=args.engine,
    )
    noun = DATASET_NOUN.get(dataset, "posts")
    key = cache_key(dataset, sampling.cache_key(), prompts, config, kind)
    out = ROOT / "cache" / "llm" / f"{key}.json"

    print(f"\n{'=' * 76}")
    print(f"{dataset.upper()}  kind={kind}")
    print(f"{'=' * 76}")
    print(f"  model        {config.model_id} ({config.dtype}, engine={config.engine})")
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

    # ---- exact reuse of the cosine run's text (FLAG-MD only; see reuse_split) ----
    reuse: dict[int, list[str]] = {}
    reuse_failed: set[int] = set()
    reuse_meta: dict = {}
    todo = subgraphs
    if args.reuse_from_cosine:
        reuse, reuse_failed, todo, reuse_meta = reuse_split(
            dataset, kind, subgraphs, sampling, prompts, config
        )
        print(f"  reuse        {len(reuse):,} texts + {len(reuse_failed):,} known format "
              f"failures reused from {reuse_meta['source_cache_key']}")
        print(f"  regenerate   {len(todo):,} subgraphs (node list differs from cosine's)")

    if args.merge:
        return merge_shards(out, key, dataset, kind, args, subgraphs, sampling, prompts,
                            config, payload, reuse, reuse_failed, reuse_meta)

    if args.num_shards > 1:
        todo = todo[args.shard::args.num_shards]
        print(f"  shard        {args.shard}/{args.num_shards}: {len(todo):,} subgraphs")
        part_path = out.with_name(f"{out.stem}.part{args.shard}of{args.num_shards}.json")
        if part_path.exists() and not args.force:
            print(f"  shard already done -> {part_path.relative_to(ROOT)} (resume: skipping)")
            return {"shard": args.shard, "skipped": True}

    enhancer = make_enhancer(config, device=args.device)
    results, stats = enhancer.enhance(
        todo, raw_texts, prompts, kind, noun=noun
    )

    if args.num_shards > 1:
        part = out.with_name(f"{out.stem}.part{args.shard}of{args.num_shards}.json")
        part.write_text(json.dumps({
            "results": {str(k): v for k, v in results.items()},
            "stats": dataclasses.asdict(stats),
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n  wrote shard {part.relative_to(ROOT)}  "
              f"({stats.succeeded:,}/{stats.total_subgraphs:,} ok, {stats.seconds:.0f}s)")
        return {"shard": args.shard}

    results = {**reuse, **results}
    overflows = stats.context_overflows
    stats = combine_stats(subgraphs, results, reuse_failed, stats.seconds)
    stats.context_overflows = overflows

    return write_cache(out, key, dataset, kind, args, sampling, prompts, config,
                       payload, results, stats, reuse_meta)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="reddit",
                        help="a dataset with prompts/<name>/, a comma-separated list, "
                             "or 'all' (= reddit,instagram)")
    parser.add_argument("--kind", default="discriminative",
                        choices=["discriminative", "residual", "both"])
    parser.add_argument("--model", default="google/gemma-2-9b-it")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                        help="hf = upstream's one-prompt transformers loop; vllm = batched "
                             "(enters the cache key; device via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--max-new-tokens", type=int, default=550)
    parser.add_argument("--truncate-chars", type=int, default=1200)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=None,
                        help="per-hop budget (default: the dataset's registry default, 10 or 3)")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--strategy", default="semantic", choices=STRATEGIES,
                        help="which sampler's subgraphs to enhance (default: FLAG's cosine)")
    add_md_args(parser)
    parser.add_argument("--reuse-from-cosine", action="store_true",
                        help="FLAG-MD: reuse the cosine run's text for every subgraph whose "
                             "node list is identical; regenerate only the rest")
    parser.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the subgraphs to generate across N workers, then --merge")
    parser.add_argument("--merge", action="store_true",
                        help="combine the N shard files (and reused text) into the final cache")
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

    datasets = (["reddit", "instagram"] if args.dataset == "all"
                else [d for d in args.dataset.split(",") if d])
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

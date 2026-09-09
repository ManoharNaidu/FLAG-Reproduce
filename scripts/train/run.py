"""Run one or more experiments.

    python -m scripts.train.run --dataset reddit --model gcn --variant baseline
    python -m scripts.train.run --dataset reddit --model gcn --variant text --seeds 5
    python -m scripts.train.run --dataset all --models gcn,gat --variants baseline,text

Writes one JSON per run to results/raw/ and refreshes results/aggregated/.

Refuses impossible experiments rather than substituting something that runs --
e.g. a FLAG variant on a dataset with no native text (Phase 9).
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

from flagbench.experiments import results as results_mod  # noqa: E402
from flagbench.experiments.runner import (  # noqa: E402
    ExperimentNotAvailable,
    run_single,
)
from flagbench.registry.registry import (  # noqa: E402
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    VARIANT_REGISTRY,
    validate,
)
from flagbench.sampling.semantic import SamplingConfig  # noqa: E402
from flagbench.training.trainer import TrainConfig  # noqa: E402


def parse_list(value: str, registry) -> list[str]:
    if value == "all":
        return sorted(registry)
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [i for i in items if i not in registry]
    if unknown:
        raise SystemExit(
            f"unknown: {unknown}. Registered: {sorted(registry)}"
        )
    return items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--datasets", dest="datasets", default="reddit")
    parser.add_argument("--model", "--models", dest="models", default="gcn")
    parser.add_argument("--variant", "--variants", dest="variants", default="baseline")
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of data seeds (paper uses 5)")
    parser.add_argument("--inits", type=int, default=1,
                        help="initialisations per seed (paper uses 5 -> 25 runs)")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--accumulation-steps", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--threshold-policy", default="validation_swept",
                        choices=["validation_swept", "argmax", "fixed"])
    parser.add_argument("--class-weighted-loss", action="store_true")

    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--sampling-strategy", default=None,
        choices=["semantic", "semantic_nothreshold", "random", "none", "feature"],
        help="override the per-variant default (baseline/+text use 'none', "
             "FLAG variants use 'semantic')",
    )

    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the matrix and exit without training")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    datasets = parse_list(args.datasets, DATASET_REGISTRY)
    models = parse_list(args.models, MODEL_REGISTRY)
    variants = parse_list(args.variants, VARIANT_REGISTRY)

    train_config = TrainConfig(
        epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
        dropout=args.dropout, weight_decay=args.weight_decay,
        accumulation_steps=args.accumulation_steps, patience=args.patience,
        early_stopping=not args.no_early_stopping,
        threshold_policy=args.threshold_policy,
        class_weighted_loss=args.class_weighted_loss,
    )
    # None lets each variant pick its own default (NS for baseline/+text,
    # SS for the FLAG variants). --sampling-strategy forces one for all.
    sampling_config = None
    if args.sampling_strategy:
        sampling_config = SamplingConfig(
            hops=args.hops, top_k=args.top_k,
            similarity_threshold=args.threshold,
            strategy=args.sampling_strategy,
        )

    combos = [
        (d, m, v, s, i)
        for d in datasets for m in models for v in variants
        for s in range(args.seeds) for i in range(args.inits)
    ]

    print(f"{'=' * 78}")
    print(f"planned: {len(datasets)} dataset(s) x {len(models)} model(s) x "
          f"{len(variants)} variant(s) x {args.seeds} seed(s) x {args.inits} "
          f"init(s) = {len(combos)} runs")
    sampling_desc = (
        sampling_config.cache_key() if sampling_config
        else "per-variant default (baseline/+text=none, flag=semantic)"
    )
    print(f"device: {args.device}   sampling: {sampling_desc}")
    print(f"{'=' * 78}\n")

    # Validate the whole matrix up front so nothing starts training only to be
    # refused halfway through.
    skipped = []
    runnable = []
    for combo in combos:
        d, m, v, _, _ = combo
        verdict = validate(d, m, v, args.device, train_config.hidden_dim)
        (runnable if verdict else skipped).append((combo, verdict))

    if skipped:
        print(f"REFUSED {len(skipped)} combination(s):\n")
        seen = set()
        for (d, m, v, _, _), verdict in skipped:
            if (d, m, v) in seen:
                continue
            seen.add((d, m, v))
            print(f"  {d}/{m}/{v}")
            for line in verdict.report().splitlines():
                print(f"    {line}")
            print()

    if args.dry_run:
        print(f"dry run: {len(runnable)} runnable, {len(skipped)} refused")
        return 0
    if not runnable:
        print("nothing to run.")
        return 1

    completed, failed = [], []
    for index, ((d, m, v, s, i), _) in enumerate(runnable, 1):
        print(f"[{index}/{len(runnable)}] {d}/{m}/{v} seed={s} init={i} ... ",
              end="", flush=True)
        try:
            result = run_single(
                dataset=d, model=m, variant=v, seed=s, init=i,
                device=args.device, train_config=train_config,
                sampling_config=sampling_config,
                save_checkpoint=args.save_checkpoint,
            )
        except ExperimentNotAvailable as exc:
            print("REFUSED")
            print(f"    {exc}")
            continue
        except FileNotFoundError as exc:
            print("MISSING INPUT")
            print(f"    {exc}")
            return 1

        if result.status == "completed":
            completed.append(result)
            print(f"AUC {result.test_auc:.4f}  F1 {result.test_f1_macro:.4f}  "
                  f"({result.training_time:.1f}s)")
        else:
            failed.append(result)
            print(f"FAILED: {result.error_message[:100]}")

    print(f"\n{'=' * 78}")
    print(f"{len(completed)} completed, {len(failed)} failed, "
          f"{len(skipped)} refused")

    summary = results_mod.aggregate_to_files()
    print(f"aggregated {summary['num_runs']} run(s) -> {summary['csv']}")

    if completed:
        table = results_mod.build_comparison_table()
        print(f"\n{results_mod.render_markdown_table(table)}")
        dest = ROOT / "results" / "tables" / "comparison.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            results_mod.render_markdown_table(table) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {dest.relative_to(ROOT)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

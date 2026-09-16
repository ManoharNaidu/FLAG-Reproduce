"""Experiment runner: one (dataset, model, variant, seed, init) run.

Phase 11's fair-comparison rule is enforced structurally: the dataset, splits,
subgraphs, metrics, selection rule and seeding are identical across models and
variants. **The only thing a variant changes is which node features feed the
GNN**, supplied by `feature_fn`. If a future variant needs to change anything
else, that is a signal to re-read Phase 11, not to add a special case here.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time

import numpy as np
import torch

from flagbench.adapters.backbone import build_backbone
from flagbench.experiments.results import RunResult
from flagbench.metrics.classification import evaluate_val_and_test
from flagbench.registry.registry import (
    get_dataset,
    get_model,
    get_variant,
    validate,
)
from flagbench.sampling.semantic import SamplingConfig, Subgraph
from flagbench.training.trainer import FinetuneConfig, SubgraphTrainer, TrainConfig
from flagbench.utils.device import resolve_device
from flagbench.utils.seeding import RunIdentity, seed_everything

ROOT = pathlib.Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


class ExperimentNotAvailable(RuntimeError):
    """The requested experiment is not meaningful. Never silently substituted."""


def load_benchmark(dataset: str) -> dict:
    path = ROOT / "data" / "benchmark" / f"flag_{dataset}" / "graph.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:\n"
            f"  python -m scripts.preprocess.build_benchmark --dataset {dataset}"
        )
    payload = torch.load(path, map_location="cpu")
    manifest_path = path.parent / "dataset_manifest.json"
    payload["_manifest"] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    return payload


def load_subgraphs(dataset: str, config: SamplingConfig) -> list[Subgraph]:
    path = (
        ROOT / "cache" / "sampling" / f"{dataset}__{config.cache_key()}.pt"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:\n"
            f"  python -m scripts.preprocess.sample_subgraphs "
            f"--dataset {dataset} --hops {config.hops} --top-k {config.top_k} "
            f"--threshold {config.similarity_threshold}"
        )
    raw = torch.load(path, map_location="cpu")
    return [
        Subgraph(
            central=item["central"], subset=item["subset"],
            edge_index=item["edge_index"], hop=item["hop"],
        )
        for item in raw
    ]


def load_text_embeddings(dataset: str, model: str = "all-MiniLM-L6-v2"):
    safe = model.replace("/", "_")
    path = ROOT / "cache" / "embeddings" / f"{dataset}__{safe}__raw.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run:\n"
            f"  python -m scripts.preprocess.encode_text --dataset {dataset}"
        )
    return torch.load(path, map_location="cpu")


def make_feature_fn(variant_key: str, payload: dict, dataset: str):
    """Return `(feature_fn, in_dim, description)` for a variant.

    This is the single point of divergence between variants.
    """
    variant = get_variant(variant_key)

    if variant.feature_source == "stored":
        features = payload["x"]
        description = (
            f"stored graph features ({features.shape[1]}d). The paper calls "
            f"these 'shallow embeddings'; measured to be 4096-d and "
            f"Llama-2-derived - see research/dataset_notes.md section 7."
        )
    elif variant.feature_source == "lm_text":
        features = load_text_embeddings(dataset)
        description = f"Sentence-BERT of raw node text ({features.shape[1]}d)"
    else:
        raise ExperimentNotAvailable(
            f"variant {variant_key!r} is dual-branch (feature_source="
            f"{variant.feature_source!r}); use make_dual_feature_fn, not "
            f"make_feature_fn."
        )

    def feature_fn(subgraph: Subgraph) -> torch.Tensor:
        return features[subgraph.subset]

    return feature_fn, int(features.shape[1]), description


# Decision D-004: the production `cache/llm/` corpus was generated at this
# reduced decode budget, not `LLMConfig()`'s paper-faithful default
# (max_new_tokens=550, truncate_chars=1200) -- a faithful-budget full-corpus
# run was estimated at ~10-14 GPU-days and judged disproportionate. Coverage
# is correspondingly low (0.6-12.4% of nodes; see D-004 for the per-cache
# table) and every `flag`/`flag_finetuned` result must be read with that in
# mind. Regenerating faithfully only requires re-running
# `scripts.llm.generate_text --force` and updating this constant.
PRODUCTION_LLM_CONFIG = {"max_new_tokens": 64, "truncate_chars": 300}


def _llm_embeddings_path(dataset: str, kind: str, sbert_model: str = "all-MiniLM-L6-v2",
                          llm_model: str = "google/gemma-2-9b-it") -> tuple[str, pathlib.Path]:
    """Locate the Sentence-BERT cache for GPU-generated LLM text.

    Recomputes the exact cache key `scripts.llm.generate_text` /
    `scripts.preprocess.encode_llm_text` would use for the CURRENT prompts,
    sampling config and LLM decode settings (decision D-004), so this can
    never silently pick up a cache built from a different prompt/model/decode
    config. See decisions D-003 and D-004 in `research/decisions.md`.
    """
    from flagbench.llm.enhance import LLMConfig, PromptSet, cache_key as llm_cache_key
    from flagbench.sampling.semantic import SamplingConfig

    variant = get_variant("flag")
    sampling = SamplingConfig(strategy=variant.default_sampling_strategy)
    prompts = PromptSet.load(dataset)
    config = LLMConfig(model_id=llm_model, **PRODUCTION_LLM_CONFIG)
    key = llm_cache_key(dataset, sampling.cache_key(), prompts, config, kind)
    safe_sbert = sbert_model.replace("/", "_")
    return key, ROOT / "cache" / "embeddings" / f"{key}__{safe_sbert}.pt"


def load_llm_embeddings(dataset: str, kind: str) -> dict[int, torch.Tensor]:
    """`{central_node_id: Tensor[k, 384]}`, `k` in `subgraph.subset` order.

    One entry per subgraph whose LLM text passed the format check (decision
    D-003 / `flagbench.llm.enhance.parse_response`) -- a missing key means
    that subgraph's generation failed, not that the cache is incomplete.
    """
    key, path = _llm_embeddings_path(dataset, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run (on a GPU, decision D-003):\n"
            f"  python -m scripts.llm.generate_text --dataset {dataset} --kind {kind}\n"
            f"then:\n"
            f"  python -m scripts.preprocess.encode_llm_text --dataset {dataset} --kind {kind}\n"
            f"(expected cache key: {key})"
        )
    return torch.load(path, map_location="cpu")


def make_dual_feature_fn(variant_key: str, payload: dict, dataset: str):
    """Return `(feature_fn, extra_feature_fn, in_dim, description)` for a
    dual-branch variant (`flag`, `flag_finetuned`).

    `feature_fn(subgraph) -> (x_raw, x_disc)` feeds the attention-fused
    `DualBranchBackbone` used for both training and evaluation of both
    variants. Where a subgraph's discriminative text failed the format check,
    `x_disc` falls back to `x_raw` -- matching upstream `test_dual.py`'s
    `hasattr(batch, "unique_embeddings")` all-or-nothing fallback exactly.

    `extra_feature_fn` is only set for `flag_finetuned`: it returns
    `(x_disc, x_common) | None` for decision D-001's extra-epoch fine-tuning
    phase, `None` when either branch's text is unavailable for that subgraph
    (see `SubgraphTrainer.finetune_extra`).
    """
    variant = get_variant(variant_key)
    if not variant.dual_branch:
        raise ExperimentNotAvailable(
            f"variant {variant_key!r} is not dual-branch; use make_feature_fn."
        )

    raw = load_text_embeddings(dataset)
    disc = load_llm_embeddings(dataset, "discriminative")

    def feature_fn(subgraph: Subgraph):
        x_raw = raw[subgraph.subset]
        d = disc.get(int(subgraph.central))
        x_disc = d if d is not None and d.shape[0] == len(subgraph.subset) else x_raw
        return x_raw, x_disc

    extra_feature_fn = None
    if variant.requires_finetuned_llm:
        common = load_llm_embeddings(dataset, "residual")

        def extra_feature_fn(subgraph: Subgraph):
            d = disc.get(int(subgraph.central))
            c = common.get(int(subgraph.central))
            n = len(subgraph.subset)
            if d is None or c is None or d.shape[0] != n or c.shape[0] != n:
                return None
            return d, c

    description = (
        f"dual-branch: raw text ({raw.shape[1]}d) + LLM discriminative text "
        f"({raw.shape[1]}d), attention-fused (models.py:DualGNN)"
    )
    return feature_fn, extra_feature_fn, int(raw.shape[1]), description


def run_single(
    dataset: str,
    model: str,
    variant: str = "baseline",
    seed: int = 0,
    init: int = 0,
    device: str = "cpu",
    train_config: TrainConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    finetune_config: FinetuneConfig | None = None,
    save_checkpoint: bool = False,
    save_result: bool = True,
) -> RunResult:
    """Run one experiment. Records a `failed` row rather than raising."""
    train_config = train_config or TrainConfig()

    model_spec = get_model(model)
    variant_spec = get_variant(variant)
    dataset_spec = get_dataset(dataset)

    # Semantic sampling is a FLAG contribution, not part of the shared protocol,
    # so `baseline` and `+text` use plain 2-hop neighbourhoods. Handing SS to the
    # baseline would give it part of the method it is a baseline for.
    if sampling_config is None:
        sampling_config = SamplingConfig(
            strategy=variant_spec.default_sampling_strategy
        )
    elif sampling_config.strategy != variant_spec.default_sampling_strategy:
        logger.info(
            "sampling strategy %r overrides variant %r default %r",
            sampling_config.strategy, variant,
            variant_spec.default_sampling_strategy,
        )

    verdict = validate(
        dataset, model, variant, device, train_config.hidden_dim
    )
    if not verdict:
        raise ExperimentNotAvailable(verdict.report())

    device_info = resolve_device(device)
    identity = RunIdentity(
        seed=seed, init=init, dataset=dataset, model=model, variant=variant
    )

    # DeviceInfo reports more than RunResult declares (memory, capability);
    # the declared fields go in columns, the rest into `extra` so nothing is lost.
    device_record = device_info.as_record()
    declared = {k: device_record.pop(k) for k in ("device", "device_name")}

    result = RunResult.new(
        dataset=dataset, model=model, variant=variant,
        impl_source=model_spec.impl_source.value,
        model_fidelity=model_spec.fidelity,
        fidelity_class=(
            model_spec.fidelity_class.value if model_spec.fidelity_class else "unknown"
        ),
        seed=seed, initialization=init,
        framework=f"torch {torch.__version__}",
        llm_finetuned=False,
        **declared,
    )
    result.extra.update(device_record)
    result.hyperparameters = train_config.as_record()
    result.sampling_config = sampling_config.as_record()
    result.threshold_policy = train_config.threshold_policy

    try:
        payload = load_benchmark(dataset)
        manifest = payload.get("_manifest", {})
        result.dataset_version = manifest.get("preprocessing_version", "")
        result.dataset_manifest_sha256 = manifest.get("output_sha256", "")

        subgraphs = load_subgraphs(dataset, sampling_config)
        by_center = {sg.central: sg for sg in subgraphs}

        extra_feature_fn = None
        if variant_spec.dual_branch:
            feature_fn, extra_feature_fn, in_dim, feature_desc = make_dual_feature_fn(
                variant, payload, dataset
            )
            result.llm_model = "google/gemma-2-9b-it"
        else:
            feature_fn, in_dim, feature_desc = make_feature_fn(
                variant, payload, dataset
            )
        result.extra["feature_source"] = feature_desc

        def split_subgraphs(mask):
            ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            return [by_center[i] for i in ids if i in by_center]

        train_sg = split_subgraphs(payload["train_mask"])
        val_sg = split_subgraphs(payload["val_mask"])
        test_sg = split_subgraphs(payload["test_mask"])
        result.extra["split_sizes"] = {
            "train": len(train_sg), "val": len(val_sg), "test": len(test_sg)
        }

        # Data seed governs shuffling; init seed governs weight initialisation.
        seed_everything(identity.stream("data"))
        net = None
        from flagbench.utils.seeding import fork_rng

        with fork_rng(identity.stream("init")):
            net = build_backbone(
                model, in_dim, out_dim=2,
                hidden_dim=train_config.hidden_dim,
                dropout=train_config.dropout,
                device=device_info.device,
                dual_branch=variant_spec.dual_branch,
            )

        trainer = SubgraphTrainer(
            net, train_config, device_info.device, feature_fn, payload["y"],
            dual_branch=variant_spec.dual_branch,
        )
        outcome = trainer.fit(train_sg, val_sg, seed=identity.stream("batch_order"))

        if variant_spec.requires_finetuned_llm:
            # Decision D-001: `+FLAG*` = extra GNN epochs under the residual +
            # orthogonality losses, LLM frozen. `finetune_extra` only keeps
            # its result if it beats the plain `+FLAG` checkpoint above.
            ft_config = finetune_config or FinetuneConfig()
            ft_outcome, ft_stats = trainer.finetune_extra(
                train_sg, val_sg, extra_feature_fn, ft_config,
                seed=identity.stream("finetune"),
            )
            if ft_outcome.best_epoch > 0:
                outcome = ft_outcome
                result.hyperparameters.update(ft_config.as_record())
            result.extra["finetune"] = {
                **ft_stats,
                "applied": ft_outcome.best_epoch > 0,
                "note": (
                    "upstream LoRA gradient path is severed; see "
                    "research/decisions.md D-001"
                ),
            }

        val_scores, val_labels = trainer.predict(val_sg)
        infer_start = time.time()
        test_scores, test_labels = trainer.predict(test_sg)
        inference_time = time.time() - infer_start

        val_eval, test_eval, threshold_record = evaluate_val_and_test(
            val_labels, val_scores, test_labels, test_scores,
            policy=train_config.threshold_policy,
        )

        result.best_epoch = outcome.best_epoch
        result.epochs_run = outcome.epochs_run
        result.stopped_early = outcome.stopped_early
        result.training_time = round(outcome.training_seconds, 2)
        result.inference_time = round(inference_time, 2)
        result.threshold = threshold_record["threshold"]

        result.validation_auc = val_eval.auc
        result.validation_f1_macro = val_eval.threshold_metrics["f1_macro"]
        result.test_auc = test_eval.auc
        result.test_f1_macro = test_eval.threshold_metrics["f1_macro"]
        result.test_precision_fraud = test_eval.threshold_metrics["precision_fraud"]
        result.test_recall_fraud = test_eval.threshold_metrics["recall_fraud"]
        result.test_f1_fraud = test_eval.threshold_metrics["f1_fraud"]
        result.test_accuracy = test_eval.threshold_metrics["accuracy"]
        result.test_ks = test_eval.ks
        result.test_ece = test_eval.ece
        result.extra["epoch_history"] = [
            {"epoch": e.epoch, "train_loss": round(e.train_loss, 5),
             "val_auc": round(e.val_auc, 5),
             "val_f1_macro": round(e.val_f1_macro, 5)}
            for e in outcome.history
        ]
        result.extra["test_label_distribution"] = test_eval.label_distribution

        if save_checkpoint:
            path = (
                ROOT / "checkpoints" / model
                / f"{dataset}__{variant}__s{seed}i{init}.pt"
            )
            net.save_checkpoint(
                path,
                dataset=dataset, variant=variant, seed=seed, init=init,
                config=train_config.as_record(),
                git_commit=result.git_commit,
                dataset_version=result.dataset_version,
                best_epoch=outcome.best_epoch,
            )
            result.checkpoint_path = str(path.relative_to(ROOT)).replace("\\", "/")

    except ExperimentNotAvailable:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed run must still be recorded
        result.status = "failed"
        result.error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("run failed: %s", result.summary())

    if save_result:
        result.save()
    return result

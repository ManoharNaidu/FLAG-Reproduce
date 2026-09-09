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
from flagbench.training.trainer import SubgraphTrainer, TrainConfig
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
            f"variant {variant_key!r} needs LLM-generated discriminative text, "
            f"which requires a GPU run of gemma-2-9b-it (decision D-003). "
            f"Not available on this machine; nothing is substituted."
        )

    def feature_fn(subgraph: Subgraph) -> torch.Tensor:
        return features[subgraph.subset]

    return feature_fn, int(features.shape[1]), description


def run_single(
    dataset: str,
    model: str,
    variant: str = "baseline",
    seed: int = 0,
    init: int = 0,
    device: str = "cpu",
    train_config: TrainConfig | None = None,
    sampling_config: SamplingConfig | None = None,
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
                dual_branch=False,
            )

        trainer = SubgraphTrainer(
            net, train_config, device_info.device, feature_fn, payload["y"]
        )
        outcome = trainer.fit(train_sg, val_sg, seed=identity.stream("batch_order"))

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

"""Subgraph-based training loop.

Mirrors what the FLAG paper describes and what `methods/flag/test.py` does, with
the gaps filled in explicitly rather than silently:

  paper : 2-hop subgraphs, "a training strategy similar to GraphSAGE",
          "accumulate the loss of 10 subgraphs before backpropagation",
          Adam at a fixed lr of 0.01, "early stopping to prevent overfitting",
          model selected by highest validation F1-macro.

  code  : accumulation_steps = 10, Adam(lr=0.01), epochs=5, and
          `--patience` is parsed but NEVER READ -- there is no early stopping.
          Selection keeps the best-so-far checkpoint over a fixed epoch budget.

We implement real early stopping (the paper says it was used) and keep the
best-checkpoint behaviour, recording both settings in the result row. See
research/flag_code_audit.md section 2.1.

One node = one subgraph = one training example. Loss is accumulated over
`accumulation_steps` subgraphs before a single optimiser step, which is what
makes the effective batch size 10 without ever batching graphs together.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from flagbench.metrics.classification import (
    evaluate,
    fit_threshold,
    threshold_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Defaults are the paper's where it states them, the code's where it does not."""

    epochs: int = 5                       # code: test.py --epochs 5
    lr: float = 0.01                      # paper AND code
    weight_decay: float = 0.0             # code: test.py --weight_decay 0
    dropout: float = 0.5                  # code
    hidden_dim: int = 32                  # code (paper says 64; decision S-001)
    accumulation_steps: int = 10          # paper AND code
    optimizer: str = "adam"               # paper AND code

    early_stopping: bool = True           # paper says used; code does not implement
    patience: int = 10                    # code parses --patience 10, never reads it
    selection_metric: str = "f1_macro"    # paper: highest validation F1-macro
    threshold_policy: str = "validation_swept"

    class_weighted_loss: bool = False
    """BWGNN uses `weight = #neg/#pos` cross-entropy. FLAG's code does NOT --
    it uses plain CrossEntropyLoss. Off by default to match FLAG."""

    shuffle_each_epoch: bool = True       # code: random.shuffle(train_loader)
    max_grad_norm: float | None = None
    log_every: int = 0

    def as_record(self) -> dict:
        return {f"cfg_{k}": v for k, v in self.__dict__.items()}


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_f1_macro: float
    val_auc: float
    val_f1_macro: float
    seconds: float


@dataclass
class TrainingOutcome:
    best_epoch: int
    best_val_metric: float
    history: list[EpochRecord] = field(default_factory=list)
    stopped_early: bool = False
    epochs_run: int = 0
    training_seconds: float = 0.0
    best_state: dict | None = None


class SubgraphTrainer:
    """Trains one backbone on a list of per-node subgraphs."""

    def __init__(
        self,
        model,
        config: TrainConfig,
        device: torch.device,
        feature_fn,
        labels: torch.Tensor,
    ):
        """
        feature_fn(subgraph) -> Tensor [num_nodes_in_subgraph, in_dim]
            Supplies node features for a subgraph. This is where the four
            variants differ, and the ONLY place they differ -- the model, the
            loop and the metrics are identical across variants, which is what
            makes the comparison fair (Phase 11).
        """
        self.model = model
        self.config = config
        self.device = device
        self.feature_fn = feature_fn
        self.labels = labels.to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        if config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.lr,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"unsupported optimizer {config.optimizer!r}")

    # -------------------------------------------------------------- helpers
    def _forward_center(self, subgraph):
        """Run the model on one subgraph, return the centre node's logits."""
        features = self.feature_fn(subgraph).to(self.device)
        edge_index = subgraph.edge_index.to(self.device)
        _, logits = self.model(features, edge_index)
        return logits[subgraph.center_position()]

    def _set_class_weights(self, subgraphs):
        if not self.config.class_weighted_loss:
            return
        labels = torch.tensor(
            [int(self.labels[sg.central]) for sg in subgraphs]
        )
        counts = torch.bincount(labels, minlength=2).float()
        if counts.min() == 0:
            return
        weight = (counts.sum() - counts) / counts
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=weight.to(self.device)
        )

    # ------------------------------------------------------------- training
    def train_epoch(self, subgraphs, rng: np.random.Generator | None = None):
        self.model.train()
        order = list(range(len(subgraphs)))
        if self.config.shuffle_each_epoch and rng is not None:
            rng.shuffle(order)

        total_loss = 0.0
        accumulated = None
        num_accumulated = 0
        predictions, targets = [], []

        self.optimizer.zero_grad()
        for step, index in enumerate(order):
            subgraph = subgraphs[index]
            logits = self._forward_center(subgraph)
            label = self.labels[subgraph.central].reshape(1)
            loss = self.criterion(logits.unsqueeze(0), label)

            accumulated = loss if accumulated is None else accumulated + loss
            num_accumulated += 1
            total_loss += float(loss.detach())
            predictions.append(int(logits.argmax(-1)))
            targets.append(int(label))

            # Note: keyed off the enumerate counter, NOT a shadowed variable.
            # Upstream train.py rebinds `i` in an inner loop, so its accumulation
            # fires on subgraph size instead (flag_code_audit.md 5.4).
            if (step + 1) % self.config.accumulation_steps == 0:
                self._step(accumulated)
                accumulated, num_accumulated = None, 0

        if accumulated is not None:
            self._step(accumulated)

        from sklearn.metrics import f1_score

        f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
        return total_loss / max(len(order), 1), f1

    def _step(self, accumulated_loss):
        accumulated_loss.backward()
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

    # ----------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, subgraphs):
        """Fraud-class probabilities and true labels for a set of subgraphs."""
        self.model.eval()
        scores, labels = [], []
        for subgraph in subgraphs:
            logits = self._forward_center(subgraph)
            scores.append(float(torch.softmax(logits, dim=-1)[1]))
            labels.append(int(self.labels[subgraph.central]))
        return np.array(scores), np.array(labels)

    @torch.no_grad()
    def embeddings(self, subgraphs):
        """Centre-node hidden embeddings -- what the paper's t-SNE visualises."""
        self.model.eval()
        out = []
        for subgraph in subgraphs:
            features = self.feature_fn(subgraph).to(self.device)
            hidden, _ = self.model(features, subgraph.edge_index.to(self.device))
            out.append(hidden[subgraph.center_position()].cpu().numpy())
        return np.stack(out)

    # ---------------------------------------------------------------- fit
    def fit(self, train_subgraphs, val_subgraphs, seed: int = 0) -> TrainingOutcome:
        rng = np.random.default_rng(seed)
        self._set_class_weights(train_subgraphs)

        outcome = TrainingOutcome(best_epoch=-1, best_val_metric=-float("inf"))
        epochs_without_improvement = 0
        start = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss, train_f1 = self.train_epoch(train_subgraphs, rng)

            val_scores, val_labels = self.predict(val_subgraphs)
            threshold, _ = fit_threshold(
                val_labels, val_scores, policy=self.config.threshold_policy
            )
            val = evaluate(val_labels, val_scores, threshold, split="val")
            val_metric = (
                val.threshold_metrics[self.config.selection_metric]
                if self.config.selection_metric in val.threshold_metrics
                else getattr(val, self.config.selection_metric)
            )

            outcome.history.append(EpochRecord(
                epoch=epoch, train_loss=train_loss, train_f1_macro=train_f1,
                val_auc=val.auc, val_f1_macro=val.threshold_metrics["f1_macro"],
                seconds=time.time() - epoch_start,
            ))
            outcome.epochs_run = epoch

            if self.config.log_every and epoch % self.config.log_every == 0:
                logger.info(
                    "epoch %d/%d loss %.4f train_f1 %.4f val_auc %.4f val_f1 %.4f",
                    epoch, self.config.epochs, train_loss, train_f1,
                    val.auc, val.threshold_metrics["f1_macro"],
                )

            if val_metric > outcome.best_val_metric:
                outcome.best_val_metric = val_metric
                outcome.best_epoch = epoch
                outcome.best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    self.config.early_stopping
                    and epochs_without_improvement >= self.config.patience
                ):
                    outcome.stopped_early = True
                    logger.info(
                        "early stopping at epoch %d (no improvement in %d)",
                        epoch, self.config.patience,
                    )
                    break

        outcome.training_seconds = time.time() - start
        if outcome.best_state is not None:
            # Always evaluate the SELECTED model, not the last one.
            self.model.load_state_dict(outcome.best_state)
        return outcome

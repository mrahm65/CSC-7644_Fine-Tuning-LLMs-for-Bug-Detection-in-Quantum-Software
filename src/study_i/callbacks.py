"""Custom HuggingFace callbacks and trainer subclasses.

* :class:`ManualEarlyStoppingCallback` - early-stops on ``eval_f1_macro``
  without requiring checkpoint saving (compatible with ``save_strategy='no'``).
* :class:`EpochLogCallback` - records per-epoch validation metrics so the
  caller can plot learning curves.
* :class:`WeightedTrainer` - drop-in :class:`Trainer` replacement that
  applies inverse-frequency class weights and label smoothing in the loss.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainerCallback


class ManualEarlyStoppingCallback(TrainerCallback):
    """Early-stop training on ``eval_f1_macro`` after ``patience`` no-ops.

    Replaces ``transformers.EarlyStoppingCallback`` which insists on
    ``load_best_model_at_end=True`` (and therefore checkpoint saving). We
    keep ``save_strategy='no'`` to avoid filling disk during cross-
    validation, so we re-implement just the patience logic here.
    """

    def __init__(self, patience: int = 4) -> None:
        self.patience = patience
        self.best_f1 = -1.0
        self.wait = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Reset state at the beginning of every fold."""
        self.best_f1 = -1.0
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Update wait counter and signal stop when ``patience`` is exceeded."""
        current_f1 = metrics.get("eval_f1_macro", 0.0) if metrics else 0.0
        if current_f1 > self.best_f1 + 1e-4:
            self.best_f1 = current_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


class EpochLogCallback(TrainerCallback):
    """Append per-epoch validation metrics to ``log_store``."""

    def __init__(self, log_store: List[Dict[str, float]]) -> None:
        self.log_store = log_store

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Snapshot eval metrics at the end of each evaluation pass."""
        if metrics:
            self.log_store.append(
                {
                    "epoch": state.epoch,
                    "eval_loss": metrics.get("eval_loss", float("nan")),
                    "eval_acc": metrics.get("eval_accuracy", float("nan")),
                    "eval_f1": metrics.get("eval_f1_macro", float("nan")),
                }
            )


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy + label smoothing."""

    _class_weights: torch.Tensor
    _label_smoothing: float = 0.0

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Move the inverse-frequency weights to the trainer's device."""
        self._class_weights = weights.to(self.args.device)

    def set_label_smoothing(self, smoothing: float) -> None:
        """Configure the label-smoothing factor used in :meth:`compute_loss`."""
        self._label_smoothing = float(smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Standard CE with optional class weights + label smoothing."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=getattr(self, "_class_weights", None),
            label_smoothing=getattr(self, "_label_smoothing", 0.0),
        )
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def oversample_to_balance(
    texts: List[str],
    labels: List[int],
    rng,
) -> Tuple[List[str], List[int]]:
    """Duplicate minority-class samples so all classes have equal counts."""
    by_label = defaultdict(list)
    for text, label in zip(texts, labels):
        by_label[label].append(text)

    target = max(len(group) for group in by_label.values())

    out_texts: List[str] = []
    out_labels: List[int] = []
    for label, group in by_label.items():
        full_repeats = target // len(group)
        remainder = target - full_repeats * len(group)
        expanded = group * full_repeats
        if remainder:
            expanded += rng.sample(group, remainder)
        out_texts.extend(expanded)
        out_labels.extend([label] * len(expanded))

    indices = list(range(len(out_texts)))
    rng.shuffle(indices)
    return [out_texts[i] for i in indices], [out_labels[i] for i in indices]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """``Trainer.compute_metrics`` callback returning accuracy + macro/weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            labels, preds, average="weighted", zero_division=0
        ),
    }

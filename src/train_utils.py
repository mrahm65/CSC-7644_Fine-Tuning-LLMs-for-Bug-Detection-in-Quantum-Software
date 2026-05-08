"""Training helper functions: oversampling, weighted loss, early stopping."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def compute_class_weights(
    labels: Sequence[int],
    num_labels: int = 2,
) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced classification.

    Returns an array ``w`` such that ``w[i] = N / (num_labels * count[i])``,
    where ``N`` is the total number of training samples and ``count[i]`` is
    the number of samples carrying label ``i``. Missing classes receive a
    weight of 1 to avoid division by zero.
    """
    counts = Counter(labels)
    # Default unseen-class frequency to 1 so the weight expression below
    # never divides by zero. Each class gets at least token weight.
    freqs = np.array(
        [counts.get(i, 1) for i in range(num_labels)], dtype=np.float32
    )
    return freqs.sum() / (num_labels * freqs)


def oversample_minority_class(
    texts: List[str],
    labels: List[int],
    rng,
) -> Tuple[List[str], List[int]]:
    """Oversample minority class samples within a training fold.

    Pulls extra copies of records belonging to under-represented classes
    until every class has the same count as the majority class. The
    resulting (texts, labels) lists are shuffled together.
    """
    by_label = defaultdict(list)
    for text, label in zip(texts, labels):
        by_label[label].append(text)

    target = max(len(group) for group in by_label.values())

    out_texts: List[str] = []
    out_labels: List[int] = []
    for label, group in by_label.items():
        full_repeats = target // len(group)
        remainder = target - full_repeats * len(group)
        # Tile the existing group as many full copies as fit, then sample
        # without replacement to top up the remainder.
        expanded = group * full_repeats
        if remainder:
            expanded += rng.sample(group, remainder)
        out_texts.extend(expanded)
        out_labels.extend([label] * len(expanded))

    # Shuffle paired indices so batches do not become class-blocked.
    indices = list(range(len(out_texts)))
    rng.shuffle(indices)
    return [out_texts[i] for i in indices], [out_labels[i] for i in indices]


# ---------------------------------------------------------------------------
# HuggingFace callback / Trainer subclass (heavy imports deferred to call-site)
# ---------------------------------------------------------------------------
def make_manual_early_stopping_callback(patience: int = 4):
    """Factory: build a callback that early-stops on ``eval_f1_macro``.

    Compatible with ``save_strategy='no'`` (no checkpoint files needed).
    """
    from transformers import TrainerCallback

    class ManualEarlyStoppingCallback(TrainerCallback):
        """Patience-based early stopping on macro-F1."""

        def __init__(self, patience: int) -> None:
            """Store ``patience`` and initialize the best-score tracker."""
            self.patience = patience
            # Initial best is below the lowest possible macro-F1 (0.0) so
            # the first evaluation always counts as an improvement.
            self.best_f1 = -1.0
            self.wait = 0

        def on_train_begin(self, args, state, control, **kwargs):
            """Reset the best-score tracker at the start of every fold."""
            self.best_f1 = -1.0
            self.wait = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Update the wait counter and stop training when patience is exceeded."""
            current_f1 = (
                metrics.get("eval_f1_macro", 0.0) if metrics else 0.0
            )
            # Tiny epsilon prevents oscillating-equal cases from resetting
            # the wait counter spuriously.
            if current_f1 > self.best_f1 + 1e-4:
                self.best_f1 = current_f1
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    control.should_training_stop = True

    return ManualEarlyStoppingCallback(patience=patience)


def make_epoch_log_callback(log_store: List[Dict]):
    """Factory: build a callback that appends per-epoch eval metrics."""
    from transformers import TrainerCallback

    class EpochLogCallback(TrainerCallback):
        """Append per-evaluation metrics to ``log_store``."""

        def __init__(self, store: List[Dict]) -> None:
            """Hold a reference to the caller-owned log list."""
            self.log_store = store

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Snapshot loss / accuracy / macro-F1 at the end of each eval."""
            if metrics:
                # ``state.epoch`` is the fractional epoch the trainer just
                # finished; storing it keeps learning-curve plots aligned.
                self.log_store.append(
                    {
                        "epoch": state.epoch,
                        "eval_loss": metrics.get("eval_loss", float("nan")),
                        "eval_acc": metrics.get(
                            "eval_accuracy", float("nan")
                        ),
                        "eval_f1": metrics.get(
                            "eval_f1_macro", float("nan")
                        ),
                    }
                )

    return EpochLogCallback(log_store)


def make_weighted_trainer_class():
    """Return a :class:`Trainer` subclass with weighted CE + label smoothing."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted cross-entropy + label smoothing."""

        _class_weights: Optional[torch.Tensor] = None
        _label_smoothing: float = 0.0

        def set_class_weights(self, weights: torch.Tensor) -> None:
            """Move the inverse-frequency weight tensor to the trainer device."""
            # The weight tensor must live on the same device as the model
            # so torch's CrossEntropyLoss can apply it without a copy.
            self._class_weights = weights.to(self.args.device)

        def set_label_smoothing(self, smoothing: float) -> None:
            """Configure the label-smoothing factor used in compute_loss()."""
            self._label_smoothing = float(smoothing)

        def compute_loss(
            self, model, inputs, return_outputs=False, **kwargs
        ):
            """Cross-entropy loss with optional class weights + smoothing.

            Pops ``labels`` from ``inputs`` (Trainer convention), runs the
            model, then applies ``torch.nn.CrossEntropyLoss`` with the
            stored class-weight tensor and label-smoothing factor.
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self._class_weights,
                label_smoothing=self._label_smoothing,
            )
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer

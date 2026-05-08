"""Load tokenizer and model backbone."""

from __future__ import annotations

from typing import Optional, Tuple

from src.data_utils import ID2LABEL, LABEL2ID, NUM_LABELS


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = NUM_LABELS,
    *,
    dropout: float = 0.2,
    label2id: Optional[dict] = None,
    id2label: Optional[dict] = None,
) -> Tuple[object, object]:
    """Load a HuggingFace tokenizer and sequence classification model.

    Args:
        model_name: HuggingFace Hub identifier (e.g. ``roberta-base``).
        num_labels: Output classes. Defaults to the binary label space.
        dropout: Hidden / attention dropout probability.
        label2id: Optional mapping from label string to id.
        id2label: Optional inverse mapping.

    Returns:
        ``(model, tokenizer)`` tuple ready to be passed to a Trainer.
    """
    # Heavy imports are deferred so that ``import src.model_utils`` works
    # in environments that have not installed transformers yet.
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    label2id = label2id or LABEL2ID
    id2label = id2label or ID2LABEL

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        ignore_mismatched_sizes=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


def get_data_collator(tokenizer):
    """Return a ``DataCollatorWithPadding`` for the given tokenizer."""
    from transformers import DataCollatorWithPadding
    return DataCollatorWithPadding(tokenizer=tokenizer)

"""Training loops for one fold and one full backbone-experiment.

The high-level entry point is :func:`run_full_experiment`, which sweeps
``cv_seeds * n_folds`` runs for a single backbone and writes per-model
artifacts (figures, JSON, CSV) under the directories on the active
:class:`~src.study_i.schemas.TrainingConfig`.

For a single fold call :func:`run_fold` directly. It returns the predicted
labels, soft-max probabilities, summary metrics, and the per-epoch log so
the caller can plot a learning curve.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    set_seed,
)

from src.study_i.callbacks import (
    EpochLogCallback,
    ManualEarlyStoppingCallback,
    WeightedTrainer,
    compute_metrics,
    oversample_to_balance,
)
from src.study_i.schemas import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    TrainingConfig,
)
from src.study_i.plotting import (
    plot_confusion_matrix,
    plot_fold_distribution,
    plot_roc_curve,
)


def _device() -> str:
    """Return ``'cuda'`` if a GPU is available, else ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_fold(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    *,
    seed: int,
    output_dir: str,
    model_name: str,
    tokenizer,
    data_collator,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], List[Dict]]:
    """Train one cross-validation fold and predict on the held-out split."""
    set_seed(seed)
    rng = random.Random(seed)
    device = _device()

    train_texts, train_labels = oversample_to_balance(
        train_texts, train_labels, rng
    )

    stratify = train_labels if len(set(train_labels)) > 1 else None
    tr_t, val_t, tr_y, val_y = train_test_split(
        train_texts, train_labels,
        test_size=config.val_split, random_state=seed, stratify=stratify,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_len)

    train_ds = Dataset.from_dict({"text": tr_t, "labels": tr_y}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_t, "labels": val_y}).map(tokenize, batched=True)
    test_ds = Dataset.from_dict(
        {"text": list(test_texts), "labels": list(test_labels)}
    ).map(tokenize, batched=True)

    hf_config = AutoConfig.from_pretrained(
        model_name, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
        ignore_mismatched_sizes=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=hf_config, ignore_mismatched_sizes=True,
    )

    counts = Counter(tr_y)
    freqs = np.array([counts.get(i, 1) for i in range(NUM_LABELS)], dtype=np.float32)
    class_weights = torch.tensor(freqs.sum() / (NUM_LABELS * freqs), dtype=torch.float)

    epoch_log: List[Dict] = []

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=config.logging_steps,
        report_to="none",
        seed=seed,
        fp16=(device == "cuda" and config.use_fp16),
        disable_tqdm=True,
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            ManualEarlyStoppingCallback(patience=config.es_patience),
            EpochLogCallback(epoch_log),
        ],
    )
    trainer.set_class_weights(class_weights)
    trainer.set_label_smoothing(config.label_smoothing)
    trainer.train()

    pred = trainer.predict(test_ds)
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=-1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs[:, 1]),
    }

    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    return y_true, y_pred, probs, metrics, epoch_log


def run_full_experiment(
    model_name: str,
    model_short: str,
    description: str,
    *,
    texts: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
    n_labeled: int,
) -> Dict[str, Any]:
    """Run all CV folds for one backbone and persist artifacts."""
    print()
    print("=" * 72)
    print(f"  EXPERIMENT: {model_short.upper()}")
    print(f"  Model     : {model_name}")
    print(f"  Notes     : {description}")
    print("=" * 72)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(config.tables_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results_all: List[Dict[str, Any]] = []
    all_epoch_logs: List[List[Dict]] = []

    for repeat_idx, cv_seed in enumerate(config.cv_seeds):
        skf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=cv_seed
        )
        print(
            f"\n=== {model_short} | Repeat {repeat_idx + 1}/"
            f"{len(config.cv_seeds)} (seed={cv_seed}) ==="
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
            scratch = os.path.join(
                config.output_dir,
                f"tmp_{model_short}_{repeat_idx}_{fold_idx}",
            )
            y_true, y_pred, probs, metrics, elog = run_fold(
                texts[train_idx].tolist(), labels[train_idx].tolist(),
                texts[test_idx].tolist(), labels[test_idx].tolist(),
                seed=cv_seed * 100 + fold_idx,
                output_dir=scratch,
                model_name=model_name,
                tokenizer=tokenizer,
                data_collator=data_collator,
                config=config,
            )
            results_all.append({
                "repeat": repeat_idx, "fold": fold_idx, "cv_seed": cv_seed,
                "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
                "probs": probs.tolist(), **metrics,
            })
            all_epoch_logs.append(elog)
            print(
                f"  fold {fold_idx + 1}/{config.n_folds} - "
                f"acc={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1_macro']:.3f}  "
                f"auc={metrics['roc_auc']:.3f}  "
                f"(n_test={len(test_idx)})"
            )

    cv_df = pd.DataFrame([
        {"repeat": r["repeat"] + 1, "fold": r["fold"] + 1,
         "accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
         "f1_weighted": r["f1_weighted"], "roc_auc": r["roc_auc"]}
        for r in results_all
    ])

    print(
        f"\n--- {model_short} | aggregate (mean +/- std across "
        f"{len(results_all)} folds) ---"
    )
    agg = cv_df[["accuracy", "f1_macro", "f1_weighted", "roc_auc"]].agg(["mean", "std"])
    print(agg.round(3))

    y_true_all = np.concatenate([r["y_true"] for r in results_all])
    y_pred_all = np.concatenate([r["y_pred"] for r in results_all])
    probs_all = np.concatenate([r["probs"] for r in results_all])
    pooled_acc = accuracy_score(y_true_all, y_pred_all)
    pooled_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    pooled_auc = roc_auc_score(y_true_all, probs_all[:, 1])

    fold_accs = np.array([r["accuracy"] for r in results_all])
    fold_f1s = np.array([r["f1_macro"] for r in results_all])
    fold_aucs = np.array([r["roc_auc"] for r in results_all])

    fmt = config.figure_format
    plot_confusion_matrix(
        y_true_all, y_pred_all,
        model_short=model_short, n_folds=len(results_all),
        pooled_acc=pooled_acc, pooled_f1=pooled_f1, pooled_auc=pooled_auc,
        output_dir=config.figures_dir, fmt=fmt,
    )
    plot_fold_distribution(
        fold_accs, fold_f1s, fold_aucs,
        model_short=model_short, output_dir=config.figures_dir, fmt=fmt,
    )
    plot_roc_curve(
        results_all, model_short=model_short,
        pooled_auc=pooled_auc, output_dir=config.figures_dir, fmt=fmt,
    )

    json_path = os.path.join(config.tables_dir, f"results_{model_short}.json")
    csv_path = os.path.join(config.tables_dir, f"per_fold_{model_short}.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({
            "task": "classical_vs_quantum_binary",
            "model": model_name, "model_short": model_short,
            "description": description, "n_samples": n_labeled,
            "cv_setup": (
                f"{config.n_folds}-fold x {len(config.cv_seeds)} "
                f"seeds = {len(results_all)} folds"
            ),
            "mean_accuracy": float(agg.loc["mean", "accuracy"]),
            "std_accuracy": float(agg.loc["std", "accuracy"]),
            "mean_f1_macro": float(agg.loc["mean", "f1_macro"]),
            "std_f1_macro": float(agg.loc["std", "f1_macro"]),
            "mean_roc_auc": float(agg.loc["mean", "roc_auc"]),
            "std_roc_auc": float(agg.loc["std", "roc_auc"]),
            "pooled_accuracy": float(pooled_acc),
            "pooled_f1_macro": float(pooled_f1),
            "pooled_roc_auc": float(pooled_auc),
            "cv_results": results_all,
        }, handle, indent=2, default=str)
    cv_df.to_csv(csv_path, index=False)

    del tokenizer, data_collator
    gc.collect()
    if _device() == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_name": model_name, "model_short": model_short,
        "description": description,
        "fold_accs": fold_accs.tolist(),
        "fold_f1s": fold_f1s.tolist(),
        "fold_aucs": fold_aucs.tolist(),
        "mean_accuracy": float(fold_accs.mean()),
        "std_accuracy": float(fold_accs.std(ddof=1)),
        "mean_f1_macro": float(fold_f1s.mean()),
        "std_f1_macro": float(fold_f1s.std(ddof=1)),
        "mean_roc_auc": float(fold_aucs.mean()),
        "std_roc_auc": float(fold_aucs.std(ddof=1)),
        "pooled_accuracy": float(pooled_acc),
        "pooled_f1_macro": float(pooled_f1),
        "pooled_roc_auc": float(pooled_auc),
        "epoch_logs": all_epoch_logs,
        "y_true_all": y_true_all.tolist(),
        "y_pred_all": y_pred_all.tolist(),
        "probs_all": probs_all.tolist(),
    }

"""Evaluation Metrics: Categorical scoring, probabilistic calibration, and result dictionary assembly."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score
)


def metrics(y, pred, labels):
    """Compute standard categorical classification metrics with failure penalization (-1)."""
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, labels=list(range(len(labels))), average='macro', zero_division=0))
    )


def hard_vote(predictions, classes):
    """Compute majority hard vote across an array of ensemble predictions."""
    stack = np.asarray(predictions)
    counts = np.stack([(stack == c).sum(axis=0) for c in range(classes)], axis=1)
    return counts.argmax(axis=1)


def probability_metrics(y, probabilities):
    """Compute Brier score, log-loss, and 10-bin expected calibration error (ECE)."""
    p = np.asarray(probabilities)
    onehot = np.eye(p.shape[1])[y]
    confidence, predicted = p.max(axis=1), p.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for b in range(10):
        mask = (confidence >= edges[b]) & ((confidence < edges[b + 1]) if b < 9 else (confidence <= 1))
        if mask.any():
            ece += mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean())
    return dict(
        brier=float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        log_loss_clipped=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
        ece_10_bin=float(ece)
    )


def score_result(y, pred, n_classes, scores=None, probabilities=None):
    """Assemble complete evaluation record including ROC AUC, PR AUC, calibration, and confusion matrix."""
    result = metrics(y, pred, list(range(n_classes)))
    result['per_class_recall'] = recall_score(
        y, pred, labels=list(range(n_classes)), average=None, zero_division=0
    ).tolist()
    result['confusion_matrix'] = confusion_matrix(
        y, pred, labels=list(range(n_classes)) + ([-1] if -1 in pred else [])
    ).tolist()
    result['predicted_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
    if n_classes == 2 and scores is not None and np.isfinite(scores).all():
        result.update(
            roc_auc=float(roc_auc_score(y, scores)),
            average_precision=float(average_precision_score(y, scores))
        )
    if probabilities is not None and np.isfinite(probabilities).all():
        result.update(probability_metrics(np.asarray(y), probabilities))
    return result

"""Decision Policy Calibration: Model output standardization and post-hoc threshold selection."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def outputs(model, x, n_classes):
    """Extract standard raw predictions, continuous decision scores, and probability estimates."""
    pred = np.asarray(model.predict(x)).ravel().astype(int)
    if hasattr(model, 'predict_proba') and getattr(model, 'probability', True):
        p = np.asarray(model.predict_proba(x))
        return dict(
            raw=pred,
            score=p[:, 1] if n_classes == 2 else p,
            probability=p,
            default_threshold=0.5
        )
    score = np.asarray(model.decision_function(x))
    if n_classes == 2:
        score = score.ravel()
    return dict(raw=pred, score=score, probability=None, default_threshold=0.0)


def choose_threshold(y_policy, scores, default, quantiles=101):
    """Select balanced-accuracy maximizing decision threshold on an independent policy split."""
    y_policy, scores = np.asarray(y_policy), np.asarray(scores)
    if not np.isfinite(scores).all() or set(y_policy) != {0, 1}:
        raise ValueError('Threshold selection requires finite binary scores with both classes')
    thresholds = np.unique(np.r_[
        default,
        np.quantile(scores, np.linspace(0, 1, quantiles)),
        np.nextafter(scores.max(), np.inf)
    ])
    values = [float(balanced_accuracy_score(y_policy, scores >= t)) for t in thresholds]
    best = max(range(len(thresholds)), key=lambda i: (values[i], -abs(thresholds[i] - default)))
    return dict(
        threshold=float(thresholds[best]),
        selection_balanced_accuracy=values[best],
        grid_size=len(thresholds),
        selection_partition='policy',
        positive_class=1
    )

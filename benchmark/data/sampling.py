"""Proportional Integer Capping & Stratification.

Implements exact proportional quota allocation to prevent scikit-learn
stratification errors when class cardinality is high or residuals are small.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def cap_indices(ids, y, cap, seed):
    """Return a proportional subset with every class retained.
    
    Ensures that every class present in y has at least one representative in the
    selected subset, balancing class deficits and excesses proportional to class counts.
    """
    ids, y = np.asarray(ids), np.asarray(y)
    if not isinstance(cap, (int, np.integer)) or cap < 1:
        raise ValueError('Sample cap must be a positive integer')
    if len(ids) <= cap:
        return ids
    classes, counts = np.unique(y[ids], return_counts=True)
    if cap < len(classes):
        raise ValueError('Sample cap must cover every class')
    rng = np.random.default_rng(seed)
    
    # Preserve existing split if scikit-learn can stratify both partitions cleanly
    if len(ids) - cap >= len(classes) and counts.min() >= 2:
        chosen, _ = train_test_split(ids, train_size=cap, stratify=y[ids], random_state=seed)
        if len(np.unique(y[chosen])) == len(classes):
            return np.asarray(chosen)
            
    target = counts * (cap / len(ids))
    quota = np.maximum(1, np.floor(target).astype(int))
    
    # Allocate by distance from proportional target, respecting class capacity
    while quota.sum() < cap:
        deficit = np.where(quota < counts, target - quota, -np.inf)
        quota[rng.choice(np.flatnonzero(deficit == deficit.max()))] += 1
    while quota.sum() > cap:
        excess = np.where(quota > 1, quota - target, -np.inf)
        quota[rng.choice(np.flatnonzero(excess == excess.max()))] -= 1
        
    chosen = np.concatenate([
        rng.choice(ids[y[ids] == c], size=int(n), replace=False)
        for c, n in zip(classes, quota)
    ])
    return rng.permutation(chosen)

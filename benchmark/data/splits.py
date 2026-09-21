"""Data Splitting: Stratified holdout and development splits (train/val/policy/test)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .sampling import cap_indices


def pilot_split(df, cfg, seed):
    """Generate train/val/test splits following pilot configuration."""
    y = df['label'].to_numpy()
    ids = np.arange(len(df))
    if '_official_split' in df:
        pool = ids[df['_official_split'].eq('train')]
        test = ids[df['_official_split'].eq('test')]
    else:
        pool, test = train_test_split(ids, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(pool, test_size=0.2, stratify=y[pool], random_state=seed)
    splits = dict(
        train=cap_indices(train, y, cfg['train_cap'], seed),
        validation=cap_indices(val, y, cfg['validation_cap'], seed),
        test=cap_indices(test, y, cfg['test_cap'], seed)
    )
    assert not (set(splits['train']) & set(splits['test']))
    assert not (set(splits['validation']) & set(splits['test']))
    for split_ids in splits.values():
        assert set(y[split_ids]) == set(y), 'A split is missing a class; raise sample caps.'
    return splits


def make_holdout(df, cfg):
    """Generate static holdout evaluation partition across training seeds; exclude known default pilot test rows."""
    y, ids = df.label.to_numpy(), np.arange(len(df))
    excluded = set()
    if cfg['exclude_pilot_tests']:
        pilot_cfg = dict(train_cap=8000, validation_cap=1000, test_cap=cfg['pilot_test_cap'])
        for seed in cfg['pilot_seeds']:
            excluded.update(pilot_split(df, pilot_cfg, seed)['test'])
    if '_official_split' in df:
        pool = ids[df['_official_split'].eq('train')]
        allowed = ids[df['_official_split'].eq('test') & ~pd.Series(ids).isin(excluded).to_numpy()]
        n_test = len(allowed)
    else:
        allowed = np.array([i for i in ids if i not in excluded])
        n_test = min(max(len(np.unique(y)), int(np.ceil(0.2 * len(df)))), len(allowed))
        pool = None
    limit = cfg['banking_test_cap'] if len(np.unique(y)) == 77 else cfg['test_cap']
    test = cap_indices(allowed, y, min(limit, n_test), cfg['holdout_seed'])
    if pool is None:
        pool = np.setdiff1d(ids, test)
    if set(y[test]) != set(y):
        raise ValueError('Holdout does not cover every class; review exclusions/caps.')
    assert not (set(test) & excluded)
    return dict(pool=np.asarray(pool), test=test, pilot_excluded=np.asarray(sorted(excluded)))


def make_split(df, holdout, cfg, seed):
    """Generate 60/20/20 train/validation/policy splits from the development pool."""
    y = df.label.to_numpy()
    rest, policy = train_test_split(
        holdout['pool'], test_size=0.2, stratify=y[holdout['pool']], random_state=seed
    )
    train, validation = train_test_split(
        rest, test_size=0.25, stratify=y[rest], random_state=seed
    )
    result = dict(
        train=cap_indices(train, y, cfg['train_cap'], seed),
        validation=cap_indices(validation, y, cfg['validation_cap'], seed),
        policy=cap_indices(policy, y, cfg['policy_cap'], seed),
        test=holdout['test']
    )
    sets = [set(v) for v in result.values()]
    for i, left in enumerate(sets):
        assert set(y[list(left)]) == set(y)
        for right in sets[i + 1:]:
            assert left.isdisjoint(right)
    return result

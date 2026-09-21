"""Bootstrap Confidence Intervals: Stratified test-case bootstrap paired across models and training seeds."""
import numpy as np
import pandas as pd
from ..config import MODELS


def paired_intervals(results, cfg):
    """Compute paired test-case bootstrap confidence intervals for Jev vs. ML comparisons."""
    rows = []
    for dataset in cfg['datasets']:
        by_model = {}
        for result in results:
            if result['dataset'] == dataset:
                by_model.setdefault(result['model'], {})[result['seed']] = result
        for jev in ['Jev ' + m for m in cfg['jev_modes']]:
            if jev not in by_model:
                continue
            for name in MODELS:
                if name not in by_model:
                    continue
                seeds = sorted(set(by_model[jev]) & set(by_model[name]))
                if seeds != sorted(cfg['seeds']):
                    continue
                reference = by_model[name][seeds[0]]
                y = np.asarray(reference['test_labels'])
                for panel in ['raw', 'adjusted']:
                    differences = []
                    for seed in seeds:
                        a, b = by_model[jev][seed], by_model[name][seed]
                        assert a['test_ids'] == b['test_ids'] == reference['test_ids']
                        assert a['test_labels'] == b['test_labels'] == reference['test_labels']
                        differences.append(
                            (np.asarray(a['predictions'][panel]) == y).astype(float) -
                            (np.asarray(b['predictions'][panel]) == y).astype(float)
                        )
                    diff = np.mean(differences, axis=0)
                    groups = [diff[y == c] for c in np.unique(y)]
                    rng = np.random.default_rng(cfg['holdout_seed'])
                    boot = np.zeros(cfg['bootstrap_samples'])
                    for group in groups:
                        boot += group[rng.integers(0, len(group), size=(len(boot), len(group)))].mean(axis=1) / len(groups)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    rows.append(dict(
                        dataset=dataset,
                        panel=panel,
                        comparison=f'{jev} minus {name}',
                        difference_pp=100 * np.mean([g.mean() for g in groups]),
                        lower_95_pp=100 * lo,
                        upper_95_pp=100 * hi,
                        seeds=len(seeds),
                        test_rows=len(y)
                    ))
    return pd.DataFrame(rows)

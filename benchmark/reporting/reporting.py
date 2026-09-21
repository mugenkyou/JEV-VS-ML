"""Benchmark Reporting: Result file aggregation, diagnostic summaries, CSV tables, and HTML reports."""
import html
import json
from pathlib import Path

import pandas as pd

from ..config import MODELS
from ..evaluation.bootstrap import paired_intervals


def read_results(root, cfg):
    """Read all completed model result JSON files across datasets and seeds."""
    results = []
    for dataset in cfg['datasets']:
        for seed in cfg['seeds']:
            folder = Path(root) / 'runs' / dataset.replace(' ', '_') / str(seed)
            for name in MODELS + ['Majority baseline'] + ['Jev ' + m for m in cfg['jev_modes']]:
                file = folder / (name + '.json')
                if file.exists():
                    results.append(json.loads(file.read_text()))
    return results


def render_results(root, cfg):
    """Aggregate raw and adjusted metrics, export summary CSVs, compute bootstrap intervals, and write HTML report."""
    root = Path(root)
    results = read_results(root, cfg)
    records = []
    for r in results:
        for panel in ['raw', 'adjusted']:
            records.append(dict(
                dataset=r['dataset'],
                model=r['model'],
                seed=r['seed'],
                panel=panel,
                **{k: v for k, v in r[panel].items() if isinstance(v, (int, float))},
                failed_test_rows=r.get('failed_test_rows', 0)
            ))
    scores = pd.DataFrame(records)
    if scores.empty:
        return {}
    scores.to_csv(root / 'scores_by_seed.csv', index=False)
    diagnostics = pd.DataFrame([
        dict(
            dataset=r['dataset'],
            seed=r['seed'],
            model=r['model'],
            backend=r.get('backend', 'ensemble/API'),
            backend_reason=r.get('backend_reason', ''),
            failed_trials=r.get('failed_trials', 0),
            failed_test_rows=r.get('failed_test_rows', 0),
            failed_policy_rows=r.get('failed_policy_rows', 0),
            threshold_status=(r.get('threshold') or {}).get('status', 'selected' if r.get('threshold') else 'not applicable'),
            resolved_models=', '.join(r.get('resolved_models', []))
        )
        for r in results
    ])
    diagnostics.to_csv(root / 'run_diagnostics.csv', index=False)
    columns = MODELS + ['Majority baseline'] + ['Jev ' + m for m in cfg['jev_modes']]
    tables, sections = {}, []
    for panel in ['raw', 'adjusted']:
        for metric in ['balanced_accuracy', 'accuracy', 'macro_f1']:
            table = pd.DataFrame('pending', index=cfg['datasets'], columns=columns)
            selected = scores[scores.panel.eq(panel)]
            for (dataset, model), group in selected.groupby(['dataset', 'model']):
                values = 100 * group[metric]
                sd = values.std(ddof=1) if len(values) > 1 else 0
                suffix = '' if len(values) == len(cfg['seeds']) else ' INCOMPLETE'
                table.loc[dataset, model] = f'{values.mean():.1f} ± {sd:.1f} (n={len(values)}){suffix}'
            table.index.name = 'dataset'
            table.to_csv(root / f'{panel}_{metric}.csv')
            tables[f'{panel}_{metric}'] = table
            sections.append(f'<h2>{html.escape(panel + " — " + metric)}</h2>' + table.to_html())
    intervals = paired_intervals(results, cfg)
    intervals.to_csv(root / 'paired_bootstrap_intervals.csv', index=False)
    report = (
        '''<!doctype html><meta charset="utf-8"><title>Jev benchmark VERSION</title>
<style>body{font:14px system-ui;margin:28px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border:1px solid #ddd;white-space:nowrap}tr:nth-child(even){background:#f5f6f8}</style>
<h1>Jev and conventional classifiers — VERSION</h1>
<p>Values are percent, mean ± sample SD across training seeds on the SAME test cases. SD is not a confidence interval.</p>
<p>Raw: each model's default decision rule, after ML validation search. Adjusted: binary thresholds learned from separate labeled policy data; multiclass unchanged. Adjusted Jev is not strictly zero-shot end to end.</p>
<p>Four predeclared ML candidates per family in benchmark/v3 mode; two in quick mode. This is a bounded-budget pipeline comparison, not a best-possible-model claim. Majority baseline is untuned. See the saved protocol and backend diagnostics for implementation differences.</p>
<p>Paired bootstrap intervals are conditional on these trained models and test sample; they exclude retraining uncertainty. Multiple comparisons are exploratory. Small Iris and Breast Cancer holdouts remain imprecise.</p>
''' + '\n'.join(sections) + '<h2>Run diagnostics — inspect failures before publishing</h2>' + diagnostics.to_html(index=False) +
        '<h2>Paired balanced-accuracy differences</h2>' + intervals.to_html(index=False)
    )
    (root / 'report.html').write_text(report.replace('VERSION', html.escape(cfg['protocol'])), encoding='utf-8')
    return tables

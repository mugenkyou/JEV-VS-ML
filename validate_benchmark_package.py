"""Import and Subsystem Smoke Test for benchmark.* Package Hierarchy.

Verifies that all subpackages, core classes, functions, and pipeline contracts
can be imported and executed without errors.
"""
import sys
import numpy as np
import pandas as pd


def test_imports():
    print("[1/8] Verifying top-level benchmark configuration...", flush=True)
    import benchmark
    from benchmark.config import configuration, ALL_DATASETS, MODELS
    assert benchmark.__version__ == "3.0.1"
    cfg = configuration('v3')
    assert cfg['protocol'] == '3.0.1'
    assert len(ALL_DATASETS) == 8
    assert len(MODELS) == 11

    print("[2/8] Verifying benchmark.data subsystem...", flush=True)
    from benchmark.data import (
        download, hf_frames, dataset_spec, prepare_data,
        cap_indices, make_holdout, make_split, pilot_split
    )
    # Test sampling / cap_indices logic
    y = np.array([0]*50 + [1]*50)
    capped = cap_indices(np.arange(100), y, 20, seed=42)
    assert len(capped) == 20
    assert (y[capped] == 0).sum() == 10
    assert (y[capped] == 1).sum() == 10

    print("[3/8] Verifying benchmark.features subsystem...", flush=True)
    from benchmark.features import Features
    # Text feature transformer test
    df_text = pd.DataFrame({'text': ['hello world', 'spam text message', 'hello test world']})
    f_text = Features({'kind': 'text'}, cfg, seed=42)
    f_text.fit(df_text, np.array([0, 1, 0]))
    t_text = f_text.transform(df_text)
    assert 'word' in t_text and 'linear' in t_text and 'tree' in t_text

    # Tabular feature transformer test
    df_tab = pd.DataFrame({'num': [1.0, 2.0, np.nan], 'cat': ['a', 'b', 'a']})
    f_tab = Features({'kind': 'tabular'}, cfg, seed=42)
    f_tab.fit(df_tab, np.array([0, 1, 0]))
    t_tab = f_tab.transform(df_tab)
    assert 'dense' in t_tab and 'cat' in t_tab

    print("[4/8] Verifying benchmark.models subsystem...", flush=True)
    from benchmark.models import (
        candidates, route, probe_cuml, fit_candidate, finish_result, train_job, train_lane
    )
    lr_candidates = candidates('Logistic regression', cfg, text=True, seed=42)
    assert len(lr_candidates) == cfg['max_trials']

    print("[5/8] Verifying benchmark.jev subsystem...", flush=True)
    from benchmark.jev import (
        digest, get_key, JevClient, few_shot_ids, make_payload, state_row
    )
    d = digest({'test': 123})
    assert isinstance(d, str) and len(d) == 64

    print("[6/8] Verifying benchmark.evaluation subsystem...", flush=True)
    from benchmark.evaluation import (
        choose_threshold, outputs, hard_vote, metrics,
        probability_metrics, score_result, paired_intervals
    )
    m = metrics([0, 1, 0], [0, 1, 1], labels=[0, 1])
    assert 'accuracy' in m and 'balanced_accuracy' in m

    th = choose_threshold([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], default=0.5)
    assert 'threshold' in th

    print("[7/8] Verifying benchmark.execution subsystem...", flush=True)
    from benchmark.execution import (
        probe_gpus, run_isolated, visible_devices, environment,
        prepare_suite, request_partition, run_jev, run_ml
    )
    env = environment()
    assert 'python' in env and 'packages' in env

    print("[8/8] Verifying benchmark.reporting subsystem...", flush=True)
    from benchmark.reporting import read_results, render_results
    assert callable(read_results) and callable(render_results)

    print("\n[+] ALL benchmark.* SUBSYSTEM IMPORTS AND INTEGRATION CHECKS PASSED!\n", flush=True)


if __name__ == '__main__':
    test_imports()

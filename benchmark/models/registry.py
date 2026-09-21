"""Model Registry: Predeclared candidate architectures and hyperparameter grids per family."""
import json
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from ..config import MODELS


def candidates(name, cfg, text, seed, gpu_id=None):
    """Return list of candidate configurations (rep, model, params, weighted) for the specified model family."""
    n, jobs = cfg['trees'], cfg['threads']

    if name == 'Logistic regression':
        rep = 'linear'
        base = LogisticRegression(max_iter=3000, random_state=seed)
        settings = [{'C': 0.3}, {'C': 3.0}]
    elif name == 'SVM':
        rep = 'linear'
        base = LinearSVC(dual='auto', max_iter=10000, random_state=seed) if text else SVC(cache_size=512)
        settings = [{'C': 0.3}, {'C': 3.0}]
    elif name == 'Decision tree':
        rep = 'tree'
        base = DecisionTreeClassifier(random_state=seed)
        settings = [{'max_depth': 10, 'min_samples_leaf': 3}, {'max_depth': None, 'min_samples_leaf': 10}]
    elif name == 'Random forest':
        rep = 'tree'
        base = RandomForestClassifier(n_estimators=n, n_jobs=jobs, random_state=seed)
        if cfg.get('speed_profile'):
            settings = [
                dict(max_depth=16, min_samples_leaf=2, max_features='sqrt'),
                dict(max_depth=24, min_samples_leaf=1, max_features=0.15)
            ]
        else:
            settings = [{'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'}, {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5}]
    elif name == 'Extra trees':
        rep = 'tree'
        base = ExtraTreesClassifier(n_estimators=n, n_jobs=jobs, random_state=seed)
        if cfg.get('speed_profile'):
            settings = [
                dict(max_depth=16, min_samples_leaf=2, max_features='sqrt'),
                dict(max_depth=24, min_samples_leaf=1, max_features=0.15)
            ]
        else:
            settings = [{'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'}, {'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5}]
    elif name == 'Naive Bayes':
        rep = 'word' if text else 'dense'
        base = MultinomialNB() if text else GaussianNB()
        settings = [{'alpha': 0.1}, {'alpha': 1.0}] if text else [{'var_smoothing': 1e-9}, {'var_smoothing': 1e-6}]
    elif name == 'Hist gradient boost':
        rep = 'knn' if (text and cfg.get('speed_profile')) else 'dense'
        base = HistGradientBoostingClassifier(max_iter=n, early_stopping=False, random_state=seed)
        if cfg.get('speed_profile'):
            base.set_params(
                max_iter=cfg['histogram_iterations'], max_bins=63,
                early_stopping=True, n_iter_no_change=cfg['early_stopping'],
                validation_fraction=0.15
            )
        settings = [{'max_leaf_nodes': 15, 'learning_rate': 0.05, 'l2_regularization': 1}, {'max_leaf_nodes': 31, 'learning_rate': 0.1, 'l2_regularization': 5}]
    elif name == 'XGBoost':
        from xgboost import XGBClassifier
        rep = 'tree'
        base = XGBClassifier(
            n_estimators=n, tree_method='hist',
            device=f'cuda:{gpu_id}' if gpu_id is not None else 'cpu',
            n_jobs=jobs, random_state=seed
        )
        settings = [
            {'max_depth': 3, 'learning_rate': 0.05, 'min_child_weight': 3, 'reg_lambda': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
            {'max_depth': 6, 'learning_rate': 0.1, 'min_child_weight': 5, 'reg_lambda': 5, 'subsample': 0.9, 'colsample_bytree': 0.9}
        ]
    elif name == 'CatBoost':
        from catboost import CatBoostClassifier
        rep = 'cat'
        base = CatBoostClassifier(
            iterations=n, verbose=False, allow_writing_files=False,
            task_type='GPU' if gpu_id is not None else 'CPU',
            thread_count=jobs, random_seed=seed,
            **({'devices': str(gpu_id)} if gpu_id is not None else {})
        )
        settings = [{'depth': 4, 'learning_rate': 0.05, 'l2_leaf_reg': 3}, {'depth': 7, 'learning_rate': 0.1, 'l2_leaf_reg': 5}]
    elif name == 'k-NN':
        return [
            ('knn', KNeighborsClassifier(n_neighbors=k, weights=w, n_jobs=jobs), dict(n_neighbors=k, weights=w), False)
            for k, w in [(5, 'distance'), (15, 'distance'), (31, 'distance'), (15, 'uniform')]
        ][:cfg['max_trials']]
    else:
        raise ValueError(f"Unknown model name: {name}")

    return [
        (rep, clone(base).set_params(**params), params, weighted)
        for params in settings for weighted in [False, True]
    ][:cfg['max_trials']]

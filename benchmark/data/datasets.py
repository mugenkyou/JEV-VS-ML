"""Dataset Ingestion, Verification, and Snapshotting.

Handles public dataset downloads, cryptographic validation, text truncation,
input-level deduplication, and Parquet snapshot generation.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer, load_iris


def write_json(path, value):
    """Write JSON file atomically using a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def download(url, root):
    """Download and cache remote resource by SHA-256 URL hash."""
    directory = Path(root) / 'downloads'
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / hashlib.sha256(url.encode()).hexdigest()
    if not path.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    raw = path.read_bytes()
    write_json(path.with_suffix('.json'), dict(url=url, sha256=hashlib.sha256(raw).hexdigest()))
    return raw


def hf_frames(repo, root):
    """Fetch parquet dataset splits from HuggingFace, pinning the exact commit hash."""
    from huggingface_hub import HfApi, hf_hub_download
    revision_file = Path(root) / ('hf_' + repo.replace('/', '_') + '.json')
    api = HfApi()
    if revision_file.exists():
        revision = json.loads(revision_file.read_text())['revision']
    else:
        revision = api.dataset_info(repo).sha
        write_json(revision_file, dict(repo=repo, revision=revision))
    files = api.list_repo_files(repo, repo_type='dataset', revision=revision)
    out = []
    for split in ['train', 'test']:
        paths = sorted(p for p in files if p.endswith('.parquet') and Path(p).name.startswith(split + '-'))
        if not paths:
            raise ValueError(f'No parquet {split} files in {repo} at {revision}')
        frame = pd.concat([
            pd.read_parquet(hf_hub_download(repo, p, repo_type='dataset', revision=revision))
            for p in paths
        ], ignore_index=True)
        frame['_official_split'] = split
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def dataset_spec(name, root):
    """Return raw dataframe, modality kind, label descriptions, task prompt, and source URL."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    if name in ['AG News', 'IMDb']:
        repo = 'fancyzhx/ag_news' if name == 'AG News' else 'stanfordnlp/imdb'
        df = hf_frames(repo, root)
        labels = (['World news', 'Sports', 'Business', 'Science and technology']
                  if name == 'AG News' else ['Negative movie review', 'Positive movie review'])
        task = ('Classify the news article by its primary topic.'
                if name == 'AG News' else 'Classify the overall sentiment of the movie review.')
        return df, 'text', labels, task, f'https://huggingface.co/datasets/{repo}'
        
    if name == 'Banking77':
        base = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/'
        labels = json.loads(download(base + 'categories.json', root))
        frames = []
        for part in ['train', 'test']:
            df = pd.read_csv(io.BytesIO(download(base + part + '.csv', root)))
            df['label'] = df['category'].map({v: i for i, v in enumerate(labels)})
            df['_official_split'] = part
            frames.append(df[['text', 'label', '_official_split']])
        return pd.concat(frames, ignore_index=True), 'text', [s.replace('_', ' ') for s in labels], 'Select the banking customer request intent.', base
        
    if name == 'SMS Spam':
        url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
        archive = zipfile.ZipFile(io.BytesIO(download(url, root)))
        lines = archive.read('SMSSpamCollection').decode('utf-8').splitlines()
        df = pd.DataFrame([s.split('\t', 1) for s in lines], columns=['category', 'text'])
        df['label'] = df.pop('category').map({'ham': 0, 'spam': 1})
        return df, 'text', ['Legitimate personal or transactional message', 'Unsolicited spam or promotional scam message'], 'Classify this SMS as legitimate or spam.', url
        
    if name in ['Iris', 'Breast Cancer']:
        bunch = load_iris(as_frame=True) if name == 'Iris' else load_breast_cancer(as_frame=True)
        df = bunch.data.copy()
        df['label'] = bunch.target
        labels = [str(x) for x in bunch.target_names]
        task = ('Identify the iris species from sepal and petal dimensions in centimetres.' if name == 'Iris' else
                'Classify a breast mass as malignant or benign from Wisconsin diagnostic cell-nuclei measurements. Mean, standard error, and worst measurements are supplied.')
        return df, 'tabular', labels, task, 'https://scikit-learn.org/stable/datasets/toy_dataset.html'
        
    if name == 'Bank Marketing':
        url = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
        z = zipfile.ZipFile(io.BytesIO(download(url, root)))
        if 'bank-full.csv' not in z.namelist():
            z = zipfile.ZipFile(io.BytesIO(z.read('bank.zip')))
        df = pd.read_csv(z.open('bank-full.csv'), sep=';')
        df['label'] = df.pop('y').map({'no': 0, 'yes': 1})
        # Call duration is unavailable prior to calling a client; excluded conservatively.
        df = df.drop(columns=['duration'])
        return df, 'tabular', ['Does not subscribe to a term deposit', 'Subscribes to a term deposit'], 'Predict term-deposit subscription from pre-call bank marketing attributes. balance is euros; pdays=-1 means no prior contact; campaign counts campaign contacts; previous counts previous contacts. Call duration is excluded.', url
        
    if name == 'Online Shoppers':
        url = 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
        z = zipfile.ZipFile(io.BytesIO(download(url, root)))
        path = next(p for p in z.namelist() if p.endswith('.csv'))
        df = pd.read_csv(z.open(path))
        df['label'] = df.pop('Revenue').astype(int)
        # PageValues is outcome-related; excluded conservatively for this benchmark.
        df = df.drop(columns=['PageValues'])
        for col in ['OperatingSystems', 'Browser', 'Region', 'TrafficType']:
            df[col] = df[col].astype(str)
        return df, 'tabular', ['Session does not result in a purchase', 'Session results in a purchase'], 'Predict purchase outcome from end-of-session browsing attributes. Page counts and durations in seconds, bounce and exit rates, month, visitor type, weekend, and SpecialDay proximity are provided. PageValues is excluded. This is retrospective session classification, not early-session forecasting.', url
        
    raise ValueError(f"Unknown dataset name: {name}")


def prepare_data(name, root, cfg):
    """Snapshot dataset into Parquet with deduplication and SHA-256 verification."""
    root = Path(root)
    folder = root / 'data' / name.replace(' ', '_')
    folder.mkdir(parents=True, exist_ok=True)
    snapshot = folder / 'snapshot.parquet'
    
    if snapshot.exists():
        df = pd.read_parquet(snapshot)
        meta = json.loads((folder / 'metadata.json').read_text())
    else:
        df, kind, labels, task, source = dataset_spec(name, root / 'sources')
        df = df.reset_index(drop=True)
        df['_row_id'] = np.arange(len(df))
        if df['label'].isna().any():
            raise ValueError(f'{name}: unmapped labels')
        if kind == 'text':
            df['text'] = df['text'].astype(str).str.replace('<br />', ' ', regex=False).str.slice(0, cfg['max_text_chars'])
        else:
            for c in df.select_dtypes(include=['bool']).columns:
                df[c] = df[c].astype(str)
                
        # Deduplicate on MODEL INPUT after truncation. Conflicting labels are dropped.
        feature_cols = [c for c in df if c not in ['label', '_row_id', '_official_split']]
        keys = df[feature_cols].astype(str).agg('\x1f'.join, axis=1)
        label_counts = df['label'].groupby(keys).nunique()
        conflicting = set(label_counts.index[label_counts > 1])
        df = df.loc[~keys.isin(conflicting)].copy()
        
        # Keep official test copy if an identical input occurs across official splits.
        if '_official_split' in df:
            df = df.sort_values('_official_split')  # test before train
        df = df.drop_duplicates(subset=feature_cols, keep='first').reset_index(drop=True)
        df['label'] = df['label'].astype(int)
        df.to_parquet(snapshot, index=False)
        meta = dict(
            name=name, kind=kind, labels=labels, task=task, source=source,
            features=feature_cols, rows=len(df),
            sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        )
        write_json(folder / 'metadata.json', meta)
        
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Dataset snapshot changed')
    return df, meta

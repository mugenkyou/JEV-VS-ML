"""Jev Request Caching: Deterministic request hashing and artifact persistence."""
import hashlib
import json
from pathlib import Path


def digest(data):
    """Compute deterministic SHA-256 hex digest for JSON-serializable structures or strings."""
    if isinstance(data, (dict, list)):
        data = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def write_json(path, data):
    """Atomically write JSON data to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)

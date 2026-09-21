"""TypeSafe Jev API Client: Rate limiting, exponential backoff, cost bounding, and response caching."""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import requests

from .cache import digest, write_json


def get_key():
    """Retrieve TypeSafe API key from environment variable or Kaggle Secrets."""
    key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret('TYPESAFE_API_KEY').strip()
        except Exception:
            raise RuntimeError(
                'Add and enable Kaggle Secret TYPESAFE_API_KEY, or set the environment variable. '
                'The key is never stored in outputs.'
            ) from None
    if not key:
        raise RuntimeError('TYPESAFE_API_KEY is empty')
    return key


class JevClient:
    """Bounded, rate-limited requests with durable per-attempt logs and response cache."""

    def __init__(self, root, cfg, key):
        self.root = Path(root)
        self.cfg = cfg
        self.key = key
        self.cache = self.root / 'jev_cache'
        self.cache.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.next_time = 0
        self.log = self.root / 'api_attempts.jsonl'
        prior = []
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                try:
                    prior.append(json.loads(line))
                except json.JSONDecodeError:
                    raise RuntimeError('Incomplete attempt log: inspect it before resuming.') from None
        self.attempts = len(prior)
        self.estimated_cost = sum(r.get('reserved_estimated_usd', 0) for r in prior)
        self.stopped = False

    def reserve(self, payload_hash, payload):
        estimate = len(json.dumps(payload).encode()) * self.cfg['estimated_usd_per_million_input_tokens'] / 1e6
        with self.lock:
            if self.stopped:
                raise RuntimeError('API stopped after an authentication or permanent request error')
            if self.attempts >= self.cfg['max_api_attempts'] or self.estimated_cost + estimate > self.cfg['max_estimated_api_usd']:
                raise RuntimeError(
                    'Configured API attempt/cost estimate limit reached; increase limits and use a new run configuration deliberately.'
                )
            self.attempts += 1
            self.estimated_cost += estimate
            wait = max(0, self.next_time - time.monotonic())
            self.next_time = max(self.next_time, time.monotonic()) + self.cfg['min_request_interval']
            with self.log.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(
                    request_hash=payload_hash,
                    attempt=self.attempts,
                    reserved_estimated_usd=estimate,
                    timestamp=time.time()
                )) + '\n')
        if wait:
            time.sleep(wait)

    def call(self, payload):
        request_hash = digest(payload)
        path = self.cache / (request_hash + '.json')
        if path.exists():
            result = json.loads(path.read_text())
            return {**result, 'cache_hit': True}
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
        started = time.perf_counter()
        last_error = ''
        for attempt in range(self.cfg['max_retries'] + 1):
            self.reserve(request_hash, payload)
            retry_after = 0
            try:
                response = self.local.session.post(
                    'https://api.typesafe.ai/v1/systemone',
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
                    json=payload,
                    timeout=(15, 90)
                )
                if response.status_code == 200:
                    body = response.json()
                    answer = body['answers']['classification']
                    keys = list(payload['questions']['classification']['criteria'])
                    probabilities = answer['probabilities']
                    p = np.array([probabilities[k] for k in keys], dtype=float)
                    if (
                        set(probabilities) != set(keys) or
                        not np.isfinite(p).all() or
                        (p < 0).any() or
                        (p > 1).any() or
                        abs(p.sum() - 1) > 0.01 or
                        answer['choice'] not in keys
                    ):
                        raise ValueError('Invalid probability vector or class')
                    result = dict(
                        ok=True,
                        prediction=keys.index(answer['choice']),
                        probabilities=(p / p.sum()).tolist(),
                        confidence=answer.get('confidence'),
                        resolved_model=body.get('model'),
                        usage=body.get('usage', {}),
                        attempts=attempt + 1,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        request_hash=request_hash
                    )
                    write_json(path, result)
                    return {**result, 'cache_hit': False}
                last_error = f'HTTP {response.status_code}'
                if response.status_code not in [408, 429, 500, 502, 503, 504]:
                    with self.lock:
                        self.stopped = True
                    raise RuntimeError(
                        f'Jev permanent error {last_error}; inspect the account/model/request. Response body deliberately omitted.'
                    )
                try:
                    retry_after = min(60, float(response.headers.get('Retry-After', 0)))
                except ValueError:
                    pass
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                last_error = type(exc).__name__
            if attempt < self.cfg['max_retries']:
                time.sleep(max(retry_after, 2 ** attempt))
        return dict(
            ok=False,
            prediction=-1,
            error=last_error,
            probabilities=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            request_hash=request_hash,
            cache_hit=False
        )

"""GPU Execution Infrastructure: Isolated sub-interpreters with explicit CUDA_VISIBLE_DEVICES."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np


def probe_gpus():
    """Check each visible CUDA ordinal independently with actual library fits."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15
        )
        names = result.stdout.strip()
        if result.returncode or not names:
            return [], 'CPU (no NVIDIA GPU detected)'
        from xgboost import XGBClassifier
        from catboost import CatBoostClassifier
        x = np.arange(80, dtype=np.float32).reshape(40, 2)
        y = np.arange(40) % 2
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        count = len([v for v in visible.split(',') if v.strip() and v.strip() != '-1']) if visible is not None else len(names.splitlines())
        usable, messages = [], []
        for gpu_id in range(count):
            try:
                m = XGBClassifier(n_estimators=2, device=f'cuda:{gpu_id}', tree_method='hist').fit(x, y)
                actual = json.loads(m.get_booster().save_config())['learner']['generic_param']['device']
                if actual != f'cuda:{gpu_id}':
                    raise RuntimeError(f'XGBoost selected {actual}, expected cuda:{gpu_id}')
                CatBoostClassifier(
                    iterations=2, task_type='GPU', devices=str(gpu_id), verbose=False, allow_writing_files=False
                ).fit(x, y)
                usable.append(gpu_id)
                messages.append(f'GPU {gpu_id}: XGBoost and CatBoost probe passed')
            except Exception as exc:
                messages.append(f'GPU {gpu_id}: excluded ({type(exc).__name__}: {str(exc)[:180]})')
        return usable, names + '\n' + '\n'.join(messages)
    except Exception as exc:
        return [], f'CPU fallback; GPU probe failed: {type(exc).__name__}: {str(exc)[:250]}'


def visible_devices():
    """Query list of visible GPU devices or UUIDs."""
    configured = os.environ.get('CUDA_VISIBLE_DEVICES')
    if configured is not None:
        return [s.strip() for s in configured.split(',') if s.strip() and s.strip() != '-1']
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15, check=True
    )
    return [s.strip() for s in result.stdout.splitlines() if s.strip()]


def child_environment(device, threads):
    """Build isolated child environment with single visible GPU and bounded thread pools."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    env['PYTHONUNBUFFERED'] = '1'
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = str(threads)
    source = str(Path(__file__).resolve().parent.parent.parent)
    env['PYTHONPATH'] = source + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def run_phase(root, cfg, devices, lanes, phase):
    """Run worker phase across isolated GPU subprocesses."""
    folder = Path(root).resolve() / 'gpu_workers' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    children = []
    started = time.monotonic()
    try:
        for lane, device in enumerate(devices):
            request = folder / f'{phase}_{lane}.json'
            result = folder / f'{phase}_{lane}.result.json'
            request.write_text(json.dumps(dict(
                root=str(Path(root).resolve()),
                cfg=cfg,
                jobs=lanes[lane],
                phase=phase,
                result=str(result),
                device=device
            )), encoding='utf-8')
            log_path = folder / f'{phase}_{lane}.log'
            writer = log_path.open('w', encoding='utf-8')
            try:
                process = subprocess.Popen(
                    [sys.executable, '-u', '-m', 'benchmark.execution.gpu', str(request)],
                    env=child_environment(device, cfg['threads']),
                    stdout=writer,
                    stderr=subprocess.STDOUT
                )
            finally:
                writer.close()
            reader = log_path.open(encoding='utf-8', errors='replace')
            children.append((process, reader, result, log_path))
        while True:
            if phase == 'probe' and time.monotonic() - started > cfg.get('gpu_probe_timeout_seconds', 180):
                raise RuntimeError(f'GPU probes timed out; inspect logs in {folder}')
            active = False
            for lane, (process, reader, result, log_path) in enumerate(children):
                chunk = reader.read()
                if chunk:
                    print(f'[GPU lane {lane} / {phase}]\n{chunk}', end='', flush=True)
                code = process.poll()
                active |= code is None
                if code not in [None, 0]:
                    print(reader.read(), end='', flush=True)
                    raise RuntimeError(
                        f'Isolated GPU {phase} failed (exit {code}). Inspect {log_path}. '
                        'No CPU fallback was applied.'
                    )
            if not active:
                break
            time.sleep(0.2)
        for _, reader, _, _ in children:
            print(reader.read(), end='', flush=True)
        return [json.loads(result.read_text(encoding='utf-8')) for _, _, result, _ in children]
    finally:
        for process, reader, _, _ in children:
            if process.poll() is None:
                process.terminate()
        for process, reader, _, _ in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.close()


def run_isolated(root, cfg, gpu_ids=None):
    """Orchestrate cuML/CUDA preflight checks and parallel training jobs across dedicated GPU workers."""
    if gpu_ids == []:
        raise RuntimeError('No GPU selected; enable Kaggle T4 x2.')
    devices = visible_devices()
    if gpu_ids is not None:
        devices = [devices[i] for i in gpu_ids]
    if not devices:
        raise RuntimeError('No visible GPU: enable Kaggle T4 x2.')
    jobs = [(name, seed) for name in cfg['datasets'] for seed in cfg['seeds']]
    count = min(len(devices), len(jobs), cfg['max_parallel_jobs'])
    if count < 1:
        raise ValueError('At least one job and worker is required')
    devices = devices[:count]
    lanes = [jobs[i::count] for i in range(count)]
    print('Isolated GPU assignments:', dict(enumerate(devices)), flush=True)
    probes = run_phase(root, cfg, devices, lanes, 'probe')
    from ..models.training import write_json
    write_json(Path(root) / 'cuml_probes.json', probes)
    results = run_phase(root, cfg, devices, lanes, 'train')
    return [r for lane in results for r in lane]


def main():
    """Subprocess worker entry point."""
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    cfg = request['cfg']
    if request['phase'] == 'probe':
        from .gpu import probe_gpus
        from ..models.backends import probe_cuml
        ids, status = probe_gpus()
        print(status, flush=True)
        if ids != [0]:
            raise RuntimeError('Native GPU probes failed in the single-device process')
        result = probe_cuml(0, cfg)
        result.update(visible_device=request['device'], local_gpu_id=0)
    elif request['phase'] == 'train':
        from ..models.training import train_lane
        cfg = dict(cfg, assigned_visible_device=request['device'])
        result = train_lane(request['jobs'], request['root'], cfg, 0)
    elif request['phase'] == 'test-environment':
        result = dict(
            visible_device=os.environ['CUDA_VISIBLE_DEVICES'],
            pid=os.getpid(),
            cuda_imported=any(k in sys.modules for k in ['cupy', 'cuml'])
        )
    else:
        raise ValueError('Unknown worker phase')
    Path(request['result']).write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()

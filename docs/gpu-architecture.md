# GPU Isolation & Multi-Device Execution Architecture

This document describes the multi-GPU hardware isolation, preflight probe verification, and child-process management implemented in **Protocol 3.0.1 (V3)** (`jevbench/gpu_process.py` and `jevbench/backends.py`).

---

## 1. Problem Statement: CUDA Context Contamination in Multi-GPU Sessions

In Python data science environments (particularly Jupyter kernels and Kaggle GPU notebooks), importing CUDA-capable libraries such as `torch`, `cupy`, `cuml`, or `xgboost` in the parent process automatically initializes the CUDA primary runtime context on GPU 0.

When multiple worker threads or subprocesses attempt to manipulate `CUDA_VISIBLE_DEVICES` or `cupy.cuda.Device()` after CUDA has already been initialized in the parent process:
1. GPU memory handles can become corrupted across processes (e.g. RAFT invalid resource handle errors in cuML).
2. Parallel worker processes may inadvertently contend for GPU 0 memory rather than utilizing secondary GPUs (e.g. Kaggle T4 $\times$ 2).
3. Silent CPU fallbacks can occur without user notice.

---

## 2. Architecture: Subprocess Process Boundaries

To guarantee complete hardware isolation, **Protocol 3.0.1** executes all GPU-accelerated operations inside dedicated child processes created via `subprocess.Popen`, with `CUDA_VISIBLE_DEVICES` configured strictly **before** the Python interpreter is launched.

```
                              ┌──────────────────────────────────────┐
                              │ Parent Process (Jupyter Notebook)    │
                              │ - No CUDA/CuPy/cuML imports          │
                              │ - Dispatches isolated worker jobs    │
                              └──────────────────┬───────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   │                                                           │
     ┌─────────────▼──────────────┐                              ┌─────────────▼──────────────┐
     │ GPU Lane 0 Subprocess      │                              │ GPU Lane 1 Subprocess      │
     │ ENV: CUDA_VISIBLE_DEVICES=0│                              │ ENV: CUDA_VISIBLE_DEVICES=1│
     ├────────────────────────────┤                              ├────────────────────────────┤
     │ 1. Device Preflight Probe  │                              │ 1. Device Preflight Probe  │
     │ 2. cuML & CUDA Imports     │                              │ 2. cuML & CUDA Imports     │
     │ 3. Train Lane Execution    │                              │ 3. Train Lane Execution    │
     └────────────────────────────┘                              └────────────────────────────┘
```

### 2.1 Child Process Environment Configuration

In `jevbench/gpu_process.py:child_environment()`, the environment for each worker process is constructed with:
* `CUDA_VISIBLE_DEVICES = <physical_gpu_id>`: Exactly one physical GPU is exposed to the child. Within the child process, this device is always virtual device `0`.
* `PYTHONUNBUFFERED = 1`: Ensures instant stdout/stderr flushing to log files.
* `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS = 2`: Restricts CPU thread pools to prevent CPU thread thrashing across parallel lanes.
* `PYTHONPATH`: Prepends the project root directory.

---

## 3. Two-Stage Execution & Preflight Barrier

Execution proceeds in two strictly sequential phases:

```
[Parent Process]
       │
       ▼
 1. Query Visible GPUs (nvidia-smi UUID inspection)
       │
       ▼
 2. PHASE 1: Probe Barrier (run_phase 'probe')
       ├── Spawn Worker 0 on GPU 0 (Dummy cuML / XGBoost / CatBoost fits)
       └── Spawn Worker 1 on GPU 1 (Dummy cuML / XGBoost / CatBoost fits)
       │
       ▼ [Both workers must exit with code 0 within 180s]
       │
 3. PHASE 2: Training Jobs (run_phase 'train')
       ├── Worker 0: Sequential dataset/seed lane
       └── Worker 1: Sequential dataset/seed lane
```

### 3.1 Preflight Probe Stage (`probe_cuml`)
Before any training jobs are dispatched:
1. Each GPU worker fits small synthetic classification benchmarks using:
   * cuML Logistic Regression (dense and sparse CSR)
   * cuML Random Forest
   * cuML k-NN (brute-force)
   * cuML SVM (RBF kernel)
   * XGBoost with `device='cuda:0'`
   * CatBoost with `task_type='GPU'`
2. Output shape, decision scores, and probability simplex sum constraints ($\sum p_i \approx 1.0$) are verified.
3. If either worker fails the probe, the entire benchmark halts immediately (`require_gpu=True`). Silent CPU fallback is explicitly disabled.

---

## 4. Backend Routing Matrix (`backends.py`)

Models are routed explicitly rather than through global monkey-patching:

| Model Family | Task / Representation | Selected Backend | Routing Reason |
|---|---|---|---|
| **Logistic Regression** | Text / Tabular | **cuML GPU** | cuML Quasi-Newton (QN) solver natively supports sparse and dense matrices. |
| **Random Forest (Unweighted)** | Text (Dense) / Tabular | **cuML GPU** | 128 quantile bins, 1 stream. |
| **Random Forest (Balanced)** | Text (Sparse) / Tabular | **scikit-learn CPU** | cuML RF lacks sample-weight support; routed to CPU to preserve balanced class weighting. |
| **k-NN** | SVD (Text) / Scaled Tabular | **cuML GPU** | cuML brute-force GPU neighbor search. |
| **SVM (Binary Tabular)** | Dense Scaled Tabular | **cuML GPU** | cuML RBF kernel SVC with 512MB cache. |
| **SVM (Text & Multiclass)** | Sparse Linear Text / Multiclass | **scikit-learn CPU** | cuML lacks sparse LinearSVC; multiclass tabular retained in scikit-learn. |
| **XGBoost** | All tasks | **CUDA Native** | Native CUDA implementation (`tree_method='hist'`). |
| **CatBoost** | All tasks | **CUDA Native** | Native CUDA implementation (`task_type='GPU'`). |
| **Hist Gradient Boost** | SVD (Text) / Tabular | **scikit-learn CPU** | CPU histogram gradient boosting (60 iterations, 63 bins). |
| **Voting Ensemble** | All tasks | **CPU Aggregation** | Averaging fitted member probabilities. |

---

## 5. Lifecycle Management & Process Reaping

To prevent orphaned background processes during notebook interrupts:
* `gpu_process.py:run_phase()` maintains an active process registry.
* In the event of an uncaught exception or keyboard interrupt, a `finally` block iterates over all active child processes:
  1. Sends `process.terminate()`.
  2. Waits up to 5 seconds for clean exit.
  3. Escalates to `process.kill()` (`SIGKILL`) if the process fails to terminate.
  4. Flushes and closes all open file descriptors for logging.

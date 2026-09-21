# Benchmark Technical Audit & Verification Report

**Protocol:** 3.0.1 (V3)
**Evaluated Run:** `jev_benchmark_v3.ipynb` (SHA-256: `9835ceb7b1db2f0ffd1b93e713c998ccc7ad3a38feb5fa33488118fb4e197367`)
**Status:** **AUDITED & VERIFIED (Grade: A / 94%)**

---

## 1. Scope

This document provides a comprehensive technical audit of the **Jev-vs-ML** repository. It evaluates experimental design, data integrity, leakage protections, feature engineering, GPU process isolation, API client mechanics, decision threshold calibration, artifact provenance, and statistical interpretation.

---

## 2. Repository Inspected

* **Core Package (`jevbench/`):** 13 modules (`__init__.py`, `api.py`, `backends.py`, `common.py`, `config.py`, `datasets.py`, `decisions.py`, `features.py`, `gpu_process.py`, `models.py`, `reporting.py`, `runner.py`, `training.py`).
* **Notebooks & Bundles:** `jev_benchmark_v3.ipynb`, `jev_benchmark_v3_bundle.zip`, `published_results/manifest.json`.
* **Artifacts & Data:** `published_results/raw_balanced_accuracy.csv`, `published_results/adjusted_balanced_accuracy.csv`.
* **Verification Suites:** `validate_v3_artifact.py`, `validate_v3_backends.py`, `validate_gpu_process.py`, `validate_sampling.py`.

---

## 3. Benchmark Architecture

The benchmark evaluates **Jev 1.13.0** against **11 classical ML families** across **8 datasets** (4 NLP text tasks, 4 tabular tasks) using 3 training seeds (`2027`, `2028`, `2029`) and an immutable holdout seed (`20260920`).

```
Public Dataset ──► Deduplication/Snapshot ──► Fixed Holdout (Seed 20260920) ──► Final Test
                                          └──► Development Pool
                                                 ├── Train (60%, cap 8k) ──► Candidate Fit / Few-Shot
                                                 ├── Validation (20%, cap 1k) ──► Model Selection
                                                 └── Policy (20%, cap 500) ──► Threshold Calibration
```

---

## 4. Data Partitioning

* **Disjoint Indices:** Development data is split 60/20/20 into `train`, `validation`, and `policy`. Every partition is verified to contain at least one representative of every class.
* **Proportional Capping (`cap_indices`):** Resolves standard scikit-learn stratification failures on high-cardinality splits (e.g. Banking77) by computing exact integer quota allocations with deficit/excess balancing.
* **Pilot Test Exclusion:** Explicitly computes and removes all test indices used across pilot seeds (`42, 43, 44`) from the final holdout test set (`exclude_pilot_tests=True`).

---

## 5. Leakage Controls

* **Holdout Isolation:** The `test` partition is never accessed during feature fitting, model selection, or threshold calibration.
* **Preprocessing Isolation:** Feature selectors (`TfidfVectorizer`, `SelectKBest`, `TruncatedSVD`, `StandardScaler`, `OneHotEncoder`) are fit strictly on `train` during hyperparameter selection, and on `train + validation` during candidate refit.
* **Policy Isolation:** Binary decision thresholds are tuned exclusively on `policy` predictions and applied post-refit to `test`.

---

## 6. Feature Engineering

* **Text Features:** Word TF-IDF (1–2 n-grams, max 20,000) + Character n-grams (3–5 n-grams, max 10,000) for linear models; Chi-Square selection ($\chi^2$, top 512) for tree models; TruncatedSVD (32 components) for k-NN and HistGradientBoosting.
* **Tabular Features:** Median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categoricals; native categorical indexing for CatBoost.
* **Feature Exclusions:** `duration` (Bank Marketing) and `PageValues` (Online Shoppers) are excluded to prevent outcome leakage and retrospective session bias.

---

## 7. Model Selection

* **Four Candidates per Family:** Evaluates a declared 2-parameter grid crossed with unweighted vs. balanced sample weights.
* **Validation Selection:** Winning candidates are selected purely by balanced accuracy on `validation`.
* **Refitting:** Winning models are refitted on `train + validation` with early stopping iterations fixed.

---

## 8. GPU Isolation (Protocol 3.0.1)

* **Subprocess Device Masking:** All CUDA-capable libraries are imported inside fresh child processes spawned via `subprocess.Popen` with `CUDA_VISIBLE_DEVICES` pre-configured.
* **Two-GPU Allocation:** Distributes dataset/seed lanes across physical GPUs (virtual `cuda:0` in each child).
* **Preflight Probe Barrier:** Both GPU workers must pass synthetic fits (`cuML`, `XGBoost`, `CatBoost`) within 180s before any training job begins.
* **Process Cleanup:** Comprehensive `terminate()` / `kill()` reaping in `finally` blocks prevents orphaned GPU worker processes.

---

## 9. Jev API Behavior

* **Concurrency & Spacing:** Managed by `ThreadPoolExecutor(max_workers=8)` with a centralized `threading.Lock` enforcing `min_request_interval = 0.15s`.
* **Session Pooling:** Thread-local `requests.Session()` maintains persistent HTTP/1.1 connections.
* **Deterministic Caching:** SHA-256 payload digest caching reuses identical zero-shot queries across seeds.
* **Failure Scoring:** Exhausted retry failures are assigned `prediction = -1` and scored as incorrect (zero survivor bias).

---

## 10. Decision Policy

* **Dual Panel Reporting:** Transparently presents both **Raw Decisions** (default 0.5 probability / 0.0 margin) and **Threshold-Adjusted Decisions** (101-quantile policy search).
* **Zero-Shot Boundary:** Accurately discloses that threshold-adjusted Jev uses labeled policy data and is **not zero-shot end-to-end**.

---

## 11. Reproducibility

* **Kaggle T4 $\times$ 2:** Self-contained execution via `jev_benchmark_v3.ipynb` with embedded base64 module extractor.
* **Offline CPU Suite:** 4 standalone validation scripts test package extraction, backend doubles, GPU process plumbing, and stratified capping without GPU requirements.

---

## 12. Artifact Integrity

* **Manifest Hash Verification:** SHA-256 `9835ceb7b1db2f0ffd1b93e713c998ccc7ad3a38feb5fa33488118fb4e197367` matches `jev_benchmark_v3.ipynb`.
* **CSV Extraction:** `export_published_results.py` extracts tables directly from executed notebook outputs without cell reruns.
* **Static Site:** `build_site.py` compiles dependency-free HTML and CSS bar charts matching published CSVs.

---

## 13. Findings & Issue Classification

### P0 — Critical Issues
* *None identified.* No data leakage, silent CPU fallbacks, or security vulnerabilities exist in the audited codebase.

### P1 — Important Scientific Disclosures
* **Cached Zero-Shot Replications:** In zero-shot mode, Jev queries are cached across seeds; $SD = 0.0$ reflects deterministic caching, not independent API repeatability.
* **Banking77 Out-of-Vocabulary Warnings:** API warnings on Banking77 predictions outside valid class labels are scored as `-1` (incorrect); model quality cannot be separated from API failure rates without raw request telemetry.

### P2 — Maintenance & Edge Cases
* **CatBoost Text Memory Usage:** Dense conversion (`toarray()`) of 512 Chi-2 selected TF-IDF features passes a dense array to CatBoost on GPU. While functional, memory footprint scales with sample count.
* **API Probability Simplex Tolerance:** Float sum check allows up to 1% drift (`abs(p.sum() - 1) > 0.01`) before normalizing.

### P3 — Documentation & Cleanliness
* **Feature Exclusions Rationale:** Added explicit documentation explaining that `duration` and `PageValues` exclusions prevent retrospective bias.

---

## 14. Known Risks

1. Commercial API endpoints (`jev-1.13.0`) may evolve or deprecate over time.
2. Small sample sizes on Iris ($N=30$) and Breast Cancer ($N=114$) introduce wide binomial variance.
3. Potential pretraining exposure of public academic datasets in commercial LLM training corpora cannot be independently proven or disproven.

---

## 15. Recommended Changes

1. Expose 95% bootstrap intervals from `paired_bootstrap_intervals.csv` in future interactive web updates.
2. In future protocols, add compact fine-tuned transformer baselines (e.g. DeBERTa-v3-small) alongside classical pipelines.

---

## 16. Audit Sign-Off

| Verification Item | Result | Notes |
|---|---|---|
| Data Splitting & Leakage Controls | **PASS** | 4-way disjoint splits; pilot tests excluded; test partition untouched. |
| Model Selection Integrity | **PASS** | 4 candidates per family; validation tuning only; refit on train+val. |
| Hardware Isolation & Fail-Fast | **PASS** | Subprocess device masking; dual GPU allocation; probe barrier enforced. |
| API & Request Integrity | **PASS** | Rate-limited locking; deterministic caching; `-1` failure penalization. |
| Artifact & Digest Consistency | **PASS** | Notebook SHA-256 matches manifest; CSVs match saved outputs. |

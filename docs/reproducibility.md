# Reproducing the Benchmark & Local Verification

This document provides complete, step-by-step instructions for reproducing the benchmark in a fresh Kaggle GPU environment and running local offline validation suites.

---

## 1. Full Benchmark Reproduction on Kaggle

The published benchmark (Protocol 3.0.1 / V3) was executed on Kaggle using two NVIDIA Tesla T4 GPUs.

### 1.1 Environment Setup
1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Under **Notebook Settings**:
   * **Accelerator:** GPU T4 x 2.
   * **Internet:** On (Enabled).
   * **Environment Preferences:** Always use latest environment (ensures pre-installed RAPIDS/cuML).
3. In **Add-ons → Secrets**, add a secret named `TYPESAFE_API_KEY` containing your TypeSafe API key, and enable notebook access to it.

### 1.2 Running the Benchmark
1. Upload and open [`jev_benchmark_v3.ipynb`](../jev_benchmark_v3.ipynb).
2. **Cell 1 (Package Extraction):** Automatically decodes and extracts the bundled `jevbench` Python source files to `/kaggle/working/jevbench_code_<hash>`.
3. **Cell 2 (Dependency Verification):** Installs required dependencies from `requirements-v2.txt`.
4. **Cell 3 (Configuration & Protocol):** Displays the frozen configuration and protocol documentation.
5. **Cell 4 (Data Preparation):** Downloads public datasets, applies deduplication, and generates reproducible splits.
6. **Cell 5 (Classical ML):** Executes GPU probes, fits all 11 classical model families across 3 seeds, and renders intermediate tables.
7. **Cell 6 (Jev API Calls):** Dispatches rate-limited Jev API requests (Zero-Shot and Few-Shot) with disk caching and retry handling.
8. **Cell 7 (Export & Reporting):** Renders final raw and threshold-adjusted result matrices, generates `report.html`, and packages the complete result ZIP archive.

---

## 2. Local Artifact & Offline Verification

You can verify the repository's source code, bundle consistency, and published score tables locally on CPU without requiring an NVIDIA GPU or making paid API calls.

### 2.1 Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2.2 Run Validation Suite

#### 1. Artifact & Manifest Consistency Check
Validates that the executed notebook outputs match the published CSVs byte-for-byte, that all embedded package hashes match, and that release bundles are uncorrupted:
```bash
python validate_v3_artifact.py
```

#### 2. Backend Routing & cuML Double Verification
Tests estimator candidate generation, parameter routing, and explicit CPU fallbacks using mock doubles:
```bash
python validate_v3_backends.py
```

#### 3. GPU Process Isolation Check
Spawns fresh child Python interpreters to verify that `CUDA_VISIBLE_DEVICES` masks are applied before library imports, child environment isolation holds, and error handling reaps child processes:
```bash
python validate_gpu_process.py
```

#### 4. Proportional Capping & Sampling Regression Check
Verifies that `cap_indices()` correctly stratifies high-cardinality splits (e.g. Banking77) without triggering scikit-learn remainder errors:
```bash
python validate_sampling.py
```

---

## 3. Regenerating Published Tables & Site

### 3.1 Regenerate CSVs from Executed Notebook
To extract the published balanced accuracy tables directly from `jev_benchmark_v3.ipynb`:
```bash
python export_published_results.py
```

### 3.2 Rebuild GitHub Pages Report
To rebuild the static HTML report (`docs/index.html`), asset bundles, and downloadable CSVs from `published_results/`:
```bash
python build_site.py
```

### 3.3 Build Fresh Unexecuted Notebook from Source
To compile an unexecuted Jupyter notebook from the current `jevbench/` Python source files:
```bash
python build_modular_notebook.py --v3
```
*Note: Generated notebooks are saved to the `generated/` directory. The builder strictly refuses to overwrite executed notebooks containing outputs.*

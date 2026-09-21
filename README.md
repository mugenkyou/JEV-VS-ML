# Jev vs. Machine Learning

[![Benchmark Report](https://img.shields.io/badge/Report-Interactive%20Website-0071e3)](https://mugenkyou.github.io/JEV-VS-ML/)
[![Architecture](https://img.shields.io/badge/Architecture-System%20Map-blueviolet)](ARCHITECTURE.md)
[![Protocol](https://img.shields.io/badge/Protocol-3.0.1%20%2F%20V3-blue)](BENCHMARK_V3.md)
[![Notebook Artifact](https://img.shields.io/badge/Artifact-Executed%20Notebook-success)](jev_benchmark_v3.ipynb)
[![Audit](https://img.shields.io/badge/Audit-Verified%20(Grade%20A)-brightgreen)](BENCHMARK_AUDIT.md)

An independent, reproducible empirical benchmark comparing **Jev 1.13.0** (a semantic classification LLM API) against **11 conventional machine learning pipelines** across eight public NLP and tabular datasets under a declared compute and data budget.

---

## 1. Overview & Research Question

Modern language model endpoints provide zero-shot and few-shot classification capabilities driven by semantic instructions rather than task-specific gradient descent. This benchmark investigates:

> **How does Jev 1.13.0 perform against tuned classical machine learning pipelines across text and tabular classification tasks when evaluated on identical held-out test cases?**

### Headline Findings
* **Strongest Advantage (IMDb Sentiment):** Jev achieves **96.3% raw / 96.1% threshold-adjusted balanced accuracy** in zero-shot mode, outperforming the strongest classical pipeline (Logistic Regression at 88.4% / 88.3%) by **+7.9 / +7.8 percentage points**.
* **Close Competitiveness (Spam & News):** Jev is competitive on **SMS Spam** (96.1% raw zero-shot vs. 95.0% Naive Bayes; 95.9% adjusted vs. 96.3% Naive Bayes) and **AG News** (87.5% zero-shot vs. 88.4% Linear SVM).
* **Intent Classification Deficit (Banking77):** On 77-class customer intent classification, Jev trails Linear SVM (**81.9% few-shot vs. 89.7% Linear SVM**).
* **Tabular Classification Deficit:** Classical pipelines substantially outperform Jev on all evaluated tabular tasks (**Bank Marketing: 73.3% Ensemble vs. 59.7% Jev; Online Shoppers: 71.2% Ensemble vs. 53.6% Jev**). On small toy datasets (**Breast Cancer, Iris**), classical models achieve 100.0% while Jev scores 92.7% and 97.0%.

---

## 2. Where Should I Start?

| Goal / Topic | Primary Resource | Module / Directory |
|---|---|---|
| **Understand the benchmark methodology** | [Methodology Guide](docs/methodology.md) | `docs/methodology.md` |
| **Understand system architecture & dataflow** | [Architecture Reference](ARCHITECTURE.md) | `ARCHITECTURE.md` |
| **Add a dataset, model, or feature pipeline** | [Contributing Guide](CONTRIBUTING.md) | `CONTRIBUTING.md` / `benchmark/` |
| **Understand GPU isolation & fail-fast probes** | [GPU Architecture Guide](docs/gpu-architecture.md) | `benchmark/execution/gpu.py` |
| **Understand Jev API client, rate limits & caching** | [API Architecture Guide](docs/api.md) | `benchmark/jev/` |
| **Reproduce the benchmark on Kaggle** | [Kaggle Master Notebook](jev_benchmark_v3.ipynb) | `jev_benchmark_v3.ipynb` |
| **Run local offline validation suite** | [Reproducibility Guide](docs/reproducibility.md) | `python validate_all.py` |
| **Review scientific audit & limitations** | [Technical Audit](BENCHMARK_AUDIT.md) / [Limitations](docs/limitations.md) | `BENCHMARK_AUDIT.md` |

---

## 3. Benchmark Design & Data Flow

The benchmark operates under **Protocol 3.0.1 (V3)** using 3 training seeds (`2027`, `2028`, `2029`) and one immutable holdout seed (`20260920`).

```
                                  ┌───────────────────────────────┐
                                  │   Raw Dataset (Public Hub)    │
                                  └──────────────┬────────────────┘
                                                 │
                                     [prepare_data / hashlib]
                                                 │
                                  ┌──────────────▼────────────────┐
                                  │ Deduplicated Snapshot Parquet │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
          ┌─────────────▼─────────────┐                     ┌─────────────▼─────────────┐
          │    Fixed Test Holdout     │                     │      Development Pool     │
          │  (Seed 20260920 / Capped) │                     │ (Pilot Test Rows Excluded)│
          └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                        │                                                 │
                        │                               ┌─────────────────┼─────────────────┐
                        │                               │                 │                 │
                        │                         ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
                        │                         │   Train   │     │Selection  │     │  Policy   │
                        │                         │ (60% pool)│     │Validation │     │Calib (20%)│
                        │                         │  Cap: 8k  │     │ (20% pool)│     │ Cap: 500  │
                        │                         └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
                        │                               │                 │                 │
                        │                               ▼                 ▼                 ▼
                        │                      [Feature Transform] [Model Selection] [Threshold Search]
                        │                               │                 │                 │
                        │                               └────────┬────────┘                 │
                        │                                        │                          │
                        │                              [Refit on Train+Val]                 │
                        │                                        │                          │
                        ├────────────────────────────────────────┴──────────────────────────┤
                        │                                                                   │
            ┌───────────▼───────────┐                                           ┌───────────▼───────────┐
            │   Classical Models    │                                           │       Jev API         │
            │  (11 Model Families)  │                                           │ (Zero-Shot/Few-Shot)  │
            └───────────┬───────────┘                                           └───────────┬───────────┘
                        │                                                                   │
                        └─────────────────────────────────┬─────────────────────────────────┘
                                                          │
                                            ┌─────────────▼─────────────┐
                                            │ Dual Reporting Panels:    │
                                            │ 1. Raw Decision Rules     │
                                            │ 2. Policy-Tuned Thresholds│
                                            └───────────────────────────┘
```

---

## 3. Published Benchmark Results

**Primary Evaluation Metric:** **Balanced Accuracy (%)** (unweighted macro-average of per-class recall).
Values are reported as $\text{mean} \pm \text{sample standard deviation}$ across 3 training seeds ($n=3$) evaluated on the same fixed holdout test cases.

### 3.1 Threshold-Adjusted Balanced Accuracy (Policy-Calibrated)
Binary classification thresholds are selected on a separate, labeled `policy` partition (up to 500 rows) for both classical models and Jev. Multiclass tasks remain unadjusted.

| Dataset | Type | Classes | Test $N$ | Jev Zero-Shot* | Jev Few-Shot* | Best Classical Pipeline | Best Classical Score | $\Delta$ (Jev vs. Best ML) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| **AG News** | Text | 4 | 1,000 | 87.5 ± 0.0 | 86.3 ± 0.6 | Linear SVM | **88.4 ± 0.3** | -0.9 pp |
| **Banking77** | Text | 77 | 1,500 | 78.9 ± 0.0 | 81.9 ± 1.7 | Linear SVM | **89.7 ± 0.6** | -7.8 pp |
| **SMS Spam** | Text | 2 | 1,000 | 95.9 ± 0.7 | 95.8 ± 0.5 | Naive Bayes | **96.3 ± 0.6** | -0.4 pp |
| **IMDb** | Text | 2 | 1,000 | **96.1 ± 0.6** | 95.5 ± 0.8 | Logistic regression | 88.3 ± 0.2 | **+7.8 pp** |
| **Bank Marketing** | Tabular | 2 | 1,000 | 59.7 ± 1.6 | 59.0 ± 2.5 | Voting ensemble | **73.3 ± 0.3** | -13.6 pp |
| **Online Shoppers** | Tabular | 2 | 1,000 | 51.8 ± 0.1 | 53.6 ± 7.3 | Voting ensemble | **71.2 ± 1.4** | -17.6 pp |
| **Breast Cancer** | Tabular | 2 | 114 | 88.4 ± 2.2 | 92.7 ± 0.9 | Logistic regression | **100.0 ± 0.0** | -7.3 pp |
| **Iris** | Tabular | 3 | 30 | 97.0 ± 0.0 | 94.5 ± 4.8 | Multiple tied | **100.0 ± 0.0** | -3.0 pp |

*\*Important Disclosure:* Threshold-adjusted Jev uses labeled policy data to fit decision thresholds and is **not zero-shot end-to-end**. Few-shot prompts contain one labeled training example per class.

### 3.2 Raw Decision Rules (Default Thresholds)
Without policy threshold tuning, predictions use default model decision thresholds (0.5 probability or 0.0 decision margin). Both panels are published to prevent presenting only the favorable decision policy.

| Dataset | Type | Classes | Test $N$ | Jev Zero-Shot | Jev Few-Shot | Best Classical Pipeline | Best Classical Score | $\Delta$ (Jev vs. Best ML) |
|---|---|---:|---:|---:|---:|---|---:|---:|
| **AG News** | Text | 4 | 1,000 | 87.5 ± 0.0 | 86.3 ± 0.6 | Linear SVM | **88.4 ± 0.3** | -0.9 pp |
| **Banking77** | Text | 77 | 1,500 | 78.9 ± 0.0 | 81.9 ± 1.7 | Linear SVM | **89.7 ± 0.6** | -7.8 pp |
| **SMS Spam** | Text | 2 | 1,000 | **96.1 ± 0.0** | 95.6 ± 0.9 | Naive Bayes | 95.0 ± 1.9 | **+1.1 pp** |
| **IMDb** | Text | 2 | 1,000 | **96.3 ± 0.0** | 95.9 ± 0.5 | Logistic regression | 88.4 ± 0.2 | **+7.9 pp** |
| **Bank Marketing** | Tabular | 2 | 1,000 | 53.4 ± 0.0 | 55.3 ± 3.1 | SVM | **71.8 ± 0.7** | -16.5 pp |
| **Online Shoppers** | Tabular | 2 | 1,000 | 51.4 ± 0.0 | 54.7 ± 9.5 | SVM | **69.1 ± 0.3** | -14.4 pp |
| **Breast Cancer** | Tabular | 2 | 114 | 61.0 ± 0.0 | 88.8 ± 5.5 | SVM / k-NN tied | **100.0 ± 0.0** | -11.2 pp |
| **Iris** | Tabular | 3 | 30 | 97.0 ± 0.0 | 94.5 ± 4.8 | Multiple tied | **100.0 ± 0.0** | -3.0 pp |

---

## 4. Evaluated Datasets & Preprocessing

The suite evaluates 4 text classification datasets and 4 tabular classification datasets:

| Dataset | Modality | Target Task | Classes | Holdout Test $N$ | Preprocessing & Exclusions | Source |
|---|---|---|---:|---:|---|---|
| **AG News** | Text | Topic classification (World, Sports, Business, Sci/Tech) | 4 | 1,000 | Hard 4,000 char truncation, input deduplication | [HuggingFace](https://huggingface.co/datasets/fancyzhx/ag_news) |
| **Banking77** | Text | Fine-grained banking customer intent | 77 | 1,500 | Hard 4,000 char truncation, input deduplication | [PolyAI-LDN](https://github.com/PolyAI-LDN/task-specific-datasets) |
| **SMS Spam** | Text | Spam vs. legitimate SMS filtering | 2 | 1,000 | Tab-delimited parsing, input deduplication | [UCI ML (228)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| **IMDb** | Text | Movie review sentiment polarity | 2 | 1,000 | HTML tag removal, 4,000 char truncation | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/imdb) |
| **Bank Marketing** | Tabular | Term-deposit subscription prediction | 2 | 1,000 | **Excludes `duration`** (call duration is unavailable prior to call contact) | [UCI ML (222)](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Online Shoppers** | Tabular | E-commerce purchase session intention | 2 | 1,000 | **Excludes `PageValues`** (retrospective revenue-linked feature) | [UCI ML (468)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| **Breast Cancer** | Tabular | Malignant vs. benign mass diagnosis | 2 | 114 | Wisconsin diagnostic cell nuclei features | [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html) |
| **Iris** | Tabular | Iris botanical species classification | 3 | 30 | Sepal and petal dimension measurements | [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html) |

*Read the [Methodology Guide](docs/methodology.md) for full data pipeline specifications.*

---

## 5. Evaluated Models & Backend Architecture

The benchmark evaluates 11 classical model families and 2 Jev operational modes. Implementations are routed explicitly to GPU or CPU:

| Model Family | Feature Representation | Execution Backend | Model Search Space |
|---|---|---|---|
| **Logistic Regression** | Word + Char TF-IDF / Scaled Tabular | **cuML GPU** | $C \in \{0.3, 3.0\} \times \{\text{unweighted}, \text{balanced}\}$ |
| **SVM** | Linear TF-IDF / RBF Tabular | **cuML GPU** (Tabular RBF) / **sklearn CPU** (Text Linear) | $C \in \{0.3, 3.0\} \times \{\text{unweighted}, \text{balanced}\}$ |
| **Decision Tree** | Top 512 Chi-2 TF-IDF / Tabular | **scikit-learn CPU** | $(\text{depth}=10, \text{leaf}=3)$ vs. $(\text{depth}=\infty, \text{leaf}=10) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Random Forest** | Dense TF-IDF / Tabular | **cuML GPU** (Unweighted) / **sklearn CPU** (Balanced) | $(\text{depth}=16, \sqrt{p})$ vs. $(\text{depth}=24, 15\% p) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Extra Trees** | Top 512 Chi-2 TF-IDF / Tabular | **scikit-learn CPU** | $(\text{depth}=16, \sqrt{p})$ vs. $(\text{depth}=24, 15\% p) \times \{\text{unweighted}, \text{balanced}\}$ |
| **k-NN** | 32 SVD Components / Scaled Tabular | **cuML GPU** | $k \in \{5, 15, 31\} \times \{\text{distance}, \text{uniform}\}$ |
| **Naive Bayes** | Multinomial (Text) / Gaussian (Tabular) | **scikit-learn CPU** | $\alpha \in \{0.1, 1.0\}$ (Text) / $\text{var\_smoothing} \in \{10^{-9}, 10^{-6}\}$ (Tabular) |
| **Hist Gradient Boost**| 32 SVD Components / Tabular | **scikit-learn CPU** | $(\text{leaves}=15, \eta=0.05)$ vs. $(\text{leaves}=31, \eta=0.1) \times \{\text{unweighted}, \text{balanced}\}$ |
| **XGBoost** | Top 512 Chi-2 TF-IDF / Tabular | **CUDA Native** | $(\text{depth}=3, \eta=0.05)$ vs. $(\text{depth}=6, \eta=0.1) \times \{\text{unweighted}, \text{balanced}\}$ |
| **CatBoost** | Native Categoricals / Dense TF-IDF | **CUDA Native** | $(\text{depth}=4, \eta=0.05)$ vs. $(\text{depth}=7, \eta=0.1) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Voting Ensemble** | Probability Average | **CPU Aggregation** | Soft voting of fitted Logistic Regression, Random Forest, and XGBoost |
| **Majority Baseline** | Untuned Prior | **scikit-learn CPU** | Constant majority-class prediction |
| **Jev Zero-Shot** | Semantic Instruction Prompt | **TypeSafe API** (`jev-1.13.0`) | Task and class semantic descriptions only |
| **Jev Few-Shot** | Semantic Prompt + 1 Example/Class | **TypeSafe API** (`jev-1.13.0`) | 1 training instance per class (2 for binary; 77 for Banking77) |

*Read the [GPU Architecture Guide](docs/gpu-architecture.md) and [API Guide](docs/api.md) for execution details.*

---

## 6. Experimental Controls & Leakage Protections

The benchmark enforces strict isolation across every phase of execution:

1. **Test Set Isolation:** The `test` partition is held out prior to all experimentation. It is never accessed during feature selection, vectorizer fitting, hyperparameter selection, or decision threshold calibration.
2. **Preprocessing Isolation:** Feature selectors (`TfidfVectorizer`, `SelectKBest(chi2)`, `TruncatedSVD`, `StandardScaler`, `OneHotEncoder`) are fit strictly on `train` during hyperparameter selection, and refitted on `train + validation` for final model fitting.
3. **Policy Partition Isolation:** Binary decision thresholds are tuned exclusively on predictions from the `policy` split and applied post-refit to `test`.
4. **Pilot Test Exclusion:** All test indices used during exploratory pilot runs (seeds `42, 43, 44`) are explicitly excluded from the holdout test set (`exclude_pilot_tests=True`).
5. **Proportional Capping Algorithm:** `cap_indices()` uses an exact integer quota balancing algorithm to guarantee that all classes are represented in capped subsets without violating scikit-learn stratification boundaries.
6. **Multi-GPU Subprocess Isolation (Protocol 3.0.1):** All GPU operations execute inside isolated child processes spawned via `subprocess.Popen` with `CUDA_VISIBLE_DEVICES` configured before library imports, preventing CUDA primary context contamination across devices.
7. **Zero Survivor Bias:** Failed API calls are assigned `prediction = -1` and scored as incorrect. Rows are never omitted from test evaluation.

---

## 7. Reproducing the Benchmark

### 7.1 Running on Kaggle (Two NVIDIA T4 GPUs)
1. Upload [`jev_benchmark_v3.ipynb`](jev_benchmark_v3.ipynb) to a new Kaggle session.
2. Under **Settings**, select **GPU T4 x 2**, enable **Internet**, and select **Always use latest environment**.
3. In **Add-ons → Secrets**, add your `TYPESAFE_API_KEY` secret.
4. Execute cells sequentially. The notebook self-extracts its bundled Python source files and executes preflight GPU probes before training.

### 7.2 Local Artifact & Code Verification (CPU)
To verify published results, checksums, and package consistency locally in a single command without requiring an NVIDIA GPU:
```bash
pip install -r requirements-dev.txt

# Run unified offline validation suite
python validate_all.py
```

Individual component validation scripts are also available:
* `python validate_v3_artifact.py`: Verifies executed notebook outputs and SHA-256 package hashes.
* `python validate_v3_backends.py`: Tests backend estimator candidate generation with constructor doubles.
* `python validate_gpu_process.py`: Tests child subprocess isolation, device masking, and process reaping.
* `python validate_sampling.py`: Tests proportional capping and stratified split regression.

### 7.3 Regenerating Artifacts & Documentation
* **Export published CSVs from notebook:** `python export_published_results.py`
* **Rebuild GitHub Pages report:** `python build_site.py`
* **Build fresh unexecuted notebook from source:** `python build_modular_notebook.py --v3`

*Read the [Reproducibility Guide](docs/reproducibility.md) and [Contributing Guide](CONTRIBUTING.md) for full instructions.*

---

## 8. Limitations & Scope of Evidence

* **Sample Standard Deviation vs. Confidence Intervals:** Reported $\pm$ values reflect training seed variation on a single shared test set, **not independent test replications or confidence intervals**.
* **Cached Zero-Shot Replications:** In zero-shot mode, Jev prompts are identical across seeds. Exact responses are cached on disk; zero standard deviation reflects deterministic caching, not API repeatability.
* **Banking77 API Warnings:** Banking77 logs warnings regarding predictions outside the valid class label set. Failed API responses are encoded as `-1` and counted as errors.
* **Small Sample Sizes:** Iris ($N=30$) and Breast Cancer ($N=114$) holdouts are small and subject to wide binomial sampling variance.
* **Bounded Compute Budget:** Classical ML models are evaluated over a bounded 4-candidate search space (150 trees max). This is a declared baseline comparison, not an exhaustive search for the theoretical upper bound of machine learning.
* **Modified Feature Sets:** Bank Marketing excludes `duration` and Online Shoppers excludes `PageValues`. Scores cannot be directly compared against academic papers using unpruned feature sets.
* **Pretraining Exposure:** Potential exposure of public academic datasets in commercial foundation model pretraining corpora cannot be independently verified or ruled out.

*Read the [Complete Limitations Document](docs/limitations.md) for detailed scientific caveats.*

---

## 9. Repository Structure

```text
Jev-vs-ML/
├── README.md                      # Main research overview & canonical results
├── ARCHITECTURE.md                # Detailed system architecture & codebase map
├── BENCHMARK_V3.md                # Protocol 3.0.1 (V3) specification & budget overrides
├── BENCHMARK_V2.md                # Common Protocol 2.0.0 specification
├── BENCHMARK_AUDIT.md             # Formal 16-section technical audit report
├── VALIDATION_V3.md               # V3 execution & verification notes
├── jev_benchmark_v3.ipynb         # Executed master Kaggle notebook with saved outputs
├── jev_benchmark_v3_bundle.zip    # Standalone release bundle (notebook + source)
│
├── jevbench/                      # Core modular Python benchmark package
│   ├── api.py                     # TypeSafe API client, rate limiting, and caching
│   ├── backends.py                # cuML and CUDA explicit routing matrix
│   ├── common.py                  # Shared utilities, digest helpers, and device probes
│   ├── config.py                  # Protocol configuration and budget definitions
│   ├── datasets.py                # Dataset ingestion, snapshots, and stratified capping
│   ├── decisions.py               # Decision threshold calibration & metrics
│   ├── features.py                # Fitted NLP & tabular feature extractors
│   ├── gpu_process.py             # Subprocess GPU isolation and lifecycle management
│   ├── models.py                  # Candidate estimator specifications
│   ├── reporting.py               # Result rendering, CSV matrices, paired intervals
│   ├── runner.py                  # Orchestration for ML and Jev evaluation
│   └── training.py                # Validation search, candidate refit, and evaluation
│
├── docs/                          # GitHub Pages interactive report & deep-dive docs
│   ├── index.html                 # Interactive benchmark report
│   ├── methodology.md             # Detailed data flow, capping & calibration math
│   ├── gpu-architecture.md        # GPU process isolation & probe barrier details
│   ├── api.md                     # Jev API networking, rate limits, and budget guards
│   ├── reproducibility.md         # Full Kaggle and local reproduction guide
│   ├── limitations.md             # Comprehensive scientific caveats and bounds
│   └── assets/                    # Styling, scripts, and release infographics
│
├── published_results/             # Verified canonical benchmark artifacts
│   ├── manifest.json              # Provenance manifest & notebook SHA-256 digest
│   ├── raw_balanced_accuracy.csv  # Raw decision rules matrix
│   └── adjusted_balanced_accuracy.csv # Policy-calibrated decision matrix
│
├── validate_v3_artifact.py        # Offline artifact and hash verification script
├── validate_v3_backends.py        # Backend routing and candidate double tests
├── validate_gpu_process.py        # Subprocess device isolation tests
├── validate_sampling.py           # Proportional capping regression tests
├── build_modular_notebook.py      # Standalone notebook generator
├── build_site.py                  # Static site generator for GitHub Pages
└── export_published_results.py    # Direct CSV table exporter
```

---

## 10. Citation & License

If you reference or build upon this benchmark, please cite this repository:

```bibtex
@misc{mugenkyou2026jevvsml,
  author = {mugenkyou},
  title = {Jev vs. Classical Machine Learning: An Eight-Dataset Empirical Benchmark},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/mugenkyou/JEV-VS-ML}},
  note = {Protocol 3.0.1 / V3}
}
```

This repository is licensed under the [MIT License](LICENSE).

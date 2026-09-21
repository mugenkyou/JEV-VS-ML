# Benchmark Methodology & Experimental Design

This document details the experimental methodology, data partitioning, feature representations, search spaces, and decision-threshold calibration procedures used in **Protocol 3.0.1 (V3)** of the Jev vs. Machine Learning benchmark.

---

## 1. Experimental Pipeline Overview

The benchmark follows a strict four-partition architecture designed to prevent data leakage across feature extraction, hyperparameter tuning, model refitting, threshold calibration, and holdout evaluation.

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

## 2. Data Ingestion, Deduplication, and Snapshotting

Every dataset is downloaded, validated, and frozen into a local Apache Parquet snapshot with a cryptographic SHA-256 digest:

1. **Text Normalization & Truncation:**
   * Text fields have HTML line breaks (`<br />`) replaced with whitespace.
   * Text strings are truncated to a hard limit of `4,000` characters (`cfg['max_text_chars'] = 4000`).
2. **Deduplication on Model Inputs:**
   * Duplicate input rows (after text truncation or tabular feature conversion) are identified across the entire dataset.
   * **Conflicting Labels:** If identical input features map to different target labels, all instances of that conflicting input are completely removed.
   * **Official Split Precedence:** If an identical input occurs in both official train and test splits, the official test copy is retained.
3. **Immutability:**
   * The deduplicated dataset is written to `data/<dataset_slug>/snapshot.parquet` alongside `metadata.json`. If the on-disk SHA-256 digest changes during execution, the runner halts immediately.

---

## 3. Partitioning & Leakage Protections

### 3.1 Four-Way Partition Architecture

For each dataset, data is partitioned into four strictly disjoint index subsets:

| Partition | Split Share | V3 Cap | Role & Allowed Operations |
|---|---|---:|---|
| **`train`** | 60% of dev pool | 8,000 rows | Used for initial candidate model fitting and preliminary feature transformation fitting. Jev few-shot examples are sampled from here. |
| **`validation`** | 20% of dev pool | 1,000 rows | Used strictly for hyperparameter selection and candidate model comparison. |
| **`policy`** | 20% of dev pool | 500 rows | Used strictly for post-refit binary threshold calibration. Never used for fitting weights. |
| **`test`** | Fixed holdout | 1,000 (1,500 for Banking77) | Completely untouched holdout. Evaluated only once per refitted model. |

### 3.2 Pilot Test Set Exclusion

To prevent historical pilot experiments from contaminating final benchmark scores:
* The test indices from exploratory pilot runs (reconstructed with seeds `42`, `43`, `44` and test cap `300`) are explicitly identified via `pilot_split()`.
* The union of all pilot test indices is computed and completely excluded from the final holdout test set (`exclude_pilot_tests=True` in `make_holdout()`).

### 3.3 Proportional Capping Algorithm (`cap_indices`)

Standard stratified sampling in scikit-learn (`train_test_split(stratify=y)`) fails when the number of discarded samples is smaller than the number of classes. For instance, in Banking77 (77 classes), reducing 8,002 development rows to an 8,000-row cap leaves a remainder of 2 rows, causing scikit-learn to throw an exception because the least populated class in the remainder has fewer than 2 instances.

To resolve this, `cap_indices()` implements an exact proportional allocation algorithm:
1. Calculates target sample count per class: $\text{target}_c = N_c \times (\text{cap} / N_{\text{total}})$.
2. Initializes integer quotas with $\text{quota}_c = \max(1, \lfloor \text{target}_c \rfloor)$.
3. Incrementally adjusts quotas using deficit and excess balancing until $\sum \text{quota}_c = \text{cap}$, ensuring every class has at least 1 representative sample.
4. Samples indices uniformly without replacement within each class quota and permutes the final selection using the specified seed.

---

## 4. Feature Extraction Pipelines

Feature transformations are encapsulated in the `Features` class (`jevbench/features.py`) and are fit exclusively on training rows:

### 4.1 NLP / Text Pipelines
* **Linear Representation (`linear`):** Concatenation of Word TF-IDF (1–2 n-grams, sublinear TF, max 20,000 features) and Character n-grams (`char_wb`, 3–5 n-grams, sublinear TF, max 10,000 features). Used for Logistic Regression and Linear SVM.
* **Tree Representation (`tree` / `dense`):** Word TF-IDF filtered by supervised Chi-Square feature selection ($\chi^2$) to top 512 features (`k=512`). Used for Decision Trees, Random Forests, Extra Trees, XGBoost, and CatBoost.
* **SVD Representation (`knn`):** Word TF-IDF reduced via TruncatedSVD to 32 components and scaled via `StandardScaler`. Used for k-NN and Histogram Gradient Boosting.

### 4.2 Tabular Pipelines
* **Standard Representation (`dense`):** Numeric columns undergo median imputation (`SimpleImputer(strategy='median')`) and standard scaling (`StandardScaler`). Categorical columns undergo most-frequent imputation and one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
* **Native Categorical Representation (`cat`):** Passed directly to CatBoost with explicit categorical column indices.

---

## 5. Model Candidate Search & Refitting Protocol

For each classical family, four predeclared candidate pipelines are evaluated:

| Model Family | Search Space (2 Parameter Grid $\times$ 2 Weighting Schemes) |
|---|---|
| **Logistic Regression** | $C \in \{0.3, 3.0\} \times \{\text{unweighted}, \text{balanced sample weights}\}$ |
| **SVM** | $C \in \{0.3, 3.0\} \times \{\text{unweighted}, \text{balanced sample weights}\}$ (Linear for text; RBF for tabular) |
| **Decision Tree** | $(\text{depth}=10, \text{min\_samples\_leaf}=3)$ vs. $(\text{unrestricted depth}, \text{min\_samples\_leaf}=10) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Random Forest** | $(\text{depth}=16, \text{min\_leaf}=2, \sqrt{p}\text{ features})$ vs. $(\text{depth}=24, \text{min\_leaf}=1, 15\%\text{ features}) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Extra Trees** | $(\text{depth}=16, \text{min\_leaf}=2, \sqrt{p}\text{ features})$ vs. $(\text{depth}=24, \text{min\_leaf}=1, 15\%\text{ features}) \times \{\text{unweighted}, \text{balanced}\}$ |
| **k-NN** | 4 neighbor configurations: $(k=5, \text{distance})$, $(k=15, \text{distance})$, $(k=31, \text{distance})$, $(k=15, \text{uniform})$ |
| **Naive Bayes** | Text: Multinomial NB ($\alpha \in \{0.1, 1.0\}$); Tabular: Gaussian NB ($\text{var\_smoothing} \in \{10^{-9}, 10^{-6}\}$) |
| **Hist Gradient Boosting** | $(\text{leaves}=15, \eta=0.05, L_2=1)$ vs. $(\text{leaves}=31, \eta=0.1, L_2=5) \times \{\text{unweighted}, \text{balanced}\}$ |
| **XGBoost** | $(\text{depth}=3, \eta=0.05, \lambda=3)$ vs. $(\text{depth}=6, \eta=0.1, \lambda=5) \times \{\text{unweighted}, \text{balanced}\}$ |
| **CatBoost** | $(\text{depth}=4, \eta=0.05, L_2=3)$ vs. $(\text{depth}=7, \eta=0.1, L_2=5) \times \{\text{unweighted}, \text{balanced}\}$ |
| **Voting Ensemble** | Equal-weight soft voting averaging predicted class probabilities from fitted Logistic Regression, Random Forest, and XGBoost |

### Selection and Refit Lifecycle
1. **Candidate Evaluation:** Each candidate is fit on `train` and evaluated on `validation`. Balanced accuracy determines the winning candidate.
2. **Refit on Train + Validation:** The winning hyperparameter configuration is refitted on the combined `train + validation` partition using a freshly fitted feature pipeline (`final_features`).
3. **Fixed Iterations:** For boosted models (XGBoost, CatBoost, HistGradientBoosting), the optimal number of iterations determined during validation early stopping is fixed during refit.

---

## 6. Decision Threshold Calibration (`decisions.py`)

In binary classification tasks with class imbalance, default decision thresholds (0.5 probability or 0.0 margin) frequently result in models predicting the majority class exclusively.

To address this without leaking test labels:
1. Candidate models produce raw continuous decision scores $s \in \mathbb{R}$ on the independent `policy` partition.
2. A grid of 101 candidate thresholds is constructed from empirical score quantiles:
   $$\mathcal{T} = \{t_{\text{default}}\} \cup \{\text{Quantile}(s_{\text{policy}}, q) \mid q \in [0, 1]\} \cup \{\max(s_{\text{policy}}) + \epsilon\}$$
3. The optimal threshold $\tau^*$ is selected to maximize balanced accuracy on `policy`:
   $$\tau^* = \arg\max_{\tau \in \mathcal{T}} \text{BalancedAccuracy}(y_{\text{policy}}, s_{\text{policy}} \ge \tau)$$
4. Ties are broken by choosing the threshold closest to the default decision threshold.
5. The selected threshold $\tau^*$ is applied to the holdout `test` predictions. Multiclass tasks remain unadjusted.

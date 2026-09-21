# Limitations & Scientific Scope

This document details the critical scientific boundaries, data constraints, hardware assumptions, and statistical caveats of the benchmark.

---

## 1. Statistical Interpretation & Uncertainty Bounds

1. **Sample Standard Deviation vs. Confidence Intervals:**
   * Results are reported as $\text{mean} \pm \text{sample standard deviation}$ across three training seeds (`2027`, `2028`, `2029`) evaluated on a **single shared holdout test set**.
   * These sample standard deviations measure variance in model training and subset sampling; **they are not confidence intervals** and do not represent three independent test populations.
2. **Zero-Shot Request Caching:**
   * In zero-shot mode, Jev receives identical prompt strings across all three seeds. Because exact requests are cached on disk, zero-shot predictions are identical across seeds ($SD = 0.0$). This reflects deterministic caching, **not independent API repeatability**.
3. **Absence of Significance Claims:**
   * No claims of formal statistical significance or hypothesis rejection are made from the displayed means.

---

## 2. Dataset Constraints & Preprocessing Exclusions

1. **Conservative Feature Exclusions:**
   * **Bank Marketing:** The `duration` attribute (call duration) is deliberately removed. Call duration is unknown prior to placing a call, making it unusable in realistic deployment.
   * **Online Shoppers:** The `PageValues` attribute is deliberately excluded as it is computed retrospectively based on revenue outcomes.
   * *Impact:* These modifications reflect realistic pre-event forecasting, but mean these results cannot be directly compared against academic papers that evaluate unpruned versions of these UCI datasets.
2. **Small Holdout Sizes:**
   * **Iris:** Contains only 30 held-out test rows (10 per class).
   * **Breast Cancer:** Contains 114 held-out test rows.
   * *Impact:* High scores on these small samples (e.g. 100.0% classical accuracy) have wide binomial sampling variance and should be interpreted as toy illustrations rather than definitive superiority.
3. **Pretraining Data Exposure:**
   * Jev 1.13.0 is a pretrained commercial language model. While public benchmarks were tokenized as unseen inputs, potential exposure of public datasets (e.g. IMDb, AG News, Iris) in foundation model pretraining corpora cannot be independently verified or ruled out.

---

## 3. Computational Budget & Search Space Bounds

1. **Bounded Classical ML Search:**
   * Classical models are evaluated over a small, predeclared grid of **four candidate configurations per family** (two hyperparameter combinations crossed with ordinary/balanced sample weights).
   * Tree counts are capped at 150 (Random Forest / Extra Trees) and 60 iterations (HistGradientBoosting).
   * *Impact:* This benchmark represents a practical, bounded-budget baseline comparison, **not an exhaustive or best-possible machine learning optimization**.
2. **Representation Trade-offs:**
   * Text tree models (Random Forest, XGBoost, CatBoost) evaluate 512 Chi-Square-selected TF-IDF features, whereas linear models evaluate up to 30,000 word and character n-grams.
   * SVD-based models (k-NN, HistGradientBoosting) use 32 latent components.
3. **Absence of Pretrained Transformer Baselines:**
   * The benchmark explicitly compares Jev against classical linear, tree-based, and distance-based estimators. It does not evaluate fine-tuned modern transformer models (e.g. DeBERTa-v3, RoBERTa) or other commercial LLM endpoints.

---

## 4. API Error Handling & Reliability

1. **Banking77 Intent Warnings:**
   * During the published Banking77 run, warnings were logged regarding predictions outside the valid label set.
   * The Jev adapter assigns `prediction = -1` to failed requests and scores them as incorrect. Without detailed per-request telemetry, API network failures cannot be fully disentangled from intrinsic model classification errors on 77-class intent queries.
2. **Cost Estimation:**
   * The recorded estimated expenditure ($\approx \$4.19$ for 38,922 requests) is an internal planning estimate based on UTF-8 payload byte heuristics, **not an official invoice or total billed charge**.

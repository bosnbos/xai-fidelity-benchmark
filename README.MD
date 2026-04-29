# Transavia Pega ADM Analysis

Analysis of Pega Adaptive Decision Manager (ADM) propensity models for Transavia. The main focus is **luggage add-on offers (L5B15 and CLUG variants) in the Email/Outbound channel**.

## Notebooks

**01 is independent.** Notebooks 03–05 all require 02 to run first. Run in order.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning: offer viability, feature importance, time-series comparison | Nothing |
| `02_data_ingestion.ipynb` | Parse raw decision JSON exports → filter to Email/Outbound/Luggage/Sales → save parquet | JSON files in `data/decisions/` |
| `03_eda.ipynb` | EDA on filtered decision data: temporal distribution, channel overview, offering breakdown, missingness, target distribution, model version churn | `02` |
| `04_surrogate.ipynb` | CatBoost surrogate + Naive Bayes baseline: depth CV, fidelity suite (R², RMSE, Spearman ρ, Kendall τ, KS), feature importances, model version analysis | `02` |
| `05_explanation.ipynb` | Three explanation methods on the CatBoost surrogate: Decision Tree (§9), SHAP (§10), LIME (§11), cross-method rank comparison (§12) | `02`, `04` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It automatically picks up all files in that folder, deduplicates on `modelExecutionID`, and overwrites the parquet. Notebooks 03–05 will then reflect the updated dataset.

## Data

All raw files are gitignored — add them locally in the correct subfolder:

| Location | Contents | Used by |
|---|---|---|
| `data/decisions/` | Pega decision JSON exports (all channels, all offerings) | Notebook 02 |
| `data/adm/` | ADM model snapshots + predictor binning JSON | Notebook 01 |
| `data/processed/` | `luggage_email_outbound.parquet` — output of notebook 02 | Notebooks 03–05 |
| `data/artifacts/{VARIANT}/` | Trained model + explanation artifacts output by notebooks 04–05 | Notebooks 05+ |

Artifact files written per variant (e.g. `data/artifacts/L5B15/`): `catboost_model.cbm`, `feature_cols.json`, `cat_cols.json`, `num_cols.json`, `train_idx.npy`, `test_idx.npy`, `fidelity.json`, `dt_model.pkl`, `dt_rules.txt`, `shap_values.npy`, `shap_importances.json`, `lime_explanations.parquet`, `lime_importances.json`, `cross_method_comparison.json`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -e .
jupyter lab
```

`pip install -e .` is a one-time step that installs the local `src/my_project` package so all modules are importable across notebooks.

## Project structure

```
├── data/
│   ├── adm/                                 # ADM snapshot files (gitignored)
│   ├── decisions/                           # Raw decision JSON exports (gitignored)
│   ├── processed/                           # Parquet output of notebook 02 (gitignored)
│   └── artifacts/                           # Trained models + explanation outputs (gitignored)
│       └── {VARIANT}/                       # One folder per variant, e.g. L5B15/
├── notebooks/
│   ├── 01_adm_analysis.ipynb                # ADM snapshots & predictor binning
│   ├── 02_data_ingestion.ipynb              # Raw JSON → filtered parquet
│   ├── 03_eda.ipynb                         # EDA on Email/Outbound decisions
│   ├── 04_surrogate.ipynb                   # CatBoost surrogate + fidelity suite
│   ├── 05_explanation.ipynb                 # DT / SHAP / LIME + cross-method comparison
│   └── archive/
│       └── exploration_original.ipynb
├── src/
│   └── my_project/
│       ├── parsing.py                       # Pega JSON parsing utilities
│       ├── features.py                      # Per-variant feature configs (VARIANT_FEATURES)
│       ├── metrics.py                       # Fidelity and stability metrics
│       ├── surrogate.py                     # CatBoost surrogate + NB baseline
│       └── explanation.py                   # DT surrogate, SHAP, LIME utilities
├── pyproject.toml
└── README.MD
```

## `src/my_project/`

An installable Python package. Five modules:

**`parsing.py`** — cracks open the nested Pega JSON. Key exports:
- `extract_l5b15_rows(df, target_names=None)` — extracts flat scoring-event rows from a raw Polars DataFrame; defaults to all entries in `LUGGAGE_VARIANTS`.
- `LUGGAGE_VARIANTS` — frozenset of luggage variant names (`L5B15`, `L5B20`, `L5B25`, `L5B30`, `L5B40`, `L5B50`, `CLUG`).
- `parse_common_inputs(record)` — unpacks the `pxCommonInputs` blob into a flat dict of customer/flight context features.

**`features.py`** — per-variant feature configuration:
- `VARIANT_FEATURES` — dict keyed by variant name (e.g. `"L5B15"`, `"CLUG"`), each entry a `VariantConfig(features, numeric)` dataclass. `features` is the ordered tuple of active Pega predictors; `numeric` is the frozenset of those Pega treats as numeric.
- `EXCLUDE_COLS` — columns always dropped before modeling: identifiers, partition constants (pyName, pyChannel, etc.), and all PII (emails, names, date of birth, loyalty numbers, booking references).
- `TARGET` — the propensity column name (`"propensity"`).

Usage:
```python
from my_project.features import VARIANT_FEATURES
cfg = VARIANT_FEATURES["L5B15"]
# cfg.features  → tuple of active predictor names
# cfg.numeric   → frozenset of numeric-typed predictors
```

**`metrics.py`** — fidelity and stability metrics:
- `fidelity_suite(y_true, y_pred)` → dict with R², RMSE, Spearman ρ, Kendall τ, KS statistic.
- `feature_ranking(importances)` → 1-based rank Series (rank 1 = most important).
- `stability_spearman(rank_a, rank_b)` → Spearman ρ between two rank vectors.
- `jaccard_at_k(rank_a, rank_b, k)` → Jaccard similarity of top-k feature sets.
- `psi(scores_a, scores_b, bins=10)` → `(psi_value, label)` where label ∈ `{"stable", "moderate", "significant"}`.

**`surrogate.py`** — model building and evaluation:
- `build_feature_matrix(df, pega_features, numeric_features)` → `(X, y, cat_cols, num_cols)`. Categorical columns get string dtype with `"__MISSING__"` for NaN; numeric columns get float with `np.nan` preserved (CatBoost handles both natively).
- `train_catboost(X_train, y_train, cat_cols, **hyperparams)` → fitted `CatBoostRegressor`.
- `NaiveBayesBaseline` — GaussianNB with quantile-binned propensity target; `fit(X_train, y_train, cat_cols, num_cols)` / `predict(X)`. Used as fidelity lower-bound robustness check.
- `evaluate_surrogate(model, X_test, y_test, model_name)` → fidelity dict.

**`explanation.py`** — three explanation methods:
- `dt_surrogate(X, y, cat_cols, num_cols, max_depth_range, ...)` → `(tree, best_depth, cv_df, encoder)`. Fits a Decision Tree via 5-fold CV Spearman ρ depth selection; uses `OrdinalEncoder` for sklearn compatibility.
- `dt_importances(tree, feature_names)` → ranked importance Series.
- `dt_rules(tree, feature_names, max_depth)` → text representation of split rules.
- `shap_importances(model, X, cat_cols)` → `(shap_values array, mean_abs Series)` via `shap.TreeExplainer`.
- `lime_explain_batch(model, X_train, X_test, cat_cols, num_cols, n_samples, n_features)` → tidy DataFrame with columns `[instance_idx, feature, importance]`.
- `aggregate_lime_importances(lime_df)` → global mean |coefficient| Series after stripping LIME condition labels back to base feature names.

---

## Research Design

### Context

Pega ADM scores customers using a **Naive Bayes** model and outputs a continuous propensity $\hat{p} \in [0,1]$. The model is treated as a black box: we have its inputs (the active predictors in `features.py`) and its outputs (the stored `propensity` column), but not its internal parameters. The goal is to explain *which features drive $\hat{p}$, by how much, and how stably* — without modifying or retraining Pega's model.

### Research questions

| | Question |
|---|---|
| **RQ1** | To what extent do SHAP, LIME, and rule-based surrogate models accurately approximate the predictions of the black-box propensity model, and does fidelity vary across explanation families? |
| **RQ2** | How stable are the feature attribution rankings produced by each explanation method across temporal splits and route subgroups? |
| **RQ3** | To what degree does observed explanation instability reflect genuine model drift rather than sensitivity of the explanation method to distributional shifts in the evaluation data? |

### XAI methods

**Global surrogate (Decision Tree)**  
A shallow regression tree is fitted directly on surrogate predictions $g(X_i)$. Decision rules are extracted as human-readable if/then conditions. Depth is selected by 5-fold CV on Spearman ρ.

**SHAP**  
A CatBoost surrogate is trained on $\hat{p}$, then `shap.TreeExplainer` computes exact, additive attributions analytically from the tree structure. The completeness axiom guarantees SHAP values sum to the model output.

**LIME**  
LIME perturbs each instance and queries the CatBoost surrogate (never Pega's live API). This is valid because the explanand is already Pega's output $\hat{p}$, and the surrogate fidelity quantifies the additional approximation error. LIME produces local linear explanations per instance; results are aggregated to global importances via mean |coefficient|.

### Surrogate model choice: why CatBoost

1. **TreeSHAP requires a tree-based model.** For a Naive Bayes surrogate, KernelSHAP would be required — a ~100× slower Monte Carlo approximation that stacks its own error on top of the surrogate approximation error.
2. **Fidelity, not architectural mimicry.** A new Naive Bayes trained only on Pega's output scores is categorically different from the original. What matters is how well the surrogate replicates Pega's input-output mapping (measured by R²); a more flexible model will do this better than one constrained by conditional independence.
3. **Native categorical handling.** Of the 23 active predictors for L5B15, the majority are categorical. CatBoost handles them via ordered target encoding natively, removing encoding choices as confounds. It also handles missing values natively, which matters for sparse IH features.

A Naive Bayes baseline (`NaiveBayesBaseline`) is included as a robustness check.

### Fidelity, stability, and drift metrics

| RQ | Metric |
|---|---|
| **RQ1 fidelity** | Global R² / RMSE / Spearman ρ / Kendall τ / KS (surrogate vs $\hat{p}$); SHAP completeness exact by axiom |
| **RQ2 stability** | Spearman rank correlation of importance vectors across splits; Jaccard@k (k = 5, 10) |
| **RQ3 drift vs sensitivity** | KS + PSI on $\hat{p}$ and key features across splits; regress explanation shift on data shift to quantify per-method sensitivity |

### Evaluation splits

- **Temporal:** early window vs late window within the data range
- **Route subgroups:** top routes by volume (`DepartureAirport` × `DestinationAirport`)

### Active predictors

L5B15 has 23 active predictors; CLUG has 22. Both are defined in `src/my_project/features.py` under `VARIANT_FEATURES`. They span four namespaces: booking context (`CustBookedFlight.BookingData.*`), flight data (`CustBookedFlight.FlightData.*`), interaction history (`IH.*`), and Pega parameters (`param::Param.*`).

---

## Stack

- **Polars** — fast DataFrame operations on large JSON files
- **pandas** — feature matrices, analysis DataFrames
- **CatBoost** — surrogate model training with native categorical and missing-value support
- **scikit-learn** — train/test split, Decision Tree, OrdinalEncoder, GaussianNB
- **SHAP** — exact TreeSHAP attributions for the CatBoost surrogate
- **LIME** — local linear explanations (model-agnostic)
- **scipy** — Spearman ρ, Kendall τ, KS test
- **Matplotlib** — static visualisations

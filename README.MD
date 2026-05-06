# Transavia Pega ADM Analysis

Analysis of Pega Adaptive Decision Manager (ADM) propensity models for Transavia. Focus is on **luggage add-on offers in the Email/Outbound channel**, with replication across variants: L5B15, CLUG, BookingDotCom, Cartrawler.

## Research Questions

| | Question |
|---|---|
| **RQ1** | How accurately do SHAP, LIME, and Decision Tree surrogates approximate the black-box propensity model? |
| **RQ2** | How stable are feature attribution rankings across temporal splits and route subgroups? |
| **RQ3** | Does observed instability reflect data drift (δ_d), model churn (δ_m), or explainer sensitivity (δ_e)? |

---

## Notebooks

Run in order. **01 and 02b are independent.** Notebooks 03–07 all require 02 first.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning | Nothing |
| `02_data_ingestion.ipynb` | Raw decision JSON → filtered parquet (L5B15 / CLUG) | `data/decisions/` |
| `02b_data_ingestion_thirdparty.ipynb` | Raw decision JSON → filtered parquet (BookingDotCom / Cartrawler) | `data/decisions/` |
| `03_eda.ipynb` | EDA: temporal distribution, channel overview, missingness, model version churn | `02` |
| `04_surrogate.ipynb` | CatBoost surrogate + Naive Bayes baseline: fidelity suite (R², RMSE, Spearman ρ, Kendall τ, KS) | `02` |
| `05_explanation.ipynb` | DT (§9), SHAP (§10), LIME (§11), cross-method rank comparison (§12) | `02`, `04` |
| `06_stability.ipynb` | Stability across temporal and route splits: Spearman ρ, Jaccard@k; within-route DT bootstrap | `02`, `04`, `05` |
| `07_attribution.ipynb` | Instability decomposition into δ_d, δ_m, δ_e; PSI bin sensitivity analysis | `02`, `04`, `05`, `06` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It picks up all files in that folder, deduplicates on `modelExecutionID`, and overwrites the parquet.

---

## Data

All raw and derived files are gitignored — add them locally.

| Location | Contents |
|---|---|
| `data/decisions/` | Raw Pega decision JSON exports |
| `data/adm/` | ADM model snapshots + predictor binning JSON |
| `data/processed/` | Parquet outputs of notebooks 02 / 02b |
| `data/artifacts/{VARIANT}/` | Trained model + explanation artifacts (one folder per variant) |

**Processed files** (`data/processed/`):  
`luggage_email_outbound.parquet`, `l5b15_decisions.parquet`, `l5b15_email_outbound.parquet`, `thirdparty_email_outbound.parquet`

**Artifact files per variant** (`data/artifacts/l5b15/`, `CLUG/`, `BookingDotCom/`, `Cartrawler/`):  
`catboost_model.cbm`, `feature_cols.json`, `cat_cols.json`, `num_cols.json`, `train_idx.npy`, `test_idx.npy`, `fidelity.json`, `dt_model.pkl`, `dt_rules.txt`, `shap_values.npy`, `shap_importances.json`, `lime_explanations.parquet`, `lime_importances.json`, `cross_method_comparison.json`

**Summary outputs** (written to `data/artifacts/l5b15/` by notebooks 06–07):  
`stability_summary.json`, `attribution_summary.json`

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -e .
jupyter lab
```

`pip install -e .` installs the local `src/my_project` package so all modules are importable across notebooks.

---

## Project Structure

```
├── data/                                    # gitignored
│   ├── adm/                                 # ADM snapshot files
│   ├── decisions/                           # Raw decision JSON exports
│   ├── processed/                           # Parquet outputs of 02 / 02b
│   └── artifacts/                           # One folder per variant
│       └── {VARIANT}/
├── notebooks/
│   ├── 01_adm_analysis.ipynb
│   ├── 02_data_ingestion.ipynb
│   ├── 02b_data_ingestion_thirdparty.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_surrogate.ipynb
│   ├── 05_explanation.ipynb
│   ├── 06_stability.ipynb
│   └── 07_attribution.ipynb
├── src/
│   └── my_project/
│       ├── parsing.py                       # Pega JSON parsing
│       ├── features.py                      # Per-variant feature configs
│       ├── metrics.py                       # Fidelity, stability, PSI metrics
│       ├── surrogate.py                     # CatBoost surrogate + NB baseline
│       └── explanation.py                   # DT, SHAP, LIME utilities
├── pyproject.toml
└── README.MD
```

---

## `src/my_project/`

**`parsing.py`**
- `extract_l5b15_rows(df, target_names=None)` — flat scoring-event rows from raw Polars DataFrame
- `LUGGAGE_VARIANTS` — frozenset of luggage variant names
- `parse_common_inputs(record)` — unpacks `pxCommonInputs` blob to flat feature dict

**`features.py`**
- `VARIANT_FEATURES` — dict keyed by variant name; each entry is `VariantConfig(features, numeric)` with the ordered tuple of active Pega predictors and frozenset of numeric ones
- `EXCLUDE_COLS` — columns always dropped (identifiers, partition constants, PII)
- `TARGET` — propensity column name

**`metrics.py`**
- `fidelity_suite(y_true, y_pred)` → R², RMSE, Spearman ρ, Kendall τ, KS
- `feature_ranking(importances)` → 1-based rank Series
- `stability_spearman(rank_a, rank_b)` → Spearman ρ between rank vectors
- `jaccard_at_k(rank_a, rank_b, k)` → Jaccard similarity of top-k feature sets
- `psi(scores_a, scores_b, bins=10)` → `(psi_value, label)`, label ∈ `{"stable", "moderate", "significant"}`

**`surrogate.py`**
- `build_feature_matrix(df, pega_features, numeric_features)` → `(X, y, cat_cols, num_cols)`
- `train_catboost(X_train, y_train, cat_cols, **hyperparams)` → fitted `CatBoostRegressor`
- `NaiveBayesBaseline` — GaussianNB with quantile-binned propensity; used as fidelity lower-bound
- `evaluate_surrogate(model, X_test, y_test, model_name)` → fidelity dict

**`explanation.py`**
- `dt_surrogate(X, y, cat_cols, num_cols, max_depth_range, ...)` → `(tree, best_depth, cv_df, encoder)`. Depth selected by 5-fold CV Spearman ρ.
- `dt_importances(tree, feature_names)` → ranked importance Series
- `dt_rules(tree, feature_names, max_depth)` → text split rules
- `shap_importances(model, X, cat_cols)` → `(shap_values, mean_abs Series)`
- `lime_explain_batch(...)` → tidy DataFrame with `[instance_idx, feature, importance]`
- `aggregate_lime_importances(lime_df)` → global mean |coefficient| Series

---

## Methods

### Why CatBoost as surrogate

Pega ADM uses Naive Bayes internally. We train a CatBoost surrogate on Pega's output propensity scores (`R² ≈ 0.92`) because:
1. **TreeSHAP requires a tree model** — KernelSHAP on NB would add ~100× compute and stack its own error
2. **Fidelity over mimicry** — a flexible surrogate reproduces Pega's input-output mapping better than one constrained by conditional independence
3. **Native categorical and missing-value handling** — no encoding choices as confounds

### Instability decomposition (RQ3)

Observed instability Δπ is decomposed as:

```
Δπ_DT  =  δ_d  +  δ_m  +  δ_e        (DT is sensitive to all three)
Δπ_SHAP ≈  δ_d  +  δ_m               (SHAP proxies system-level shift)
δ_e    =  Δπ_DT − Δπ_SHAP            (residual = explainer sensitivity)
```

- **δ_d** (data drift) — PSI on propensity score distribution (5 equal-frequency bins)
- **δ_m** (model churn) — PSI on modelPerformance (AUC) distribution between splits
- **δ_e** (explainer sensitivity) — DT residual after accounting for δ_d + δ_m

Route subgroups act as a controlled natural experiment: shared time window (same model snapshots active) keeps δ_m ≈ 0 and δ_d low, isolating δ_e. A within-route DT bootstrap (N=10 seeds) validates that between-route instability exceeds DT fitting noise.

### Evaluation splits

- **Temporal** — early vs late window within each variant's data range
- **Route subgroups** — top routes by volume (`DepartureAirport × DestinationAirport`)
- **Replication variants** — CLUG, BookingDotCom, Cartrawler (same pipeline as L5B15)

---

## Stack

- **Polars** — JSON parsing
- **pandas** — feature matrices and analysis
- **CatBoost** — surrogate model
- **scikit-learn** — Decision Tree, OrdinalEncoder, GaussianNB, CV
- **SHAP** — exact TreeSHAP attributions
- **LIME** — local linear explanations
- **scipy** — Spearman ρ, Kendall τ, KS test
- **Matplotlib / Plotly** — visualisations

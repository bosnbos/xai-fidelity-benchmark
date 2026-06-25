# Transavia Pega ADM Analysis

Replication code for a master's thesis on Explainable AI (XAI) evaluation in a production Customer Decision Hub (CDH). The empirical setting is a large European airline using Pega Adaptive Decision Manager (ADM) to score luggage add-on offers in the Email/Outbound channel. The primary analysis focuses on a 15 kg luggage add-on (L5B15), with replication across three further variants: CLUG, BookingDotCom, Cartrawler.

## Research Questions

| | Question |
|---|---|
| **RQ1** | How faithfully do the predictive models used by each explanation method reproduce the Pega propensity scores on a held-out test set, as measured by a suite of rank-based, pointwise, and distributional fidelity metrics? |
| **RQ2** | How consistent are the feature rankings produced by each explanation method across temporal, route, and customer culture splits? |
| **RQ3** | When feature attribution rankings shift across splits, what is the dominant source: changes in the underlying Pega model, shifts in the input data distribution, or sensitivity of the explanation method itself? |
| **RQ4** | Do the fidelity and stability patterns observed for the primary luggage offer model replicate across three further offer variants, including a generic luggage offer and two third-party partner offers? |
| **RQ5** | Which features most strongly drive the offer-level propensity scores, how does importance distribute across feature groups, and what do the resulting attributions imply for marketing strategy and data governance in a production CDH? |

---

## Key Findings

**RQ1 — Surrogate fidelity.** CatBoost achieves R² ≈ 0.92 and Spearman ρ ≈ 0.96 on the L5B15 held-out test set, confirming that downstream attributions are grounded in a faithful representation of the Pega model.

**RQ2 — Explanation stability.** SHAP and LIME produce consistent feature rankings across all split types (temporal, route, culture). The decision tree is consistently the least stable method; its instability is driven by refit variance rather than genuine model or data change.

**RQ3 — Instability attribution.** Under controlled conditions (route and culture splits), the dominant source of observed instability is the explanation method itself, specifically the decision tree's refit sensitivity. Under temporal splits, large propensity drift and near-complete model-version turnover co-occur with DT divergence; SHAP and LIME continue to agree throughout.

**RQ4 — Cross-offer replication.** The fidelity and stability patterns replicate across all four variants. Temporal instability is a structural consequence of Pega ADM's continuous online learning: any time window long enough to span meaningful population dynamics also spans many micro-updates of the underlying Naive Bayes.

**RQ5 — Feature drivers.** The L5B15 propensity score is dominated by contact recency and frequency and trip context, not customer identity. No customer-level attributes were retained as active predictors, so the model scores contact fatigue rather than who the customer is. A secondary finding is that the staff-standby flag acts as a data-completeness proxy rather than a behavioural signal.

---

## Notebooks

Notebooks are numbered in execution order. 01 and 02b are independent of each other; 03 onwards depend on 02.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning | Nothing |
| `02_data_ingestion.ipynb` | Raw decision JSON → filtered parquet (L5B15 / CLUG) | `data/decisions/` |
| `02b_data_ingestion_thirdparty.ipynb` | Raw decision JSON → filtered parquet (BookingDotCom / Cartrawler) | `data/decisions/` |
| `03_eda.ipynb` | Temporal distribution, channel overview, missingness, model version churn | `02` |
| `04_surrogate_selection.ipynb` | Surrogate architecture selection on L5B15 across five candidate models | `02` |
| `05_surrogate_fit.ipynb` | Refit selected CatBoost per offer variant; produces cross-variant fidelity table | `02`, `04` |
| `06_explanation.ipynb` | DT, SHAP, LIME explanations and cross-method rank comparison | `02`, `05` |
| `07_stability.ipynb` | Stability across temporal, route, and culture splits; DT bootstrap validation | `02`, `05`, `06` |
| `08_attribution.ipynb` | Instability decomposition into data drift, model churn, and explainer sensitivity | `02`, `05`, `06`, `07` |
| `08b_attribution_replication.ipynb` | RQ4 — same attribution pipeline applied to CLUG, BookingDotCom, Cartrawler | `02`, `02b`, `05`, `06`, `07` |
| `09_vendor_hds_analysis.ipynb` | RQ5 supplementary — Pega HDS-style XGBoost importance compared against SHAP/LIME/DT | `02`, `06` |

---

## Scripts

All scripts read artifacts saved by the notebooks and write outputs to `thesis/tables/` or `data/artifacts/`. No notebook re-run is needed to regenerate tables after a formatting change.

| Script | Produces |
|---|---|
| `build_thesis_tables.py` | All `.tex` tables in `thesis/tables/` from JSON/CSV artifacts |
| `build_rq1_table.py` | `data/artifacts/rq1_explainer_fidelity.csv` (joins per-variant fidelity and DT model files) |
| `build_rq5_feature_analysis.py` | RQ5 feature importance `.tex` tables |
| `build_metric_selection_table.py` | `data/artifacts/metric_selection.csv` |
| `build_missingness_table.py` | `data/artifacts/missingness.csv` |
| `plot_feature_importance_heatmap.py` | `thesis/figures/feature_importance_heatmap.png` |
| `plot_shap_top_by_category.py` | SHAP bar charts grouped by feature category |
| `plot_shap_waterfall.py` | SHAP waterfall plots for individual instances |
| `run_explanation_all_variants.py` | Runs notebook 06 explanation pipeline across all variants |
| `stratify_by_technique.py` | `data/artifacts/stratified_by_technique.json` |
| `stratify_l5b15_stability.py` | Stability stratification artifacts for L5B15 |

---

## Data

All raw and derived data files are gitignored and must be added locally.

| Location | Contents |
|---|---|
| `data/decisions/` | Raw Pega decision JSON exports |
| `data/adm/` | ADM model snapshots and predictor binning JSON |
| `data/processed/` | Parquet outputs of notebooks 02 and 02b |
| `data/artifacts/{VARIANT}/` | Trained model and explanation artifacts, one folder per variant |

**Processed parquet files** (`data/processed/`): `luggage_email_outbound.parquet`, `l5b15_decisions.parquet`, `l5b15_email_outbound.parquet`, `thirdparty_email_outbound.parquet`

**Artifact files per variant** (`data/artifacts/l5b15/`, `CLUG/`, `BookingDotCom/`, `Cartrawler/`): `catboost_model.cbm`, `feature_cols.json`, `cat_cols.json`, `num_cols.json`, `train_idx.npy`, `test_idx.npy`, `fidelity.json`, `dt_model.pkl`, `dt_rules.txt`, `shap_values.npy`, `shap_importances.json`, `lime_explanations.parquet`, `lime_importances.json`, `cross_method_comparison.json`, `stability_summary.json`, `attribution_summary.json`

**Cross-variant outputs** (`data/artifacts/`): `surrogate_fidelity_per_variant.csv`

---

## Setup

```bash
uv sync
uv run jupyter lab
```

---

## Project Structure

```
├── data/                                    # gitignored
│   ├── adm/
│   ├── decisions/
│   ├── processed/
│   └── artifacts/{VARIANT}/
├── notebooks/
│   ├── 01_adm_analysis.ipynb
│   ├── 02_data_ingestion.ipynb
│   ├── 02b_data_ingestion_thirdparty.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_surrogate_selection.ipynb
│   ├── 05_surrogate_fit.ipynb
│   ├── 06_explanation.ipynb
│   ├── 07_stability.ipynb
│   ├── 08_attribution.ipynb
│   ├── 08b_attribution_replication.ipynb
│   └── 09_vendor_hds_analysis.ipynb
├── scripts/
├── src/
│   └── my_project/
│       ├── parsing.py
│       ├── features.py
│       ├── metrics.py
│       ├── surrogate.py
│       └── explanation.py
├── thesis/
│   ├── thesis.tex
│   ├── references.bib
│   ├── tables/                              # generated by scripts/
│   └── figures/
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
- `psi(scores_a, scores_b, bins=10)` → `(psi_value, label)`, label in `{"stable", "moderate", "significant"}`

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

Pega ADM uses Naive Bayes internally. A CatBoost surrogate is trained on Pega's output propensity scores (R² ≈ 0.92) for three reasons: TreeSHAP requires a tree model; a flexible surrogate reproduces Pega's input-output mapping better than one constrained by conditional independence; and CatBoost handles categoricals and missing values natively without encoding choices as confounds.

### Instability decomposition (RQ3)

Observed instability between two splits is decomposed into three sources:

| Source | Metric | Threshold |
|---|---|---|
| Data drift (δ_d) | KS statistic on propensity score distribution | KS ≥ 0.10 |
| Model churn (δ_m) | Jaccard overlap on `modelVersion` IDs active in each split | J_v ≤ 0.10 |
| Explainer sensitivity (δ_e) | Two mechanisms measured against SHAP as deterministic reference | reported, not thresholded |

KS is used over PSI because Pega ADM's isotonic calibration produces a step-function propensity, making PSI bin-sensitive. Jaccard on model-version IDs is used because the `modelPerformance` field carries heavy ties and is not a continuous performance measurement.

Two mechanisms of explainer sensitivity are distinguished, both expressed as gaps from the SHAP baseline:

- `δ_e^DT = ρ_SHAP − ρ_DT` — DT refit sensitivity: the tree is re-fitted independently on each split, so any ranking gap reflects structural refit variance.
- `δ_e^LIME = ρ_SHAP − ρ_LIME` — LIME sampling sensitivity: LIME shares the fixed-surrogate property with SHAP, so any gap is local-perturbation noise.

### Evaluation splits

| Split type | Description |
|---|---|
| Temporal | Early vs late window within each variant's data range |
| Route | Top 4 routes by volume (DepartureAirport × DestinationAirport), paired adjacently |
| Culture | fr-FR vs nl-NL bookings within the same time window |
| Replication | CLUG, BookingDotCom, Cartrawler run through the same pipeline as L5B15 |

Route and culture splits share the same pool of active model snapshots by construction, making them a controlled natural experiment where δ_m ≈ 0 and δ_d is typically small, isolating δ_e.

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

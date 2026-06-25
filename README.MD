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

**RQ3 — Instability attribution.** Under controlled conditions (route and culture splits, where model-version overlap is near 1 by construction), the dominant source of observed instability is the explanation method itself, specifically the decision tree's refit sensitivity. Under temporal splits, large propensity drift and near-complete model-version turnover co-occur with DT divergence; SHAP and LIME continue to agree throughout.

**RQ4 — Cross-offer replication.** The fidelity and stability patterns replicate across all four variants. Temporal instability is a structural consequence of Pega ADM's continuous online learning: any time window long enough to span meaningful population dynamics also spans many micro-updates of the underlying Naive Bayes, so temporal splits structurally combine data drift and snapshot turnover regardless of the offer.

**RQ5 — Feature drivers.** The L5B15 propensity score is dominated by contact recency and frequency (contact history features) and trip context, not customer identity. No customer-level attributes were retained as active predictors, so the model scores contact fatigue rather than who the customer is. A secondary finding is that the staff-standby flag acts as a data-completeness proxy rather than a behavioural signal, illustrating that high importance and behavioural meaningfulness are not the same thing in a self-selecting feature space.

---

## Notebooks

Run in order. **01 and 02b are independent.** Notebooks 03-08b all require 02 first.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning | Nothing |
| `02_data_ingestion.ipynb` | Raw decision JSON → filtered parquet (L5B15 / CLUG) | `data/decisions/` |
| `02b_data_ingestion_thirdparty.ipynb` | Raw decision JSON → filtered parquet (BookingDotCom / Cartrawler) | `data/decisions/` |
| `03_eda.ipynb` | EDA: temporal distribution, channel overview, missingness, model version churn | `02` |
| `04_surrogate_selection.ipynb` | Surrogate architecture selection on L5B15: Linear / GaussianNB / RandomForest / LightGBM / CatBoost on the full fidelity suite | `02` |
| `05_surrogate_fit.ipynb` | Refit the selected CatBoost per offer variant; saves per-variant artifacts + cross-variant fidelity table | `02`, `04` |
| `06_explanation.ipynb` | DT, SHAP, LIME explanations and cross-method rank comparison | `02`, `05` |
| `07_stability.ipynb` | Stability across temporal, route, and culture splits: Spearman ρ, Jaccard@k; within-sample DT bootstrap | `02`, `05`, `06` |
| `08_attribution.ipynb` | Instability decomposition into δ_d (KS), δ_m (J_v), δ_e^DT, δ_e^LIME; PSI bin-sensitivity diagnostics | `02`, `05`, `06`, `07` |
| `08b_attribution_replication.ipynb` | RQ4 — same attribution pipeline applied to CLUG, BookingDotCom, Cartrawler | `02`, `02b`, `05`, `06`, `07` |
| `09_vendor_hds_analysis.ipynb` | RQ5 supplementary — mirrors Pega's HDS-style XGBoost feature importance; compares rankings against SHAP/LIME/DT | `02`, `06` |
| `10_pega_nb_importance.ipynb` | RQ5 supplementary — re-implements Pega's native NB importance formula directly from predictor-binning snapshots | `data/adm/` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It picks up all files in that folder, deduplicates on `modelExecutionID`, and overwrites the parquet.

---

## Thesis tables

All thesis-ready LaTeX tables are produced by **`scripts/build_thesis_tables.py`**, which reads the JSON / CSV artifacts saved by the notebooks (in `data/artifacts/`) and writes the `.tex` files directly into **`thesis/tables/``. The thesis references these files via `\input{tables/<name>}`, so a rebuild updates the thesis tables in place.

The notebooks compute and persist numbers (JSON / CSV); the script handles all formatting (captions, column headers, rounding). Separating the two means thesis-table tweaks never require re-running expensive notebook computations.

```bash
uv run python scripts/build_thesis_tables.py            # rebuild every table
uv run python scripts/build_thesis_tables.py --list     # list available targets
uv run python scripts/build_thesis_tables.py attribution_l5b15 stability_matrix
```

**Run the script whenever:**
- you re-run a notebook and want the corresponding `.tex` refreshed (numbers changed)
- you tweak a caption, column header, or rounding inside the script (formatting changed)

**Main-text tables**

| Target | Reads (under `data/artifacts/`) | Writes (under `thesis/tables/`) |
|---|---|---|
| `surrogate_fidelity` | `surrogate_fidelity_per_variant.csv` | `surrogate_fidelity_per_variant.tex` |
| `rq1_explainer_fidelity` | `rq1_explainer_fidelity.csv` | `rq1_explainer_fidelity.tex` |
| `stability_matrix` | `l5b15/stability_summary.json` | `stability_matrix.tex` |
| `attribution_l5b15` | `l5b15/attribution_summary.json` | `attribution_summary.tex` |
| `attribution_replication` | `attribution_summary_replication.json` | `attribution_summary_replication.tex` |

**Appendix tables** (in narrative order)

| Target | Reads (under `data/artifacts/`) | Writes (under `thesis/tables/`) |
|---|---|---|
| `missingness` | `missingness.csv` | `missingness.tex` |
| `depth_sensitivity` | `l5b15/depth_sensitivity_table.csv` | `depth_sensitivity_table.tex` |
| `metric_selection` | `metric_selection.csv` | `metric_selection.tex` |
| `surrogate_comparison` | `l5b15/surrogate_comparison.csv` | `surrogate_comparison.tex` |
| `psi_propensity` | `l5b15/psi_bin_sensitivity_propensity.csv` | `psi_bin_sensitivity_propensity.tex` |
| `psi_auc` | `l5b15/psi_bin_sensitivity_auc.csv` | `psi_bin_sensitivity_auc.tex` |
| `jaccard_sensitivity` | `l5b15/attribution_summary.json` | `jaccard_threshold_sensitivity.pdf` + `.png` |
| `bootstrap_dt_summary` | `l5b15/bootstrap_dt_routes.json` | `bootstrap_dt_summary.tex` |
| `top10_per_split` | `l5b15/top10_per_split.csv` | `top10_per_split.tex` |

`rq1_explainer_fidelity.csv` is built by `scripts/build_rq1_table.py` (joins per-variant `fidelity.json` and `dt_model.pkl`); run that once when surrogate/DT models change, then `build_thesis_tables.py` for the LaTeX.

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

**Summary outputs** (written to `data/artifacts/l5b15/` by notebooks 07-08):
`stability_summary.json`, `attribution_summary.json`

**Cross-variant outputs** (`data/artifacts/`):
`surrogate_fidelity_per_variant.csv` (from notebook 05)

---

## Setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync
uv run jupyter lab
```

`uv sync` installs all dependencies and the local `src/my_project` package. All notebooks use `uv run` or can be opened directly in the activated environment.

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
│   ├── 04_surrogate_selection.ipynb
│   ├── 05_surrogate_fit.ipynb
│   ├── 06_explanation.ipynb
│   ├── 07_stability.ipynb
│   ├── 08_attribution.ipynb
│   ├── 08b_attribution_replication.ipynb
│   ├── 09_vendor_hds_analysis.ipynb
│   └── 10_pega_nb_importance.ipynb
├── scripts/
│   ├── build_thesis_tables.py               # Main table builder (reads artifacts, writes .tex)
│   ├── build_rq1_table.py                   # Builds rq1_explainer_fidelity.csv
│   ├── build_rq5_feature_analysis.py        # Builds RQ5 feature importance tables
│   ├── build_metric_selection_table.py
│   ├── build_missingness_table.py
│   ├── plot_feature_importance_heatmap.py
│   ├── plot_shap_top_by_category.py
│   ├── plot_shap_waterfall.py
│   ├── run_explanation_all_variants.py
│   ├── stratify_by_technique.py
│   └── stratify_l5b15_stability.py
├── src/
│   └── my_project/
│       ├── parsing.py                       # Pega JSON parsing
│       ├── features.py                      # Per-variant feature configs
│       ├── metrics.py                       # Fidelity, stability, PSI metrics
│       ├── surrogate.py                     # CatBoost surrogate + NB baseline
│       └── explanation.py                   # DT, SHAP, LIME utilities
├── thesis/
│   ├── thesis.tex
│   ├── references.bib
│   ├── tables/                              # LaTeX tables (generated by scripts/)
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

Pega ADM uses Naive Bayes internally. We train a CatBoost surrogate on Pega's output propensity scores (R² ≈ 0.92) because:
1. **TreeSHAP requires a tree model** — KernelSHAP on NB would add ~100× compute and stack its own error
2. **Fidelity over mimicry** — a flexible surrogate reproduces Pega's input-output mapping better than one constrained by conditional independence
3. **Native categorical and missing-value handling** — no encoding choices as confounds

### Instability decomposition (RQ3)

Observed instability between two splits is decomposed into three sources:

| Source | Metric | Threshold (significant) |
|---|---|---|
| **δ_d** (data drift) | Kolmogorov-Smirnov statistic on propensity score distribution | KS ≥ 0.10 |
| **δ_m** (model churn) | Jaccard overlap on the set of `modelVersion` IDs active in each split | J_v ≤ 0.10 |
| **δ_e** (explainer sensitivity) | Two mechanisms vs SHAP as deterministic reference (see below) | reported, not classified |

**Why KS over PSI.** Pega ADM's isotonic calibration produces a step-function propensity (clustered values), which makes PSI bin-sensitive. KS operates on empirical CDFs with no binning parameter. The bin-sensitivity check is in `08_attribution.ipynb` §A1 and exported as `psi_bin_sensitivity_propensity.tex`.

**Two mechanisms of explainer sensitivity.** SHAP is the reference because TreeSHAP is deterministic given a fixed CatBoost surrogate.

- **δ_e^DT = ρ_SHAP − ρ_DT** — DT refit sensitivity: the tree is re-fitted independently on each split.
- **δ_e^LIME = ρ_SHAP − ρ_LIME** — LIME sampling sensitivity: LIME shares the fixed-surrogate property with SHAP, so any gap is local-perturbation noise.

### Evaluation splits

- **Temporal** — early vs late window within each variant's data range
- **Route subgroups** — top 4 routes by volume (`DepartureAirport × DestinationAirport`), paired adjacently
- **Culture** — fr-FR vs nl-NL bookings (same time window as route splits)
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

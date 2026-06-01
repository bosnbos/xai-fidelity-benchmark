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

Run in order. **01 and 02b are independent.** Notebooks 03–08 all require 02 first.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning | Nothing |
| `02_data_ingestion.ipynb` | Raw decision JSON → filtered parquet (L5B15 / CLUG) | `data/decisions/` |
| `02b_data_ingestion_thirdparty.ipynb` | Raw decision JSON → filtered parquet (BookingDotCom / Cartrawler) | `data/decisions/` |
| `03_eda.ipynb` | EDA: temporal distribution, channel overview, missingness, model version churn | `02` |
| `04_surrogate_selection.ipynb` | Surrogate architecture selection on L5B15: Linear / GaussianNB / RandomForest / LightGBM / CatBoost on the full fidelity suite | `02` |
| `05_surrogate_fit.ipynb` | Refit the selected CatBoost per offer variant; saves per-variant artifacts + cross-variant fidelity table for the Results chapter | `02`, `04` |
| `06_explanation.ipynb` | DT (§9), SHAP (§10), LIME (§11), cross-method rank comparison (§12) | `02`, `05` |
| `07_stability.ipynb` | Stability across temporal, route, and culture splits: Spearman ρ, Jaccard@k; within-sample DT bootstrap | `02`, `05`, `06` |
| `08_attribution.ipynb` | Instability decomposition into δ_d (KS), δ_m (J_v), δ_e^DT, δ_e^LIME; PSI bin-sensitivity diagnostics | `02`, `05`, `06`, `07` |
| `08b_attribution_replication.ipynb` | RQ4 — same attribution pipeline applied to CLUG, BookingDotCom, Cartrawler | `02`, `02b`, `05`, `06`, `07` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It picks up all files in that folder, deduplicates on `modelExecutionID`, and overwrites the parquet.

---

## Thesis tables

All thesis-ready LaTeX tables are produced by **`scripts/build_thesis_tables.py`**, which reads the JSON / CSV artefacts saved by the notebooks (in `data/artifacts/`) and writes the `.tex` files into the repo-root **`tables/`** folder. Everything ends up in one flat directory so you can drag-and-drop the whole folder into an Overleaf project's `tables/`.

The notebooks compute and persist numbers (JSON / CSV); the script handles all formatting (captions, column headers, rounding). Separating the two means thesis-table tweaks never require re-running expensive notebook computations.

```bash
uv run python scripts/build_thesis_tables.py            # rebuild every table
uv run python scripts/build_thesis_tables.py --list     # list available targets
uv run python scripts/build_thesis_tables.py attribution_l5b15 stability_matrix
```

**Run the script whenever:**
- you re-run a notebook and want the corresponding `.tex` refreshed (numbers changed)
- you tweak a caption, column header, or rounding inside the script (formatting changed)

You do **not** need to re-run notebooks just to update tables — the script reads the already-saved CSV/JSON. Notebooks only need a re-run when the underlying analysis changes (new data, hyperparameter tweak, model refit, new metric).

**Main-text tables**

| Target | Reads (under `data/artifacts/`) | Writes (under `tables/`) |
|---|---|---|
| `surrogate_fidelity` | `surrogate_fidelity_per_variant.csv` | `surrogate_fidelity_per_variant.tex` |
| `rq1_explainer_fidelity` | `rq1_explainer_fidelity.csv` | `rq1_explainer_fidelity.tex` |
| `stability_matrix` | `l5b15/stability_summary.json` | `stability_matrix.tex` |
| `attribution_l5b15` | `l5b15/attribution_summary.json` | `attribution_summary.tex` |
| `attribution_replication` | `attribution_summary_replication.json` | `attribution_summary_replication.tex` |

**Appendix tables** (in narrative order)

| Target | Reads (under `data/artifacts/`) | Writes (under `tables/`) |
|---|---|---|
| `missingness` | `missingness.csv` (produced by `scripts/build_missingness_table.py`) | `missingness.tex` |
| `depth_sensitivity` | `l5b15/depth_sensitivity_table.csv` | `depth_sensitivity_table.tex` |
| `metric_selection` | `metric_selection.csv` (produced by `scripts/build_metric_selection_table.py`) | `metric_selection.tex` |
| `surrogate_comparison` | `l5b15/surrogate_comparison.csv` | `surrogate_comparison.tex` |
| `psi_propensity` | `l5b15/psi_bin_sensitivity_propensity.csv` | `psi_bin_sensitivity_propensity.tex` |
| `psi_auc` | `l5b15/psi_bin_sensitivity_auc.csv` | `psi_bin_sensitivity_auc.tex` |
| `jaccard_sensitivity` | `l5b15/attribution_summary.json` | `jaccard_threshold_sensitivity.pdf` + `.png` (figure, not table) |
| `bootstrap_dt_summary` | `l5b15/bootstrap_dt_routes.json` | `bootstrap_dt_summary.tex` |
| `top10_per_split` | `l5b15/top10_per_split.csv` | `top10_per_split.tex` |

Default rounding is 3 decimal places; RMSE keeps 4 dp because some variants are near $10^{-4}$.

**Overleaf workflow:** upload (or drag-and-drop) the contents of `tables/` into a folder of the same name in the Overleaf project, then reference any table with `\input{tables/<name>}`.

`rq1_explainer_fidelity.csv` itself is built by `scripts/build_rq1_table.py` (it joins per-variant `fidelity.json` and `dt_model.pkl`); run that once when surrogate/DT models change, then `build_thesis_tables.py` for the LaTeX.

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

**Summary outputs** (written to `data/artifacts/l5b15/` by notebooks 07–08):  
`stability_summary.json`, `attribution_summary.json`

**Cross-variant outputs** (`data/artifacts/`):  
`surrogate_fidelity_per_variant.csv` (from notebook 05)

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
│   ├── 04_surrogate_selection.ipynb
│   ├── 05_surrogate_fit.ipynb
│   ├── 06_explanation.ipynb
│   ├── 07_stability.ipynb
│   └── 08_attribution.ipynb
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

Observed instability between two splits is decomposed into three sources, each measured with a metric matched to the underlying object:

| Source | Metric | Threshold (significant) |
|---|---|---|
| **δ_d** (data drift) | Kolmogorov–Smirnov statistic on the propensity score distribution (bin-free, non-parametric) | KS ≥ 0.10 |
| **δ_m** (model churn) | Jaccard overlap on the set of `modelVersion` IDs active in each split (lower = more churn) | J_v ≤ 0.10 |
| **δ_e** (explainer sensitivity) | Decomposed into two mechanisms vs SHAP as the deterministic reference, see below | reported, not classified |

**Why KS over PSI.** Pega ADM's isotonic calibration produces a step-function propensity (clustered values), which makes PSI bin-sensitive — increasing the bin count inflates PSI artificially. KS operates directly on empirical CDFs with no binning parameter. The bin-sensitivity check is documented in `08_attribution.ipynb` §A1 and exported as `psi_bin_sensitivity_propensity.tex`.

**Why Jaccard over PSI-on-AUC.** Pega's `modelPerformance` field is effectively a categorical snapshot identifier with heavy ties, not a continuous performance measurement. Jaccard answers the actual question (snapshot overlap between splits) directly; the threshold ≤ 0.10 sits in an empty band of the bimodal J_v distribution and classifications are invariant for τ ∈ [0.05, 0.40] (§A4). Note that J_v measures *snapshot turnover* rather than parameter shift directly, since UUIDs rotate on every micro-update; for same-window splits the production scorer uses the same active snapshots by construction, so any reasonable churn proxy returns ≈ 0 there.

**Two mechanisms of explainer sensitivity.** SHAP is the reference because TreeSHAP is deterministic given a fixed CatBoost surrogate. The other two explainers are compared against it:

- **δ_e^DT = ρ_SHAP − ρ_DT** — DT *refit sensitivity*: the tree structure is re-fitted independently on each split.
- **δ_e^LIME = ρ_SHAP − ρ_LIME** — LIME *sampling sensitivity*: LIME shares the fixed-surrogate property with SHAP, so any gap is local-perturbation noise, not refit.

Both gaps are reported as descriptive evidence; **the classifier uses only KS_prop and J_v**. A large δ_e^DT validates an "Explainer" label; a small δ_e^DT qualifies a "Dist. shift" / "Model churn" / "Both" label as also clean of explainer effects.

**Effect-size only.** With n in the thousands per split, p-values are sample-size driven and uninformative. The KS statistic and Jaccard overlap *are* the effect sizes; we threshold them directly.

### Evaluation splits

- **Temporal** — early vs late window within each variant's data range
- **Route subgroups** — top 4 routes by volume (`DepartureAirport × DestinationAirport`): ORY→NCE, ORY→OPO, NCE→ORY, TLS→ORY, paired adjacently
- **Culture** — fr-FR vs nl-NL bookings (shares same time window, like route splits)
- **Replication variants** — CLUG, BookingDotCom, Cartrawler (same pipeline as L5B15)

Route and culture splits share the production model snapshots by construction (same time window) and act as a controlled natural experiment: δ_m ≈ 0 and δ_d typically small, isolating δ_e. A within-sample DT bootstrap (N=10 seeds per sub-sample) validates that observed between-split instability exceeds DT refit noise (§14b/d in `07_stability.ipynb`, summary in `bootstrap_dt_summary.tex`).

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

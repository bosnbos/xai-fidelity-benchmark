# Transavia Pega ADM Analysis

Analysis of Pega Adaptive Decision Manager (ADM) propensity models for Transavia. The main focus is **luggage add-on offers (L5B15/L5B20/L5B25/L5B30/L5B40/L5B50/CLUG) in the Email/Outbound channel**.

## Notebooks

**01 is independent.** Notebooks 03, 04, and 05 all require 02 to run first.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning: offer viability, feature importance, time-series comparison | Nothing |
| `02_data_ingestion.ipynb` | Parse raw decision JSON exports → extract all luggage variants → filter to Email/Outbound/Luggage/Sales → save parquet | JSON files in `data/decisions/` |
| `03_eda.ipynb` | EDA on the filtered decision data: temporal distribution, channel overview, offering breakdown, missingness, target distribution, correlations, cardinality | `02` |
| `04_modeling.ipynb` | CatBoost surrogate model, evaluation (RMSE, Spearman rank), feature importance, model version drift | `02` |
| `05_modeling.ipynb` | Alternative modeling approach on the same filtered dataset | `02` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It automatically picks up all files in that folder, deduplicates on `modelExecutionID` (so overlapping exports are never double-counted), and overwrites the parquet. Notebooks 03–05 will then reflect the updated dataset.

## Data

All raw files are gitignored — add them locally in the correct subfolder:

| Location | Contents | Used by |
|---|---|---|
| `data/decisions/` | Pega decision JSON exports (all channels, all offerings) | Notebook 02 |
| `data/adm/` | ADM model snapshots + predictor binning JSON | Notebook 01 |
| `data/processed/` | `luggage_email_outbound.parquet` — output of notebook 02 | Notebooks 03–05 |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -e .
jupyter lab
```

`pip install -e .` is a one-time step that installs the local `src/my_project` package into the virtual environment, making the shared parsing utilities importable across notebooks.

## Project structure

```
├── data/
│   ├── adm/                                 # ADM snapshot files (gitignored)
│   ├── decisions/                           # Raw decision JSON exports (gitignored)
│   └── processed/                           # Parquet output (gitignored)
├── notebooks/
│   ├── 01_adm_analysis.ipynb                # ADM snapshots & predictor binning
│   ├── 02_data_ingestion.ipynb              # Raw JSON → filtered parquet
│   ├── 03_eda.ipynb                         # EDA on Email/Outbound decisions
│   ├── 04_modeling.ipynb                    # CatBoost surrogate model
│   ├── 05_modeling.ipynb                    # Alternative modeling approach
│   └── archive/
│       └── exploration_original.ipynb
├── src/
│   └── my_project/
│       ├── parsing.py                       # Shared JSON parsing functions
│       └── features.py                      # Shared feature config (TARGET, EXCLUDE_COLS, PEGA_FEATURES)
├── pyproject.toml
└── README.MD
```

### `src/`?

`src/my_project/` is a small installable Python package with two modules:

**`parsing.py`** — functions that crack open the nested Pega JSON. Key exports:
- `extract_l5b15_rows(df, target_names=None)` — extracts flat scoring-event rows from a raw Polars DataFrame. Defaults to all entries in `LUGGAGE_VARIANTS`; pass a string or list to narrow to specific products. Also extracts `pxDecisionTime` (the actual Pega interaction timestamp) from `pxDecisionResults`.
- `LUGGAGE_VARIANTS` — frozenset of all luggage weight variant names (`L5B15`, `L5B20`, `L5B25`, `L5B30`, `L5B40`, `L5B50`, `CLUG`).
- `parse_common_inputs(record)` — unpacks the `pxCommonInputs` blob into a flat dict of customer/flight context features.
- `extract_filtered_model(df, ...)` — flexible extraction with per-field filter arguments.

**`features.py`** — single source of truth for modeling configuration:
- `TARGET` — the propensity column name used as the model target (`"propensity"`).
- `EXCLUDE_COLS` — list of columns to drop before modeling: identifiers, partition constants, and all PII (emails, names, date of birth, loyalty numbers, booking references). Importing this in every modeling notebook guarantees PII never enters the feature matrix.
- `PEGA_FEATURES` — template list of candidate Pega predictor variables grouped by namespace (`CustBookedFlight`, `Customer`, `IH`, `param`). Entries are commented out by default; uncomment the variables actually used in the live Pega model to restrict the feature set.

`pip install -e .` is a one-time step that registers both modules so `from my_project.parsing import ...` and `from my_project.features import ...` work in any notebook using the `.venv` kernel.

## Research Design

### Context

Pega ADM scores customers using a **Naive Bayes** model and outputs a continuous propensity $\hat{p} \in [0,1]$. The model is treated as a black box: we have its inputs (the 30 active predictors in `features.py`) and its outputs (the stored `propensity` column), but not its internal parameters. The goal is to explain *which features drive $\hat{p}$, by how much, and how stably* — without modifying or retraining Pega's model.

### Research questions

| | Question |
|---|---|
| **RQ1** | To what extent do SHAP, LIME, and rule-based surrogate models accurately approximate the predictions of the black-box propensity model, and does fidelity vary across explanation families? |
| **RQ2** | How stable are the feature attribution rankings produced by each explanation method across temporal splits and route subgroups? |
| **RQ3** | To what degree does observed explanation instability reflect genuine model drift rather than sensitivity of the explanation method to distributional shifts in the evaluation data? |

### XAI methods

**Global surrogate (Decision Tree)**
A shallow regression tree (max depth 4–6) is fitted directly on the stored propensity scores. Decision rules are extracted as human-readable if/then conditions. Fidelity is measured globally with R² and RMSE against $\hat{p}$. No live model queries are needed.

**SHAP**
A CatBoost surrogate is trained on $\hat{p}$, then TreeSHAP computes exact, additive feature attributions analytically from the tree structure. The completeness axiom guarantees SHAP values sum to the model output. No live model queries are needed — TreeSHAP works directly on the surrogate.

**LIME**
LIME is model-agnostic and requires a callable `predict(X)` function. The CatBoost surrogate serves as that oracle — LIME perturbs each instance and queries the surrogate, never Pega's live API. This is valid because the explanand is already Pega's output $\hat{p}$, and the surrogate R² quantifies the approximation error introduced. LIME produces local linear explanations per instance; fidelity is measured as the local R² of the linear fit.

### Surrogate model choice: why CatBoost, not Naive Bayes

A natural question is whether the surrogate should mirror Pega's own model architecture (Naive Bayes). It should not, for three reasons:

**1. TreeSHAP requires a tree-based model.**
TreeSHAP computes exact Shapley values in polynomial time by exploiting the tree structure. For a Naive Bayes surrogate, KernelSHAP would be required instead — a model-agnostic Monte Carlo approximation that is ~100× slower and introduces its own approximation error on top of the surrogate approximation error. Using Naive Bayes as surrogate would therefore stack two sources of inexactness, which directly undermines RQ1.

**2. The surrogate's job is fidelity, not architectural mimicry.**
Pega's Naive Bayes was trained on historical ground-truth purchase outcomes using Pega-specific evidence weighting, smoothing, and feature preprocessing that are not available here. A new Naive Bayes trained only on Pega's *output scores* is a categorically different model — the shared architecture provides no special proximity to the original. What matters is how well the surrogate replicates Pega's input-output mapping (measured by R²). A more flexible model that can capture feature interactions will do this better than one constrained by the conditional independence assumption.

**3. Naive Bayes is directly interpretable and does not need SHAP.**
Feature contributions in Naive Bayes are readable directly from log-likelihood ratios. Applying SHAP to a Naive Bayes surrogate is methodologically redundant — SHAP on a surrogate is motivated precisely because the surrogate is complex enough to need decomposition.

**Why CatBoost specifically over XGBoost or LightGBM.**
Of the 30 active predictors, the majority are categorical (`CustBookedFlight.BookingData.CultureCode`, `pxLastGroupID`, `FlightData.AircraftType`, etc.). CatBoost handles categoricals natively via ordered target encoding, removing the need for manual preprocessing choices (one-hot, ordinal, frequency encoding) that would otherwise become confounds in the explanation. CatBoost also handles missing values natively, which is relevant given sparse interaction history features. XGBoost and LightGBM require explicit encoding decisions that would need separate justification.

**Robustness check.** A Naive Bayes surrogate is included as a robustness baseline. If its R² against $\hat{p}$ is comparable to CatBoost, the architectural choice has limited impact on fidelity. If it is lower, this confirms that the conditional independence assumption constrains the surrogate and that CatBoost is the principled choice.

### Fidelity, stability and drift metrics

| RQ | Metric |
|---|---|
| **RQ1 fidelity** | Global R² / RMSE (surrogate vs $\hat{p}$); mean local R² ± std (LIME); SHAP completeness is exact by axiom |
| **RQ2 stability** | Spearman rank correlation of feature importance vectors across splits; Jaccard@k (k = 5, 10) for top-k overlap |
| **RQ3 drift vs sensitivity** | KS test + Population Stability Index (PSI) on $\hat{p}$ and key features across splits; regress explanation shift on data shift to quantify per-method sensitivity ratio |

### Evaluation splits

- **Temporal:** early window (Mar 9 – Apr 5) vs late window (Apr 6 – Apr 20)
- **Route subgroups:** top routes by volume using `CustBookedFlight.FlightData.DepartureAirport` × `FlightData.DestinationAirport`

### Active predictors (L5B15 Email/Outbound model)

The 30 predictors confirmed active in Pega's live Naive Bayes model are defined in `src/my_project/features.py` (`PEGA_FEATURES`). They span four namespaces: booking context (`CustBookedFlight.BookingData.*`), flight data (`CustBookedFlight.FlightData.*`), interaction history (`IH.*`), and Pega parameters (`param::Param.*`).

---

## Stack

- **Polars** for fast DataFrame operations on large JSON files
- **CatBoost** for surrogate model training with native categorical support
- **Plotly** and **Matplotlib** for interactive and static visualisations
- **scikit-learn** for train/test split and regression metrics

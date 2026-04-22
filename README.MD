# Transavia Pega ADM Analysis

Analysis of Pega Adaptive Decision Manager (ADM) propensity models for Transavia. The main focus is the **L5B15 luggage offer, Email/Outbound channel**.

## Notebooks

**01 is independent.** Notebooks 03, 04, and 05 all require 02 to run first.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning: offer viability, feature importance, time-series comparison | Nothing |
| `02_data_ingestion.ipynb` | Parse raw decision JSON exports → filter to Email/Outbound/Luggage/Sales → save parquet | JSON files in `data/decisions/` |
| `03_eda.ipynb` | EDA on the filtered decision data: missingness, target distribution, correlations, cardinality | `02` |
| `04_modeling.ipynb` | CatBoost surrogate model, evaluation (RMSE, Spearman rank), feature importance, model version drift | `02` |
| `05_modeling.ipynb` | Alternative modeling approach on the same filtered dataset | `02` |

### Adding new data

Drop new Pega decision JSON exports into `data/decisions/` and re-run `02_data_ingestion.ipynb`. It automatically picks up all files in that folder, deduplicates on `modelExecutionID` (so overlapping exports are never double-counted), and overwrites the parquet. Notebooks 03–05 will then reflect the updated dataset.

## Data

All raw files are gitignored — add them locally in the correct subfolder:

| Location | Contents | Used by |
|---|---|---|
| `data/decisions/` | Pega decision JSON exports (L5B15 scoring events) | Notebook 02 |
| `data/adm/` | ADM model snapshots + predictor binning JSON | Notebook 01 |
| `data/processed/` | `l5b15_email_outbound.parquet` — output of notebook 02 | Notebooks 03–05 |

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
│       └── parsing.py                       # Shared JSON parsing functions
├── pyproject.toml
└── README.MD
```

### What is `src/`?

`src/my_project/` is a small installable Python package containing `parsing.py` — the functions that crack open the nested Pega JSON (`extract_l5b15_rows`, `parse_common_inputs`, etc.). Keeping these in a package means you fix or extend the parsing logic in one place rather than in every notebook. `pip install -e .` registers the package so `from my_project.parsing import ...` works in any notebook using the `.venv` kernel.

## Stack

- **Polars** for fast DataFrame operations on large JSON files
- **CatBoost** for surrogate model training with native categorical support
- **Plotly** and **Matplotlib** for interactive and static visualisations
- **scikit-learn** for train/test split and regression metrics

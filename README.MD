# Transavia Pega ADM Analysis

Analysis of Pega Adaptive Decision Manager (ADM) propensity models for Transavia, focused on the L5B15 luggage offer and OnlineFlightCheckinOpen check-in offer.

## Notebooks

There are four notebooks. **Run order matters for the decision data notebooks** — see the dependency note below.

| Notebook | Purpose | Depends on |
|---|---|---|
| `01_adm_analysis.ipynb` | Inspect ADM model snapshots and predictor binning data: offer viability checks, feature importance, time-series comparison of two offers | Nothing — run independently |
| `02_data_ingestion.ipynb` | Parse raw Pega decision JSON exports, extract L5B15 scoring events, deduplicate, and save to `data/processed/l5b15_decisions.parquet` | Raw JSON files in `data/` |
| `03_eda.ipynb` | Exploratory analysis of the processed decision data: missingness, target distribution, correlations, categorical cardinality | `02` must run first |
| `04_modeling.ipynb` | CatBoost surrogate model, evaluation (RMSE, Spearman rank), feature importance, model version drift analysis | `02` must run first |

**01 is independent** — it reads the ADM snapshot files directly and has nothing to do with the decision data pipeline.

**02 → 03 and 04** — notebooks 03 and 04 read from `data/processed/l5b15_decisions.parquet`, which 02 creates. Always run 02 first, or after adding new data.

### Adding new data

Drop additional `data*.json` files into `data/` and re-run `02_data_ingestion.ipynb`. It automatically picks up all matching files and deduplicates on `modelExecutionID`, so overlapping exports are never double-counted. Notebooks 03 and 04 will then reflect the updated dataset.

## Data

Raw JSON files go in `data/` (gitignored — add locally):

| File | Contents |
|---|---|
| `data-model-snapshop.json` | ~161k ADM model snapshot records (used by notebook 01) |
| `data-Predictor-binning-snapshot.json` | ~222k predictor binning records (used by notebook 01) |
| `data*.json` | Decision-level records with nested scoring events (used by notebook 02) |

Processed output is saved to `data/processed/l5b15_decisions.parquet` by notebook 02.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -e .
jupyter lab
```

The `-e .` installs the local `src/my_project` package so notebooks can import shared parsing utilities.

## Project structure

```
├── data/                                    # Raw JSON files (gitignored — add locally)
│   └── processed/                           # Parquet output from notebook 02 (gitignored)
├── notebooks/
│   ├── 01_adm_analysis.ipynb                # ADM snapshots & predictor binning
│   ├── 02_data_ingestion.ipynb              # Raw JSON → parquet
│   ├── 03_eda.ipynb                         # EDA on decision data
│   ├── 04_modeling.ipynb                    # Surrogate model & drift analysis
│   └── archive/
│       └── exploration_original.ipynb       # Original monolithic notebook
├── src/
│   └── my_project/
│       └── parsing.py                       # Shared JSON parsing functions
├── pyproject.toml                           # Project metadata and dependencies
└── README.MD
```

### What is `src/`?

`src/my_project/` is a small Python package that holds code shared across notebooks. Right now it contains `parsing.py` — the functions that extract L5B15 scoring events from nested Pega JSON (`extract_l5b15_rows`, `parse_common_inputs`, etc.). Keeping these in a package rather than copy-pasting them into every notebook means you only need to fix or extend the logic in one place. Running `pip install -e .` makes the package importable as `from my_project.parsing import ...`.

## Stack

- **Polars** for fast DataFrame operations on large JSON files
- **CatBoost** for surrogate model training with native categorical support
- **Plotly** and **Matplotlib** for interactive and static visualisations
- **scikit-learn** for train/test split and regression metrics

"""Surrogate model building, training, and evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from my_project.features import VARIANT_FEATURES
from my_project.metrics import fidelity_suite
from my_project.parsing import THIRDPARTY_VARIANTS

_NAN_MARKERS = frozenset({"", "NaN", "nan", "None", "none", "null", "NULL"})


def build_feature_matrix(
    df: pd.DataFrame,
    pega_features: list[str],
    numeric_features: frozenset[str],
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare (X, y, cat_cols, num_cols) from a processed decisions DataFrame.

    Restricts X to features present in both pega_features and df.columns.
    - cat_cols get string dtype with "__MISSING__" for NaN (CatBoost native format).
    - num_cols get float dtype with np.nan preserved (CatBoost handles natively).

    Returns (X, y, cat_cols, num_cols).
    """
    active = [f for f in pega_features if f in df.columns]
    X = df[active].copy()
    y = df["propensity"].astype(float)

    num_cols = [f for f in active if f in numeric_features]
    cat_cols = [f for f in active if f not in numeric_features]

    for col in num_cols:
        X[col] = pd.to_numeric(
            X[col].replace(list(_NAN_MARKERS), np.nan), errors="coerce"
        )
    for col in cat_cols:
        X[col] = (
            X[col]
            .replace(list(_NAN_MARKERS), np.nan)
            .astype("string")
            .fillna("__MISSING__")
        )

    return X, y, cat_cols, num_cols


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list[str],
    **hyperparams,
) -> CatBoostRegressor:
    """Fit a CatBoost surrogate. Any kwarg overrides the defaults."""
    defaults: dict = {
        "iterations":    500,
        "depth":         6,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "eval_metric":   "RMSE",
        "random_seed":   42,
        "verbose":       False,
    }
    defaults.update(hyperparams)
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    model = CatBoostRegressor(**defaults)
    model.fit(X_train, y_train, cat_features=cat_indices)
    return model


class NaiveBayesBaseline:
    """Naive Bayes regression baseline with a sklearn-compatible predict() interface.

    Discretises propensity into quantile bins, fits GaussianNB as a classifier,
    then recovers continuous predictions via class-probability-weighted bin means.
    Mirrors the NB architecture of Pega ADM without matching its predictor encoding;
    used purely as a fidelity lower-bound comparison for CatBoost.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._model: GaussianNB | None = None
        self._encoders: dict[str, LabelEncoder] = {}
        self._bin_means: np.ndarray | None = None
        self._col_means: dict[str, float] = {}
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cat_cols: list[str],
        num_cols: list[str],
    ) -> "NaiveBayesBaseline":
        self._cat_cols = list(cat_cols)
        self._num_cols = list(num_cols)

        y_binned = pd.qcut(y_train, q=self.n_bins, labels=False, duplicates="drop").astype(int)
        self._bin_means = y_train.groupby(y_binned).mean().sort_index().values

        X_enc = self._encode(X_train, fit=True)
        self._model = GaussianNB()
        self._model.fit(X_enc, y_binned)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_enc = self._encode(X, fit=False)
        probs = self._model.predict_proba(X_enc)
        return probs @ self._bin_means

    def _encode(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []

        for col in self._cat_cols:
            s = X[col].astype(str)
            if fit:
                le = LabelEncoder()
                parts.append(le.fit_transform(s).reshape(-1, 1))
                self._encoders[col] = le
            else:
                le = self._encoders[col]
                known = set(le.classes_)
                fallback = le.classes_[0]
                s_safe = s.map(lambda v, k=known, fb=fallback: v if v in k else fb)
                parts.append(le.transform(s_safe).reshape(-1, 1))

        for col in self._num_cols:
            vals = pd.to_numeric(X[col], errors="coerce")
            if fit:
                self._col_means[col] = float(vals.mean())
            parts.append(vals.fillna(self._col_means[col]).values.reshape(-1, 1))

        return np.hstack(parts) if parts else np.zeros((len(X), 1))


def evaluate_surrogate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict:
    """Call model.predict(X_test) and return fidelity_suite annotated with model_name."""
    pred = np.asarray(model.predict(X_test))
    result = fidelity_suite(y_test.values, pred)
    result["model"] = model_name
    return result


def cv_select_depth(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list[str],
    depths: range | list[int] = range(1, 16),
    n_splits: int = 5,
    random_state: int = 42,
    cv_iterations: int = 200,
    selection: str = "1sd",
) -> tuple[int, pd.DataFrame]:
    """K-fold CV depth selection by mean Spearman ρ on the validation folds.

    `cv_iterations` is smaller than the default 500 to speed up the depth sweep —
    the relative ranking of depths is stable well before convergence. The final
    model should be refit at full iterations after selection.

    `selection`:
        "max" – classic argmax of mean Spearman ρ (strict CV-optimal).
        "1sd" – Breiman-style parsimony rule: smallest depth whose mean ρ is
                within one fold-to-fold standard deviation of the maximum.
                Prefers simpler models when deeper trees give noise-level gains.

    Returns (best_depth, results_df) where results_df is indexed by depth with
    columns ["mean_rho", "std_rho"]. results_df.attrs["max_depth"] also records
    the argmax depth so callers can show both.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for depth in depths:
        fold_rhos = []
        for tr_idx, val_idx in kf.split(X_train):
            m = train_catboost(
                X_train.iloc[tr_idx], y_train.iloc[tr_idx], cat_cols,
                depth=depth, iterations=cv_iterations,
            )
            rho, _ = spearmanr(y_train.iloc[val_idx], m.predict(X_train.iloc[val_idx]))
            fold_rhos.append(rho)
        rows.append({
            "depth":    int(depth),
            "mean_rho": round(float(np.mean(fold_rhos)), 4),
            "std_rho":  round(float(np.std(fold_rhos)),  4),
        })
    results = pd.DataFrame(rows).set_index("depth")
    max_depth = int(results["mean_rho"].idxmax())
    results.attrs["max_depth"] = max_depth

    if selection == "max":
        best_depth = max_depth
    elif selection == "1sd":
        max_mean = results.loc[max_depth, "mean_rho"]
        max_std  = results.loc[max_depth, "std_rho"]
        in_band  = results[results["mean_rho"] >= max_mean - max_std]
        best_depth = int(in_band.index.min())
    else:
        raise ValueError(f"selection must be 'max' or '1sd', got {selection!r}")

    return best_depth, results


def fit_surrogate_for_variant(
    variant: str,
    processed_dir: Path,
    artifact_root: Path,
    cv_depth: bool = True,
    final_iterations: int = 500,
    technique: str = "0.0",
) -> dict:
    """End-to-end per-variant pipeline used by 05_surrogate_fit.

    Steps: load correct parquet → filter to `pyName == variant` and
    `modelTechnique == technique` → drop absent/constant features from the
    variant's `VARIANT_FEATURES` entry → `build_feature_matrix` →
    stratified 80/20 split on propensity deciles → optional per-variant
    depth CV → final CatBoost fit at full iterations → save the full
    artifact set under `artifact_root / variant /`.

    `technique` defaults to "0.0" — the production-decision model Pega
    actually uses to choose offers (confirmed with the Transavia data
    team). Every scoring event in the export is evaluated by two
    parallel techniques: this production model (label degenerate to
    "0.0" in our exports, almost certainly a gradient-boosted scorer)
    and a NaiveBayes audit/shadow model. NaiveBayes scores carry online
    learning state invisible to the surrogate (R² ceiling ~0.4); the
    production model is feature-deterministic and gives R² ~0.9. See
    notebook 03 §6.10 for the rationale and breakdown.

    Returns a dict with everything the calling notebook needs for plots
    and the cross-variant summary (model, y_test, y_pred, importances,
    sizes, CV result, fidelity metrics).
    """
    cfg = VARIANT_FEATURES[variant]
    src = ("thirdparty_email_outbound.parquet"
           if variant in THIRDPARTY_VARIANTS
           else "luggage_email_outbound.parquet")

    df = pd.read_parquet(processed_dir / src)
    df = df[(df["pyName"] == variant) & (df["modelTechnique"] == technique)].reset_index(drop=True)

    present = [f for f in cfg.features if f in df.columns]
    active  = [f for f in present if df[f].nunique(dropna=False) > 1]
    dropped = sorted(set(cfg.features) - set(active))

    X, y, cat_cols, num_cols = build_feature_matrix(df, active, cfg.numeric)

    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bins,
    )

    if cv_depth:
        best_depth, cv_results = cv_select_depth(X_train, y_train, cat_cols)
    else:
        best_depth, cv_results = 6, None  # train_catboost default

    model    = train_catboost(X_train, y_train, cat_cols,
                              depth=best_depth, iterations=final_iterations)
    y_pred   = model.predict(X_test)
    fidelity = fidelity_suite(y_test.values, y_pred)

    importances = pd.Series(
        model.get_feature_importance(),
        index=X_train.columns,
        name="importance",
    ).sort_values(ascending=False)

    art_dir = artifact_root / variant
    art_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(art_dir / "catboost_model.cbm"))
    (art_dir / "feature_cols.json").write_text(json.dumps(list(X.columns)))
    (art_dir / "cat_cols.json").write_text(json.dumps(cat_cols))
    (art_dir / "num_cols.json").write_text(json.dumps(num_cols))
    np.save(art_dir / "train_idx.npy", X_train.index.to_numpy())
    np.save(art_dir / "test_idx.npy",  X_test.index.to_numpy())
    (art_dir / "fidelity.json").write_text(
        json.dumps({**fidelity, "model": f"CatBoost ({variant})",
                    "depth": best_depth}, indent=2)
    )

    return {
        "variant":     variant,
        "n":           len(df),
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "n_features":  X.shape[1],
        "n_cat":       len(cat_cols),
        "n_num":       len(num_cols),
        "dropped":     dropped,
        "best_depth":  best_depth,
        "cv_results":  cv_results,
        "model":       model,
        "y_test":      y_test,
        "y_pred":      y_pred,
        "importances": importances,
        **fidelity,
    }

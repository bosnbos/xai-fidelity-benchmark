"""Surrogate model building, training, and evaluation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from my_project.metrics import fidelity_suite

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

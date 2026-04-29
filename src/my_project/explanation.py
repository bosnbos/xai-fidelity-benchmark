"""Explanation utilities: Decision Tree surrogate, SHAP, and LIME."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor, export_text
from scipy.stats import spearmanr as _spearmanr

from my_project.metrics import feature_ranking


# ── Decision Tree surrogate ────────────────────────────────────────────────

def _encode_for_sklearn(
    X: pd.DataFrame,
    cat_cols: list[str],
    encoder: OrdinalEncoder | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, OrdinalEncoder]:
    """Ordinal-encode categorical columns; leave numerics as-is."""
    X_num = X.copy()
    if fit:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        X_num[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
    else:
        X_num[cat_cols] = encoder.transform(X[cat_cols].astype(str))
    return X_num.values.astype(float), encoder


def dt_surrogate(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
    max_depth_range: range = range(1, 11),
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[DecisionTreeRegressor, int, pd.DataFrame, OrdinalEncoder]:
    """Fit a Decision Tree surrogate via CV depth selection on Spearman ρ.

    Returns (tree, best_depth, cv_df, encoder).
    cv_df has columns [mean_rho, std_rho] indexed by depth.
    The returned tree is refitted on the full X with best_depth.
    """
    X_enc, encoder = _encode_for_sklearn(X, cat_cols, fit=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for depth in max_depth_range:
        fold_rhos = []
        for tr_idx, val_idx in kf.split(X_enc):
            dt = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
            dt.fit(X_enc[tr_idx], y.iloc[tr_idx])
            rho, _ = _spearmanr(y.iloc[val_idx], dt.predict(X_enc[val_idx]))
            fold_rhos.append(rho)
        rows.append({
            "depth":    depth,
            "mean_rho": round(float(np.mean(fold_rhos)), 4),
            "std_rho":  round(float(np.std(fold_rhos)),  4),
        })

    cv_df = pd.DataFrame(rows).set_index("depth")
    best_depth = int(cv_df["mean_rho"].idxmax())

    tree = DecisionTreeRegressor(max_depth=best_depth, random_state=random_state)
    tree.fit(X_enc, y)
    return tree, best_depth, cv_df, encoder


def dt_importances(tree: DecisionTreeRegressor, feature_names: list[str]) -> pd.Series:
    """Return feature importances from a fitted Decision Tree as a ranked Series."""
    imp = pd.Series(tree.feature_importances_, index=feature_names, name="importance")
    return imp.sort_values(ascending=False)


def dt_rules(
    tree: DecisionTreeRegressor,
    feature_names: list[str],
    max_depth: int | None = None,
) -> str:
    """Return a text representation of the decision tree rules."""
    return export_text(tree, feature_names=list(feature_names), max_depth=max_depth)


# ── SHAP ──────────────────────────────────────────────────────────────────

def shap_importances(
    model,
    X: pd.DataFrame,
    cat_cols: list[str],
    check_additivity: bool = False,
) -> tuple[np.ndarray, pd.Series]:
    """Compute SHAP values via TreeExplainer on a CatBoost model.

    Returns (shap_values array of shape (n, p), mean_abs pd.Series ranked by importance).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    return shap_values, mean_abs


# ── LIME ──────────────────────────────────────────────────────────────────

def _make_lime_predict_fn(model, feature_names: list[str], cat_cols: list[str]):
    """Return a predict function that converts LIME's float-encoded array back to
    the string format CatBoost expects for categorical columns."""
    cat_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    def predict_fn(X_arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X_arr, columns=feature_names)
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return model.predict(df).astype(float)

    return predict_fn


def lime_explain_batch(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    n_samples: int = 500,
    n_features: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Explain every row in X_test with LIME and return a tidy DataFrame.

    Columns: [instance_idx, feature, importance].
    n_features defaults to X_test.shape[1] (all features).
    """
    feature_names = list(X_test.columns)
    n_features = n_features or len(feature_names)
    cat_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    # LIME needs a numeric training array; represent categoricals by their
    # integer codes (LIME uses them only to compute perturbation distributions).
    X_train_num = X_train.copy()
    X_test_num = X_test.copy()
    cat_maps: dict[str, dict] = {}
    for col in cat_cols:
        codes = {v: i for i, v in enumerate(X_train[col].unique())}
        cat_maps[col] = codes
        X_train_num[col] = X_train[col].map(codes).fillna(-1).astype(float)
        X_test_num[col] = X_test[col].map(codes).fillna(-1).astype(float)

    explainer = LimeTabularExplainer(
        training_data=X_train_num.values.astype(float),
        feature_names=feature_names,
        categorical_features=cat_indices,
        mode="regression",
        random_state=random_state,
    )

    predict_fn = _make_lime_predict_fn(model, feature_names, cat_cols)

    rows = []
    for i in range(len(X_test_num)):
        exp = explainer.explain_instance(
            X_test_num.iloc[i].values.astype(float),
            predict_fn,
            num_features=n_features,
            num_samples=n_samples,
        )
        for feat_label, importance in exp.as_list():
            rows.append({"instance_idx": i, "feature": feat_label, "importance": importance})

    return pd.DataFrame(rows)


def aggregate_lime_importances(lime_df: pd.DataFrame) -> pd.Series:
    """Aggregate per-instance LIME importances into a global mean |importance| ranking.

    lime_df must have columns [instance_idx, feature, importance].
    Returns a pd.Series indexed by feature, sorted descending.
    """
    agg = (
        lime_df.assign(abs_importance=lambda d: d["importance"].abs())
        .groupby("feature")["abs_importance"]
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_lime")
    )
    return agg

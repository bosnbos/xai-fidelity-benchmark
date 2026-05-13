"""Explanation utilities: Decision Tree surrogate, SHAP, and LIME."""
from __future__ import annotations

import re as _re

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

class PegaBinEncoder:
    """Encodes categorical features using Pega ADM's learned bin groupings.

    Each Pega symbolic predictor is divided into bins whose members share
    similar lift (Z-ratio). Using these bins instead of an arbitrary OrdinalEncoder
    means every DT split corresponds to a semantically valid Pega category boundary,
    e.g. DestinationAirport in {MISSING | AMS | ORY | all others} instead of
    alphabetical code <= 73.

    Exposes a categories_ attribute compatible with annotate_dt_rules so that
    split thresholds are decoded to the actual category sets per bin.
    """

    def __init__(
        self,
        pega_bins: dict[str, dict[int, list[str]]],
        cat_cols: list[str],
        feature_name_map: dict[str, str] | None = None,
    ) -> None:
        """
        pega_bins        : output of parsing.load_pega_bins()
        cat_cols         : ordered list of categorical column names in the feature matrix
        feature_name_map : optional {col_name → pega_predictor_name} for name mismatches
        """
        self.cat_cols = list(cat_cols)
        fmap = feature_name_map or {}

        self._value_to_bin: dict[str, dict[str, int]] = {}
        self.categories_: list[list[str]] = []   # [bin_label, ...] per feature

        for col in cat_cols:
            pega_name = fmap.get(col, col.replace("param::", ""))
            if pega_name in pega_bins:
                sorted_bins = sorted(pega_bins[pega_name].items())   # [(pega_idx, [vals]), ...]
                v2b: dict[str, int] = {}
                labels: list[str] = []
                for local_idx, (_, values) in enumerate(sorted_bins):
                    for v in values:
                        v2b[v] = local_idx
                        if v == "MISSING":              # Pega's missing marker
                            v2b["__MISSING__"] = local_idx
                    if len(values) <= 4:
                        labels.append(", ".join(values))
                    else:
                        labels.append(", ".join(values[:3]) + f", ... (+{len(values) - 3})")
                self._value_to_bin[col] = v2b
                self.categories_.append(labels)
            else:
                self._value_to_bin[col] = {}
                self.categories_.append(["__all__"])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        for col in self.cat_cols:
            mapping = self._value_to_bin.get(col, {})
            if mapping:
                n_bins = len(self.categories_[self.cat_cols.index(col)])
                X_enc[col] = (
                    X[col].astype(str)
                    .map(lambda v, m=mapping, n=n_bins - 1: float(m.get(v, n)))
                )
            else:
                X_enc[col] = 0.0
        return X_enc


def _encode_for_sklearn(
    X: pd.DataFrame,
    cat_cols: list[str],
    encoder: OrdinalEncoder | PegaBinEncoder | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, OrdinalEncoder | PegaBinEncoder]:
    """Encode categorical columns for sklearn; leave numerics as-is.

    If encoder is a PegaBinEncoder, uses Pega's learned bin groupings.
    If encoder is None and fit=True, fits and applies an OrdinalEncoder.
    """
    X_num = X.copy()
    if isinstance(encoder, PegaBinEncoder):
        X_num[cat_cols] = encoder.transform(X[cat_cols])
    elif fit:
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
    encoder: OrdinalEncoder | PegaBinEncoder | None = None,
    selection: str = "1sd",
) -> tuple[DecisionTreeRegressor, int, pd.DataFrame, OrdinalEncoder | PegaBinEncoder]:
    """Fit a Decision Tree surrogate via CV depth selection on Spearman ρ.

    `selection`:
        "max" – classic argmax of mean Spearman ρ.
        "1sd" – Breiman's parsimony rule: smallest depth whose mean ρ is
                within one fold-to-fold standard deviation of the max.
                Matches the rule used for the CatBoost surrogate
                (`cv_select_depth` in surrogate.py) and aligns with the DT's
                interpretability goal by favouring shallower trees.

    Returns (tree, best_depth, cv_df, encoder).
    cv_df has columns [mean_rho, std_rho] indexed by depth.
    The returned tree is refitted on the full X with best_depth.

    Pass a PegaBinEncoder as encoder to use Pega's learned bin groupings
    instead of the default alphabetical OrdinalEncoder.
    """
    if encoder is None:
        X_enc, encoder = _encode_for_sklearn(X, cat_cols, fit=True)
    else:
        X_enc, _ = _encode_for_sklearn(X, cat_cols, encoder=encoder)

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
    max_depth = int(cv_df["mean_rho"].idxmax())
    if selection == "max":
        best_depth = max_depth
    elif selection == "1sd":
        max_mean   = cv_df.loc[max_depth, "mean_rho"]
        max_std    = cv_df.loc[max_depth, "std_rho"]
        in_band    = cv_df[cv_df["mean_rho"] >= max_mean - max_std]
        best_depth = int(in_band.index.min())
    else:
        raise ValueError(f"selection must be 'max' or '1sd', got {selection!r}")

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
    # export_text requires an explicit int for max_depth in newer sklearn versions.
    depth = max_depth if max_depth is not None else tree.get_depth()
    return export_text(tree, feature_names=list(feature_names), max_depth=depth)


def annotate_dt_rules(
    rules_text: str,
    cat_cols: list[str],
    encoder: OrdinalEncoder,
) -> str:
    """Replace ordinal-encoded thresholds with readable category sets.

    OrdinalEncoder assigns integer codes 0, 1, 2, ... to sorted unique category
    values. A split 'col <= k.5' therefore means 'col is one of the first (k+1)
    categories alphabetically'. This function decodes that back to the actual
    category strings so rules like 'BundleName <= 5.50' become
    'BundleName in {BUNDLE_A, BUNDLE_B, ...}'.

    Numeric features are left unchanged.
    """
    cat_categories: dict[str, list[str]] = {
        col: list(encoder.categories_[i]) for i, col in enumerate(cat_cols)
    }

    lines = []
    for line in rules_text.splitlines():
        # Each split line produced by export_text looks like:
        #   "|--- feature_name <= 5.50"  or  "|   |--- feature_name > 2.50"
        m = _re.match(r'^(.*?\|---\s+)(.*?)\s*(<=|>)\s*([\d.]+)\s*$', line)
        if m:
            prefix, feature, op, threshold = (
                m.group(1), m.group(2), m.group(3), float(m.group(4))
            )
            if feature in cat_categories:
                cats = cat_categories[feature]
                k = int(threshold)          # 5.50 → 5
                chosen = cats[:k + 1] if op == "<=" else cats[k + 1:]
                if len(chosen) <= 5:
                    cat_str = "{" + ", ".join(chosen) + "}"
                else:
                    cat_str = "{" + ", ".join(chosen[:4]) + f", ... +{len(chosen) - 4} more}}"
                lines.append(f"{prefix}{feature} in {cat_str}")
                continue
        lines.append(line)
    return "\n".join(lines)


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

    Uses discretize_continuous=True (quartile binning) to avoid scipy.stats.truncnorm
    scale=0 errors that occur when numerical features have zero or NaN-derived variance.
    Categorical columns are integer-coded for LIME and decoded back to their original
    string values before each CatBoost prediction call.
    """
    feature_names = list(X_test.columns)
    n_features = n_features or len(feature_names)
    cat_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    # Encode categoricals as integer codes; store reverse map for predict_fn.
    cat_decoders: dict[str, list] = {}
    for col in cat_cols:
        unique_vals = list(X_train[col].unique())
        cat_decoders[col] = unique_vals
        code_map = {v: i for i, v in enumerate(unique_vals)}
        X_train_enc[col] = X_train[col].map(code_map).fillna(0).astype(float)
        X_test_enc[col] = X_test[col].map(code_map).fillna(0).astype(float)

    # Fill NaN in numerical columns with training column mean so LIME sees no NaN.
    col_means: dict[str, float] = {}
    for col in num_cols:
        if col in X_train_enc.columns:
            mean_val = float(np.nanmean(X_train_enc[col].values))
            col_means[col] = 0.0 if np.isnan(mean_val) else mean_val
            X_train_enc[col] = X_train_enc[col].fillna(col_means[col])
            X_test_enc[col] = X_test_enc[col].fillna(col_means[col])

    def predict_fn(X_arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X_arr, columns=feature_names)
        for col in cat_cols:
            decoder = cat_decoders[col]
            codes = df[col].round().astype(int).clip(0, len(decoder) - 1)
            df[col] = codes.map(lambda i, d=decoder: d[i])
        return model.predict(df).astype(float)

    # discretize_continuous=True uses quartile binning instead of truncnorm sampling,
    # which avoids scale=0 errors for constant or near-constant numerical features.
    explainer = LimeTabularExplainer(
        training_data=X_train_enc.values.astype(float),
        feature_names=feature_names,
        categorical_features=cat_indices,
        discretize_continuous=True,
        mode="regression",
        random_state=random_state,
    )

    rows = []
    for i in range(len(X_test_enc)):
        exp = explainer.explain_instance(
            X_test_enc.iloc[i].values.astype(float),
            predict_fn,
            num_features=n_features,
            num_samples=n_samples,
        )
        for feat_label, importance in exp.as_list():
            rows.append({"instance_idx": i, "feature": feat_label, "importance": importance})

    return pd.DataFrame(rows)


def compute_split_rankings(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    cb_model,
    X_background: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    pega_enc: PegaBinEncoder,
    lime_sample: int = 500,
    lime_n_samples: int = 200,
    random_state: int = 42,
) -> dict:
    """Compute DT, SHAP, and LIME importance rankings for a data subset.

    DT is fitted directly on y_sub (Pega propensity scores), introducing one
    layer of approximation. SHAP and LIME use cb_model as the prediction oracle.
    LIME is computed on a random sample of lime_sample instances for feasibility.

    Returns {"dt": Series, "shap": Series, "lime": Series}.
    """
    feature_names = list(X_sub.columns)

    dt, _, _, _ = dt_surrogate(
        X_sub, y_sub, cat_cols, num_cols,
        max_depth_range=range(1, 9), n_splits=5,
        encoder=pega_enc, random_state=random_state,
    )
    dt_imp = dt_importances(dt, feature_names)

    _, shap_imp = shap_importances(cb_model, X_sub, cat_cols)

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(X_sub), size=min(lime_sample, len(X_sub)), replace=False)
    X_lime = X_sub.iloc[sample_idx]
    lime_df = lime_explain_batch(
        cb_model, X_background, X_lime, cat_cols, num_cols,
        n_samples=lime_n_samples, n_features=len(feature_names),
    )
    lime_df["base_feature"] = lime_df["feature"].apply(
        lambda lbl: next(
            (fn for fn in sorted(feature_names, key=len, reverse=True) if fn in lbl), lbl
        )
    )
    lime_imp = (
        lime_df.assign(abs_imp=lambda d: d["importance"].abs())
        .groupby("base_feature")["abs_imp"].mean()
        .sort_values(ascending=False)
        .rename("mean_abs_lime")
    )

    return {"dt": dt_imp, "shap": shap_imp, "lime": lime_imp}


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

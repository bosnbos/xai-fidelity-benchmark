"""Fidelity and stability metrics for surrogate evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, ks_2samp, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def fidelity_suite(y_true, y_pred) -> dict[str, float]:
    """R², RMSE, Spearman ρ, Kendall τ, and KS statistic in one call.

    Returns a flat dict that serialises as a single CSV row.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    ks,  _ = ks_2samp(y_true, y_pred)
    return {
        "r2":           round(float(r2_score(y_true, y_pred)), 4),
        "rmse":         round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "spearman_rho": round(float(rho), 4),
        "kendall_tau":  round(float(tau), 4),
        "ks_stat":      round(float(ks), 4),
    }


def feature_ranking(importances: pd.Series) -> pd.Series:
    """Convert importance scores to 1-based ranks (highest importance = rank 1)."""
    return importances.rank(ascending=False, method="min").astype(int)


def stability_spearman(rank_a: pd.Series, rank_b: pd.Series) -> float:
    """Spearman ρ between two feature rank Series aligned on their index."""
    aligned = pd.DataFrame({"a": rank_a, "b": rank_b}).dropna()
    if len(aligned) < 3:
        return float("nan")
    rho, _ = spearmanr(aligned["a"], aligned["b"])
    return float(rho)


def jaccard_at_k(rank_a: pd.Series, rank_b: pd.Series, k: int) -> float:
    """Jaccard similarity of the top-k feature sets from two rank Series."""
    top_a = set(rank_a.nsmallest(k).index)
    top_b = set(rank_b.nsmallest(k).index)
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def stability_row(
    rankings_a: dict,
    rankings_b: dict,
    label_a: str,
    label_b: str,
    n_a: int,
    n_b: int,
    shared_features: list[str],
) -> list[dict]:
    """Return one stability result dict per method for a split pair.

    rankings_a / rankings_b must be dicts with keys "dt", "shap", "lime",
    each mapping to a pd.Series of importance scores indexed by feature name.
    """
    rows = []
    for method in ["DT", "SHAP", "LIME"]:
        key = method.lower()
        imp_a = rankings_a[key].reindex(shared_features).fillna(0)
        imp_b = rankings_b[key].reindex(shared_features).fillna(0)
        rank_a = feature_ranking(imp_a)
        rank_b = feature_ranking(imp_b)
        rows.append({
            "split":      f"{label_a} vs {label_b}",
            "method":     method,
            "n_a":        n_a,
            "n_b":        n_b,
            "Spearman ρ": round(stability_spearman(rank_a, rank_b), 4),
            "Jaccard@5":  round(jaccard_at_k(rank_a, rank_b, k=5),  4),
            "Jaccard@10": round(jaccard_at_k(rank_a, rank_b, k=10), 4),
        })
    return rows


_PSI_THRESHOLDS = [
    (0.10, "stable"),
    (0.25, "moderate"),
    (float("inf"), "significant"),
]


def psi(scores_a, scores_b, bins: int = 10) -> tuple[float, str]:
    """Population Stability Index between two score distributions.

    scores_a is the reference; scores_b is the comparison population.
    Bins are derived from the reference distribution via percentiles to
    avoid empty reference bins.

    Returns (psi_value, label) where label ∈ {"stable", "moderate", "significant"}.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    breakpoints = np.nanpercentile(scores_a, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0, "stable"

    def _fracs(x: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(x, bins=breakpoints)
        counts = np.where(counts == 0, 1e-6, counts.astype(float))
        return counts / counts.sum()

    p_ref = _fracs(scores_a)
    p_cur = _fracs(scores_b)
    psi_val = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
    label = next(lbl for thresh, lbl in _PSI_THRESHOLDS if psi_val < thresh)
    return psi_val, label

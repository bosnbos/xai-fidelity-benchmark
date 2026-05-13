"""Re-run the surrogate + DT/SHAP explanations stratified by modelTechnique.

Trains a separate CatBoost surrogate for each (variant, technique) pair, computes
DT and SHAP global importances, and reports cross-method agreement (Spearman rho,
Jaccard@5, Jaccard@10) within each stratum and across strata.

Compares to the mixed-data baseline saved in data/artifacts/{variant}/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from my_project.features import VARIANT_FEATURES
from my_project.surrogate import (
    build_feature_matrix,
    train_catboost,
    evaluate_surrogate,
)
from my_project.explanation import (
    dt_surrogate,
    dt_importances,
    shap_importances,
)
from my_project.metrics import (
    feature_ranking,
    stability_spearman,
    jaccard_at_k,
)


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed"
ART = REPO / "data" / "artifacts"

VARIANT_TO_FILE = {
    "L5B15": "luggage_email_outbound.parquet",
    "CLUG":  "luggage_email_outbound.parquet",
    "BookingDotCom": "thirdparty_email_outbound.parquet",
    "Cartrawler":    "thirdparty_email_outbound.parquet",
}


def compare(rank_a: pd.Series, rank_b: pd.Series, shared: list[str]) -> dict:
    a = feature_ranking(rank_a.reindex(shared).fillna(0))
    b = feature_ranking(rank_b.reindex(shared).fillna(0))
    return {
        "spearman_rho": round(stability_spearman(a, b), 4),
        "jaccard_5":    round(jaccard_at_k(a, b, 5),     4),
        "jaccard_10":   round(jaccard_at_k(a, b, 10),    4),
    }


def run_variant(variant: str, results: dict) -> None:
    cfg = VARIANT_FEATURES[variant]
    pega_features = list(cfg.features)
    numeric_features = cfg.numeric

    raw = pd.read_parquet(DATA / VARIANT_TO_FILE[variant])
    df_var = raw[raw["pyName"] == variant].copy()

    techniques = df_var["modelTechnique"].dropna().unique().tolist()
    print(f"\n{'='*70}\n{variant}  (n_total={len(df_var):,})  techniques={techniques}\n{'='*70}")

    per_tech = {}
    for tech in techniques:
        sub = df_var[df_var["modelTechnique"] == tech].copy()
        if len(sub) < 200:
            print(f"  skip {tech!r}: n={len(sub)} too small")
            continue

        X, y, cat_cols, num_cols = build_feature_matrix(sub, pega_features, numeric_features)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        cb = train_catboost(X_tr, y_tr, cat_cols, iterations=400, depth=6, learning_rate=0.05)
        fidelity = evaluate_surrogate(cb, X_te, y_te, model_name=f"CatBoost ({variant}/{tech})")

        # SHAP on a manageable test slice
        X_shap = X_te.iloc[:min(2000, len(X_te))]
        _, shap_imp = shap_importances(cb, X_shap, cat_cols)

        # DT on the full subset (CV depth selection 1..8)
        dt, best_depth, _, _ = dt_surrogate(
            X, y, cat_cols, num_cols,
            max_depth_range=range(1, 9), n_splits=5, random_state=42,
        )
        dt_imp = dt_importances(dt, list(X.columns))

        shared = sorted(set(shap_imp.index) & set(dt_imp.index))
        cross = compare(shap_imp, dt_imp, shared)

        print(f"\n  technique = {tech!r}  (n={len(sub):,})")
        print(f"    fidelity  : R2={fidelity['r2']:.4f}  rho={fidelity['spearman_rho']:.4f}  rmse={fidelity['rmse']:.6f}")
        print(f"    DT depth  : {best_depth}")
        print(f"    DT vs SHAP: rho={cross['spearman_rho']:.4f}  J@5={cross['jaccard_5']}  J@10={cross['jaccard_10']}")
        print(f"    top-5 SHAP: {list(shap_imp.head(5).index)}")
        print(f"    top-5 DT  : {list(dt_imp.head(5).index)}")

        per_tech[str(tech)] = {
            "n": int(len(sub)),
            "fidelity": fidelity,
            "best_dt_depth": best_depth,
            "dt_vs_shap": cross,
            "top5_shap": list(shap_imp.head(5).index),
            "top5_dt":   list(dt_imp.head(5).index),
            "shap_full": shap_imp.to_dict(),
            "dt_full":   dt_imp.to_dict(),
        }

    # Cross-technique agreement: do the two techniques produce the same SHAP rankings?
    if len(per_tech) == 2:
        keys = list(per_tech.keys())
        shap_a = pd.Series(per_tech[keys[0]]["shap_full"])
        shap_b = pd.Series(per_tech[keys[1]]["shap_full"])
        shared_ab = sorted(set(shap_a.index) & set(shap_b.index))
        cross_tech_shap = compare(shap_a, shap_b, shared_ab)
        dt_a = pd.Series(per_tech[keys[0]]["dt_full"])
        dt_b = pd.Series(per_tech[keys[1]]["dt_full"])
        shared_ab_dt = sorted(set(dt_a.index) & set(dt_b.index))
        cross_tech_dt = compare(dt_a, dt_b, shared_ab_dt)
        print(f"\n  cross-technique agreement ({keys[0]} vs {keys[1]}):")
        print(f"    SHAP-vs-SHAP: {cross_tech_shap}")
        print(f"    DT-vs-DT    : {cross_tech_dt}")
        results[variant] = {
            "per_technique": per_tech,
            "cross_technique_shap": cross_tech_shap,
            "cross_technique_dt":   cross_tech_dt,
        }
    else:
        results[variant] = {"per_technique": per_tech}


def main():
    results: dict = {}
    for v in ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]:
        run_variant(v, results)

    out = REPO / "data" / "artifacts" / "stratified_by_technique.json"
    # strip the full importance dicts before saving to keep file small
    slim = {}
    for v, d in results.items():
        slim[v] = {
            "per_technique": {
                t: {k: v for k, v in r.items() if k not in ("shap_full","dt_full")}
                for t, r in d["per_technique"].items()
            },
            "cross_technique_shap": d.get("cross_technique_shap"),
            "cross_technique_dt":   d.get("cross_technique_dt"),
        }
    with open(out, "w") as f:
        json.dump(slim, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

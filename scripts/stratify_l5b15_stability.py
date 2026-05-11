"""Stratified stability + attribution analysis for L5B15, one technique at a time.

Mirrors notebooks 07 and 08 but partitions the data by `modelTechnique` first,
training a separate CatBoost surrogate per technique and running every split
within the technique. Outputs:

  data/artifacts/l5b15/stratified_stability_attribution.json

Use to test whether the route-/culture-split "explainer-driven" attribution
(δ_e dominant) holds up once the two parallel ADM techniques are separated.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from my_project.explanation import (
    PegaBinEncoder,
    compute_split_rankings,
)
from my_project.features import PEGA_MODEL_IDS, VARIANT_FEATURES
from my_project.metrics import (
    feature_ranking,
    jaccard_at_k,
    psi,
    stability_row,
    stability_spearman,
)
from my_project.parsing import load_pega_bins
from my_project.surrogate import build_feature_matrix, train_catboost

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed"
ART = REPO / "data" / "artifacts"
ADM_BINNING = REPO / "data" / "adm" / "data-Predictor-binning-snapshot.json"

VARIANT = "L5B15"
PSI_BINS = 5

# LIME tuning — smaller than notebook 07 (500/200) so all 8 splits across two
# techniques fit in a reasonable runtime. Methodology unchanged.
LIME_SAMPLE = 300
LIME_N_SAMPLES = 150

MIN_ROUTE_OBS_PER_TECH = 250
TOP_N_ROUTES_PER_TECH = 3

CULTURE_COL = "CustBookedFlight.BookingData.CultureCode"
CULTURE_CODES = ["fr-FR", "nl-NL"]


def _stability_per_method(rankings_a, rankings_b, label_a, label_b, n_a, n_b, shared):
    return stability_row(rankings_a, rankings_b, label_a, label_b, n_a, n_b, shared)


def _classify_primary_source(psi_d, ks_m, ks_p, delta_e):
    """Same heuristic as notebook 08 §20."""
    sig_m = (ks_m >= 0.10) and (ks_p < 0.05)
    sig_d = psi_d >= 0.10
    sig_e = delta_e >= 0.10
    if sig_d and sig_m:
        return "Both (δ_d + δ_m)"
    if sig_m and not sig_d:
        return "Model churn (δ_m)"
    if sig_d and not sig_m:
        return "Distribution shift (δ_d)"
    if sig_e and not sig_d and not sig_m:
        return "Explainer (δ_e)"
    return "Mixed"


def run_for_technique(tech: str, df_full: pd.DataFrame, X_full: pd.DataFrame,
                      y_full: pd.Series, meta_full: pd.DataFrame,
                      cat_cols: list[str], num_cols: list[str],
                      pega_enc: PegaBinEncoder) -> dict:
    print(f"\n{'='*70}\n{VARIANT} — technique = {tech!r}\n{'='*70}")
    mask_tech = df_full["modelTechnique"] == tech
    df = df_full[mask_tech].reset_index(drop=True)
    X = X_full[mask_tech].reset_index(drop=True)
    y = y_full[mask_tech].reset_index(drop=True)
    meta = meta_full[mask_tech].reset_index(drop=True)

    print(f"  rows: {len(df):,}")

    # Train per-technique CatBoost (so SHAP/LIME use the right oracle)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cb = train_catboost(X_tr, y_tr, cat_cols, iterations=400, depth=6, learning_rate=0.05)

    # Background for LIME = the train half (matches the convention in notebook 07)
    X_background = X_tr

    output: dict = {"n_total": int(len(df)), "stability": [], "attribution": []}

    # ── §13 Temporal split (within technique) ──────────────────────────────
    median_t = meta["pxDecisionTime"].median()
    mask_early = meta["pxDecisionTime"] <= median_t
    mask_late = meta["pxDecisionTime"] > median_t
    n_early, n_late = int(mask_early.sum()), int(mask_late.sum())
    print(f"\n  temporal split @ {median_t}: early={n_early:,}, late={n_late:,}")

    print("    rankings: early …")
    r_early = compute_split_rankings(
        X[mask_early], y[mask_early], cb, X_background, cat_cols, num_cols, pega_enc,
        lime_sample=LIME_SAMPLE, lime_n_samples=LIME_N_SAMPLES,
    )
    print("    rankings: late …")
    r_late = compute_split_rankings(
        X[mask_late], y[mask_late], cb, X_background, cat_cols, num_cols, pega_enc,
        lime_sample=LIME_SAMPLE, lime_n_samples=LIME_N_SAMPLES,
    )
    shared = list(X.columns)
    temporal_rows = stability_row(r_early, r_late, "early", "late", n_early, n_late, shared)
    for r in temporal_rows:
        r["split"] = f"L5B15/{tech} temporal"
    output["stability"].extend(temporal_rows)

    # Attribution for temporal split
    psi_d, _ = psi(y[mask_early], y[mask_late], bins=PSI_BINS)
    auc_a = meta.loc[mask_early, "modelPerformance"].dropna()
    auc_b = meta.loc[mask_late, "modelPerformance"].dropna()
    ks_m, ks_p = ks_2samp(auc_a, auc_b)
    rho = {row["method"]: row["Spearman ρ"] for row in temporal_rows}
    delta_e = max(0.0, rho["SHAP"] - rho["DT"])
    output["attribution"].append({
        "split": f"L5B15/{tech} temporal",
        "PSI (δ_d)": round(psi_d, 4),
        "KS_AUC (δ_m)": round(float(ks_m), 4),
        "KS p": round(float(ks_p), 4),
        "ρ_DT": rho["DT"], "ρ_SHAP": rho["SHAP"], "ρ_LIME": rho["LIME"],
        "Δρ_DT": round(1.0 - rho["DT"], 4),
        "Δρ_SHAP": round(1.0 - rho["SHAP"], 4),
        "δ_e (residual)": round(delta_e, 4),
        "Primary source": _classify_primary_source(psi_d, ks_m, ks_p, delta_e),
    })

    # ── §14 Route subgroups (within technique) ─────────────────────────────
    route_counts = meta["route"].value_counts()
    top_routes = route_counts[route_counts >= MIN_ROUTE_OBS_PER_TECH] \
                    .head(TOP_N_ROUTES_PER_TECH).index.tolist()
    print(f"\n  top-{TOP_N_ROUTES_PER_TECH} routes (≥{MIN_ROUTE_OBS_PER_TECH} obs): {top_routes}")

    route_rankings = {}
    for route in top_routes:
        mask = meta["route"] == route
        print(f"    rankings: {route}  (n={int(mask.sum()):,}) …")
        route_rankings[route] = compute_split_rankings(
            X[mask], y[mask], cb, X_background, cat_cols, num_cols, pega_enc,
            lime_sample=min(LIME_SAMPLE, int(mask.sum())),
            lime_n_samples=LIME_N_SAMPLES,
        )

    for i in range(len(top_routes) - 1):
        r1, r2 = top_routes[i], top_routes[i + 1]
        m1, m2 = (meta["route"] == r1), (meta["route"] == r2)
        rows = stability_row(route_rankings[r1], route_rankings[r2],
                             r1, r2, int(m1.sum()), int(m2.sum()), shared)
        for r in rows:
            r["split"] = f"L5B15/{tech} {r1} vs {r2}"
        output["stability"].extend(rows)

        # Attribution for this route pair
        psi_d, _ = psi(y[m1], y[m2], bins=PSI_BINS)
        auc_a = meta.loc[m1, "modelPerformance"].dropna()
        auc_b = meta.loc[m2, "modelPerformance"].dropna()
        ks_m, ks_p = ks_2samp(auc_a, auc_b) if len(auc_a) > 0 and len(auc_b) > 0 else (0.0, 1.0)
        rho_pair = {r_["method"]: r_["Spearman ρ"] for r_ in rows}
        delta_e = max(0.0, rho_pair["SHAP"] - rho_pair["DT"])
        output["attribution"].append({
            "split": f"L5B15/{tech} {r1} vs {r2}",
            "PSI (δ_d)": round(psi_d, 4),
            "KS_AUC (δ_m)": round(float(ks_m), 4),
            "KS p": round(float(ks_p), 4),
            "ρ_DT": rho_pair["DT"], "ρ_SHAP": rho_pair["SHAP"], "ρ_LIME": rho_pair["LIME"],
            "Δρ_DT": round(1.0 - rho_pair["DT"], 4),
            "Δρ_SHAP": round(1.0 - rho_pair["SHAP"], 4),
            "δ_e (residual)": round(delta_e, 4),
            "Primary source": _classify_primary_source(psi_d, ks_m, ks_p, delta_e),
        })

    # ── §14c Culture subgroup ──────────────────────────────────────────────
    culture_subsets = {}
    for code in CULTURE_CODES:
        mask = (df[CULTURE_COL] == code)
        culture_subsets[code] = mask
        print(f"  culture {code}: n={int(mask.sum()):,}")

    if all(culture_subsets[c].sum() >= 200 for c in CULTURE_CODES):
        culture_rankings = {}
        for code in CULTURE_CODES:
            mask = culture_subsets[code]
            print(f"    rankings: {code} …")
            culture_rankings[code] = compute_split_rankings(
                X[mask], y[mask], cb, X_background, cat_cols, num_cols, pega_enc,
                lime_sample=min(LIME_SAMPLE, int(mask.sum())),
                lime_n_samples=LIME_N_SAMPLES,
            )

        c1, c2 = CULTURE_CODES
        rows = stability_row(culture_rankings[c1], culture_rankings[c2],
                             c1, c2,
                             int(culture_subsets[c1].sum()),
                             int(culture_subsets[c2].sum()), shared)
        for r in rows:
            r["split"] = f"L5B15/{tech} {c1} vs {c2}"
        output["stability"].extend(rows)

        m1, m2 = culture_subsets[c1], culture_subsets[c2]
        psi_d, _ = psi(y[m1], y[m2], bins=PSI_BINS)
        auc_a = meta.loc[m1, "modelPerformance"].dropna()
        auc_b = meta.loc[m2, "modelPerformance"].dropna()
        ks_m, ks_p = ks_2samp(auc_a, auc_b) if len(auc_a) > 0 and len(auc_b) > 0 else (0.0, 1.0)
        rho_pair = {r_["method"]: r_["Spearman ρ"] for r_ in rows}
        delta_e = max(0.0, rho_pair["SHAP"] - rho_pair["DT"])
        output["attribution"].append({
            "split": f"L5B15/{tech} {c1} vs {c2}",
            "PSI (δ_d)": round(psi_d, 4),
            "KS_AUC (δ_m)": round(float(ks_m), 4),
            "KS p": round(float(ks_p), 4),
            "ρ_DT": rho_pair["DT"], "ρ_SHAP": rho_pair["SHAP"], "ρ_LIME": rho_pair["LIME"],
            "Δρ_DT": round(1.0 - rho_pair["DT"], 4),
            "Δρ_SHAP": round(1.0 - rho_pair["SHAP"], 4),
            "δ_e (residual)": round(delta_e, 4),
            "Primary source": _classify_primary_source(psi_d, ks_m, ks_p, delta_e),
        })
    else:
        print(f"  skip culture split: insufficient rows in fr-FR or nl-NL within {tech}")

    return output


def main():
    cfg = VARIANT_FEATURES[VARIANT]
    pega_features = list(cfg.features)
    numeric_features = cfg.numeric

    print("Loading L5B15 data …")
    raw = pd.read_parquet(DATA / "luggage_email_outbound.parquet")
    raw["pxDecisionTime"] = pd.to_datetime(raw["pxDecisionTime"], utc=True, errors="coerce")
    df_full = raw[raw["pyName"] == VARIANT].reset_index(drop=True)
    print(f"  L5B15 rows: {len(df_full):,}")

    X_full, y_full, cat_cols, num_cols = build_feature_matrix(
        df_full, pega_features, numeric_features
    )

    meta_full = df_full[[
        "pxDecisionTime", "modelEvidence", "modelPerformance", "modelVersion",
        "CustBookedFlight.FlightData.DepartureAirport",
        "CustBookedFlight.FlightData.DestinationAirport",
    ]].copy()
    meta_full["route"] = (
        meta_full["CustBookedFlight.FlightData.DepartureAirport"].astype(str)
        + "->"
        + meta_full["CustBookedFlight.FlightData.DestinationAirport"].astype(str)
    )

    print("Loading Pega bins for DT encoder …")
    pega_bins = load_pega_bins(ADM_BINNING, PEGA_MODEL_IDS[VARIANT])
    pega_enc = PegaBinEncoder(pega_bins, cat_cols)
    print(f"  bins for {len(pega_bins)} predictors")

    techniques = sorted(df_full["modelTechnique"].dropna().unique().tolist())
    print(f"  techniques: {techniques}")

    out: dict = {"variant": VARIANT, "lime_sample": LIME_SAMPLE,
                 "lime_n_samples": LIME_N_SAMPLES, "psi_bins": PSI_BINS,
                 "by_technique": {}}

    for tech in techniques:
        out["by_technique"][str(tech)] = run_for_technique(
            tech, df_full, X_full, y_full, meta_full,
            cat_cols, num_cols, pega_enc,
        )

    out_path = ART / VARIANT.lower() / "stratified_stability_attribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Print compact summary
    print("\n" + "="*70)
    print("STABILITY SUMMARY (Spearman ρ by method × split × technique)")
    print("="*70)
    for tech, d in out["by_technique"].items():
        print(f"\n  technique = {tech!r}")
        for row in d["stability"]:
            print(f"    {row['split']:55s} {row['method']:5s}  "
                  f"ρ={row['Spearman ρ']:.4f}  J@5={row['Jaccard@5']}  J@10={row['Jaccard@10']}")

    print("\n" + "="*70)
    print("ATTRIBUTION SUMMARY")
    print("="*70)
    for tech, d in out["by_technique"].items():
        print(f"\n  technique = {tech!r}")
        for row in d["attribution"]:
            print(f"    {row['split']:55s}")
            print(f"      PSI(δ_d)={row['PSI (δ_d)']}  KS_AUC(δ_m)={row['KS_AUC (δ_m)']}  "
                  f"δ_e={row['δ_e (residual)']}  → {row['Primary source']}")


if __name__ == "__main__":
    main()

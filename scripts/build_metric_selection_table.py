#!/usr/bin/env python3
"""Build the model-version-churn metric-selection CSV.

Computes three candidate metrics on the ``modelPerformance`` (AUC)
distribution between split halves, for the same splits used in the main
analysis:

    AUC PSI         — equal-frequency PSI with B = 5 bins
    delta_mu_AUC    — absolute difference in mean AUC between halves
    KS, KS_p        — two-sample Kolmogorov-Smirnov + p-value

Rows: one per (variant temporal) for all four variants, plus three L5B15
adjacent route pairs (matching the routes used in nb 07 / nb 08).

Writes:
    data/artifacts/metric_selection.csv

The CSV is consumed by ``scripts/build_thesis_tables.py`` (target
``metric_selection``) to produce ``tables/metric_selection.tex``.

Run from repo root:
    uv run python scripts/build_metric_selection_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

REPO     = Path(__file__).resolve().parent.parent
ART_ROOT = REPO / "data" / "artifacts"
DATA_DIR = REPO / "data" / "processed"

sys.path.insert(0, str(REPO / "src"))
from my_project.parsing import THIRDPARTY_VARIANTS  # noqa: E402
from my_project.metrics import psi                  # noqa: E402

VARIANTS    = ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]
TOP_ROUTES  = ["ORY->NCE", "ORY->OPO", "NCE->ORY", "TLS->ORY"]
PSI_BINS    = 5


def _load_variant_df(variant: str) -> pd.DataFrame:
    src = ("thirdparty_email_outbound.parquet"
           if variant in THIRDPARTY_VARIANTS
           else "luggage_email_outbound.parquet")
    df = pd.read_parquet(DATA_DIR / src)
    df = df[df["pyName"] == variant].reset_index(drop=True)
    df["pxDecisionTime"] = pd.to_datetime(df["pxDecisionTime"], utc=True,
                                          errors="coerce")
    df["route"] = (
        df["CustBookedFlight.FlightData.DepartureAirport"].astype(str)
        + "->"
        + df["CustBookedFlight.FlightData.DestinationAirport"].astype(str)
    )
    return df


def _metrics(s_a: pd.Series, s_b: pd.Series) -> dict[str, float]:
    psi_v, _ = psi(s_a, s_b, bins=PSI_BINS)
    mean_diff = abs(float(s_b.mean()) - float(s_a.mean()))
    ks_stat, ks_p = ks_2samp(s_a, s_b)
    return {
        "AUC_PSI":      round(float(psi_v),    4),
        "delta_mu_AUC": round(float(mean_diff), 4),
        "KS":           round(float(ks_stat),  4),
        "KS_p":         round(float(ks_p),     4),
    }


def main() -> int:
    print("Computing AUC-churn metric comparison...")
    rows: list[dict] = []

    # Temporal: one row per variant
    for v in VARIANTS:
        df = _load_variant_df(v)
        split_t = df["pxDecisionTime"].median()
        early = df.loc[df["pxDecisionTime"] <= split_t, "modelPerformance"].dropna()
        late  = df.loc[df["pxDecisionTime"] >  split_t, "modelPerformance"].dropna()
        m = _metrics(early, late)
        rows.append({"split": f"{v} temporal", "split_type": "temporal", **m})
        print(f"  {v} temporal  KS={m['KS']:.4f}  AUC_PSI={m['AUC_PSI']:.4f}")

    # Route pairs: L5B15 only
    df_l5 = _load_variant_df("L5B15")
    route_masks = {r: df_l5["route"] == r for r in TOP_ROUTES}
    for i in range(len(TOP_ROUTES) - 1):
        r1, r2 = TOP_ROUTES[i], TOP_ROUTES[i + 1]
        s_a = df_l5.loc[route_masks[r1], "modelPerformance"].dropna()
        s_b = df_l5.loc[route_masks[r2], "modelPerformance"].dropna()
        if len(s_a) < 50 or len(s_b) < 50:
            print(f"  SKIP {r1} vs {r2}: too few records ({len(s_a)} vs {len(s_b)})")
            continue
        m = _metrics(s_a, s_b)
        rows.append({"split": f"{r1} vs {r2}", "split_type": "route", **m})
        print(f"  {r1} vs {r2}  KS={m['KS']:.4f}  AUC_PSI={m['AUC_PSI']:.4f}")

    df_out = pd.DataFrame(rows)
    out_path = ART_ROOT / "metric_selection.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print()
    print(df_out.to_string(index=False))
    print()
    print("LaTeX table: uv run python scripts/build_thesis_tables.py metric_selection")
    return 0


if __name__ == "__main__":
    sys.exit(main())

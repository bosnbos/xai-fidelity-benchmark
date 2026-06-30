#!/usr/bin/env python3
"""Build the per-variant feature-group missingness CSV.

For every variant, loads its parquet and its saved
``feature_cols.json``, buckets each active predictor into one of five
groups by name prefix, and computes the mean fraction of missing values
across the group's columns.

Writes:
    data/artifacts/missingness.csv

The CSV is consumed by ``scripts/build_thesis_tables.py`` (target
``missingness``) to produce ``tables/missingness.tex``.

Group definitions (edit the `_GROUP_RULES` list below to retune):

    Scoring parameters   — predictor name starts with "Param."
    Contact history      — predictor name starts with "IH."
    Customer attributes  — predictor name starts with "Customer."
    Flight data          — under CustBookedFlight.FlightData / .Flight
    Booking context      — everything else under CustBookedFlight.*

Run from repo root:
    uv run python scripts/build_missingness_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO      = Path(__file__).resolve().parent.parent
ART_ROOT  = REPO / "data" / "artifacts"
DATA_DIR  = REPO / "data" / "processed"

sys.path.insert(0, str(REPO / "src"))
from my_project.parsing import THIRDPARTY_VARIANTS  # noqa: E402

VARIANTS = ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]

# Order matters for the output table; first matching rule wins.
_GROUP_RULES: list[tuple[str, str]] = [
    # CatBoost prefixes Param.* features with `param::` in feature_cols.json
    ("Scoring parameters",   "param::"),
    ("Scoring parameters",   "Param."),
    ("Contact history",      "IH."),
    ("Customer attributes",  "Customer."),
    ("Flight data",          "CustBookedFlight.FlightData."),
    ("Flight data",          "CustBookedFlight.Flight."),
    # BookingData sub-namespace → booking context; all other CustBookedFlight.* → flight data
    ("Booking context",      "CustBookedFlight.BookingData."),
    ("Flight data",          "CustBookedFlight."),
]


def _strip_catboost_prefix(feat: str) -> str:
    """Strip CatBoost's `param::` annotation to recover the actual column name."""
    return feat.removeprefix("param::")

_GROUP_ORDER = [
    "Booking context",
    "Flight data",
    "Contact history",
    "Scoring parameters",
    "Customer attributes",
]

_GROUP_DESC = {
    "Booking context":     "Fare class, journey type, product class, culture",
    "Flight data":         "Route, operator, aircraft type",
    "Contact history":     "Contact frequency and recency across channels",
    "Scoring parameters":  "Parameters passed at scoring time",
    "Customer attributes": "Demographics and loyalty profile",
}


def _categorize(feat: str) -> str:
    for group, prefix in _GROUP_RULES:
        if feat.startswith(prefix):
            return group
    return "Booking context"  # default fallback


def _load_variant_df(variant: str) -> pd.DataFrame:
    src = ("thirdparty_email_outbound.parquet"
           if variant in THIRDPARTY_VARIANTS
           else "luggage_email_outbound.parquet")
    df = pd.read_parquet(DATA_DIR / src)
    return df[df["pyName"] == variant].reset_index(drop=True)


def _compute_one(variant: str) -> dict[str, tuple[int | None, float | None]]:
    df = _load_variant_df(variant)
    feature_cols = json.loads(
        (ART_ROOT / variant / "feature_cols.json").read_text()
    )

    by_group: dict[str, list[str]] = {}
    for f in feature_cols:
        by_group.setdefault(_categorize(f), []).append(f)

    out: dict[str, tuple[int | None, float | None]] = {}
    for g in _GROUP_ORDER:
        feats = by_group.get(g, [])
        if not feats:
            out[g] = (None, None)
            continue
        cols = [_strip_catboost_prefix(f) for f in feats]
        present = [c for c in cols if c in df.columns]
        if not present:
            out[g] = (len(feats), None)
            continue
        miss = float(df[present].isna().mean().mean())
        out[g] = (len(feats), round(miss, 4))
    return out


def main() -> int:
    print("Computing missingness per variant per feature group...")
    per_variant = {}
    for v in VARIANTS:
        print(f"  {v}")
        per_variant[v] = _compute_one(v)

    rows = []
    for g in _GROUP_ORDER:
        row = {"group": g, "description": _GROUP_DESC[g]}
        for v in VARIANTS:
            n, miss = per_variant[v][g]
            row[f"{v}_n"]    = n
            row[f"{v}_miss"] = miss
        rows.append(row)
    df_out = pd.DataFrame(rows)

    out_path = ART_ROOT / "missingness.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print()
    print(df_out.to_string(index=False))
    print()
    print("LaTeX table: uv run python scripts/build_thesis_tables.py missingness")
    return 0


if __name__ == "__main__":
    sys.exit(main())

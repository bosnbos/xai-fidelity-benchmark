"""Build the RQ1 results table: per-variant fidelity of each explanation method.

For every variant in {L5B15, CLUG, BookingDotCom, Cartrawler}:
  - CatBoost surrogate fidelity is read from data/artifacts/{V}/fidelity.json
    (already produced by 05_surrogate_fit). Anchors both SHAP and LIME.
  - Decision Tree fidelity is read from data/artifacts/{V}/dt_fidelity.json if
    present (saved by the updated 06 §9.2). Otherwise we compute it directly
    from the saved dt_model.pkl + test indices, so this script works even if
    06 was last run before the dt_fidelity.json save line was added.

Saves:
  data/artifacts/rq1_explainer_fidelity.csv

The CatBoost row covers both SHAP and LIME because both methods anchor on the
same surrogate; their predictive fidelity to Pega ADM is therefore identical
to the surrogate's. This is noted in the table caption.

The thesis LaTeX table is produced by `scripts/build_thesis_tables.py`
(target: `rq1_explainer_fidelity`), which reads the CSV written here.

Run from repo root:
    uv run python scripts/build_rq1_table.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO     = Path(__file__).resolve().parent.parent
ART_ROOT = REPO / "data" / "artifacts"
DATA_DIR = REPO / "data" / "processed"
VARIANTS = ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]

sys.path.insert(0, str(REPO / "src"))
from my_project.explanation import _encode_for_sklearn  # noqa: E402
from my_project.features    import VARIANT_FEATURES     # noqa: E402
from my_project.metrics     import fidelity_suite       # noqa: E402
from my_project.parsing     import THIRDPARTY_VARIANTS  # noqa: E402
from my_project.surrogate   import build_feature_matrix # noqa: E402


def _compute_dt_fidelity_from_pkl(variant: str) -> dict:
    """Re-evaluate DT fidelity from the saved dt_model.pkl + test indices.

    Used when dt_fidelity.json hasn't been written yet by 06. The DT model
    and its encoder are deterministic given the saved indices, so this
    matches what 06's §9.2 would print.
    """
    art_dir = ART_ROOT / variant

    # Load the parquet with the same filters 05 / 06 use
    src = ("thirdparty_email_outbound.parquet"
           if variant in THIRDPARTY_VARIANTS
           else "luggage_email_outbound.parquet")
    df = pd.read_parquet(DATA_DIR / src)
    df = df[(df["pyName"] == variant) & (df["modelTechnique"] == "0.0")].reset_index(drop=True)

    # Rebuild X using the saved training feature set (same fix as 06)
    feature_cols = json.loads((art_dir / "feature_cols.json").read_text())
    cfg          = VARIANT_FEATURES[variant]
    X, y, cat_cols, _ = build_feature_matrix(df, feature_cols, cfg.numeric)

    test_idx = np.load(art_dir / "test_idx.npy")
    X_test   = X.iloc[test_idx]
    y_test   = y.iloc[test_idx]

    with open(art_dir / "dt_model.pkl", "rb") as f:
        dt_obj = pickle.load(f)
    dt, encoder = dt_obj["tree"], dt_obj["encoder"]

    X_test_enc, _ = _encode_for_sklearn(X_test, cat_cols, encoder=encoder, fit=False)
    pred = dt.predict(X_test_enc)
    return fidelity_suite(y_test.values, pred)


def _load_dt_fidelity(variant: str) -> dict:
    """Prefer saved dt_fidelity.json; fall back to computing from dt_model.pkl."""
    path = ART_ROOT / variant / "dt_fidelity.json"
    if path.exists():
        print(f"  {variant:<14} DT fidelity ← {path.name}")
        return json.loads(path.read_text())
    print(f"  {variant:<14} DT fidelity computed from dt_model.pkl")
    return _compute_dt_fidelity_from_pkl(variant)


def main() -> int:
    print("Loading fidelity per variant...")
    rows: list[dict] = []
    for v in VARIANTS:
        cb_fid = json.loads((ART_ROOT / v / "fidelity.json").read_text())
        dt_fid = _load_dt_fidelity(v)
        for label, fid in [
            ("Decision Tree",           dt_fid),
            ("CatBoost (SHAP + LIME)",  cb_fid),
        ]:
            rows.append({
                "Variant":    v,
                "Method":     label,
                "R²":         fid["r2"],
                "RMSE":       fid["rmse"],
                "Spearman ρ": fid["spearman_rho"],
                "Kendall τ":  fid["kendall_tau"],
                "KS":         fid["ks_stat"],
            })

    rq1_df = pd.DataFrame(rows)
    print()
    print("RQ1 — Explainer fidelity to Pega ADM propensity scores")
    print(rq1_df.to_string(index=False))

    out_csv = ART_ROOT / "rq1_explainer_fidelity.csv"
    rq1_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print("LaTeX table: run `uv run python scripts/build_thesis_tables.py "
          "rq1_explainer_fidelity` to regenerate the .tex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

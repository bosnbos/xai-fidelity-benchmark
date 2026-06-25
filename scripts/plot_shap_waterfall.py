"""Local SHAP waterfall plot for a single customer (l5b15 by default).

Loads the pre-computed SHAP values from `data/artifacts/<variant>/shap_values.npy`
and the fitted CatBoost surrogate to obtain the TreeExplainer expected value.
Selects one customer from the test split and renders shap.plots.waterfall()
with short, readable feature labels coloured by category.

Customer selection strategies (--pick):
  high    – highest predicted propensity in the test set (default)
  low     – lowest predicted propensity in the test set
  median  – customer whose predicted propensity is closest to the test-set median
  mixed   – highest variance in individual SHAP contributions (most push-and-pull)
  <int>   – explicit zero-based index into the test split

Output: `presentations/figures/shap_waterfall_<pick>_<variant>.pdf`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from my_project.parsing import THIRDPARTY_VARIANTS
from my_project.surrogate import build_feature_matrix
from my_project.features import VARIANT_FEATURES

# ── Category colours (consistent with global importance plot) ─────────────
CATEGORY_COLORS = {
    "IH":              "#2CA02C",
    "Booking context": "#1F77B4",
    "Strategy param":  "#FFA500",
    "Other":           "#7F7F7F",
}

# ── Exhaustive short-label lookup for all l5b15 features ─────────────────
# Keys are raw feature column names; values are compact display labels with
# a 4-char category prefix ([IH], [BC], [SP]) so the waterfall bars have
# enough horizontal room and the category is readable without colour tricks.
SHORT_LABELS: dict[str, str] = {
    # Booking context (BookingData namespace) ---------------------------------
    "CustBookedFlight.BookingData.BookingMonth":         "[BC] Booking month",
    "CustBookedFlight.BookingData.BookerGender":         "[BC] Gender",
    "CustBookedFlight.BookingData.CultureCode":          "[BC] Culture",
    "CustBookedFlight.BookingData.FlightInboundArrival": "[BC] Inbound arrival",
    # Flight data (flight/passenger attributes) ------------------------------
    "CustBookedFlight.Language":                         "[FD] Language",
    "CustBookedFlight.Journey":                          "[FD] Journey",
    "CustBookedFlight.FlightNumberOperatorIATA":         "[FD] Operator (IATA)",
    "CustBookedFlight.SeatNumber":                       "[FD] Seat number",
    "CustBookedFlight.IsStaffStandBy":                   "[FD] Staff standby",
    "CustBookedFlight.FlightData.AirlineCodeIATA":       "[FD] Airline",
    "CustBookedFlight.FlightData.DestinationAirport":    "[FD] Destination",
    # IH — Email outbound ---------------------------------------------------
    "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount":      "[IH] Eml Out·Pend count",
    "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince":   "[IH] Eml Out·Pend days",
    "IH.Email.Outbound.Pending.pxLastGroupID":                 "[IH] Eml Out·Pend group",
    "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince": "[IH] Eml Out·Deliv days",
    "IH.Email.Outbound.Delivered.pxLastGroupID":               "[IH] Eml Out·Deliv group",
    # IH — Email inbound ----------------------------------------------------
    "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince":    "[IH] Eml In·Pend days",
    "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince":    "[IH] Eml In·Click days",
    # IH — Push outbound ----------------------------------------------------
    "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount":       "[IH] Push Out·Pend count",
    "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince":    "[IH] Push Out·Pend days",
    "IH.Push.Outbound.Pending.pxLastGroupID":                  "[IH] Push Out·Pend group",
    # IH — Event ------------------------------------------------------------
    "IH.Event.Outbound.RealTimeEvent.pyHistoricalOutcomeCount": "[IH] Event RT count",
    # Strategy param --------------------------------------------------------
    "param::Param.BundleName": "[SP] Bundle",
}

# Category colour keyed to the label prefix (for legend)
PREFIX_COLORS: dict[str, str] = {
    "[IH]": CATEGORY_COLORS["IH"],
    "[BC]": CATEGORY_COLORS["Booking context"],
    "[FD]": "#D62728",
    "[SP]": CATEGORY_COLORS["Strategy param"],
}


def categorize(feature: str) -> str:
    if feature.startswith("IH."):
        return "IH"
    if feature.startswith("CustBookedFlight."):
        return "Booking context"
    if feature.startswith("param::") or feature.startswith("Param."):
        return "Strategy param"
    return "Other"


def shorten(feature: str) -> str:
    return SHORT_LABELS.get(feature, feature.split(".")[-1])


def pick_instance(shap_values: np.ndarray, preds: np.ndarray, strategy: str) -> int:
    if strategy == "high":
        return int(np.argmax(preds))
    if strategy == "low":
        return int(np.argmin(preds))
    if strategy == "median":
        med = float(np.median(preds))
        return int(np.argmin(np.abs(preds - med)))
    if strategy == "mixed":
        # Maximise min(sum_positive, |sum_negative|) so both sides are substantial
        pos_sum = np.sum(np.maximum(shap_values, 0), axis=1)
        neg_abs = np.abs(np.sum(np.minimum(shap_values, 0), axis=1))
        return int(np.argmax(np.minimum(pos_sum, neg_abs)))
    try:
        idx = int(strategy)
    except ValueError:
        raise ValueError(f"Unknown --pick value: {strategy!r}")
    if not (0 <= idx < len(preds)):
        raise IndexError(f"--pick {idx} out of range (test set has {len(preds)} rows)")
    return idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant",     default="l5b15")
    p.add_argument("--pick",        default="high",
                   help="Customer selection: high | low | median | mixed | <int>")
    p.add_argument("--max-display", type=int, default=15)
    args = p.parse_args()

    variant_key = args.variant.upper()
    art = REPO / "data" / "artifacts" / args.variant

    # ── Load data (same pipeline as notebook 06) ──────────────────────────
    processed_dir  = REPO / "data" / "processed"
    processed_file = processed_dir / (
        "thirdparty_email_outbound.parquet" if variant_key in THIRDPARTY_VARIANTS
        else "luggage_email_outbound.parquet"
    )
    df = pd.read_parquet(processed_file)
    df = df[(df["pyName"] == variant_key) & (df["modelTechnique"] == "0.0")].reset_index(drop=True)

    cfg            = VARIANT_FEATURES[variant_key]
    saved_features = json.loads((art / "feature_cols.json").read_text())
    X, y, cat_cols, num_cols = build_feature_matrix(df, saved_features, cfg.numeric)

    test_idx = np.load(art / "test_idx.npy")
    X_test   = X.iloc[test_idx].reset_index(drop=True)

    # ── Load model ────────────────────────────────────────────────────────
    cb_model = CatBoostRegressor()
    cb_model.load_model(str(art / "catboost_model.cbm"))

    # Use saved SHAP values for customer selection only. Fresh SHAP values are
    # recomputed for the chosen instance so the waterfall's base + SHAP = f(x)
    # identity holds exactly.
    shap_values_saved = np.load(art / "shap_values.npy")
    preds             = cb_model.predict(X_test).astype(float)

    # ── Select and recompute SHAP for chosen instance ────────────────────
    i         = pick_instance(shap_values_saved, preds, args.pick)
    explainer = shap.TreeExplainer(cb_model)
    sv_i      = explainer.shap_values(X_test.iloc[[i]])[0]
    ev        = explainer.expected_value
    base      = float(ev[0] if hasattr(ev, "__len__") else ev)
    pred_i    = preds[i]

    print(f"Customer index    : {i}")
    print(f"Predicted p̂      : {pred_i:.4f}")
    print(f"Base rate E[f(x)] : {base:.4f}")
    print(f"SHAP sum + base   : {base + sv_i.sum():.4f}  (≈ predicted p̂)")

    # ── Build short names and category mapping ────────────────────────────
    raw_names   = list(X_test.columns)
    short_names = [shorten(f) for f in raw_names]

    explanation = shap.Explanation(
        values        = sv_i,
        base_values   = base,
        data          = X_test.iloc[i].values,
        feature_names = short_names,
    )

    # ── Patch SHAP bar-label precision (default "%+0.02f" strips to "+0") ─
    import shap.plots._waterfall as _wf_mod
    import shap.utils._general as _shap_utils

    _orig_fmt = _shap_utils.format_value

    def _fmt3(s, fmt):
        if fmt == "%+0.02f":
            fmt = "%+0.03f"
        return _orig_fmt(s, fmt)

    _shap_utils.format_value = _fmt3
    _wf_mod.format_value     = _fmt3

    # ── Render waterfall ──────────────────────────────────────────────────
    n_shown = min(args.max_display, len(sv_i))
    fig, ax = plt.subplots(figsize=(11, 0.42 * n_shown + 2.5))
    plt.sca(ax)

    shap.plots.waterfall(explanation, max_display=args.max_display, show=False)
    _shap_utils.format_value = _orig_fmt   # restore
    _wf_mod.format_value     = _orig_fmt
    ax = plt.gca()

    # Post-render xlim expansion: SHAP sets the correct xlim for any mix of
    # positive/negative bars; we just add padding on both sides so bar-value
    # text labels are never clipped at the axes boundary.
    xlo, xhi = ax.get_xlim()
    x_range  = xhi - xlo
    ax.set_xlim(xlo - x_range * 0.05, xhi + x_range * 0.20)

    # ── Text-only legend (colour patches don't render; prefix is self-explanatory)
    prefix_labels = {
        "[IH]": "Interaction history",
        "[BC]": "Booking context",
        "[FD]": "Flight data",
        "[SP]": "Scoring parameter",
    }
    shown_prefixes = {lbl[:4] for lbl in short_names if lbl[:4] in prefix_labels}
    legend_lines = [
        f"{p}  {prefix_labels[p]}"
        for p in prefix_labels
        if p in shown_prefixes
    ]
    ax.text(
        0.99, 0.02, "\n".join(legend_lines),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.9),
    )

    ax.set_title(
        f"SHAP local explanation — customer #{i}"
        f"\n$\\hat{{p}}$ = {pred_i:.4f}  |  $E[f(x)]$ = {base:.4f}",
        fontsize=10, pad=6,
    )
    ax.spines[["top", "right"]].set_visible(False)

    out = REPO / "presentations" / "figures" / f"shap_waterfall_{args.pick}_{args.variant}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""RQ5 feature-substance analysis for the L5B15 model.

Produces three artifacts that answer "what features drive the propensity
score, and what does that mean for the business" using ONLY the three
ad-hoc explanation methods (SHAP, LIME, decision-tree surrogate). Pega's
own predictor importance is deliberately NOT used, to preserve the thesis's
black-box framing.

Outputs (CSV → data/artifacts/l5b15/, LaTeX → thesis/tables/):

  1. rq5_feature_importance      Ranked per-feature importance share across
                                 SHAP / LIME / DT, with group tag + mean rank.
  2. rq5_group_importance        Importance aggregated by feature group
                                 (% of each method's total).
  3. rq5_staffstandby_proxy      Test of whether CustBookedFlight.IsStaffStandBy
                                 *populated-ness* proxies for customer-data
                                 availability (the missingness-proxy hypothesis).

Run from repo root:
    .venv/bin/python scripts/build_rq5_feature_analysis.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "data" / "artifacts" / "l5b15"
DATA = REPO / "data" / "processed"
TABLES = REPO / "thesis" / "tables"

sys.path.insert(0, str(REPO / "src"))

# --- feature-group mapping: identical to plot_feature_importance_heatmap.py ---
# (so the group story matches the figure already in the thesis)
GROUP_FULL = {
    "CH": "Contact history",
    "BC": "Booking context",
    "FD": "Flight data",
    "SP": "Scoring parameter",
}


def group_of(name: str) -> str:
    if name.startswith("IH."):
        return "CH"
    if name.startswith("param::"):
        return "SP"
    if name.startswith("CustBookedFlight.BookingData."):
        return "BC"
    if name.startswith("CustBookedFlight."):  # incl. FlightData.*, SeatNumber, IsStaffStandBy
        return "FD"
    return "FD"


def short(name: str) -> str:
    """Readable but still-real feature label (anonymise later if needed)."""
    return (name
            .replace("CustBookedFlight.BookingData.", "")
            .replace("CustBookedFlight.FlightData.", "")
            .replace("CustBookedFlight.", "")
            .replace("IH.Email.Outbound.", "Email.Out.")
            .replace("IH.Email.Inbound.", "Email.In.")
            .replace("IH.Push.Outbound.", "Push.Out.")
            .replace("IH.Event.Outbound.", "Event.Out.")
            .replace("param::Param.", "Param.")
            .replace(".pxLastOutcomeTime.DaysSince", ".DaysSinceLast")
            .replace(".pyHistoricalOutcomeCount", ".HistCount")
            .replace(".pxLastGroupID", ".LastGroup"))


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------------------------------------------------------------------------
# Load the three methods' overall importances, normalised to shares (sum = 1).
# ---------------------------------------------------------------------------
def load_importances() -> pd.DataFrame:
    shap = json.loads((ART / "shap_importances.json").read_text())
    lime = json.loads((ART / "lime_importances.json").read_text())
    feature_cols = json.loads((ART / "feature_cols.json").read_text())

    dt_bundle = pickle.load(open(ART / "dt_model.pkl", "rb"))
    dt_imp = np.asarray(dt_bundle["tree"].feature_importances_, dtype=float)
    if len(dt_imp) != len(feature_cols):
        raise RuntimeError(
            f"DT importance length {len(dt_imp)} != feature_cols {len(feature_cols)}")
    dt = dict(zip(feature_cols, dt_imp))

    feats = sorted(set(shap) | set(lime) | set(dt))

    def norm(d: dict[str, float]) -> dict[str, float]:
        tot = sum(d.get(f, 0.0) for f in feats) or 1.0
        return {f: d.get(f, 0.0) / tot for f in feats}

    s, l, d = norm(shap), norm(lime), norm(dt)
    df = pd.DataFrame({
        "feature": feats,
        "group": [group_of(f) for f in feats],
        "label": [short(f) for f in feats],
        "shap": [s[f] for f in feats],
        "lime": [l[f] for f in feats],
        "dt": [d[f] for f in feats],
    })
    for m in ("shap", "lime", "dt"):
        df[f"rank_{m}"] = df[m].rank(ascending=False, method="min").astype(int)
    # Order by the SHAP/LIME consensus only. RQ2/RQ3 established SHAP and LIME as
    # the reliable methods and the decision tree as the unstable one, so the two
    # reliable methods govern the ranking; the DT share is retained as a
    # corroborating column but deliberately excluded from the ordering rather
    # than averaged in with equal weight.
    df["sl_rank"] = df[["rank_shap", "rank_lime"]].mean(axis=1)
    return df.sort_values(["sl_rank", "rank_shap"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Per-feature ranking table
# ---------------------------------------------------------------------------
def build_feature_table(df: pd.DataFrame) -> None:
    df.to_csv(ART / "rq5_feature_importance.csv", index=False)

    lines = [
        r"\begin{table}[ht]",
        r"\caption{L5B15 feature-importance shares across the three explanation "
        r"methods, ordered by the SHAP/LIME consensus rank (the mean of the SHAP "
        r"and LIME within-method ranks). The ordering uses only the two methods "
        r"that RQ2 and RQ3 found reliable; the decision-tree share is reported as "
        r"a corroborating column but does not drive the ordering. Each method's "
        r"importances are normalised to sum to 100\%. Group tags: CH~=~contact "
        r"history, FD~=~flight data, BC~=~booking context, SP~=~scoring parameter.}",
        r"\label{tab:rq5_feature_importance}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Feature & Grp & SHAP\% & LIME\% & DT\% & SL rank \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{_tex_escape(r['label'])} & {r['group']} & "
            f"{100*r['shap']:.1f} & {100*r['lime']:.1f} & {100*r['dt']:.1f} & "
            f"{r['sl_rank']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "rq5_feature_importance.tex").write_text("\n".join(lines))
    print("\n=== 1. Per-feature importance (top 12 by SHAP/LIME consensus rank) ===")
    print(df.head(12)[["label", "group", "shap", "lime", "dt", "sl_rank"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------------------
# 2. Group aggregation table
# ---------------------------------------------------------------------------
def build_group_table(df: pd.DataFrame) -> None:
    g = df.groupby("group")[["shap", "lime", "dt"]].sum()
    g = g.reindex(["CH", "FD", "BC", "SP"]).fillna(0.0)
    g["n_feat"] = df.groupby("group").size().reindex(g.index).fillna(0).astype(int)
    g.to_csv(ART / "rq5_group_importance.csv")

    lines = [
        r"\begin{table}[ht]",
        r"\caption{L5B15 feature-importance aggregated by feature group, as a "
        r"percentage of each method's total importance. Contact-history (recency "
        r"and frequency of prior outbound contact) and flight-level features "
        r"dominate across all three methods; customer-level attributes contribute "
        r"nothing because none survive Pega's active-predictor selection for L5B15.}",
        r"\label{tab:rq5_group_importance}",
        r"\centering",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Feature group & \# feat & SHAP\% & LIME\% & DT\% \\",
        r"\midrule",
    ]
    for tag, row in g.iterrows():
        lines.append(
            f"{GROUP_FULL[tag]} & {int(row['n_feat'])} & "
            f"{100*row['shap']:.1f} & {100*row['lime']:.1f} & {100*row['dt']:.1f} \\\\")
    lines += [
        r"\midrule",
        f"Customer attributes & 0 & 0.0 & 0.0 & 0.0 \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "rq5_group_importance.tex").write_text("\n".join(lines))
    print("\n=== 2. Group importance (% of method total) ===")
    print((100 * g[["shap", "lime", "dt"]]).round(1).to_string())


# ---------------------------------------------------------------------------
# 3. IsStaffStandBy missingness-proxy test
# ---------------------------------------------------------------------------
def _populated(s: pd.Series) -> pd.Series:
    """A field counts as 'populated' if not null and not blank/whitespace."""
    str_s = s.astype("string")
    return s.notna() & (str_s.str.strip().fillna("") != "")


def staffstandby_proxy() -> None:
    from scipy import stats

    df = pd.read_parquet(DATA / "luggage_email_outbound.parquet")
    df = df[df["pyName"] == "L5B15"].reset_index(drop=True)
    n = len(df)

    SB = "CustBookedFlight.IsStaffStandBy"
    s_pop = _populated(df[SB])                       # field explicitly written vs blank/null
    rate = float(s_pop.mean())

    # Value distribution: is the flag itself almost never True among populated rows?
    def _is_true(v) -> bool:
        if pd.isna(v):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return str(v).strip().lower() in {"true", "1", "t", "y", "yes"}
    sb_true = df[SB].map(_is_true)
    true_rate_overall = float(sb_true.mean())
    true_rate_among_pop = float(sb_true[s_pop].mean()) if s_pop.any() else float("nan")

    # Customer-level features that were NOT kept active for L5B15 (the dropped
    # profile). Exclude the same-namespace twin of the flagged predictor.
    cust_cols = [c for c in df.columns
                 if c.startswith("Customer.") and c != "Customer.PreviousFlight.IsStaffStandBy"]

    # Y = "returning-customer" indicator: any previous-flight history.
    pf_cols = [c for c in cust_cols if c.startswith("Customer.PreviousFlight.")]
    pf_present = df[pf_cols].notna().any(axis=1)

    # C = broad customer-completeness, computed over customer fields OUTSIDE
    # the previous-flight namespace so r_pb (Z vs C) and V (Z vs Y) test
    # disjoint slices of the customer namespace and don't share data.
    comp_cols = [c for c in cust_cols if not c.startswith("Customer.PreviousFlight.")]
    completeness = df[comp_cols].notna().mean(axis=1)

    # Continuous association: populated-ness vs customer-data completeness
    comp_pop = float(completeness[s_pop].mean())
    comp_mis = float(completeness[~s_pop].mean())
    r_pb, p_pb = stats.pointbiserialr(s_pop.astype(int), completeness)

    # Binary association: populated-ness vs presence of a previous-flight profile
    p_pf_pop = float(pf_present[s_pop].mean())
    p_pf_mis = float(pf_present[~s_pop].mean())
    risk_ratio = p_pf_pop / p_pf_mis if p_pf_mis > 0 else float("nan")

    ct = pd.crosstab(s_pop, pf_present)
    if min(ct.shape) > 1:
        chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    else:
        p_chi, cramers_v = float("nan"), float("nan")

    summary = pd.DataFrame([
        ("n (L5B15 records)", f"{n:,}"),
        ("IsStaffStandBy populated rate", f"{rate:.3f}"),
        ("P(IsStaffStandBy = True) overall", f"{true_rate_overall:.4f}"),
        ("P(IsStaffStandBy = True | populated)", f"{true_rate_among_pop:.4f}"),
        ("customer fields in $C$ (excl.\\ prev-flight)", f"{len(comp_cols)}"),
        ("previous-flight fields in $Y$", f"{len(pf_cols)}"),
        ("mean customer-completeness | populated", f"{comp_pop:.3f}"),
        ("mean customer-completeness | blank/null", f"{comp_mis:.3f}"),
        ("P(previous-flight profile | populated)", f"{p_pf_pop:.3f}"),
        ("P(previous-flight profile | blank/null)", f"{p_pf_mis:.3f}"),
        ("risk ratio", f"{risk_ratio:.1f}x"),
        ("point-biserial r (populated vs completeness)", f"{r_pb:.3f} (p={p_pb:.1e})"),
        ("Cramér's V (populated vs prev-flight profile)", f"{cramers_v:.3f} (p={p_chi:.1e})"),
    ], columns=["quantity", "value"])
    summary.to_csv(ART / "rq5_staffstandby_proxy.csv", index=False)

    lines = [
        r"\begin{table}[ht]",
        r"\caption{Test of the missingness-proxy hypothesis for "
        r"\texttt{IsStaffStandBy} on L5B15. The flag is never \texttt{true} in "
        r"the sample, and what varies across records is whether the field is "
        r"populated at all; where it is populated, customer-profile data is "
        r"far more likely to be present, indicating that the surrogate (and the "
        r"underlying model) keys on the field's \emph{presence} as a proxy for "
        r"record completeness rather than on staff/standby status itself.}",
        r"\label{tab:rq5_staffstandby_proxy}",
        r"\centering",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(f"{_tex_escape(r['quantity'])} & {_tex_escape(str(r['value']))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "rq5_staffstandby_proxy.tex").write_text("\n".join(lines))

    print("\n=== 3. IsStaffStandBy missingness-proxy ===")
    print(summary.to_string(index=False))


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_importances()
    build_feature_table(df)
    build_group_table(df)
    staffstandby_proxy()
    print("\nDone. CSVs → data/artifacts/l5b15/ , LaTeX → thesis/tables/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

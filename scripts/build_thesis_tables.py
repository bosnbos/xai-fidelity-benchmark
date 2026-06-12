#!/usr/bin/env python3
"""Generate all thesis-ready LaTeX tables from saved JSON/CSV artefacts.

Each table is produced by one function in this module, reading from
``data/artifacts/`` and writing the ``.tex`` directly into ``thesis/tables/``
(the same folder ``build_rq5_feature_analysis.py`` writes to). Notebooks
produce the JSON/CSV; this script handles the formatting. Writing straight
into ``thesis/tables/`` means the thesis always references the freshly built
tables — no separate copy step.

This separation keeps the notebooks focused on scientific computation and
makes thesis-table tweaks fast: edit a caption or a rounding spec here and
re-run the script in seconds; no need to re-fit the surrogate or re-run
the DT bootstrap.

Convention: numeric values are rounded to 3 decimals by default. RMSE is
shown at 4 decimals because the third-party-offer variants have values
near 1e-4 that would round to 0.000 at 3 dp.

Usage
-----
    uv run python scripts/build_thesis_tables.py                  # build all
    uv run python scripts/build_thesis_tables.py attribution_l5b15  # one
    uv run python scripts/build_thesis_tables.py --list             # list targets
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

REPO   = Path(__file__).resolve().parent.parent
ART    = REPO / "data" / "artifacts"
L5     = ART / "l5b15"
TABLES = REPO / "thesis" / "tables"   # .tex outputs land directly in the thesis

# Default rounding
_NUM = "{:.3f}"
_RMSE = "{:.4f}"   # RMSE keeps 4dp because some variants are ~1e-4

# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

_GREEK_MAP = {
    "δ": r"\delta",
    "ρ": r"\rho",
    "π": r"\pi",
    "Δ": r"\Delta",
    "τ": r"\tau",
}
_GREEK_PATTERN = re.compile(
    r"(" + "|".join(re.escape(g) for g in _GREEK_MAP) + r")"
    r"((?:_\w+|\^\w+)?)"
)
_UNICODE_OTHER = {
    "→": r"$\to$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "∧": r"$\land$",
    "²": r"$^2$",
}


def _post(s: str) -> str:
    """Replace stray unicode math glyphs with LaTeX commands.

    Greek letters are matched together with any trailing `_X` / `^X` token
    so the underscore stays inside math mode (`δ_d` → `$\\delta_d$`,
    not `$\\delta$_d` which would break the build).
    """
    def _greek_repl(m: re.Match) -> str:
        g, sub = m.group(1), m.group(2) or ""
        return f"${_GREEK_MAP[g]}{sub}$"
    s = _GREEK_PATTERN.sub(_greek_repl, s)
    for k, v in _UNICODE_OTHER.items():
        s = s.replace(k, v)
    return s


def _write(path: Path, latex: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_post(latex))
    print(f"  → {path.relative_to(REPO)}  ({len(latex):,} chars)")


# ─────────────────────────────────────────────────────────────────────────
#  Appendix tables
# ─────────────────────────────────────────────────────────────────────────

def build_missingness() -> None:
    """Appendix — per-variant feature-group missingness profile."""
    df = pd.read_csv(ART / "missingness.csv")

    def _fmt_n(x):
        return "--" if pd.isna(x) else f"{int(x)}"

    def _fmt_miss(x):
        return "--" if pd.isna(x) else f"{x*100:.1f}\\%"

    # Build the body manually so we can keep the wide compact layout
    variants = ["L5B15", "CLUG", "BookingDotCom", "Cartrawler"]
    body_rows = []
    for _, r in df.iterrows():
        cells = [r["group"], r["description"]]
        for v in variants:
            cells.append(_fmt_n(r[f"{v}_n"]))
            cells.append(_fmt_miss(r[f"{v}_miss"]))
        body_rows.append(" & ".join(cells) + r" \\")

    body = "\n".join(body_rows)
    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{Feature groups, descriptions, and mean missingness "
        "profile across all analysed offer variants. Active predictor counts "
        "($n$) vary because Pega ADM selects predictors independently per "
        "model based on evidential contribution. Customer attributes appear "
        "only in the BookingDotCom variant. BDC = BookingDotCom; "
        "CTL = Cartrawler.}\n"
        "\\label{tab:missingness}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{lp{3.2cm}rrrrrrrr}\n"
        "\\toprule\n"
        " & & \\multicolumn{2}{c}{L5B15} & \\multicolumn{2}{c}{CLUG} "
        "& \\multicolumn{2}{c}{BDC} & \\multicolumn{2}{c}{CTL} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} "
        "\\cmidrule(lr){9-10}\n"
        "Feature group & Description & $n$ & Miss. & $n$ & Miss. & "
        "$n$ & Miss. & $n$ & Miss. \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "missingness.tex", latex)


def build_depth_sensitivity() -> None:
    """Appendix — L5B15 CatBoost depth selection (2-column, bold selected row)."""
    df = pd.read_csv(L5 / "depth_sensitivity_table.csv")

    selected = df.loc[df["Selected (1-15)"] == "yes", "Depth"].iloc[0]

    body_rows = []
    for _, r in df.iterrows():
        d_str    = f"{int(r['Depth'])}"
        mean_str = f"{r['Mean ρ (1-15)']:.3f}"
        std_str  = f"{r['Std ρ (1-15)']:.3f}"
        if r["Depth"] == selected:
            d_str    = f"\\textbf{{{d_str}}}"
            mean_str = f"\\textbf{{{mean_str}}}"
            std_str  = f"\\textbf{{{std_str}}}"
        body_rows.append(f"{d_str} & {mean_str} & {std_str} \\\\")
    body = "\n".join(body_rows)

    cv_max_depth = int(df.loc[df["Mean ρ (1-15)"].idxmax(), "Depth"])

    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{Five-fold cross-validation Spearman~$\\rho_S$ over depths "
        f"1--15 for the L5B15 CatBoost surrogate. CV-maximum is at depth "
        f"{cv_max_depth}; the 1-SD parsimony rule selects depth {int(selected)} (bold).}}\n"
        "\\label{tab:depth_sensitivity}\n"
        "\\centering\n"
        "\\begin{tabular}{rrr}\n"
        "\\toprule\n"
        "Depth & $\\bar{\\rho}_S$ & $\\sigma_{\\rho_S}$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "depth_sensitivity_table.tex", latex)


def build_metric_selection() -> None:
    """Appendix — δ_m metric selection (AUC PSI vs |Δμ| vs KS) across variants."""
    df = pd.read_csv(ART / "metric_selection.csv")

    def _fmt_ksp(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"

    body_rows = []
    # Group by split_type with subheaders
    for kind, label in [("temporal", "Temporal splits"),
                        ("route",    "Route splits (L5B15)")]:
        sub = df[df["split_type"] == kind]
        if sub.empty:
            continue
        body_rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for _, r in sub.iterrows():
            split = r["split"].replace("->", r"$\to$")
            body_rows.append(
                f"{split} & {r['AUC_PSI']:.3f} & {r['delta_mu_AUC']:.3f} & "
                f"{r['KS']:.3f} & {_fmt_ksp(r['KS_p'])} \\\\"
            )
        if kind == "temporal":
            body_rows.append("\\midrule")
    body = "\n".join(body_rows)

    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{Comparison of candidate metrics for model version churn "
        "($\\delta_m$) across all splits. AUC PSI uses equal-frequency bins "
        "($B = 5$) on the modelPerformance distribution. "
        "$\\Delta\\mu_{\\text{AUC}}$ is the absolute difference in mean AUC "
        "between splits. KS is the two-sample Kolmogorov--Smirnov statistic on "
        "the AUC distribution. KS cleanly separates temporal splits "
        "(KS $\\approx 0.33$--$0.50$, $p \\approx 0$) from route splits "
        "(KS $\\approx 0.04$, $p > 0.39$), making it the most informative of "
        "the three AUC-based candidates. These metrics corroborate the Jaccard "
        "overlap on model-version identifiers, which is used as the $\\delta_m$ "
        "proxy in the main analysis.}\n"
        "\\label{tab:metric_selection}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Split & AUC PSI & $\\Delta\\mu_{\\text{AUC}}$ & KS & KS $p$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "metric_selection.tex", latex)


_MODEL_NAME_MAP = {
    "LinearRegression": "Linear regression",
    "GaussianNB":       "Naive Bayes",
    "RandomForest":     "Random Forest",
}


def build_surrogate_comparison() -> None:
    """Appendix — surrogate architecture comparison on L5B15."""
    df = pd.read_csv(L5 / "surrogate_comparison.csv")
    df["Model"] = df["Model"].map(lambda m: _MODEL_NAME_MAP.get(m, m))

    # Detect column names defensively (notebook uses these labels)
    rename = {
        "R²":          r"$R^2$",
        "Spearman ρ":  r"Spearman $\rho$",
        "Kendall τ":   r"Kendall $\tau$",
        "KS statistic": "KS statistic",
        "Model":       "Model",
        "model":       "Model",
        "RMSE":        "RMSE",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Bold best value per metric
    metric_cols = {
        r"$R^2$":         ("max", _NUM),
        "RMSE":           ("min", _RMSE),
        r"Spearman $\rho$": ("max", _NUM),
        r"Kendall $\tau$":  ("max", _NUM),
        "KS statistic":   ("min", _NUM),
    }
    metric_cols = {k: v for k, v in metric_cols.items() if k in df.columns}

    body_rows = []
    best_idx = {c: (df[c].idxmax() if direction == "max" else df[c].idxmin())
                for c, (direction, _) in metric_cols.items()}

    for idx, r in df.iterrows():
        cells = [str(r["Model"])]
        for c, (_, fmt) in metric_cols.items():
            val = fmt.format(r[c])
            if idx == best_idx[c]:
                val = f"$\\mathbf{{{val.strip('$')}}}$" if val.startswith("$") else f"\\textbf{{{val}}}"
            cells.append(val)
        body_rows.append(" & ".join(cells) + r" \\")

    body = "\n".join(body_rows)
    col_spec = "l" + "r" * len(metric_cols)
    header = "Model & " + " & ".join(metric_cols.keys()) + r" \\"

    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{Surrogate architecture comparison on the L5B15 held-out "
        "test set (stratified 80/20 split). Five candidates are evaluated on "
        "rank-based fidelity (Spearman~$\\rho$, Kendall~$\\tau$), pointwise "
        "accuracy ($R^2$, RMSE), and distributional alignment (KS statistic). "
        "Best value per metric in bold.}\n"
        "\\label{tab:surrogate_comparison}\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "surrogate_comparison.tex", latex)


# ─────────────────────────────────────────────────────────────────────────
#  RQ tables (main text)
# ─────────────────────────────────────────────────────────────────────────

def build_surrogate_fidelity() -> None:
    """RQ1 — per-variant CatBoost surrogate fidelity on the held-out 20%."""
    df = pd.read_csv(ART / "surrogate_fidelity_per_variant.csv")
    latex = (
        df.style
        .format({
            "n":            "{:,.0f}",
            "CV-max":       "{:.0f}",
            "Depth":        "{:.0f}",
            "R²":           _NUM,
            "RMSE":         _RMSE,
            "Spearman ρ":   _NUM,
            "Kendall τ":    _NUM,
            "KS statistic": _NUM,
        })
        .hide(axis="index")
        .set_caption(
            "Per-variant CatBoost surrogate fidelity on the held-out 20\\%. "
            "CV-max is the depth maximising 5-fold CV Spearman~$\\rho$; "
            "Depth is the 1-SD parsimony pick actually used for the final "
            "model fit."
        )
        .to_latex(hrules=True, label="tab:surrogate_fidelity_per_variant")
    )
    _write(TABLES / "surrogate_fidelity_per_variant.tex", latex)


def build_rq1_explainer_fidelity() -> None:
    """RQ1 — explainer fidelity to Pega ADM propensity per variant."""
    df = pd.read_csv(ART / "rq1_explainer_fidelity.csv")
    latex = (
        df.style
        .format({
            "R²":         _NUM,
            "RMSE":       _RMSE,
            "Spearman ρ": _NUM,
            "Kendall τ":  _NUM,
            "KS":         _NUM,
        })
        .hide(axis="index")
        .set_caption(
            "Predictive fidelity to Pega ADM propensity scores per variant, "
            "on the held-out 20\\%. The Decision Tree is fit directly to the "
            "logged scores; the CatBoost row applies to both SHAP and LIME, "
            "which explain the same surrogate and therefore share its "
            "predictive fidelity to Pega. R$^2$ and RMSE measure "
            "absolute-value agreement; Spearman~$\\rho$ and Kendall~$\\tau$ "
            "measure rank agreement; the Kolmogorov--Smirnov statistic "
            "measures distributional agreement (lower is better, 0 = "
            "identical distributions)."
        )
        .to_latex(hrules=True, label="tab:rq1_explainer_fidelity")
    )
    _write(TABLES / "rq1_explainer_fidelity.tex", latex)


def build_stability_matrix() -> None:
    """RQ2 — L5B15 stability matrix: per-method (ρ, J5, J10) per split."""
    df = pd.read_json(L5 / "stability_summary.json")

    # Map split label → short row label matching the thesis prose
    short_label = {
        "L5B15 temporal":          "Temporal",
        "ORY->NCE vs ORY->OPO":    r"Route pair 1",
        "ORY->OPO vs NCE->ORY":    r"Route pair 2",
        "NCE->ORY vs TLS->ORY":    r"Route pair 3",
        "fr-FR vs nl-NL":          "Culture (FR / NL)",
    }
    df["split"] = df["split"].map(lambda s: short_label.get(s, s.replace("->", r"$\to$")))

    # Pivot to a 9-column block: SHAP/LIME/DT × (ρ, J5, J10)
    pivot = df.pivot_table(
        index="split",
        columns="method",
        values=["Spearman ρ", "Jaccard@5", "Jaccard@10"],
    )
    # Reorder columns: SHAP first, then LIME, then DT; each block (ρ, J5, J10)
    methods_order  = ["SHAP", "LIME", "DT"]
    metrics_order  = ["Spearman ρ", "Jaccard@5", "Jaccard@10"]
    cols = [(m, meth) for meth in methods_order for m in metrics_order]
    pivot = pivot.reindex(columns=cols)

    # Order rows: temporal first, then 3 route pairs, then culture
    row_order = ["Temporal", "Route pair 1", "Route pair 2", "Route pair 3",
                 "Culture (FR / NL)"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    # Render as raw LaTeX (Styler.to_latex doesn't handle multi-column headers well)
    body_rows = []
    for split, r in pivot.iterrows():
        cells = [split]
        for meth in methods_order:
            for metric in metrics_order:
                v = r.get((metric, meth), float("nan"))
                cells.append(f"{v:.3f}" if pd.notna(v) else "--")
        body_rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(body_rows)

    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{Stability of feature importance rankings across L5B15 "
        "splits. $\\rho_S$ measures full-ranking consistency; $J_5$ and "
        "$J_{10}$ measure top-$k$ overlap. Route pair numbering follows "
        "Table~\\ref{tab:route_pairs}.}\n"
        "\\label{tab:stability_l5b15}\n"
        "\\centering\n"
        "\\begin{tabular}{l rrr rrr rrr}\n"
        "\\toprule\n"
        " & \\multicolumn{3}{c}{SHAP} & \\multicolumn{3}{c}{LIME} & "
        "\\multicolumn{3}{c}{Decision Tree} \\\\\n"
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
        "Split & $\\rho_S$ & $J_5$ & $J_{10}$ & $\\rho_S$ & $J_5$ & "
        "$J_{10}$ & $\\rho_S$ & $J_5$ & $J_{10}$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "stability_matrix.tex", latex)


def build_attribution_l5b15() -> None:
    """RQ3 — L5B15 instability attribution summary."""
    df = pd.read_json(L5 / "attribution_summary.json")

    short_split = {
        "L5B15 temporal":          "L5B15 temporal",
        "ORY->NCE vs ORY->OPO":    r"Route pair 1 (ORY$\to$NCE vs ORY$\to$OPO)",
        "ORY->OPO vs NCE->ORY":    r"Route pair 2 (ORY$\to$OPO vs NCE$\to$ORY)",
        "NCE->ORY vs TLS->ORY":    r"Route pair 3 (NCE$\to$ORY vs TLS$\to$ORY)",
        "fr-FR vs nl-NL":          "Culture (fr-FR vs nl-NL)",
    }

    cols = ["split", "KS_prop (δ_d)", "J_v (δ_m)", "δ_e^DT", "δ_e^LIME",
            "Primary source"]
    df = df[cols].copy()
    df["split"] = df["split"].map(lambda s: short_split.get(s, s.replace("->", r"$\to$")))

    # Enforce a stable row order: temporal, route pairs 1/2/3, then culture
    row_order = [
        "L5B15 temporal",
        r"Route pair 1 (ORY$\to$NCE vs ORY$\to$OPO)",
        r"Route pair 2 (ORY$\to$OPO vs NCE$\to$ORY)",
        r"Route pair 3 (NCE$\to$ORY vs TLS$\to$ORY)",
        "Culture (fr-FR vs nl-NL)",
    ]
    df = df.set_index("split").reindex([r for r in row_order if r in set(df["split"])]).reset_index()

    df = df.rename(columns={
        "split":          "Split",
        "KS_prop (δ_d)":  r"$\delta_d$",
        "J_v (δ_m)":      r"$\delta_m$",
        "δ_e^DT":         r"$\delta_e^{\mathrm{DT}}$",
        "δ_e^LIME":       r"$\delta_e^{\mathrm{LIME}}$",
        "Primary source": "Source",
    })
    float_cols = [c for c in df.columns if c not in ("Split", "Source")]
    latex = (
        df.style
        .format({c: _NUM for c in float_cols})
        .hide(axis="index")
        .set_caption(
            "L5B15 instability attribution. $\\delta_d$ is the "
            "Kolmogorov--Smirnov statistic on the propensity distribution; "
            "flagged if $\\geq 0.10$. $\\delta_m$ is the Jaccard overlap on "
            "the set of model versions active in each "
            "sub-population; flagged if $\\leq 0.10$. "
            "$\\delta_e^{\\mathrm{DT}} = \\rho_{\\mathrm{SHAP}} - "
            "\\rho_{\\mathrm{DT}}$ and $\\delta_e^{\\mathrm{LIME}} = "
            "\\rho_{\\mathrm{SHAP}} - \\rho_{\\mathrm{LIME}}$ are the refit "
            "and sampling sensitivity gaps respectively; both are "
            "descriptive and not used for source classification. Source "
            "assignment follows Table~\\ref{tab:attribution_rule}."
        )
        .to_latex(hrules=True, label="tab:attribution_l5b15")
    )
    _write(TABLES / "attribution_summary.tex", latex)


def build_attribution_replication() -> None:
    """RQ4 — replication attribution summary across variants, grouped by split."""
    df = pd.read_json(ART / "attribution_summary_replication.json")

    # Variant display order (closest-to-L5B15 product context first)
    variant_order = ["CLUG", "BookingDotCom", "Cartrawler"]

    # Subgroup definitions: (header label, predicate on the raw split string)
    groups = [
        ("Temporal splits",
         lambda s, v: s == f"{v} temporal"),
        (r"Route pair 1 (ORY$\to$NCE vs ORY$\to$OPO)",
         lambda s, v: s == "ORY->NCE vs ORY->OPO"),
        (r"Route pair 2 (ORY$\to$OPO vs NCE$\to$ORY)",
         lambda s, v: s == "ORY->OPO vs NCE->ORY"),
        (r"Route pair 3 (NCE$\to$ORY vs TLS$\to$ORY)",
         lambda s, v: s == "NCE->ORY vs TLS->ORY"),
        ("Culture (fr-FR vs nl-NL)",
         lambda s, v: s == "fr-FR vs nl-NL"),
    ]

    body_rows: list[str] = []
    n_cols = 6  # Variant, δ_d, δ_m, δ_e^DT, δ_e^LIME, Source

    for gi, (group_label, pred) in enumerate(groups):
        sub = df[df.apply(lambda r: pred(r["split"], r["variant"]), axis=1)].copy()
        if sub.empty:
            continue
        # Order variants
        sub["_order"] = sub["variant"].map({v: i for i, v in enumerate(variant_order)})
        sub = sub.sort_values("_order")

        body_rows.append(
            f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_label}}}}} \\\\"
        )
        for _, r in sub.iterrows():
            cells = [
                r["variant"],
                f"{r['KS_prop (δ_d)']:.3f}",
                f"{r['J_v (δ_m)']:.3f}",
                f"{r['δ_e^DT']:.3f}",
                f"{r['δ_e^LIME']:.3f}",
                r["Primary source"],
            ]
            body_rows.append(" & ".join(cells) + r" \\")
        if gi < len(groups) - 1:
            body_rows.append(r"\midrule")
    body = "\n".join(body_rows)

    latex = (
        "\\begin{table}[ht]\n"
        "\\caption{RQ4 replication: attribution summary for CLUG, "
        "BookingDotCom, and Cartrawler, grouped by split. Routes are pinned "
        "to the L5B15 set (ORY$\\to$NCE, ORY$\\to$OPO, NCE$\\to$ORY, "
        "TLS$\\to$ORY) for direct comparability. Metric definitions and "
        "thresholds match Table~\\ref{tab:attribution_l5b15}: $\\delta_d$ "
        "from the Kolmogorov--Smirnov statistic on propensity, flagged if "
        "$\\geq 0.10$; $\\delta_m$ from Jaccard overlap on the set of "
        "active model versions, flagged if $\\leq 0.10$. "
        "$\\delta_e^{\\mathrm{DT}}$ and $\\delta_e^{\\mathrm{LIME}}$ are "
        "descriptive and not used for source classification.}\n"
        "\\label{tab:attribution_replication}\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrl}\n"
        "\\toprule\n"
        "Variant & $\\delta_d$ & $\\delta_m$ & "
        "$\\delta_e^{\\mathrm{DT}}$ & $\\delta_e^{\\mathrm{LIME}}$ & "
        "Source \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write(TABLES / "attribution_summary_replication.tex", latex)


def build_bootstrap_dt_summary() -> None:
    """Appendix — within-route DT bootstrap validation summary."""
    summary = json.loads(
        (L5 / "bootstrap_dt_routes.json").read_text()
    )["_summary"]

    def _classify(g: float) -> str:
        if g > 0.10: return "Genuine signal"
        if g > 0.03: return "Small effect"
        return "Within noise floor"

    rows = []
    for label, s in summary["routes_per_pair"].items():
        rows.append({
            "Split":          label.replace("->", r"$\to$"),
            "Within (mean)":  s["mean_within"],
            "Between":        s["between"],
            "Gap":            s["gap"],
            "Classification": _classify(s["gap"]),
        })
    rows.append({
        "Split":          "Temporal (early vs late)",
        "Within (mean)":  summary["temporal"]["mean_within"],
        "Between":        summary["temporal"]["between"],
        "Gap":            summary["temporal"]["gap"],
        "Classification": _classify(summary["temporal"]["gap"]),
    })
    rows.append({
        "Split":          "Culture (fr-FR vs nl-NL)",
        "Within (mean)":  summary["culture"]["mean_within"],
        "Between":        summary["culture"]["between"],
        "Gap":            summary["culture"]["gap"],
        "Classification": _classify(summary["culture"]["gap"]),
    })
    rows.append({
        "Split":          "Routes (aggregate)",
        "Within (mean)":  summary["routes_aggregate"]["mean_within"],
        "Between":        summary["routes_aggregate"]["mean_between"],
        "Gap":            summary["routes_aggregate"]["gap"],
        "Classification": _classify(summary["routes_aggregate"]["gap"]),
    })

    df = pd.DataFrame(rows)
    latex = (
        df.style
        .format({
            "Within (mean)": _NUM,
            "Between":       _NUM,
            "Gap":           "{:+.3f}",
        })
        .hide(axis="index")
        .set_caption(
            "Within-sample DT reproducibility versus observed between-split "
            "$\\rho_S^{\\mathrm{DT}}$. For each sub-sample the DT surrogate "
            "was re-fitted on identical data with $N=10$ random seeds, "
            "yielding 45 pairwise Spearman~$\\rho$ values per sub-sample. "
            "Within (mean) is the average of those means across the two "
            "sub-samples of each split. Gap $=$ Within $-$ Between. "
            "Classification: $>0.10$ genuine signal; $0.03$ to $0.10$ small "
            "effect; $\\leq 0.03$ indistinguishable from refit noise."
        )
        .to_latex(hrules=True, label="tab:bootstrap_dt")
    )
    _write(TABLES / "bootstrap_dt_summary.tex", latex)


def build_top10_per_split() -> None:
    """RQ2 — top-10 features per split per method (long-format CSV → wide)."""
    df = pd.read_csv(L5 / "top10_per_split.csv")
    wide_rows = {}
    for (split, method), grp in df.groupby(["split", "method"]):
        grp = grp.sort_values("rank")
        cell = " \\newline ".join(f"{r}. {f}" for r, f in zip(grp["rank"], grp["feature"]))
        wide_rows.setdefault(split, {})[method] = cell

    splits_in_order = list(df["split"].drop_duplicates())
    wide = pd.DataFrame.from_dict(
        {s: wide_rows[s] for s in splits_in_order if s in wide_rows},
        orient="index",
    )[["DT", "SHAP", "LIME"]]
    wide.index.name = "split"
    wide.index = wide.index.map(lambda s: s.replace("->", r"$\to$"))

    latex = (
        wide.reset_index().style
        .hide(axis="index")
        .set_caption(
            "Top-10 feature attributions per L5B15 split per explanation "
            "method."
        )
        .to_latex(hrules=True, label="tab:top10_l5b15_per_split")
    )
    latex = latex.replace(r"\textbackslash newline", r"\newline")
    latex = latex.replace(
        r"\begin{tabular}{llll}",
        r"\begin{tabular}{l p{4.2cm} p{4.2cm} p{4.2cm}}",
    )
    _write(TABLES / "top10_per_split.tex", latex)


# Compact column labels for the PSI sensitivity tables.
_PSI_COL_MAP = {
    "L5B15 temporal":         "Temporal",
    "ORY->NCE vs ORY->OPO":   "Route pair 1",
    "ORY->OPO vs NCE->ORY":   "Route pair 2",
    "NCE->ORY vs TLS->ORY":   "Route pair 3",
    "fr-FR vs nl-NL":         "Culture",
}
_PSI_ROUTE_LEGEND = (
    "Route pair 1: ORY$\\to$NCE vs ORY$\\to$OPO; "
    "Route pair 2: ORY$\\to$OPO vs NCE$\\to$ORY; "
    "Route pair 3: NCE$\\to$ORY vs TLS$\\to$ORY; "
    "Culture: fr-FR vs nl-NL."
)


def _relabel_psi_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_PSI_COL_MAP.get(c, c.replace("->", r"$\to$"))
                  for c in df.columns]
    return df


def build_jaccard_sensitivity() -> None:
    """Appendix — L5B15 J_v strip plot (figure, not table).

    Visualises the bimodal distribution of observed $J_v$ values across the
    L5B15 splits, with candidate thresholds marked. Justifies the choice of
    $\\tau=0.10$ as sitting in the empty band between the temporal cluster
    near 0 and the in-window cluster near 0.5.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_json(L5 / "attribution_summary.json")
    j = df.set_index("split")["J_v (δ_m)"].sort_values()

    short_labels = {
        "L5B15 temporal":          "Temporal",
        "ORY->NCE vs ORY->OPO":    "Route pair 1",
        "ORY->OPO vs NCE->ORY":    "Route pair 2",
        "NCE->ORY vs TLS->ORY":    "Route pair 3",
        "fr-FR vs nl-NL":          "Culture",
    }

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    xs = j.values
    ys = np.full_like(xs, 0.05, dtype=float)
    ax.scatter(xs, ys, s=130, color="steelblue", alpha=0.9,
               edgecolor="white", linewidth=1.0, zorder=3)

    # Annotate; stagger vertical offset when points are close together
    sorted_items = list(j.items())
    last_x = -1.0
    offset_toggle = 1
    for split_name, jv in sorted_items:
        label = short_labels.get(split_name, split_name)
        if jv - last_x < 0.05:
            offset_toggle *= -1
        dy = 18 if offset_toggle > 0 else 38
        ax.annotate(
            label, (jv, 0.05),
            xytext=(0, dy), textcoords="offset points",
            ha="center", fontsize=9, rotation=20,
        )
        last_x = jv

    # Candidate threshold lines
    for tau, color, lbl in [
        (0.05, "#bbbbbb", r"$\tau=0.05$"),
        (0.10, "darkorange", r"$\tau=0.10$ (chosen)"),
        (0.25, "#bbbbbb", r"$\tau=0.25$"),
        (0.50, "#999999", r"$\tau=0.50$ (alt.)"),
    ]:
        ax.axvline(tau, color=color, linestyle="--", linewidth=1.1, alpha=0.85)
        ax.text(tau, -0.10, lbl, ha="center", va="top",
                fontsize=8.5, color=color)

    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-0.28, 0.55)
    ax.set_yticks([])
    ax.set_xlabel(r"$J_v$ (Jaccard overlap on modelVersion sets)")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_pdf = TABLES / "jaccard_threshold_sensitivity.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_pdf.relative_to(REPO)}")
    print(f"  → {out_png.relative_to(REPO)}")


def build_psi_propensity() -> None:
    """Appendix — propensity-PSI sensitivity to bin count."""
    df = pd.read_csv(L5 / "psi_bin_sensitivity_propensity.csv").set_index("bins")
    df = _relabel_psi_cols(df)
    latex = (
        df.reset_index().style
        .format({c: _NUM for c in df.columns})
        .format({"bins": "{:d}"})
        .hide(axis="index")
        .set_caption(
            "Propensity-PSI sensitivity to bin count. Each row corresponds "
            "to a different choice of PSI bin count (equal-frequency on the "
            "reference half); columns are the splits used in the main "
            "analysis. PSI values rise systematically with bin count, "
            "particularly for the temporal split, because Pega ADM's "
            "isotonic calibration produces a step-function propensity: "
            "increasing the bin count splits clustered values unevenly "
            "between the two halves and inflates PSI artificially. This "
            "motivates the choice of the bin-free KS statistic on "
            "propensity as the $\\delta_d$ metric in the main analysis. "
            + _PSI_ROUTE_LEGEND
        )
        .to_latex(hrules=True, label="tab:psi_bin_sensitivity_propensity")
    )
    _write(TABLES / "psi_bin_sensitivity_propensity.tex", latex)


def build_psi_auc() -> None:
    """Appendix — AUC-PSI sensitivity to bin count."""
    df = pd.read_csv(L5 / "psi_bin_sensitivity_auc.csv").set_index("bins")
    df = _relabel_psi_cols(df)
    latex = (
        df.reset_index().style
        .format({c: _NUM for c in df.columns})
        .format({"bins": "{:d}"})
        .hide(axis="index")
        .set_caption(
            "AUC-PSI sensitivity to bin count. Same construction as "
            "Table~\\ref{tab:psi_bin_sensitivity_propensity} but applied to "
            "the \\texttt{modelPerformance} (AUC) distribution. AUC-PSI is "
            "more stable across bin choices than propensity-PSI; PSI would "
            "have been an acceptable $\\delta_m$ proxy on this distribution. "
            "Jaccard overlap on model versions is preferred in the "
            "main analysis because it answers the snapshot-overlap question "
            "directly without forcing a categorical mixture identifier into "
            "a continuous-distance frame. "
            + _PSI_ROUTE_LEGEND
        )
        .to_latex(hrules=True, label="tab:psi_bin_sensitivity_auc")
    )
    _write(TABLES / "psi_bin_sensitivity_auc.tex", latex)


# ─────────────────────────────────────────────────────────────────────────
#  Dispatch
# ─────────────────────────────────────────────────────────────────────────

BUILDERS: dict[str, Callable[[], None]] = {
    # Main-text RQ tables
    "surrogate_fidelity":      build_surrogate_fidelity,
    "rq1_explainer_fidelity":  build_rq1_explainer_fidelity,
    "stability_matrix":        build_stability_matrix,
    "attribution_l5b15":       build_attribution_l5b15,
    "attribution_replication": build_attribution_replication,
    # Appendix tables (in narrative order)
    "missingness":             build_missingness,
    "depth_sensitivity":       build_depth_sensitivity,
    "metric_selection":        build_metric_selection,
    "surrogate_comparison":    build_surrogate_comparison,
    "psi_propensity":          build_psi_propensity,
    "psi_auc":                 build_psi_auc,
    "jaccard_sensitivity":     build_jaccard_sensitivity,
    "bootstrap_dt_summary":    build_bootstrap_dt_summary,
    "top10_per_split":         build_top10_per_split,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tables", nargs="*", default=["all"],
        help="Table names to build (default: all). See --list.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available table names and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for k in BUILDERS:
            print(k)
        return

    if args.tables == ["all"]:
        targets = list(BUILDERS)
    else:
        unknown = [t for t in args.tables if t not in BUILDERS]
        if unknown:
            print(f"Unknown tables: {unknown}", file=sys.stderr)
            print(f"Available: {list(BUILDERS)}", file=sys.stderr)
            sys.exit(1)
        targets = args.tables

    failures: list[tuple[str, str]] = []
    for name in targets:
        print(f"\n[{name}]")
        try:
            BUILDERS[name]()
        except FileNotFoundError as e:
            print(f"  SKIP — missing input: {e}")
            failures.append((name, f"missing input: {e}"))
        except Exception as e:
            print(f"  ERROR — {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))

    print(f"\n{'='*60}")
    print(f"Built {len(targets) - len(failures)}/{len(targets)} tables.")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

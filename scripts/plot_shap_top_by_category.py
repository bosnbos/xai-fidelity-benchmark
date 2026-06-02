"""Top-N SHAP feature-importance chart, coloured by category.

Reads `data/artifacts/<variant>/shap_importances.json`, classifies each feature
by its dot-prefix into IH / Booking context / Strategy param / Customer, and
plots a horizontal bar chart sorted by absolute SHAP value.

Output: `presentations/figures/shap_top<N>_<variant>_by_category.pdf`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

CATEGORY_COLORS = {
    "IH":               "#2CA02C",
    "Booking context":  "#1F77B4",
    "Strategy param":   "#FFA500",
    "Customer":         "#A100FF",
    "Other":            "#7F7F7F",
}


def categorize(feature: str) -> str:
    if feature.startswith("IH."):
        return "IH"
    if feature.startswith("CustBookedFlight."):
        return "Booking context"
    if feature.startswith("param::") or feature.startswith("Param."):
        return "Strategy param"
    if feature.startswith("Customer."):
        return "Customer"
    return "Other"


def pretty(feature: str) -> str:
    """Strip the dot-prefix and make the label more readable."""
    f = feature
    if f.startswith("param::Param."):
        f = f[len("param::Param."):]
    elif "." in f:
        f = f.split(".", 1)[1]
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="l5b15")
    p.add_argument("--top", type=int, default=15)
    args = p.parse_args()

    src = REPO / "data" / "artifacts" / args.variant / "shap_importances.json"
    out = REPO / "presentations" / "figures" / f"shap_top{args.top}_{args.variant}_by_category.pdf"

    shap = json.load(open(src))
    ranked = sorted(shap.items(), key=lambda kv: -abs(kv[1]))[: args.top]
    ranked.reverse()  # so largest sits on top in the bar chart

    labels = [pretty(f) for f, _ in ranked]
    values = [abs(v) for _, v in ranked]
    cats = [categorize(f) for f, _ in ranked]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(ranked) + 1.0))
    ax.barh(range(len(ranked)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"Mean $|$SHAP$|$", fontsize=10)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend in category order, but only categories that appear in the chart
    present = []
    seen = set()
    for c in cats:
        if c not in seen:
            present.append(c)
            seen.add(c)
    handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in present]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

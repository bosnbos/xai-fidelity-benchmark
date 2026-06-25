"""Top-N global SHAP feature-importance chart, coloured by category.

Reads `data/artifacts/<variant>/shap_importances.json`, maps each feature to
a short label with a category prefix ([IH], [BC], [SP]), and plots a
horizontal bar chart sorted by mean |SHAP| value.

Output: `presentations/figures/shap_top<N>_<variant>_by_category.pdf`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

CATEGORY_COLORS = {
    "IH":                "#2CA02C",
    "Booking context":   "#1F77B4",
    "Scoring parameter": "#FFA500",
    "Other":             "#7F7F7F",
}

# Short labels with category prefix — mirrors plot_shap_waterfall.py
SHORT_LABELS: dict[str, str] = {
    # Booking context
    "CustBookedFlight.BookingData.BookingMonth":         "[BC] Booking month",
    "CustBookedFlight.BookingData.BookerGender":         "[BC] Gender",
    "CustBookedFlight.BookingData.CultureCode":          "[BC] Culture",
    "CustBookedFlight.BookingData.FlightInboundArrival": "[BC] Inbound arrival",
    "CustBookedFlight.Language":                         "[BC] Language",
    "CustBookedFlight.Journey":                          "[BC] Journey",
    "CustBookedFlight.FlightNumberOperatorIATA":         "[BC] Operator (IATA)",
    "CustBookedFlight.SeatNumber":                       "[BC] Seat number",
    "CustBookedFlight.IsStaffStandBy":                   "[BC] Staff standby",
    "CustBookedFlight.FlightData.AirlineCodeIATA":       "[BC] Airline",
    "CustBookedFlight.FlightData.DestinationAirport":    "[BC] Destination",
    # IH — Email outbound
    "IH.Email.Outbound.Pending.pyHistoricalOutcomeCount":      "[IH] Eml Out·Pend count",
    "IH.Email.Outbound.Pending.pxLastOutcomeTime.DaysSince":   "[IH] Eml Out·Pend days",
    "IH.Email.Outbound.Pending.pxLastGroupID":                 "[IH] Eml Out·Pend group",
    "IH.Email.Outbound.Delivered.pxLastOutcomeTime.DaysSince": "[IH] Eml Out·Deliv days",
    "IH.Email.Outbound.Delivered.pxLastGroupID":               "[IH] Eml Out·Deliv group",
    # IH — Email inbound
    "IH.Email.Inbound.Pending.pxLastOutcomeTime.DaysSince":    "[IH] Eml In·Pend days",
    "IH.Email.Inbound.Clicked.pxLastOutcomeTime.DaysSince":    "[IH] Eml In·Click days",
    # IH — Push outbound
    "IH.Push.Outbound.Pending.pyHistoricalOutcomeCount":       "[IH] Push Out·Pend count",
    "IH.Push.Outbound.Pending.pxLastOutcomeTime.DaysSince":    "[IH] Push Out·Pend days",
    "IH.Push.Outbound.Pending.pxLastGroupID":                  "[IH] Push Out·Pend group",
    # IH — Event
    "IH.Event.Outbound.RealTimeEvent.pyHistoricalOutcomeCount": "[IH] Event RT count",
    # Scoring parameter
    "param::Param.BundleName": "[SP] Bundle",
}

PREFIX_TO_CATEGORY = {
    "[IH]": "IH",
    "[BC]": "Booking context",
    "[SP]": "Scoring parameter",
}


def shorten(feature: str) -> str:
    if feature in SHORT_LABELS:
        return SHORT_LABELS[feature]
    # Fallback for unknown features: strip namespace prefix
    if "." in feature:
        return feature.split(".", 1)[1]
    return feature


def categorize(feature: str) -> str:
    label = shorten(feature)
    return PREFIX_TO_CATEGORY.get(label[:4], "Other")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="l5b15")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    src = REPO / "data" / "artifacts" / args.variant / "shap_importances.json"
    out = (
        REPO / "presentations" / "figures"
        / f"shap_top{args.top}_{args.variant}_by_category.pdf"
    )

    raw = json.load(open(src))
    ranked = sorted(raw.items(), key=lambda kv: -abs(kv[1]))[: args.top]
    ranked.reverse()   # largest at top of the horizontal bar chart

    labels  = [shorten(f) for f, _ in ranked]
    values  = [abs(v) for _, v in ranked]
    cats    = [categorize(f) for f, _ in ranked]
    colors  = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(ranked) + 1.2))
    bars = ax.barh(range(len(ranked)), values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.set_xlabel(r"Mean $|$SHAP$|$", fontsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.4f}" if x != 0 else "0")
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend — only categories present in the chart
    present: list[str] = []
    seen: set[str] = set()
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

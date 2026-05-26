"""Three-panel feature-importance heatmap (SHAP / LIME / DT) across L5B15 splits.

Reads:  data/artifacts/L5B15/per_split_rankings.json
Writes: data/artifacts/L5B15/feature_importance_heatmap.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART_DIR = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "L5B15"
SRC = ART_DIR / "per_split_rankings.json"

# Display order: early/late, four routes, two culture codes.
SPLIT_ORDER = [
    ("L5B15 early", "early"),
    ("L5B15 late",  "late"),
    ("ORY->NCE",    "ORY→NCE"),
    ("ORY->OPO",    "ORY→OPO"),
    ("NCE->ORY",    "NCE→ORY"),
    ("TLS->ORY",    "TLS→ORY"),
    ("fr-FR",       "fr-FR"),
    ("nl-NL",       "nl-NL"),
]
METHODS = [("shap", "SHAP", "Blues"),
           ("lime", "LIME", "Blues"),
           ("dt",   "Decision Tree", "Blues")]

# Feature-group definitions. Tag is used as a prefix on the y labels so the
# grouping is readable in greyscale; the matching colour tints the tick text
# and the legend swatches.
GROUPS = {
    "CH": ("Contact history",    "#1f77b4"),
    "BC": ("Booking context",    "#2ca02c"),
    "FD": ("Flight data",        "#d62728"),
    "SP": ("Scoring parameter",  "#9467bd"),
}


def group_of(feature_name: str) -> str:
    """Map a raw Pega feature name to one of CH / BC / FD / SP."""
    n = feature_name
    if n.startswith("IH."):
        return "CH"
    if n.startswith("param::"):
        return "SP"
    if n.startswith("CustBookedFlight.BookingData."):
        return "BC"
    if n.startswith("CustBookedFlight.FlightData.") or n.startswith("CustBookedFlight."):
        return "FD"
    # Fallback — keep grouped with flight data rather than crashing.
    return "FD"


def short(name: str) -> str:
    s = (name
         .replace("CustBookedFlight.BookingData.", "")
         .replace("CustBookedFlight.FlightData.", "")
         .replace("CustBookedFlight.", "")
         .replace("IH.Email.Outbound.", "Email.Out.")
         .replace("IH.Email.Inbound.",  "Email.In.")
         .replace("IH.Push.Outbound.",  "Push.Out.")
         .replace("param::Param.",      "Param.")
         .replace(".pxLastOutcomeTime.DaysSince", ".LastOut.Days")
         .replace(".pyHistoricalOutcomeCount",    ".HistCount")
         .replace(".pxLastGroupID",               ".LastGroup"))
    return s


def main() -> None:
    data = json.loads(SRC.read_text())

    # Collect the union of features that appear in any (split, method).
    features = sorted({
        f for split in data.values() for m in split.values() for f in m.keys()
    })

    # Build a (feature × split) matrix per method.
    split_keys = [k for k, _ in SPLIT_ORDER]
    split_lbls = [lbl for _, lbl in SPLIT_ORDER]
    mats: dict[str, pd.DataFrame] = {}
    for m_key, _, _ in METHODS:
        M = pd.DataFrame(
            {lbl: [data[k][m_key].get(f, np.nan) for f in features]
             for k, lbl in zip(split_keys, split_lbls)},
            index=features,
        ).fillna(0.0)
        mats[m_key] = M

    # Per-panel normalisation to [0, 1] by panel max.
    norm = {m: (M / M.values.max() if M.values.max() > 0 else M) for m, M in mats.items()}

    # Order features by mean of the panel-normalised values across all
    # cells (so cross-method scale differences are removed).
    stacked = pd.concat(norm.values(), axis=1)
    order = stacked.mean(axis=1).sort_values(ascending=False).index.tolist()
    groups = [group_of(f) for f in order]
    named_labels = [f"[{g}] {short(f)}" for f, g in zip(order, groups)]
    anon_labels  = [f"[{g}] Feature {i + 1}" for i, g in enumerate(groups)]

    # Render both versions (named for internal use, anonymised for sharing).
    _render(norm, order, named_labels, split_lbls, groups,
            left_margin=0.32, suffix="")
    _render(norm, order, anon_labels, split_lbls, groups,
            left_margin=0.18, suffix="_anonymised")


def _render(norm, order, y_labels, split_lbls, groups, *,
            left_margin: float, suffix: str) -> None:
    """Draw and save one variant of the heatmap.

    left_margin is tuned to whichever label form is being drawn — anonymised
    "Feature N" labels are short and need less left padding than the named
    feature labels. `groups` aligns with `order` and colours the y-tick labels.
    """
    n_feat = len(order)
    fig_w_in = 22 / 2.54                       # 22 cm
    fig_h_in = max(6.0, 0.32 * n_feat + 1.9)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.035],
        wspace=0.10,
        left=left_margin, right=0.92, top=0.92, bottom=0.22,
    )
    ax_shap = fig.add_subplot(gs[0, 0])
    ax_lime = fig.add_subplot(gs[0, 1])
    ax_dt   = fig.add_subplot(gs[0, 2])
    cax     = fig.add_subplot(gs[0, 3])
    axes = [ax_shap, ax_lime, ax_dt]

    last_im = None
    for ax, (m_key, m_title, cmap) in zip(axes, METHODS):
        M = norm[m_key].loc[order, split_lbls]
        last_im = ax.imshow(
            M.values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(m_title, fontsize=11, pad=6)
        ax.set_xticks(range(len(split_lbls)))
        ax.set_xticklabels(split_lbls, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_yticklabels(y_labels, fontsize=8)
    # Tint each y-tick label by its feature group.
    for tick_label, g in zip(axes[0].get_yticklabels(), groups):
        tick_label.set_color(GROUPS[g][1])
    for ax in axes[1:]:
        ax.set_yticklabels([])

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Importance (normalised per panel)", fontsize=8)
    cbar.outline.set_visible(False)

    # Legend mapping group tags to full names.
    legend_handles = [
        mpatches.Patch(color=colour, label=f"[{tag}] {full}")
        for tag, (full, colour) in GROUPS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle("L5B15 feature importance across splits", fontsize=12)

    out_pdf = ART_DIR / f"feature_importance_heatmap{suffix}.pdf"
    out_png = ART_DIR / f"feature_importance_heatmap{suffix}.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

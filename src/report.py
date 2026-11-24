"""Static figures for the README / petition exhibit.

Usage:
    python src/report.py --outputs outputs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FORECAST, NATIONAL, ALERT = "#2a78d6", "#eb6834", "#e34948"
NAMES = {"LOC-I": "Loss of control in flight", "RE": "Runway excursion", "CFIT": "Controlled flight into terrain",
         "MAC": "Mid-air / near mid-air", "RI": "Runway incursion"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def backtest_figure(nat, path):
    cats = ["LOC-I", "RE", "CFIT", "MAC", "RI"]
    fig, axes = plt.subplots(len(cats), 1, figsize=(9, 11), sharex=True, facecolor=SURFACE)
    nat = nat.assign(t=nat.year + (nat.qtr - 1) / 4)
    for ax, c in zip(axes, cats):
        g = nat[nat.category == c].sort_values("t")
        style(ax)
        ax.fill_between(g.t, g.lower95, g.upper95, color=FORECAST, alpha=0.15, linewidth=0,
                        label="95% forecast range")
        ax.plot(g.t, g.pred_ensemble, color=FORECAST, linewidth=2, label="Forecast (made 2 quarters ahead)")
        ax.plot(g.t, g.y, color=INK, linewidth=1.2, marker="o", markersize=3, label="Actual NTSB cases")
        a = g[g.alert_spike]
        ax.scatter(a.t, a.y, s=70, facecolors="none", edgecolors=ALERT, linewidths=2, zorder=5,
                   label="Spike alert (above 95% range)")
        ax.set_title(NAMES[c], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_ylim(bottom=0)
    ri = nat[(nat.category == "RI") & nat.alert_spike & (nat.year == 2023)]
    if not ri.empty:
        r = ri.iloc[0]
        axes[-1].annotate("Q1 2023: JFK, Austin, Burbank, Boston\nclose calls (FAA Safety Summit, Mar 2023)",
                          xy=(r.t, r.y), xytext=(r.t + 0.6, r.y - 0.5), fontsize=8, color=INK,
                          arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.supylabel("Cases per quarter (U.S. total)", color=MUTED, fontsize=9)
    fig.suptitle("Walk-forward forecasts of U.S. high-risk aviation occurrences, 2018–2026",
                 x=0.06, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.06, 0.005, "Source: NTSB aviation database (Nov 2026). Each year is forecast by models "
             "trained only on earlier years.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path, dpi=200)


def skill_figure(met, path):
    cats = ["LOC-I", "RE", "CFIT", "MAC", "RI"]
    m = met[(met.model == "ensemble") & met.category.isin(cats)]
    piv = m.pivot(index="category", columns="level", values="RMSE_skill_vs_MA4_pct").reindex(cats)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    y = range(len(cats))
    h = 0.36
    b1 = ax.barh([i - h / 2 for i in y], piv.division, height=h - 0.03, color=FORECAST, label="Regional (9 divisions)")
    b2 = ax.barh([i + h / 2 for i in y], piv.national, height=h - 0.03, color=NATIONAL, label="National")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_width()
            if v >= 0:
                ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:+.0f}%",
                        va="center", ha="left", fontsize=8, color=INK)
            else:  # label sits just right of zero so it never collides with the category names
                ax.text(1, b.get_y() + b.get_height() / 2, f"{v:+.0f}%",
                        va="center", ha="left", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(list(y), [NAMES[c] for c in cats], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Reduction in forecast error (RMSE) vs. 4-quarter moving average", color=MUTED, fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Where the model beats a simple baseline (2018–2026 backtest)", loc="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    out = Path(ap.parse_args().outputs)
    (out / "figures").mkdir(exist_ok=True)
    backtest_figure(pd.read_csv(out / "backtest_national.csv"), out / "figures" / "backtest_national.png")
    skill_figure(pd.read_csv(out / "metrics.csv"), out / "figures" / "skill_vs_baseline.png")
    print("saved figures to", out / "figures")


if __name__ == "__main__":
    main()

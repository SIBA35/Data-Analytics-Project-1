"""Clean, consistent charts for the executive report and dashboard (one colour palette everywhere)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

NAVY, TEAL, AMBER, CORAL, PURPLE = "#1F3A5F", "#2A9D8F", "#E9A23B", "#E76F51", "#7B5EA7"
GREY, LIGHTGREY, INK, BG = "#8A94A6", "#D9DFE8", "#1B2733", "#FFFFFF"
TYPE_COLORS = {"Apartment": NAVY, "Independent House": AMBER, "Villa": TEAL}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

RC = {"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": LIGHTGREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": BG, "axes.facecolor": BG,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.grid": False, "axes.spines.left": True, "axes.spines.bottom": True}


def _clean(ax, grid="x"):
    ax.spines["left"].set_color(LIGHTGREY); ax.spines["bottom"].set_color(LIGHTGREY)
    if grid:
        ax.grid(axis=grid, color="#EEF1F6", linewidth=1); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _pct(x, _pos=None):
    return "0%" if abs(x) < 1e-9 else f"{x:+.0f}%"


# ------------------------------------------------------------------ report charts


def fig_locality_ppsf(df):
    m = df.groupby("Location").PPSF.median().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = [CORAL if i < 2 else TEAL if i >= len(m) - 2 else NAVY for i in range(len(m))]
    ax.barh(m.index, m.values, color=colors, height=0.62)
    for y, v in enumerate(m.values):
        ax.text(v + m.max() * 0.012, y, f"{v:,.0f}", va="center", fontsize=9.5)
    med = df.PPSF.median(); ax.axvline(med, color=GREY, ls="--", lw=1)
    ax.set_xlim(0, m.max() * 1.15); ax.set_xlabel(f"Median price per sqft (INR); dashed line = overall median {med:,.0f}")
    ax.set_title(f"{m.index[-1]} is priciest, {m.index[0]} cheapest"); _clean(ax); fig.tight_layout(); return fig


def fig_area_price(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for t, c in TYPE_COLORS.items():
        s = df[df.Type == t]; ax.scatter(s.Area, s.Price_Cr, s=20, alpha=.6, color=c, label=t, linewidths=0)
    b = np.polyfit(df.Area, df.Price_Cr, 1); xs = np.linspace(df.Area.min(), df.Area.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK, ls="--", lw=1.3, label="Trend")
    ax.set_xlabel("Area (sqft)"); ax.set_ylabel("Price (INR crore)"); ax.set_title("Bigger homes cost more, almost linearly")
    ax.legend(loc="upper left", fontsize=9, ncol=2); _clean(ax, "y"); fig.tight_layout(); return fig


def fig_type_ppsf(df):
    order = [t for t in TYPE_COLORS if t in df.Type.unique()]
    data = [df.loc[df.Type == t, "PPSF"].values for t in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bp = ax.boxplot(data, patch_artist=True, widths=.5, medianprops=dict(color="white", lw=2), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY), flierprops=dict(marker="o", markersize=3, markerfacecolor=GREY, markeredgecolor="none"))
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(TYPE_COLORS[t]); patch.set_edgecolor(TYPE_COLORS[t])
    for i, d in enumerate(data, 1):
        ax.text(i, np.median(d), f"{np.median(d):,.0f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK, bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order); ax.set_ylabel("Price per sqft (INR)")
    ax.set_title("Villas sell at a premium per sqft"); _clean(ax, "y"); fig.tight_layout(); return fig


def fig_time_check(df):
    m = df.groupby("Month").PPSF.median(); med = df.PPSF.median()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhspan(med * 0.9, med * 1.1, color="#EEF2F7", zorder=0); ax.axhline(med, color=GREY, ls="--", lw=1)
    ax.plot(m.index, m.values, color=TEAL, lw=2.2, marker="o", ms=5)
    ax.text(12.3, med, " overall\n median", color=GREY, fontsize=8.5, va="center")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS, fontsize=9); ax.set_xlim(0.6, 13.2)
    ax.set_ylim(med * 0.75, med * 1.25); ax.set_ylabel("Median price per sqft (INR)")
    ax.set_title("No seasonality: monthly price stays within +/-10%"); _clean(ax, "y"); fig.tight_layout(); return fig








def fig_price_hist(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(df.Price_Cr, bins=22, color=NAVY, edgecolor="white", linewidth=.8)
    med = df.Price_Cr.median(); ax.axvline(med, color=AMBER, lw=2.2)
    ax.text(med + 0.05, ax.get_ylim()[1] * 0.93, f"median INR {med:.2f} cr", color=AMBER, fontsize=10, fontweight="bold", va="top")
    ax.set_xlabel("Price (INR crore)"); ax.set_ylabel("Listings"); q1, q3 = df.Price_Cr.quantile([.25, .75]); ax.set_title(f"Half of all homes cost between INR {q1:.1f} and {q3:.1f} crore")
    _clean(ax, "y"); fig.tight_layout(); return fig


def fig_type_mix(df):
    c = df.Type.value_counts().reindex([t for t in TYPE_COLORS if t in df.Type.unique()])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(c.index, c.values, color=[TYPE_COLORS[t] for t in c.index], width=.55)
    for i, v in enumerate(c.values):
        ax.text(i, v + c.max() * .015, f"{v}  ({v / c.sum() * 100:.0f}%)", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, c.max() * 1.15); ax.set_ylabel("Listings"); ax.set_title("Apartments make up three of every four listings")
    _clean(ax, "y"); fig.tight_layout(); return fig


def fig_bedrooms(df):
    g = df.groupby("Bedrooms").agg(m=("Price_Cr", "median"), n=("Price", "size"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(g.index.astype(str) + " BHK", g.m, color=TEAL, width=.55)
    for i, (m, n) in enumerate(zip(g.m, g.n)):
        ax.text(i, m + g.m.max() * .015, f"{m:.2f}", ha="center", fontsize=10.5, fontweight="bold"); ax.text(i, g.m.max() * .03, f"n={n}", ha="center", color="white", fontsize=9)
    ax.set_ylim(0, g.m.max() * 1.15); ax.set_ylabel("Median price (INR crore)"); ax.set_title("Price rises with every bedroom up to 4 BHK")
    _clean(ax, "y"); fig.tight_layout(); return fig


def fig_age_ppsf(df):
    g = df.groupby("Age_Bucket", observed=True).PPSF.median()
    cols = [TEAL, "#6BB5A0", AMBER, CORAL][: len(g)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(g.index.astype(str), g.values, color=cols, width=.55)
    for i, v in enumerate(g.values):
        ax.text(i, v + g.max() * .015, f"{v:,.0f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, g.max() * 1.15); ax.set_xlabel("Building age (years)"); ax.set_ylabel("Median price per sqft (INR)")
    ax.set_title("Older buildings sell for less per sqft"); _clean(ax, "y"); fig.tight_layout(); return fig


def fig_value_drivers(vd):
    t = vd.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, (lab, r) in enumerate(t.iterrows()):
        sig = not (r.low <= 0 <= r.high); c = (TEAL if r.pct > 0 else CORAL) if sig else GREY
        ax.plot([r.low, r.high], [i, i], color=c, lw=3.2, solid_capstyle="round"); ax.plot(r.pct, i, "o", color=c, ms=10, zorder=3)
        ax.text(max(r.high, 0) + 0.8, i, f"{r.pct:+.1f}%", va="center", fontsize=10.5, fontweight="bold", color=c)
    ax.axvline(0, color=INK, lw=1); ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct)); ax.set_xlim(min(t.low.min() - 2, -12), max(t.high.max() + 7, 28))
    ax.set_xlabel("Change in price per sqft, all else equal"); ax.set_xticks(range(-10, 31, 10))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color=TEAL, lw=3, label="raises price"), Line2D([0], [0], marker="o", color=CORAL, lw=3, label="lowers price"),
                       Line2D([0], [0], marker="o", color=GREY, lw=3, label="no measurable effect")], loc="lower right", fontsize=9)
    ax.set_title("What adds to or takes away from value"); _clean(ax); fig.tight_layout(); return fig


def fig_heatmap_loc_type(df):
    p = df.pivot_table(index="Location", columns="Type", values="PPSF", aggfunc="median")
    p = p.reindex(df.groupby("Location").PPSF.median().sort_values(ascending=False).index)[[t for t in TYPE_COLORS if t in df.Type.unique()]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pal", ["#EAF4F3", TEAL, NAVY])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(p.values, cmap=cmap, aspect="auto", vmin=np.nanmin(p.values), vmax=np.nanmax(p.values))
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns); ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index)
    lo, hi = np.nanmin(p.values), np.nanmax(p.values)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if (v - lo) / (hi - lo) > .45 else INK)
    ax.set_title("Median price per sqft (INR) by locality and type"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); return fig


def _styled(fn):
    import functools
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with mpl.rc_context(RC):
            return fn(*a, **k)
    return wrapper


for _n in [n for n in list(globals()) if n.startswith("fig_")]:
    globals()[_n] = _styled(globals()[_n])

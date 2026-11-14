# SYNTHETIC CONTROL + PRE-TREATMENT FIT VALIDATION (MSPE / RMSPE)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # set to "TkAgg"/remove for pop-up plots
import matplotlib.pyplot as plt
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest


def main():
    # ---------------- 1. SETTINGS (EDIT THESE) ----------------
    DATA_PATH      = "data/properties_clean.csv"  # file that HAS the date column
    DATE_COL       = "observation_date"        # "Observation Date" after name cleaning
    UNIT_COL       = "location"                # units to compare
    OUTCOME        = "price_per_sqft"          # what we track over time
    FREQ           = "M"                       # "M" monthly (data covers only Jan-Dec 2025)
    TREATED        = "Baner"                   # <-- treated location (from project brief)
    TREATMENT_DATE = "2025-07-01"              # <-- when treatment started (from brief)
    FILL_GAPS      = True                      # fill empty location-months by linear interpolation
    MIN_PRE_PERIODS = 4                        # need enough history before treatment
    OUT_DIR        = "synth_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- 2. BUILD THE PANEL ----------------
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if DATE_COL not in df.columns:
        raise KeyError(f"'{DATE_COL}' not found. Columns are: {df.columns.tolist()}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    print("Rows with invalid/missing date (dropped):", df[DATE_COL].isna().sum())
    df = df.dropna(subset=[DATE_COL, OUTCOME, UNIT_COL])
    df["period"] = df[DATE_COL].dt.to_period(FREQ)

    panel = df.groupby([UNIT_COL, "period"], as_index=False)[OUTCOME].mean()
    wide = panel.pivot(index="period", columns=UNIT_COL, values=OUTCOME)
    wide = wide.reindex(pd.period_range(wide.index.min(), wide.index.max(), freq=FREQ))
    wide.index.name = "period"
    print("Empty location-periods before filling:", int(wide.isna().sum().sum()))
    if FILL_GAPS:
        wide = wide.interpolate(limit_direction="both")   # fills gaps from neighbouring months

    # keep only units observed in EVERY period (synthetic control needs a balanced panel)
    complete = wide.dropna(axis=1)
    dropped = [c for c in wide.columns if c not in complete.columns]
    print("Units dropped (gaps in some periods):", dropped)
    if TREATED not in complete.columns:
        raise ValueError(f"Treated unit '{TREATED}' has missing periods - use a coarser FREQ.")

    periods = list(complete.index)
    t_index = {p: i for i, p in enumerate(periods, start=1)}
    T0_period = pd.Period(TREATMENT_DATE, freq=FREQ)
    T0 = t_index[T0_period]                       # first treated period (number)
    pre_t  = list(range(1, T0))
    post_t = list(range(T0, len(periods) + 1))
    print(f"Periods: {len(periods)} | pre: {len(pre_t)} | post: {len(post_t)} | donors: {complete.shape[1]-1}")
    if len(pre_t) < MIN_PRE_PERIODS or len(post_t) < 1:
        raise ValueError("Need more pre-treatment periods and at least 1 post period.")

    long = (complete.reset_index().melt(id_vars="period", var_name=UNIT_COL, value_name=OUTCOME))
    long["t"] = long["period"].map(t_index)
    long = long.drop(columns="period")
    controls = [c for c in complete.columns if c != TREATED]

    # ---------------- 3. FIT THE SYNTHETIC CONTROL ----------------
    dp = Dataprep(
        foo=long,
        predictors=[OUTCOME], predictors_op="mean", time_predictors_prior=pre_t,
        special_predictors=[(OUTCOME, [t], "mean") for t in pre_t],   # match every pre period
        dependent=OUTCOME, unit_variable=UNIT_COL, time_variable="t",
        treatment_identifier=TREATED, controls_identifier=controls,
        time_optimize_ssr=pre_t,
    )
    synth = Synth()
    synth.fit(dataprep=dp)

    w = synth.weights()
    print("\nDonor weights (non-zero):")
    print(w[w > 0.001].round(3).to_string())

    # ---------------- 4. SYNTHETIC PATH + GAP ----------------
    Y_treated = complete[TREATED].to_numpy()
    Y_donors  = complete[controls].to_numpy()
    w_vec     = np.array([w[c] for c in controls])
    Y_synth   = Y_donors @ w_vec
    gap       = pd.Series(Y_treated - Y_synth, index=range(1, len(periods) + 1))

    # ---------------- 5. PRE-TREATMENT FIT VALIDATION ----------------
    pre_mspe  = (gap.loc[pre_t] ** 2).mean()
    pre_rmspe = np.sqrt(pre_mspe)
    pre_mean  = Y_treated[:len(pre_t)].mean()
    rmspe_pct = 100 * pre_rmspe / pre_mean
    post_mspe = (gap.loc[post_t] ** 2).mean()
    ratio     = post_mspe / pre_mspe

    print("\n========== PRE-TREATMENT FIT ==========")
    print(f"Pre-treatment MSPE  : {pre_mspe:,.2f}")
    print(f"Pre-treatment RMSPE : {pre_rmspe:,.2f}  ({OUTCOME} units)")
    print(f"RMSPE as % of treated pre-mean: {rmspe_pct:.2f}%")
    print(f"pysyncon cross-check MSPE     : {synth.mspe():,.2f}")
    print(f"Post-treatment MSPE : {post_mspe:,.2f}")
    print(f"Post/Pre MSPE ratio : {ratio:.2f}")
    # Rule of thumb only: a pre-fit error under ~5% of the level is usually considered a good fit
    print("Fit verdict (rule of thumb <5%):", "GOOD" if rmspe_pct < 5 else "WEAK - interpret with caution")

    # ---------------- 6. PLACEBO TEST (is the treated gap unusual?) ----------------
    pt = PlaceboTest()
    pt.fit(dataprep=dp, scm=synth,
           scm_options={"optim_method": "Nelder-Mead", "optim_initial": "equal"})

    def mspe_parts(g):
        pre = (g[g.index < T0] ** 2).mean()
        post = (g[g.index >= T0] ** 2).mean()
        return pre, post, post / pre

    rows = [{"unit": TREATED, "pre_MSPE": pre_mspe, "post_MSPE": post_mspe, "ratio": ratio}]
    for c in pt.gaps.columns:
        p, q, r = mspe_parts(pt.gaps[c])
        rows.append({"unit": c, "pre_MSPE": p, "post_MSPE": q, "ratio": r})
    tbl = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
    tbl.to_csv(f"{OUT_DIR}/mspe_table.csv", index=False)
    print("\nMSPE table (all units):")
    print(tbl.round(2).to_string(index=False))

    rank = int(tbl.index[tbl["unit"] == TREATED][0]) + 1
    print(f"\nTreated unit ratio rank: {rank} of {len(tbl)}  -> placebo p-value = {rank/len(tbl):.3f}")

    # ---------------- 7. PLOTS ----------------
    x = [str(p) for p in periods]
    plt.figure(figsize=(10, 5))
    plt.plot(x, Y_treated, "o-", label=f"{TREATED} (actual)")
    plt.plot(x, Y_synth, "s--", label="Synthetic control")
    plt.axvline(x[T0 - 1], color="red", ls="--", label="Treatment start")
    plt.title(f"Actual vs Synthetic {TREATED}  |  Pre RMSPE = {pre_rmspe:,.0f} ({rmspe_pct:.1f}%)")
    plt.xlabel("Period"); plt.ylabel(OUTCOME); plt.xticks(rotation=45); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/synthetic_vs_actual.png", dpi=130); plt.close()

    plt.figure(figsize=(10, 5))
    for c in pt.gaps.columns:
        plt.plot(x, pt.gaps[c].values, color="lightgray")
    plt.plot(x, gap.values, color="black", lw=2.5, label=f"{TREATED} gap")
    plt.axhline(0, color="k", lw=0.8); plt.axvline(x[T0 - 1], color="red", ls="--")
    plt.title("Placebo Gaps (gray = other locations)")
    plt.xlabel("Period"); plt.ylabel("Actual - Synthetic"); plt.xticks(rotation=45); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/placebo_gaps.png", dpi=130); plt.close()

    print(f"\nSaved to ./{OUT_DIR}/ : synthetic_vs_actual.png, placebo_gaps.png, mspe_table.csv")


# Required on Windows: pysyncon's placebo test starts extra processes,
# and without this guard each one would re-run the whole script.
if __name__ == "__main__":
    main()
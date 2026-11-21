import matplotlib.pyplot as plt
import streamlit as st

import ui
import viz
from data import load_data
from report import conclusions, kpis, locality_table, value_drivers

st.set_page_config(page_title="Pune Property Market - Executive Report", layout="wide")
ui.inject_css()


@st.cache_data
def get_data():
    return load_data()


def show(fig):
    st.pyplot(fig); plt.close(fig)


try:
    df = get_data()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()

k = kpis(df); lt = locality_table(df); vd = value_drivers(df)
top, bot = lt.index[0], lt.index[-1]

ui.hero("Pune Property Market: Executive Report", f"Full dataset of {k['listings']} listings across {k['localities']} localities",
        f"Listings observed {df.Date.min():%d %b %Y} to {df.Date.max():%d %b %Y}")
ui.kpi_cards([("Listings", f"{k['listings']:,}", f"{k['localities']} localities"),
              ("Total listed value", f"INR {k['total_cr']:,.0f} cr", "all 500 listings"),
              ("Median price", f"INR {k['median_cr']:.2f} cr", f"range {k['pmin']:.2f} to {k['pmax']:.2f} cr"),
              ("Median price per sqft", f"INR {k['ppsf']:,.0f}", f"{bot} to {top}: {lt.Median_ppsf.iloc[-1]:,.0f} to {lt.Median_ppsf.iloc[0]:,.0f}")])

ui.section(1, "Market snapshot", "Price levels and the mix of homes on the market.")
a, b = st.columns(2)
with a:
    show(viz.fig_price_hist(df))
with b:
    show(viz.fig_type_mix(df))

ui.section(2, "Locality performance", "Where a home is located is the largest driver of price per sqft.")
a, b = st.columns([3, 2])
with a:
    show(viz.fig_locality_ppsf(df))
with b:
    rows = [[loc, f"{r.Listings:.0f}", f"{r.Median_price_cr:.2f}", f"{r.Median_ppsf:,.0f}", f"{r.vs_market:+.0f}%"] for loc, r in lt.iterrows()]
    colors = {(i, 4): ("#2A9D8F" if r.vs_market >= 0 else "#E76F51") for i, (_, r) in enumerate(lt.iterrows())}
    ui.table(rows, ["Locality", "Listings", "Median price (cr)", "Price / sqft", "vs market"], colors)
ui.callout(f"{top} is {lt.vs_market.iloc[0]:.0f}% above the market median per sqft; {bot} is {abs(lt.vs_market.iloc[-1]):.0f}% below it.", "info", "Takeaway:")

ui.section(3, "Property profile", "How size, bedrooms, type and age relate to price.")
a, b = st.columns(2)
with a:
    show(viz.fig_area_price(df))
with b:
    show(viz.fig_bedrooms(df))
a, b = st.columns(2)
with a:
    show(viz.fig_type_ppsf(df))
with b:
    show(viz.fig_age_ppsf(df))

ui.section(4, "What adds to or takes away from value", "Effect of each factor on price per sqft with location, type and the other factors held equal.")
show(viz.fig_value_drivers(vd))

ui.section(5, "Segment map", "Price per sqft for every locality and property type.")
show(viz.fig_heatmap_loc_type(df))

ui.section(6, "Timing", "Monthly price per sqft across the observation period.")
a, b = st.columns([3, 2])
with a:
    show(viz.fig_time_check(df))
with b:
    ui.callout("Price per sqft shows no seasonal pattern across the year.", "info", "Result:")

ui.section(7, "Conclusion", "What the full dataset tells us.")
ui.conclusion_cards(conclusions(df, vd))

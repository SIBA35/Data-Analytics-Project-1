# Libraries import
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Real Estate Pricing Simulator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. COLOR THEME AND CSS
# ==========================================================

BG = "#F5F7FB"
WHITE = "#FFFFFF"
BLUE = "#2563EB"
GREEN = "#10B981"
PURPLE = "#8B5CF6"
ORANGE = "#F59E0B"
DARK = "#1E293B"
MUTED = "#64748B"
BORDER = "#E2E8F0"

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FB;
    color: #1E293B;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #1E293B !important;
}

p, label, span {
    color: #475569;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}

[data-testid="stMetricValue"] {
    color: #1E293B;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: #64748B;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}

.stButton button {
    background: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

.stButton button:hover {
    background: #1D4ED8;
    color: white;
}

hr {
    border-color: #E2E8F0;
}

/* =========================================
   PROPERTY DETAIL SLICERS
========================================= */

/* Selectbox gradient background */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: linear-gradient(
        135deg,
        #EFF6FF,
        #DBEAFE
    ) !important;

    border: 1.5px solid #93C5FD !important;
    border-radius: 9px !important;
    min-height: 42px;
    transition: all 0.25s ease;
}

/* Selectbox hover */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border: 1.5px solid #2563EB !important;
    background: linear-gradient(
        135deg,
        #DBEAFE,
        #BFDBFE
    ) !important;
}

/* Selectbox focus */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border: 2px solid #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
}

/* Selected text */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #1E3A8A !important;
    font-weight: 600;
}

/* Number input styling */
[data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: linear-gradient(
        135deg,
        #EFF6FF,
        #DBEAFE
    ) !important;

    border: 1.5px solid #93C5FD !important;
    border-radius: 9px !important;
    transition: all 0.25s ease;
}

/* Number input hover and focus */
[data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="input"]:hover {
    border-color: #2563EB !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border: 2px solid #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12);
}

/* Number input text */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    color: #1E3A8A !important;
    font-weight: 600;
}

/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] {
    background: #DBEAFE !important;
}

/* Slider active track and thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #2563EB !important;
    border: 2px solid white !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15);
}

/* Dropdown menu */
[data-baseweb="popover"] ul {
    background: #FFFFFF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 9px !important;
}

[data-baseweb="popover"] li:hover {
    background: #DBEAFE !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #1E293B !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("../data/properties_clean.csv")

    numeric_cols = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "floor_number",
        "age",
        "dist_city",
        "price",
        "price_per_sqft"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    required = [
        "location",
        "property_type",
        "area_sqft",
        "price",
        "price_per_sqft"
    ]

    data = data.dropna(subset=required)

    data = data[
        (data["area_sqft"] > 0) &
        (data["price"] > 0) &
        (data["price_per_sqft"] > 0)
    ].copy()

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

if df.empty:
    st.error("No valid property data available.")
    st.stop()


# ==========================================================
# 4. HELPER FUNCTIONS
# ==========================================================

def money(value):
    return f"₹{value:,.0f}"


def lakh(value):
    if value >= 10000000:
        return f"₹{value/10000000:.2f} Cr"
    return f"₹{value/100000:.2f} L"


def comparable_properties(data, settings):

    # First identify location and property type
    base = data[
        (data["location"] == settings["location"]) &
        (data["property_type"] == settings["type"])
    ].copy()

    if len(base) < 3:
        base = data[
            data["property_type"] == settings["type"]
        ].copy()

    if len(base) < 3:
        base = data.copy()

    comps = base.copy()
    applied = []

    conditions = [
        ("Bedrooms", "bedrooms", settings["bedrooms"]),
        ("Bathrooms", "bathrooms", settings["bathrooms"]),
        ("Furnishing", "furnishing", settings["furnishing"]),
        ("Parking", "parking", settings["parking"])
    ]

    for label, column, value in conditions:

        if column not in comps.columns:
            continue

        filtered = comps[comps[column] == value]

        if len(filtered) >= 3:
            comps = filtered
            applied.append(label)

    # Similar property area
    filtered = comps[
        (comps["area_sqft"] >= settings["area"] * 0.70) &
        (comps["area_sqft"] <= settings["area"] * 1.30)
    ]

    if len(filtered) >= 3:
        comps = filtered
        applied.append("Area")

    # Similar age
    if "age" in comps.columns:
        filtered = comps[
            (comps["age"] - settings["age"]).abs() <= 5
        ]

        if len(filtered) >= 3:
            comps = filtered
            applied.append("Age")

    # Similar floor
    if "floor_number" in comps.columns:
        filtered = comps[
            (comps["floor_number"] - settings["floor"]).abs() <= 3
        ]

        if len(filtered) >= 3:
            comps = filtered
            applied.append("Floor")

    # If no comparable records remain, use broader market data
    if comps.empty:
        comps = base.copy()

    return comps, applied


# ==========================================================
# 5. CHART STYLE
# ==========================================================

def style_chart(fig, height=300):

    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            color="#334155",
            size=11
        ),
        margin=dict(
            l=15,
            r=15,
            t=45,
            b=25
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            gridcolor="#EDF0F5",
            zeroline=False
        )
    )

    return fig


# ==========================================================
# 6. SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>REAL ESTATE<br>PRICING SIMULATOR</h1>",
        unsafe_allow_html=True
    )

    st.caption("Smarter Decisions. Better Investments.")

    st.divider()

    st.markdown("### ⚙️ Property Details")

    location = st.selectbox(
        "📍 Location",
        sorted(df["location"].dropna().unique())
    )

    property_type = st.selectbox(
        "🏠 Property Type",
        sorted(df["property_type"].dropna().unique())
    )

    min_area = max(100, int(df["area_sqft"].min()))
    max_area = max(min_area, int(df["area_sqft"].max()))

    area = st.number_input(
        "Area (sq. ft.)",
        min_value=min_area,
        max_value=max_area,
        value=min(1000, max_area),
        step=50
    )

    bedroom_values = sorted(
        df["bedrooms"].dropna().unique().astype(int)
    )

    bedrooms = st.selectbox(
        "🛏️ Bedrooms",
        bedroom_values
    )

    bathroom_values = sorted(
        df["bathrooms"].dropna().unique().astype(int)
    )

    bathrooms = st.selectbox(
        "🚿 Bathrooms",
        bathroom_values
    )

    floor_min = int(df["floor_number"].min())
    floor_max = int(df["floor_number"].max())

    floor = st.number_input(
        "Floor Number",
        min_value=floor_min,
        max_value=floor_max,
        value=min(3, floor_max)
    )

    age_min = int(df["age"].min())
    age_max = int(df["age"].max())

    age = st.number_input(
        "Property Age (Years)",
        min_value=age_min,
        max_value=age_max,
        value=min(5, age_max)
    )

    furnishing = st.selectbox(
        "🛋️ Furnishing",
        sorted(df["furnishing"].dropna().unique())
    )

    parking = st.selectbox(
        "🚗 Parking",
        sorted(df["parking"].dropna().unique())
    )

    st.divider()

    st.caption("Adjust the property market price")

    adjustment = st.slider(
        "Market Adjustment (%)",
        -30,
        30,
        0
    )

    st.markdown(
        """
        <div style="background:#EFF6FF;
                    padding:12px;
                    border-radius:9px;">
        <small style="color:#1D4ED8;">
        Change property details to instantly
        update the estimated price.
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# 7. PRICING CALCULATION
# ==========================================================

settings = {
    "location": location,
    "type": property_type,
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floor": floor,
    "age": age,
    "furnishing": furnishing,
    "parking": parking
}

comps, applied_filters = comparable_properties(
    df, settings
)

benchmark_ppsf = comps["price_per_sqft"].median()

base_price = area * benchmark_ppsf

final_price = base_price * (
    1 + adjustment / 100
)

final_ppsf = final_price / area

market_median = df["price"].median()

market_range_low = area * comps["price_per_sqft"].quantile(0.25)
market_range_high = area * comps["price_per_sqft"].quantile(0.75)

# Market comparison for selected location
location_data_selected = df[
    df["location"] == location
]

if not location_data_selected.empty:
    location_median = location_data_selected["price"].median()
else:
    location_median = market_median

difference = (
    (final_price / location_median) - 1
) * 100 if location_median else 0


# ==========================================================
# 8. HEADER
# ==========================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        "<p style='color:#2563EB;font-weight:700;"
        "letter-spacing:1px;margin-bottom:0;'>"
        "PROPERTY PRICE PREDICTION</p>",
        unsafe_allow_html=True
    )

    st.title("Pricing Simulator")

    st.write(
        "Enter property details and explore how different "
        "factors affect the estimated value of your property."
    )


st.divider()


# ==========================================================
# 9. KPI CARDS
# ==========================================================

st.subheader("Property Market Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "🏠 Estimated Property Value",
        lakh(final_price),
        f"{difference:+.1f}% vs location median"
    )

with k2:
    st.metric(
        "₹ Price per Sq.ft.",
        money(final_ppsf),
        f"{adjustment:+d}% adjustment"
    )

with k3:
    st.metric(
        "📊 Property Market Range",
        f"{lakh(market_range_low)} – {lakh(market_range_high)}",
        "Comparable properties"
    )

with k4:
    st.metric(
        "🔎 Comparable Records",
        f"{len(comps)}",
        "Properties matched"
    )

st.write("")


# ==========================================================
# 10. PREDICTION RESULT + PRICE COMPARISON
# ==========================================================

left, right = st.columns([1, 1])

with left:

    with st.container(border=True):

        st.subheader("✅ Prediction Result")

        st.caption(
            "Estimated property value based on your inputs."
        )

        st.markdown(
            f"""
            <h1 style="color:#059669;font-size:35px;">
            {money(final_price)}
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        a, b = st.columns(2)

        a.metric(
            "Price per Sq.ft.",
            money(final_ppsf)
        )

        b.metric(
            "Property Area",
            f"{area:,} sqft"
        )

        st.info(
            f"This estimate uses {len(comps)} comparable "
            f"properties in the selected market."
        )

        st.caption(
            "Matching characteristics: " +
            (", ".join(applied_filters)
             if applied_filters else "Broader market comparison")
        )


with right:

    with st.container(border=True):

        st.subheader("📊 Price Comparison")

        location_comparison = (
            df.groupby("location", as_index=False)
            .agg(
                Market_Price=("price", "median")
            )
            .sort_values(
                "Market_Price",
                ascending=False
            )
            .head(5)
        )

        location_comparison = location_comparison.sort_values(
            "Market_Price"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=location_comparison["location"],
                y=location_comparison["Market_Price"],
                name="Market Average",
                marker_color="#60A5FA"
            )
        )

        fig.add_trace(
            go.Bar(
                x=[location],
                y=[final_price],
                name="Current Property",
                marker_color="#10B981"
            )
        )

        fig.update_layout(
            barmode="group",
            title="Median Property Price by Location",
            yaxis_title="Property Price (₹)",
            xaxis_title="Location"
        )

        style_chart(fig, 300)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================================
# 11. WHAT-IF ANALYSIS
# ==========================================================

st.divider()

st.subheader("💡 What-if Analysis")

st.caption(
    "Explore how changing property features affects "
    "the estimated property value."
)

what_left, what_right = st.columns([1.2, 1])

with what_left:

    with st.container(border=True):

        st.markdown("### Change Property Area")

        st.write(
            "Adjust the property area to simulate "
            "an alternative valuation."
        )

        scenario_area = st.slider(
            "New Property Area (sq. ft.)",
            min_area,
            max_area,
            int(area),
            step=10,
            key="whatif_area"
        )

        scenario_adjustment = st.slider(
            "Scenario Market Adjustment (%)",
            -30,
            30,
            int(adjustment),
            key="whatif_adjustment"
        )

        scenario_base = scenario_area * benchmark_ppsf

        scenario_price = scenario_base * (
            1 + scenario_adjustment / 100
        )

        scenario_change = scenario_price - final_price

        change_percent = (
            scenario_change / final_price * 100
            if final_price else 0
        )

        st.markdown(
            f"""
            <div style="
            background:#EFF6FF;
            border:1px solid #BFDBFE;
            padding:18px;
            border-radius:12px;">
                <p style="color:#475569;margin:0;">
                Simulated Property Value
                </p>
                <h2 style="color:#2563EB;margin:5px 0;">
                {money(scenario_price)}
                </h2>
                <p style="color:#059669;margin:0;">
                {change_percent:+.2f}% compared with current estimate
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


with what_right:

    with st.container(border=True):

        st.markdown("### Scenario Comparison")

        scenario_df = pd.DataFrame({
            "Scenario": [
                "Original",
                "Modified"
            ],
            "Estimated Price": [
                final_price,
                scenario_price
            ]
        })

        fig = px.bar(
            scenario_df,
            x="Scenario",
            y="Estimated Price",
            color="Scenario",
            text="Estimated Price",
            color_discrete_map={
                "Original": "#94A3B8",
                "Modified": "#10B981"
            }
        )

        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside"
        )

        style_chart(fig, 300)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================================
# 12. FEATURE IMPACT ANALYSIS
# ==========================================================

st.divider()

st.subheader("📈 Feature Impact")

st.caption(
    "Shows how the median price per sq.ft. varies "
    "across different property characteristics."
)

feature_frames = []

# Location
for col, label in [
    ("location", "Location"),
    ("property_type", "Property Type"),
    ("furnishing", "Furnishing"),
    ("parking", "Parking"),
    ("bedrooms", "Bedrooms"),
    ("bathrooms", "Bathrooms")
]:

    if col not in df.columns:
        continue

    grouped = (
        df.groupby(col)["price_per_sqft"]
        .median()
        .dropna()
    )

    if len(grouped) < 2:
        continue

    selected = settings.get(col, None)

    if selected is None:
        continue

    selected_value = grouped.get(selected, np.nan)

    if pd.isna(selected_value):
        continue

    impact = (
        (selected_value / df["price_per_sqft"].median()) - 1
    ) * 100

    feature_frames.append({
        "Feature": label,
        "Impact": impact
    })

# Area comparison using area groups
area_data = df.copy()

area_data["Area Group"] = pd.cut(
    area_data["area_sqft"],
    bins=4,
    duplicates="drop"
)

area_groups = (
    area_data.groupby(
        "Area Group",
        observed=True
    )["price_per_sqft"]
    .median()
    .dropna()
)

if len(area_groups) > 1:

    selected_group = pd.cut(
        [area],
        bins=area_data["Area Group"].cat.categories
    )[0]

    selected_ppsf = area_groups.get(
        selected_group, np.nan
    )

    if pd.notna(selected_ppsf):

        feature_frames.append({
            "Feature": "Area",
            "Impact": (
                selected_ppsf /
                df["price_per_sqft"].median() - 1
            ) * 100
        })


feature_df = pd.DataFrame(feature_frames)

impact_left, impact_right = st.columns([1.3, 1])

with impact_left:

    with st.container(border=True):

        st.markdown("### Property Feature Analysis")

        if not feature_df.empty:

            feature_df = feature_df.sort_values(
                "Impact",
                ascending=True
            )

            fig = px.bar(
                feature_df,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Impact",
                color_continuous_scale=[
                    "#F59E0B",
                    "#93C5FD",
                    "#10B981"
                ],
                text="Impact"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="#94A3B8"
            )

            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Difference from Overall Median (%)",
                yaxis_title=""
            )

            style_chart(fig, 330)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("Not enough category variation for feature analysis.")


with impact_right:

    with st.container(border=True):

        st.markdown("### Property Information")

        info = {
            "Location": location,
            "Property Type": property_type,
            "Area": f"{area:,} sqft",
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floor": floor,
            "Age": f"{age} years",
            "Furnishing": furnishing,
            "Parking": parking
        }

        for key, value in info.items():

            st.markdown(
                f"""
                <div style="
                display:flex;
                justify-content:space-between;
                padding:9px 3px;
                border-bottom:1px solid #E2E8F0;">
                <span style="color:#64748B;">{key}</span>
                <b style="color:#1E293B;">{value}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================================
# 13. COMPARABLE PROPERTY TABLE
# ==========================================================

st.divider()

st.subheader("🔍 Comparable Properties")

st.caption(
    "Historical property records used for the pricing estimate."
)

table_columns = [
    "property_id",
    "location",
    "property_type",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "price",
    "price_per_sqft"
]

st.dataframe(
    comps[table_columns].sort_values(
        "price",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ==========================================================
# 14. DOWNLOAD REPORT
# ==========================================================

report = pd.DataFrame({
    "Parameter": [
        "Location",
        "Property Type",
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Floor",
        "Age",
        "Furnishing",
        "Parking",
        "Market Adjustment",
        "Estimated Property Price",
        "Price per Sqft",
        "Comparable Properties"
    ],
    "Value": [
        location,
        property_type,
        area,
        bedrooms,
        bathrooms,
        floor,
        age,
        furnishing,
        parking,
        adjustment,
        final_price,
        final_ppsf,
        len(comps)
    ]
})

st.download_button(
    "⬇️ Download Pricing Report",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name="real_estate_pricing_report.csv",
    mime="text/csv"
)


# ==========================================================
# 15. FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:10px;">
        <h4 style="color:#2563EB;">
        Real Estate Dynamic Pricing Simulator
        </h4>
        <p style="color:#94A3B8;font-size:12px;">
        Built with Python | Streamlit | Pandas | Plotly
        </p>
        <p style="color:#94A3B8;font-size:11px;">
        Estimated values are analytical estimates based on
        historical comparable property records.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

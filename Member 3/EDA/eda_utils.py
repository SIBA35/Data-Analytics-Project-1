"""Reusable helpers for the Pune property price EDA.

Keep this file in the same folder as the notebook and the Excel dataset.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="deep")

DEFAULT_PATH = "DOC-20261110-WA0011.xlsx"   # relative path: file sits next to the notebook
RENAME = {
    "Area (sqft)": "Area",
    "Age (Years)": "Age",
    "Distance to City Centre (km)": "Distance",
    "Price (INR)": "Price",
    "Floor Number": "Floor",
    "Property Type": "Type",
}
NUM_COLS = ["Area", "Bedrooms", "Bathrooms", "Floor", "Age", "Distance", "Price"]
CAT_COLS = ["Location", "Type", "Furnishing", "Parking"]


# ----------------------------------------------------------------- loading
def clean_columns(columns):
    """Strip BOM / mojibake BOM ('ï»¿') and stray spaces from column names."""
    return [c.replace("\ufeff", "").replace("ï»¿", "").strip() for c in columns]


def load_data(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the Excel file in the same folder as the notebook "
            f"(current folder: {Path.cwd()})."
        )
    df = pd.read_excel(path)
    df.columns = clean_columns(df.columns)
    return df.rename(columns=RENAME)


def add_features(df):
    df = df.copy()
    df["Price_Cr"] = df["Price"] / 1e7
    df["PPSF"] = df["Price"] / df["Area"]
    df["log_price"] = np.log(df["Price"])
    df["log_ppsf"] = np.log(df["PPSF"])
    lo, hi = df["PPSF"].quantile([0.01, 0.99])
    df["PPSF_capped"] = df["PPSF"].clip(lo, hi)          # 1st/99th percentile capping
    df["Bath_Bed_Ratio"] = df["Bathrooms"] / df["Bedrooms"]
    df["Age_Bucket"] = pd.cut(df["Age"], [-1, 5, 10, 20, 30], labels=["0-5", "6-10", "11-20", "21-30"])
    df["Area_Bucket"] = pd.cut(df["Area"], [0, 1000, 1500, 2000, 3000],
                               labels=["<1000", "1000-1500", "1500-2000", "2000+"])
    return df


# ----------------------------------------------------------------- checks
def quality_report(df):
    return {
        "rows": len(df), "columns": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["Property ID"].duplicated().sum()),
        "more_baths_than_beds": int((df["Bathrooms"] > df["Bedrooms"]).sum()),
        "houses": int((df["Type"] == "Independent House").sum()),
        "houses_above_ground_floor": int(((df["Type"] == "Independent House") & (df["Floor"] > 0)).sum()),
    }


def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def vif_table(df, cols):
    X = df[cols].astype(float)
    out = {}
    for c in cols:
        y = X[c].values
        Z = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        r2 = 1 - ((y - Z @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF").round(2)


def skew_table(df):
    """Skewness of price per sqft, raw vs log vs capped, overall and by location."""
    rows = {"ALL": {"PPSF": df.PPSF.skew(), "log_ppsf": df.log_ppsf.skew(), "PPSF_capped": df.PPSF_capped.skew()}}
    for loc, g in df.groupby("Location"):
        rows[loc] = {"PPSF": g.PPSF.skew(), "log_ppsf": g.log_ppsf.skew(), "PPSF_capped": g.PPSF_capped.skew()}
    return pd.DataFrame(rows).T.round(2)


def group_tests(df, target="Price"):
    rows = []
    for c in CAT_COLS:
        groups = [g[target].values for _, g in df.groupby(c)]
        rows.append((c, stats.f_oneway(*groups).pvalue, stats.kruskal(*groups).pvalue))
    return pd.DataFrame(rows, columns=["Feature", "ANOVA p", "Kruskal-Wallis p"]).set_index("Feature")


def type_effect_after_area(df):
    """Does Property Type still matter once Area is accounted for? (residual test)"""
    x, y = np.log(df["Area"].values), df["log_price"].values
    Z = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(Z, y, rcond=None)[0]
    resid = pd.Series(y - Z @ b, index=df.index)
    groups = [resid[df["Type"] == t].values for t in df["Type"].unique()]
    return {"raw_kruskal_p": stats.kruskal(*[g.log_price.values for _, g in df.groupby("Type")]).pvalue,
            "after_area_kruskal_p": stats.kruskal(*groups).pvalue,
            "after_area_anova_p": stats.f_oneway(*groups).pvalue}


def baseline_r2(df, n_splits=30, seed=0):
    """Mean test R^2 (log price) over random 80/20 splits for different feature sets."""
    sets = {
        "Area only": ["Area"],
        "Area + Location": ["Area", "Location"],
        "Area + Location + Type + Age": ["Area", "Location", "Type", "Age"],
        "All features": ["Area", "Bedrooms", "Bathrooms", "Floor", "Age", "Distance",
                         "Location", "Type", "Furnishing", "Parking"],
    }
    y = df["log_price"].values
    rng = np.random.default_rng(seed)
    res = {}
    for name, cols in sets.items():
        X = pd.get_dummies(df[cols], drop_first=True).astype(float).values
        Z = np.column_stack([np.ones(len(X)), X])
        scores = []
        for _ in range(n_splits):
            idx = rng.permutation(len(Z)); k = int(0.8 * len(Z)); tr, te = idx[:k], idx[k:]
            b = np.linalg.lstsq(Z[tr], y[tr], rcond=None)[0]
            p = Z[te] @ b
            scores.append(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
        res[name] = float(np.mean(scores))
    return pd.Series(res, name="Mean test R2 (log price)").round(3)


# ----------------------------------------------------------------- plots
def plot_price_distribution(df):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(df.Price_Cr, kde=True, ax=ax[0]); ax[0].set_title("Price (INR crore)")
    sns.histplot(df.log_price, kde=True, ax=ax[1]); ax[1].set_title("log(Price)")
    stats.probplot(df.Price, plot=ax[2]); ax[2].set_title("Q-Q plot (Price)")
    fig.tight_layout(); return fig


def plot_numeric_distributions(df):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for a, c in zip(axs.ravel(), ["Area", "Bedrooms", "Bathrooms", "Floor", "Age", "Distance"]):
        sns.histplot(df[c], kde=c in ["Area", "Distance", "Age"], ax=a,
                     bins=20 if c in ["Area", "Distance", "Age", "Floor"] else 5); a.set_title(c)
    fig.tight_layout(); return fig


def plot_categorical_counts(df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for a, c in zip(axs.ravel(), CAT_COLS):
        sns.countplot(data=df, y=c, order=df[c].value_counts().index, ax=a); a.set_title(f"{c} counts")
    fig.tight_layout(); return fig


def plot_price_by_category(df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for a, c in zip(axs.ravel(), CAT_COLS):
        order = df.groupby(c).Price_Cr.median().sort_values(ascending=False).index
        sns.boxplot(data=df, y=c, x="Price_Cr", order=order, ax=a)
        a.set_title(f"Price by {c}"); a.set_xlabel("Price (INR crore)")
    fig.tight_layout(); return fig


def plot_correlation(df):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df[NUM_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation matrix"); fig.tight_layout(); return fig


def plot_scatter(df):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for a, c in zip(axs.ravel(), ["Area", "Bedrooms", "Bathrooms", "Age", "Distance", "Floor"]):
        sns.scatterplot(data=df, x=c, y="Price_Cr", hue="Type", alpha=.6, ax=a, legend=a is axs[0, 0])
        a.set_title(f"{c} vs Price")
    fig.tight_layout(); return fig


def plot_ppsf_by_location(df):
    order = df.groupby("Location").PPSF.median().sort_values(ascending=False).index
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(data=df, x="Location", y="PPSF", order=order, ax=ax[0]); ax[0].tick_params(axis="x", rotation=45)
    ax[0].set_title("Price per sqft by location (raw)")
    sns.boxplot(data=df, x="Location", y="log_ppsf", order=order, ax=ax[1]); ax[1].tick_params(axis="x", rotation=45)
    ax[1].set_title("log(Price per sqft) by location")
    fig.tight_layout(); return fig


def plot_location_type_heatmap(df):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df.pivot_table(index="Location", columns="Type", values="Price_Cr", aggfunc="median"),
                annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title("Median price (INR crore): Location x Type"); fig.tight_layout(); return fig


def plot_age_bedrooms(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=df, x="Age_Bucket", y="PPSF", ax=ax[0]); ax[0].set_title("Price per sqft by building age")
    sns.boxplot(data=df, x="Bedrooms", y="Price_Cr", ax=ax[1]); ax[1].set_title("Price by bedrooms")
    fig.tight_layout(); return fig


def plot_outliers(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=df.Price_Cr, ax=ax[0]); ax[0].set_title("Price outliers")
    sns.boxplot(x=df.PPSF, ax=ax[1]); ax[1].set_title("Price per sqft outliers")
    fig.tight_layout(); return fig

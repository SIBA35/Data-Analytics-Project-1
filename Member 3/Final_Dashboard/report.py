"""Figures and text for the executive dashboard (all computed from the data)."""
import numpy as np
import pandas as pd


def kpis(df):
    return {"listings": len(df), "total_cr": df.Price.sum() / 1e7, "median_cr": df.Price_Cr.median(), "ppsf": df.PPSF.median(),
            "pmin": df.Price_Cr.min(), "pmax": df.Price_Cr.max(), "localities": df.Location.nunique()}


def locality_table(df):
    t = df.groupby("Location").agg(Listings=("Price", "size"), Median_price_cr=("Price_Cr", "median"), Median_ppsf=("PPSF", "median"),
                                   Median_area=("Area", "median")).sort_values("Median_ppsf", ascending=False)
    t["vs_market"] = (t.Median_ppsf / df.PPSF.median() - 1) * 100
    return t


def value_drivers(df):
    """Effect of each factor on price per sqft, holding location, type, age, distance, floor, furnishing and parking equal."""
    X = pd.get_dummies(df[["Location", "Type", "Furnishing", "Parking"]], drop_first=True).astype(float)
    X["Age"], X["Distance"], X["Floor"], X["Area"] = df.Age, df.Distance, df.Floor, df.Area / 1000
    Z = np.column_stack([np.ones(len(X)), X.values]); y = df.log_ppsf.values
    b = np.linalg.lstsq(Z, y, rcond=None)[0]; r = y - Z @ b
    se = np.sqrt(np.diag((r @ r / (len(y) - Z.shape[1])) * np.linalg.inv(Z.T @ Z)))
    idx = {c: i + 1 for i, c in enumerate(X.columns)}
    spec = [("Villa vs apartment", "Type_Villa", 1), ("Independent house vs apartment", "Type_Independent House", 1),
            ("Parking available", "Parking_Yes", 1), ("Semi-furnished vs furnished", "Furnishing_Semi-Furnished", 1),
            ("Unfurnished vs furnished", "Furnishing_Unfurnished", 1), ("10 more years of age", "Age", 10),
            ("10 km farther from centre", "Distance", 10), ("10 floors higher", "Floor", 10)]
    rows = []
    for label, col, k in spec:
        c, s = b[idx[col]] * k, se[idx[col]] * k
        rows.append({"factor": label, "pct": (np.exp(c) - 1) * 100, "low": (np.exp(c - 1.96 * s) - 1) * 100,
                     "high": (np.exp(c + 1.96 * s) - 1) * 100, "t": c / s})
    return pd.DataFrame(rows).set_index("factor")


def conclusions(df, vd):
    k = kpis(df); lt = locality_table(df); top, bot = lt.index[0], lt.index[-1]
    sz = df.groupby("Area_Bucket", observed=True).PPSF.median(); mo = df.groupby("Month").PPSF.median(); med = df.PPSF.median()
    g = lambda f: vd.loc[f, "pct"]
    return [
        ("Where a home is sets its price tier",
         f"Median price per sqft runs from INR {lt.Median_ppsf.iloc[-1]:,.0f} in {bot} to INR {lt.Median_ppsf.iloc[0]:,.0f} in {top}, "
         f"so {top} commands about {(lt.Median_ppsf.iloc[0] / lt.Median_ppsf.iloc[-1] - 1) * 100:.0f}% more per sqft. {lt.index[1]} follows closely at INR {lt.Median_ppsf.iloc[1]:,.0f}."),
        ("Size sets the base price",
         f"Price per sqft is almost the same across size bands (INR {sz.min():,.0f} to {sz.max():,.0f}), so total price rises in step with area. "
         f"Homes above 2,000 sqft have a median price of INR {df[df.Area > 2000].Price_Cr.median():.2f} crore against INR {df[df.Area < 1000].Price_Cr.median():.2f} crore below 1,000 sqft."),
        ("Villas and houses earn a premium",
         f"With location, age and amenities held equal, villas sell for about {g('Villa vs apartment'):.0f}% more per sqft than apartments and independent houses about {g('Independent house vs apartment'):.0f}% more."),
        ("Age and distance weigh on value; amenities lift it",
         f"Ten more years of age is linked to about {abs(g('10 more years of age')):.0f}% lower price per sqft, and ten more kilometres from the centre to about {abs(g('10 km farther from centre')):.0f}% lower. "
         f"Parking is linked to about {g('Parking available'):.0f}% higher, and unfurnished homes sell about {abs(g('Unfurnished vs furnished')):.0f}% below furnished ones. Floor level makes no measurable difference."),
        ("Prices show no seasonal pattern",
         f"Monthly median price per sqft stays between INR {mo.min():,.0f} and INR {mo.max():,.0f}, within {max(abs(mo.max() / med - 1), abs(mo.min() / med - 1)) * 100:.0f}% of the overall median of INR {med:,.0f}."),
        ("Scope of these findings",
         f"The results describe differences across {k['listings']} listings, each observed once. They show what is priced higher or lower, not what would happen if a price were changed."),
    ]

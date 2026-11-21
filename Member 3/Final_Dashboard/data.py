"""Data loading for the executive dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {"Area (sqft)": "Area", "Age (Years)": "Age", "Distance to City Centre (km)": "Distance", "Price (INR)": "Price",
          "Floor Number": "Floor", "Property Type": "Type", "Observation Date": "Date", "Observation Time": "Time"}


def load_data():
    folder = Path(__file__).resolve().parent
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Excel dataset found in {folder}.")
    df = pd.read_excel(files[0])
    df.columns = [c.replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price_Cr"] = df["Price"] / 1e7
    df["PPSF"] = df["Price"] / df["Area"]
    df["log_ppsf"] = np.log(df["PPSF"])
    df["Month"] = df["Date"].dt.month
    df["Age_Bucket"] = pd.cut(df["Age"], [-1, 5, 10, 20, 30], labels=["0-5", "6-10", "11-20", "21-30"])
    df["Area_Bucket"] = pd.cut(df["Area"], [0, 1000, 1500, 2000, 3000], labels=["<1000", "1000-1500", "1500-2000", "2000+"])
    return df

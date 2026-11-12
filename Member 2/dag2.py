# 1. IMPORT LIBRARIES
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dowhy import CausalModel
import re
import networkx as nx   # installed automatically with dowhy

warnings.filterwarnings("ignore")

# Set to False if you are running in Jupyter and want plots shown inline
SAVE_ONLY = True
if SAVE_ONLY:
    matplotlib.use("Agg")

# ---- paths (edit DATA_PATH if your CSV is elsewhere) ----
DATA_PATH = r"C:/Users/Arun nayak/Data-Analytics-Project-1/data/properties_clean.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "properties_clean.csv"      # fallback: same folder as this script
OUT_DIR = "week1_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def out(name):
    return os.path.join(OUT_DIR, name)

def finish_plot(name):
    plt.tight_layout()
    plt.savefig(out(name), dpi=120, bbox_inches="tight")
    if not SAVE_ONLY:
        plt.show()
    plt.close()

# ==========================================
# DATA INGESTION AND CLEANING
# ==========================================
df = pd.read_csv(DATA_PATH)

print("Original Dataset:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Clean column names
df.columns = (
    df.columns.str.strip().str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# Clean text columns (stray spaces / inconsistent case)
cat_columns = ["location", "property_type", "furnishing", "parking"]
for col in cat_columns:
    df[col] = df[col].astype(str).str.strip().str.title()

# Remove duplicates & missing values
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset="property_id", inplace=True)
df.dropna(inplace=True)

# Convert numerical columns (names matched to YOUR dataset)
numeric_columns = [
    "area_sqft", "bedrooms", "bathrooms", "floor_number",
    "age", "dist_city", "price",
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

# Basic sanity filter
df = df[(df["price"] > 0) & (df["area_sqft"] > 0)]

# Recalculate price per sqft (dataset already has it; this keeps it consistent)
df["price_per_sqft"] = df["price"] / df["area_sqft"]

print("\nCleaned Dataset:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# ---- Regional market analysis ----
regional_data = (
    df.groupby("location")
    .agg(
        average_price=("price", "mean"),
        average_price_per_sqft=("price_per_sqft", "mean"),
        total_properties=("property_id", "count"),
    )
    .sort_values("average_price_per_sqft", ascending=False)
    .reset_index()
)
print("\nRegional Market Analysis:")
print(regional_data.round(0))
regional_data.to_csv(out("regional_market_analysis.csv"), index=False)

# Chart 1: average price by region
plt.figure(figsize=(10, 5))
sns.barplot(data=regional_data, x="location", y="average_price", color="steelblue")
plt.title("Average Property Price by Region")
plt.ylabel("Average Price (INR)")
plt.xticks(rotation=45)
finish_plot("avg_price_by_region.png")

# Chart 2: price per sqft by region (fairer comparison across area sizes)
plt.figure(figsize=(10, 5))
sns.barplot(data=regional_data, x="location", y="average_price_per_sqft", color="darkorange")
plt.title("Average Price per Sqft by Region")
plt.ylabel("INR per sqft")
plt.xticks(rotation=45)
finish_plot("avg_ppsf_by_region.png")

df.to_csv(out("cleaned_dataset.csv"), index=False)

# ==========================================
# CAUSAL DAG USING DOWHY
# ==========================================
df["parking_treatment"] = df["parking"].map({"Yes": 1, "No": 0})
# Furnishing as an ordered numeric level (DoWhy needs numeric effect modifiers)
df["furnishing_level"] = df["furnishing"].map(
    {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
)

causal_cols = [
    "parking_treatment", "price", "area_sqft", "bedrooms", "bathrooms",
    "floor_number", "age", "dist_city", "location", "property_type", "furnishing_level",
]
causal_df = df.dropna(subset=causal_cols)[causal_cols].copy()

# NOTE: price_per_sqft and property_id are deliberately NOT in the DAG.
# price_per_sqft is calculated FROM price (the outcome), so adjusting for it
# would bias the estimate.

dag = """
digraph {
    location -> property_type;
    location -> dist_city;
    location -> price;
    location -> parking_treatment;

    property_type -> area_sqft;
    property_type -> floor_number;
    property_type -> parking_treatment;
    property_type -> price;

    area_sqft -> bedrooms;
    area_sqft -> bathrooms;
    area_sqft -> parking_treatment;
    area_sqft -> price;

    bedrooms -> price;
    bathrooms -> price;

    floor_number -> price;
    age -> furnishing_level;
    age -> parking_treatment;
    age -> price;
    dist_city -> parking_treatment;
    dist_city -> price;
    furnishing_level -> price;

    parking_treatment -> price;
}
"""

# Visualize + save the DAG as PNG (networkx + matplotlib, no Graphviz needed)
edges = re.findall(r"(\w+)\s*->\s*(\w+)", dag)
G = nx.DiGraph(edges)
for layer, nodes in enumerate(nx.topological_generations(G)):
    for n in nodes:
        G.nodes[n]["layer"] = layer
pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
node_colors = ["tomato" if n == "parking_treatment" else
               "gold" if n == "price" else "lightblue" for n in G.nodes]

plt.figure(figsize=(14, 8))
nx.draw_networkx(G, pos, node_color=node_colors, node_size=2600, font_size=8,
                 arrows=True, arrowsize=14, edge_color="gray",
                 connectionstyle="arc3,rad=0.08")
plt.title("Causal DAG (red = treatment, yellow = outcome)")
plt.axis("off")
plt.savefig(out("causal_dag.png"), dpi=130, bbox_inches="tight")
if not SAVE_ONLY:
    plt.show()
plt.close()
print("\nDAG image saved:", out("causal_dag.png"))

# Create DoWhy model
model = CausalModel(
    data=causal_df,
    treatment="parking_treatment",
    outcome="price",
    graph=dag,
)

# Identify causal effect (backdoor criterion)
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\nIdentified Causal Effect:")
print(identified_estimand)

# Estimate effect: linear regression adjusting for the backdoor set
# (categorical columns are one-hot encoded automatically by DoWhy)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
)
print("\nEstimated causal effect of parking on price (INR):", round(estimate.value))

# Naive (unadjusted) difference for comparison
naive = (causal_df.groupby("parking_treatment")["price"].mean())
print("Naive difference (Yes - No):", round(naive[1] - naive[0]))

# Refutation tests: a trustworthy estimate should survive these
for refuter in ["random_common_cause", "placebo_treatment_refuter"]:
    try:
        res = model.refute_estimate(
            identified_estimand, estimate,
            method_name=refuter,
            **({"placebo_type": "permute"} if refuter == "placebo_treatment_refuter" else {}),
        )
        print(f"\nRefutation - {refuter}:")
        print(res)
    except Exception as e:
        print(f"\nRefutation {refuter} skipped:", e)

# ==========================================
# PRE-TREATMENT AND TREATMENT PERIOD
# ==========================================
# NOTE: this dataset has NO transaction dates, so this is only a
# DEMONSTRATION split by row order, not a real before/after separation.
split_point = int(len(df) * 0.70)
pre_treatment_data = df.iloc[:split_point].copy()
treatment_data = df.iloc[split_point:].copy()

print("\nPre-Treatment Records:", len(pre_treatment_data))
print("Treatment Records:", len(treatment_data))

baseline_price = pre_treatment_data["price"].mean()
print("Baseline Average Price (INR):", round(baseline_price))

plt.figure(figsize=(10, 5))
plt.plot(pre_treatment_data["price"].values, label="Pre-Treatment (Demo)", alpha=0.8)
plt.plot(range(split_point, len(df)), treatment_data["price"].values,
         label="Treatment (Demo)", alpha=0.8)
plt.axvline(split_point, color="red", linestyle="--", label="Demonstration Split")
plt.axhline(baseline_price, color="grey", linestyle=":", label="Baseline mean")
plt.title("Demonstration of Treatment Period Separation")
plt.xlabel("Record Index")
plt.ylabel("Property Price (INR)")
plt.legend()
finish_plot("treatment_period_demo.png")

pre_treatment_data.to_csv(out("pre_treatment_data.csv"), index=False)
treatment_data.to_csv(out("treatment_data.csv"), index=False)


print("WEEK 1 TASKS EXECUTED")
print(f"All files saved in ./{OUT_DIR}/ :")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dowhy import CausalModel
from graphviz import Source
from IPython.display import display

# ==========================================
# TASK 1: DATA INGESTION AND CLEANING
# ==========================================

# Load Excel dataset
df = pd.read_excel('C:/Users/Arun nayak/Data-Analytics-Project-1/Dataset/DOC-20261110-WA0011.xlsx')

print("Original Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Clean column names 
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Convert numerical columns
numeric_columns = [
    'area_sqft',
    'bedrooms',
    'bathrooms',
    'floor_number',
    'age_years',
    'distance_to_city_centre_km',
    'price_inr'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Create price per square foot
df['price_per_sqft'] = (
    df['price_inr'] / df['area_sqft']
)

print("\nCleaned Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# Regional market analysis
regional_data = df.groupby('location').agg(
    average_price=('price_inr', 'mean'),
    total_properties=('property_id', 'count')
).reset_index()

print("\nRegional Market Analysis:")
print(regional_data)

# Visualize regional prices
plt.figure(figsize=(10, 5))
sns.barplot(
    data=regional_data,
    x='location',
    y='average_price'
)
plt.title("Average Property Price by Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)


# ==========================================
# TASK 2: CAUSAL DAG USING DOWHY
# ==========================================

# Create treatment variable
df['parking_treatment'] = df['parking'].map({
    'Yes': 1,
    'No': 0
})

# Prepare causal dataset
causal_df = df.dropna(subset=[
    'parking_treatment',
    'price_inr',
    'area_sqft',
    'bedrooms',
    'bathrooms',
    'age_years',
    'location',
    'property_type'
]).copy()

# Define DAG
dag = """
digraph {
    area_sqft -> parking_treatment;
    area_sqft -> price_inr;

    bedrooms -> parking_treatment;
    bedrooms -> price_inr;

    bathrooms -> parking_treatment;
    bathrooms -> price_inr;

    age_years -> parking_treatment;
    age_years -> price_inr;

    location -> parking_treatment;
    location -> price_inr;

    property_type -> parking_treatment;
    property_type -> price_inr;

    parking_treatment -> price_inr;
}
"""

# Visualize DAG
graph = Source(dag)
display(graph)

graph.render(
    'causal_dag',
    format='png',
    cleanup=True
)

# Create DoWhy model
model = CausalModel(
    data=causal_df,
    treatment='parking_treatment',
    outcome='price_inr',
    graph=dag
)

# Identify causal effect
identified_estimand = model.identify_effect(
    proceed_when_unidentifiable=True
)

print("\nIdentified Causal Effect:")
print(identified_estimand)


# ==========================================
# TASK 3: PRE-TREATMENT AND TREATMENT PERIOD
# ==========================================

# NOTE:
# Your current dataset has no transaction dates.
# This is only a demonstration split, NOT a real
# historical treatment-period separation.

# Divide records for demonstration only
split_point = int(len(df) * 0.70)

pre_treatment_data = df.iloc[:split_point].copy()

treatment_data = df.iloc[split_point:].copy()

print("\nPre-Treatment Dataset:")
print(pre_treatment_data.head())

print("\nTreatment Dataset:")
print(treatment_data.head())

print("\nPre-Treatment Records:",
      len(pre_treatment_data))

print("Treatment Records:",
      len(treatment_data))

# Calculate baseline average price
baseline_price = pre_treatment_data['price_inr'].mean()

print("\nBaseline Average Price:", baseline_price)

# Visualize demonstration split
plt.figure(figsize=(10, 5))

plt.plot(
    pre_treatment_data['price_inr'].values,
    label='Pre-Treatment (Demo)'
)

plt.plot(
    range(split_point, len(df)),
    treatment_data['price_inr'].values,
    label='Treatment (Demo)'
)

plt.axvline(
    split_point,
    color='red',
    linestyle='--',
    label='Demonstration Split'
)

plt.title("Demonstration of Treatment Period Separation")
plt.xlabel("Record Index")
plt.ylabel("Property Price (INR)")
plt.legend()
plt.tight_layout()
plt.show()

# Save separated datasets
pre_treatment_data.to_csv(
    'pre_treatment_data.csv',
    index=False
)

treatment_data.to_csv(
    'treatment_data.csv',
    index=False
)


# ==========================================
# WEEK 1 COMPLETION
# ==========================================

print("\n================================")
print("WEEK 1 TASKS EXECUTED")
print("================================")

print("1. Data Ingestion Completed")
print("2. Causal DAG Created")
print("3. Demonstration Period Separation Completed")

print("\nGenerated Files:")
print("- cleaned_dataset.csv")
print("- causal_dag.png")
print("- pre_treatment_data.csv")
print("- treatment_data.csv")
print("- regional market analysis is available above")

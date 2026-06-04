# ============================================================
# PROJECT 2: Data Classification Using AI
# STEP 1: Load and Understand the Dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# --- Load the Iris Dataset ---
iris = load_iris()

# Convert to a DataFrame for easier exploration
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 55)
print("        IRIS DATASET - EXPLORATION REPORT")
print("=" * 55)

print(f"\n📦 Total Samples  : {df.shape[0]}")
print(f"📐 Features       : {df.shape[1] - 2}")
print(f"🌸 Classes        : {df['species_name'].nunique()} → {list(df['species_name'].unique())}")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Class Distribution ---")
print(df['species_name'].value_counts())

print("\n--- Statistical Summary ---")
print(df[iris.feature_names].describe().round(2))

print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n✅ Step 1 Complete: Dataset loaded and understood.")

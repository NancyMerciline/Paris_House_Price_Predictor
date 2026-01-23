import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# =========================
# PROJECT PATH CONFIGURATION
# =========================
# Define the project root directory (2 levels above this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print("PROJECT_ROOT =", PROJECT_ROOT)

# Define path to the raw DVF dataset (government data)
DATA_PATH1 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2020-S2.txt"
DATA_PATH2 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2021.txt"
DATA_PATH3 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2022.txt"
DATA_PATH4 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2023.txt"
DATA_PATH5 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2024.txt"
DATA_PATH6 = PROJECT_ROOT / "data" / "raw" / "ValeursFoncieres-2025-S1.txt"


# =========================
# LOAD RAW DATA
# =========================
# Load the DVF dataset using '|' as separator
df1 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)
df2 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)
df3 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)
df4 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)
df5 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)
df6 = pd.read_csv(DATA_PATH1, sep="|", low_memory=False)

#Add Column year
df1["year"] = 2020
df2["year"] = 2021
df3["year"] = 2022
df4["year"] = 2023
df5["year"] = 2024
df6["year"] = 2025

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index= True)

# Display first rows and dataset shape
print(df.head)
print(df.shape)


# =========================
# REMOVE EMPTY COLUMNS
# =========================
# Count columns before cleaning
nb_avant = df.shape[1]

# Drop columns that are 100% empty
df.dropna(axis=1, how="all", inplace=True)

# Count columns after cleaning
nb_apres = df.shape[1]

print(f"Colonnes supprimées : {nb_avant - nb_apres}")
print(df.head)
print(df.shape)


# =========================
# LIST ALL REMAINING COLUMNS
# =========================
# Enumerate all remaining columns with index
for i, col in enumerate(df.columns):
    print(f"{i} - {col}")


# =========================
# FILTER DATA FOR PARIS ONLY
# =========================
# Check number of transactions for Paris (department code = 75)
print(df[df["Code departement"] == "75"].shape)

# Keep only Paris transactions
df = df[df["Code departement"] == "75"]


# =========================
# DATA TYPE CLEANING
# =========================
# Convert 'Valeur fonciere' from string to float
df["Valeur fonciere"] = (
    df["Valeur fonciere"]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# Convert 'Surface reelle bati' to numeric (invalid values become NaN)
df["Surface reelle bati"] = pd.to_numeric(
    df["Surface reelle bati"], errors="coerce"
)

# Remove invalid transactions (negative or zero values)
df = df[
    (df["Valeur fonciere"] > 0) &
    (df["Surface reelle bati"] > 0)
]


# =========================
# CREATE TARGET VARIABLE: PRICE PER m²
# =========================
# Compute price per square meter
df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

# Remove extreme outliers
df = df[
    (df["prix_m2"] > 500) &
    (df["prix_m2"] < 20000)
]

# Basic statistics on price per m²
df["prix_m2"].describe()


# =========================
# PRICE DISTRIBUTION (RAW)
# =========================
plt.figure(figsize=(8, 4))
sns.histplot(df["prix_m2"], bins=50, kde=True)
plt.title("Distribution du prix au m² (Paris)")
plt.xlabel("Prix au m² (€)")
plt.ylabel("Nombre de transactions")
plt.show()


# =========================
# LOG TRANSFORMATION OF TARGET
# =========================
# Apply logarithmic transformation to stabilize variance
df["log_prix_m2"] = np.log(df["prix_m2"])

# Plot distribution after log transformation
plt.figure(figsize=(8, 4))
sns.histplot(df["log_prix_m2"], bins=50, kde=True)
plt.title("Distribution du log(prix au m²) – Paris")
plt.xlabel("log(Prix au m²)")
plt.ylabel("Nombre de transactions")
plt.show()


# =========================
# MISSING VALUES ANALYSIS
# =========================
# Display data types
df.dtypes

# Compute missing value rate per column
missing_rate = df.isna().mean().sort_values(ascending=False)
print(missing_rate)


# =========================
# DROP USELESS COLUMNS (VERY SPARSE)
# =========================
# Columns with more than 95% missing values
cols_to_drop = [
    "No Volume",
    "Prefixe de section",
    "Nature culture speciale",
    "Surface Carrez du 5eme lot",
    "Surface Carrez du 4eme lot",
    "5eme lot",
    "Surface Carrez du 3eme lot",
    "Surface terrain",
    "Nature culture",
    "4eme lot",
    "B/T/Q",
    "3eme lot",
    "Surface Carrez du 2eme lot",
    "2eme lot",
    "Surface Carrez du 1er lot"
]

# Drop these columns if they exist
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

print(df.head)
print(df.shape)


# =========================
# SAVE CLEAN DATASET
# =========================
# Create processed data directory if it doesn't exist
out_dir = PROJECT_ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

# Save cleaned Paris-only dataset
out_file = out_dir / "data_clean_paris.csv"
df.to_csv(out_file, index=False)

print(out_file)
print(out_file.exists())
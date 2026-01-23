import warnings
warnings.filterwarnings("ignore")  # Ignore non-critical warnings for cleaner logs

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# =========================
# CONFIGURATION
# =========================

# Define project root dynamically (robust to execution location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to cleaned dataset (Paris only)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_clean_paris.csv"

# Target variable (log-transformed price per m²)
TARGET = "log_prix_m2"

# Output directories for processed data and reports
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================

# Load cleaned dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)

# Ensure target exists
assert TARGET in df.columns, f"Target '{TARGET}' not found."

# ---------------------------------------------------------
# IMPORTANT: Remove leakage columns BEFORE any feature selection / modeling
# prix_m2 is directly related to the target log_prix_m2
# Valeur fonciere is also used to compute prix_m2
# Keeping them would cause data leakage and unrealistic performance.
# ---------------------------------------------------------
LEAK_COLS = ["prix_m2", "Valeur fonciere"]
df.drop(columns=[c for c in LEAK_COLS if c in df.columns], inplace=True, errors="ignore")

# =========================
# 1) RULE-BASED FEATURE REMOVAL (BEFORE MODELING)
# =========================

# Drop purely technical / identifier / address-related columns
# These variables do not add predictive value for ML
DROP_ALWAYS = [
    "No disposition", "No voie", "No plan", "Code voie",
    "Section", "Code commune", "Commune", "Voie",
    "Identifiant de document", "Reference document", "Identifiant local"
]
df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], inplace=True, errors="ignore")

# Drop constant columns (e.g. Code departement = 75 for Paris only)
constant_cols = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
if constant_cols:
    print("Constant columns removed:", constant_cols)
    df.drop(columns=constant_cols, inplace=True)

# Drop columns with too many missing values (≥ 95%)
missing_rate = df.isna().mean()
to_drop_missing = missing_rate[missing_rate >= 0.95].index.tolist()
if to_drop_missing:
    print("Columns removed (>=95% missing):", to_drop_missing)
    df.drop(columns=to_drop_missing, inplace=True)

# Save missing value diagnostics
missing_rate.sort_values(ascending=False).to_csv(
    REPORT_DIR / "missing_rate.csv",
    header=["missing_rate"]
)

# =========================
# 2) CORRELATION ANALYSIS (NUMERIC FEATURES)
# =========================

# Select numeric features excluding target
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET]

if len(num_cols) > 0:
    # Compute correlation between numeric features and target
    corr_target = (
        df[num_cols + [TARGET]]
        .corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    # Save correlation values
    corr_target.to_csv(
        REPORT_DIR / "corr_with_target.csv",
        header=["corr_with_target"]
    )

    print("\nTop absolute correlations with target:")
    print(corr_target.head(15))

    # Plot top correlations for interpretability
    top = corr_target.head(20)[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top.index, top.values)
    plt.title("Top numeric correlations with target")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_corr_with_target.png", dpi=150)
    plt.close()

# =========================
# 3) MODEL-BASED FEATURE IMPORTANCE (LIGHT VERSION)
# =========================

# Remove rows with missing target
df = df[df[TARGET].notna()].copy()

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

# Preprocessing pipeline
# - Numeric: median imputation
# - Categorical: most frequent + OneHotEncoding (sparse for efficiency)
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ],
    remainder="drop"
)

# Random Forest model (light configuration for performance)
model = RandomForestRegressor(
    n_estimators=100,   # Reduced for faster execution
    max_depth=20,       # Limit depth to prevent overfitting
    random_state=42,
    n_jobs=-1
)

# Full pipeline
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipe.fit(X_train, y_train)

# Evaluate model
pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"\nMAE (on log_prix_m2) = {mae:.4f}")

# Extract feature importances from Random Forest
rf = pipe.named_steps["model"]
importances = rf.feature_importances_

# Retrieve feature names after preprocessing
feature_names = []
feature_names.extend(numeric_features)
if categorical_features:
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(categorical_features).tolist())

# Create importance dataframe
imp_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
)

# Save feature importances
imp_df.to_csv(REPORT_DIR / "rf_feature_importance.csv", index=False)

print("\nTop 20 feature importances (Random Forest):")
print(imp_df.head(20))

# =========================
# 4) FINAL FEATURE SELECTION (ORIGINAL FEATURES)
# =========================

# Keep top 40% most important features
threshold = imp_df["importance"].quantile(0.60)
top_features = imp_df[imp_df["importance"] >= threshold]["feature"].tolist()

# Map back to original feature names
selected_original = set()
for f in top_features:
    if f in numeric_features:
        selected_original.add(f)
    else:
        base = f.split("_")[0]
        if base in categorical_features:
            selected_original.add(base)

selected_original = sorted(selected_original)

# Save selected features
pd.Series(
    selected_original,
    name="selected_features"
).to_csv(REPORT_DIR / "selected_features.csv", index=False)

# Create final dataset ready for modeling
df_final = df[selected_original + [TARGET]].copy()
df_final.to_csv(OUT_DIR / "data_model_ready_paris.csv", index=False)

print("\n✅ Saved:")
print("-", REPORT_DIR / "corr_with_target.csv")
print("-", REPORT_DIR / "rf_feature_importance.csv")
print("-", REPORT_DIR / "selected_features.csv")
print("-", OUT_DIR / "data_model_ready_paris.csv")
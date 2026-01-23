# 03_train_model_targetenc.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

# =========================
# CONFIGURATION
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_model_ready_paris.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "log_prix_m2"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found. Columns: {list(df.columns)}")

df = df[df[TARGET].notna()].copy()

# IMPORTANT: remove leakage columns if they still exist
LEAK_COLS = ["prix_m2", "Valeur fonciere"]
df.drop(columns=[c for c in LEAK_COLS if c in df.columns], inplace=True, errors="ignore")

# =========================
# BUILD ARRONDISSEMENT (01..20)
# =========================
if "Code postal" not in df.columns:
    raise ValueError("Column 'Code postal' is missing from dataset.")

# Convert Code postal to string and extract last 2 digits -> arrondissement (01..20)
df["Code postal"] = df["Code postal"].astype(str).str.strip()
df["Code postal"] = df["Code postal"].str.replace(".0", "", regex=False)  # if came as float text
df["Code postal"] = df["Code postal"].str.zfill(5)

df["arrondissement"] = df["Code postal"].str[-2:]  # "01".."20"

# Keep only valid Paris arrondissements
valid_arr = {f"{i:02d}" for i in range(1, 21)}
df = df[df["arrondissement"].isin(valid_arr)].copy()

# =========================
# FEATURES
# =========================
NUM_COLS = ["Surface reelle bati", "Nombre pieces principales"]
CAT_COLS = ["arrondissement", "Type local"]

missing = (set(NUM_COLS + CAT_COLS + [TARGET]) - set(df.columns))
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# Clean types
df["Surface reelle bati"] = pd.to_numeric(df["Surface reelle bati"], errors="coerce")
df["Nombre pieces principales"] = pd.to_numeric(df["Nombre pieces principales"], errors="coerce")
df = df.dropna(subset=NUM_COLS + CAT_COLS + [TARGET]).copy()

# =========================
# TARGET ENCODING (METHOD 1)
# =========================
def target_encode(train_df, col, target, smoothing=20):
    """
    Returns:
      - mapping dict: category -> smoothed mean(target)
      - global_mean: overall mean(target)
    """
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    mapping = smooth.to_dict()
    return mapping, global_mean

# =========================
# SPLIT
# =========================
X = df[NUM_COLS + CAT_COLS].copy()
y = df[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = X_train.copy()
train_df[TARGET] = y_train.values

# Create target-encoding maps (trained ONLY on train split)
arr_map, global_mean = target_encode(train_df, "arrondissement", TARGET, smoothing=20)
type_map, _ = target_encode(train_df, "Type local", TARGET, smoothing=20)

def apply_te(df_part):
    out = df_part.copy()
    out["arrondissement_te"] = out["arrondissement"].map(arr_map).fillna(global_mean)
    out["type_local_te"] = out["Type local"].map(type_map).fillna(global_mean)
    return out

X_train_te = apply_te(X_train)
X_test_te = apply_te(X_test)

FINAL_FEATURES = ["Surface reelle bati", "Nombre pieces principales", "arrondissement_te", "type_local_te"]

# =========================
# MODEL
# =========================
model = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.05,
    max_iter=400,
    random_state=42
)

model.fit(X_train_te[FINAL_FEATURES], y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_te[FINAL_FEATURES])

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE (log_prix_m2)")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# =========================
# SAVE BUNDLE
# =========================
bundle = {
    "model": model,
    "FINAL_FEATURES": FINAL_FEATURES,
    "arr_map": arr_map,
    "type_map": type_map,
    "global_mean": float(global_mean)
}

model_path = MODEL_DIR / "model_prix_m2_paris_targetenc.pkl"
joblib.dump(bundle, model_path)

print("\n✅ Model bundle saved:", model_path)

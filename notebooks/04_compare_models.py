# 04_compare_models.py
# Compare several ML approaches on Paris DVF (log_prix_m2 target)
# including your approach: Target Encoding + HistGradientBoosting

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

from scipy import sparse


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_clean_paris.csv"
TARGET = "log_prix_m2"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# =========================
# UTILITIES
# =========================
def safe_exp(x):
    # avoid overflow just in case
    return np.exp(np.clip(x, -50, 50))


def mae_price_m2_scorer(y_true_log, y_pred_log):
    """
    MAE computed in original €/m² space by exponentiating log_prix_m2.
    """
    y_true = safe_exp(np.asarray(y_true_log))
    y_pred = safe_exp(np.asarray(y_pred_log))
    return mean_absolute_error(y_true, y_pred)


MAE_LOG = make_scorer(mean_absolute_error, greater_is_better=False)
RMSE_LOG = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
R2 = make_scorer(r2_score)
MAE_EURO = make_scorer(mae_price_m2_scorer, greater_is_better=False)


class ToDense(BaseEstimator, TransformerMixin):
    """
    Convert sparse matrix to dense (needed for HistGradientBoostingRegressor).
    Apply ONLY where required to avoid memory blow-up.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding with smoothing.
    Safe inside CV because fit() only sees y of train fold.
    """
    def __init__(self, cols, smoothing=20):
        self.cols = cols
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.global_mean_ = float(y.mean())
        self.maps_ = {}

        for col in self.cols:
            stats = pd.DataFrame({"x": X[col], "y": y}).groupby("x")["y"].agg(["mean", "count"])
            smooth = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (stats["count"] + self.smoothing)
            self.maps_[col] = smooth.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.copy()

        for col in self.cols:
            out[col + "_te"] = out[col].map(self.maps_[col]).fillna(self.global_mean_)
            out.drop(columns=[col], inplace=True)

        return out


# =========================
# 1) LOAD + CLEAN MINIMUM
# =========================
df = pd.read_csv(DATA_PATH, low_memory=False)
if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found. Columns: {list(df.columns)}")

# Remove leakage columns if present
LEAK_COLS = ["prix_m2", "Valeur fonciere"]
df.drop(columns=[c for c in LEAK_COLS if c in df.columns], inplace=True, errors="ignore")

# Build arrondissement from Code postal (01..20)
if "Code postal" not in df.columns:
    raise ValueError("Missing column: 'Code postal'")

df["Code postal"] = (
    df["Code postal"].astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(5)
)
df["arrondissement"] = df["Code postal"].str[-2:]
valid_arr = {f"{i:02d}" for i in range(1, 21)}
df = df[df["arrondissement"].isin(valid_arr)].copy()

# Use same main features as your final model (fair comparison)
USE_NUM = ["Surface reelle bati", "Nombre pieces principales"]
USE_CAT = ["arrondissement", "Type local"]

needed = set(USE_NUM + USE_CAT + [TARGET])
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean types + drop missing
df["Surface reelle bati"] = pd.to_numeric(df["Surface reelle bati"], errors="coerce")
df["Nombre pieces principales"] = pd.to_numeric(df["Nombre pieces principales"], errors="coerce")
df = df.dropna(subset=USE_NUM + USE_CAT + [TARGET]).copy()

X = df[USE_NUM + USE_CAT].copy()
y = df[TARGET].astype(float).copy()

print("Loaded:", DATA_PATH)
print("Final shape used:", df.shape)

# Final split for realistic test evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)


# =========================
# 2) PREPROCESSORS
# =========================
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # keep sparse for memory
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_ohe = ColumnTransformer(
    transformers=[
        ("num", num_pipe, USE_NUM),
        ("cat", cat_pipe, USE_CAT),
    ],
    remainder="drop"
)

preprocess_te = Pipeline(steps=[
    ("te", SmoothedTargetEncoder(cols=USE_CAT, smoothing=20)),
    ("imputer", SimpleImputer(strategy="median")),
])


# =========================
# 3) MODELS TO COMPARE
# =========================
models = {
    "Baseline_Mean": Pipeline([
        ("preprocess", preprocess_ohe),
        ("model", DummyRegressor(strategy="mean"))
    ]),

    "Ridge_OHE": Pipeline([
        ("preprocess", preprocess_ohe),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ]),

    "RandomForest_OHE": Pipeline([
        ("preprocess", preprocess_ohe),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    "ExtraTrees_OHE": Pipeline([
        ("preprocess", preprocess_ohe),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),

    # YOURS: Target Encoding + HGB (dense numeric output, OK)
    "HGB_TargetEnc (YOURS)": Pipeline([
        ("preprocess", preprocess_te),
        ("model", HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=400,
            random_state=RANDOM_STATE
        ))
    ]),

    # HGB with OHE needs dense conversion because HGB doesn't accept sparse
    "HGB_OHE": Pipeline([
        ("preprocess", preprocess_ohe),
        ("todense", ToDense()),
        ("model", HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=400,
            random_state=RANDOM_STATE
        ))
    ]),
}


# =========================
# 4) CROSS-VALIDATION COMPARISON
# =========================
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "MAE_log": MAE_LOG,         # negative in sklearn convention
    "RMSE_log": RMSE_LOG,       # negative
    "R2": R2,
    "MAE_euro": MAE_EURO        # negative
}

results = []
for name, pipe in models.items():
    t0 = time.time()
    cv_out = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise"  # keep strict; change to np.nan if you want it to continue
    )
    dt = time.time() - t0

    row = {
        "model": name,
        "cv_MAE_log": -np.mean(cv_out["test_MAE_log"]),
        "cv_RMSE_log": -np.mean(cv_out["test_RMSE_log"]),
        "cv_R2": np.mean(cv_out["test_R2"]),
        "cv_MAE_€/m²": -np.mean(cv_out["test_MAE_euro"]),
        "fit_time_sec": dt
    }
    results.append(row)

res_df = pd.DataFrame(results).sort_values(by="cv_MAE_€/m²", ascending=True)
print("\n=== CV COMPARISON (train split only) ===")
print(res_df.to_string(index=False))

res_path = REPORT_DIR / "model_comparison_cv.csv"
res_df.to_csv(res_path, index=False)
print("\nSaved:", res_path)


# =========================
# 5) FINAL TEST EVALUATION (hold-out)
# =========================
final_rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae_log = mean_absolute_error(y_test, pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    # Convert back to €/m²
    y_test_e = safe_exp(y_test.values)
    pred_e = safe_exp(pred)
    mae_euro = mean_absolute_error(y_test_e, pred_e)
    rmse_euro = np.sqrt(mean_squared_error(y_test_e, pred_e))

    final_rows.append({
        "model": name,
        "test_MAE_log": mae_log,
        "test_RMSE_log": rmse_log,
        "test_R2": r2,
        "test_MAE_€/m²": mae_euro,
        "test_RMSE_€/m²": rmse_euro
    })

final_df = pd.DataFrame(final_rows).sort_values(by="test_MAE_€/m²", ascending=True)
print("\n=== FINAL TEST COMPARISON (hold-out) ===")
print(final_df.to_string(index=False))

final_path = REPORT_DIR / "model_comparison_test.csv"
final_df.to_csv(final_path, index=False)
print("\nSaved:", final_path)

best = final_df.iloc[0]
print("\n🏆 BEST (by test MAE €/m²):", best["model"])

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import joblib

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="Paris Real Estate Estimator", layout="centered")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_prix_m2_paris_targetenc.pkl"

# =========================
# LOAD MODEL BUNDLE
# =========================
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

st.title("🏠 Paris Real Estate Price Estimator (€/m²)")
st.caption("Target-encoding model trained on Paris DVF data — predicts price per m².")

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
FINAL_FEATURES = bundle["FINAL_FEATURES"]
arr_map = bundle["arr_map"]
type_map = bundle["type_map"]
global_mean = float(bundle["global_mean"])

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Property inputs")

arr_choices = [f"{i:02d}" for i in range(1, 21)]
arrondissement = st.sidebar.selectbox("Arrondissement", arr_choices, index=0)

# If you want a fixed list of property types, keep it simple:
# Otherwise, you can load it from a CSV. Here we keep common ones:
type_choices = [
    "Appartement",
    "Maison",
    "Local industriel. commercial ou assimilé"
]
property_type = st.sidebar.selectbox("Property type", type_choices, index=0)

surface = st.sidebar.number_input(
    "Built surface area (m²)",
    min_value=5.0, max_value=400.0, value=40.0, step=1.0
)

rooms = st.sidebar.number_input(
    "Number of main rooms",
    min_value=1, max_value=15, value=2, step=1
)

# =========================
# BUILD INPUT FOR MODEL
# =========================
# Target-encoding (same logic as training)
arr_te = arr_map.get(arrondissement, global_mean)
type_te = type_map.get(property_type, global_mean)

X_input = pd.DataFrame([{
    "Surface reelle bati": float(surface),
    "Nombre pieces principales": int(rooms),
    "arrondissement_te": float(arr_te),
    "type_local_te": float(type_te),
}])

# =========================
# PREDICTION
# =========================
if st.sidebar.button("📌 Estimate"):
    pred_log = float(model.predict(X_input[FINAL_FEATURES])[0])
    price_per_m2 = float(np.exp(pred_log))
    total_price = price_per_m2 * float(surface)

    st.subheader("Estimated price")
    c1, c2 = st.columns(2)

    c1.metric("Estimated €/m²", f"{price_per_m2:,.0f} €".replace(",", " "))
    c2.metric("Estimated total price", f"{total_price:,.0f} €".replace(",", " "))

    st.write("**Inputs used:**")
    st.dataframe(X_input, use_container_width=True)

    st.info(
        f"Encoding used → Arrondissement TE: {arr_te:.4f} | Type TE: {type_te:.4f}"
    )
else:
    st.info("Choose the parameters on the left, then click **Estimate**.")

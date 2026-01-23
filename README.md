# 🏠 Paris Real Estate Price Estimator (€/m²)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)


Machine Learning project to estimate **real estate prices in Paris (€/m²)** using **official French government data (DVF)** and an interactive **Streamlit web application**.

---

## 📌 Project Overview

This project uses **real real-estate transaction data** from the French government (DVF – *Demande de Valeurs Foncières*) to build a **robust and realistic price prediction model** for Paris properties.

The workflow covers the **entire Data Science pipeline**:
- Data cleaning & preprocessing
- Feature analysis & selection
- Machine Learning model training
- Deployment with Streamlit

Special attention is given to **location modeling (Paris arrondissements)** to ensure realistic price differences between districts.

---

## 🎯 Objectives

- Use **official DVF open data**
- Build a **clean and reproducible ML pipeline**
- Correctly model **location effects (arrondissements)**
- Avoid data leakage
- Deliver an **interactive price estimator**
- Showcase applied **Data Science & ML skills**

---

## 🗂️ Project Structure

```text
house_pricing_ml/
│
├── Data/
│ ├── raw/ # Raw DVF files (2020–2025)      # Raw DVF files (not included – see Data section)
│ ├── processed/ # Cleaned and model-ready datasets
│
├── notebooks/
│ ├── 01_clean_data.py # Data cleaning & preprocessing
│ ├── 02_feature_analysis.py # Feature analysis & selection
│ ├── 03_train_model.py # Model training
│ ├── app.py # Streamlit application
│
├── models/
│ ├── model_prix_m2_paris_targetenc.pkl
│
├── reports/
│ ├── corr_with_target.csv
│ ├── rf_feature_importance.csv
│ ├── selected_features.csv
│
├── requirements.txt
├── README.md
```

---
## 📊 Data Source

This project is based on **DVF – Demande de Valeurs Foncières**, the official French government database containing all real estate transactions recorded by notaries.

The dataset provides detailed information on:
- Transaction dates
- Sale prices
- Property types
- Surface areas
- Number of rooms
- Geographic location (city, postal code, department)

### Official source

The data is published and maintained by the French government on the national open data portal:

https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/

This source guarantees:
- Reliability and accuracy
- Real transaction data
- Open and reusable data under public license

---

## ⚠️ Raw Data Availability

Raw DVF files are **not included in this repository** due to GitHub’s file size limitations (individual files exceed 100 MB).

### How to reproduce the project:

1. Download DVF data from the official source above  
2. Place the raw files in the following directory:
```bash
Data/raw/
```
3. Run the data processing and modeling pipeline:
- `01_clean_data.py`
- `02_feature_analysis.py`
- `03_train_model.py`

All processed datasets and models can be regenerated locally from the raw data.

## 📊 Data Description

- **Source**: DVF – Demande de Valeurs Foncières (French Government)
- **Geographic scope**: Paris (75)
- **Time range**:
  - Training data: 2020–2024
  - Validation / testing: 2024–2025
- **Target variable**:
  - `log_prix_m2` (log-transformed price per square meter)

### Main Features Used
- `Surface reelle bati`
- `Nombre pieces principales`
- `Type local`
- `Arrondissement` (engineered from postal code)

---

## 🧹 Step 1 — Data Cleaning

Script: `01_clean_data.py`

Main operations:
- Load raw DVF files
- Filter Paris transactions only
- Drop empty and irrelevant columns
- Convert prices and surfaces to numeric
- Remove outliers
- Compute:
  - `prix_m2`
  - `log_prix_m2`
- Save cleaned data to `data/processed/`

---

## 🔎 Step 2 — Feature Analysis

Script: `02_feature_analysis.py`

Performed analyses:
- Missing value diagnostics
- Removal of constant and technical columns
- Correlation analysis with the target
- Random Forest feature importance
- Leakage detection (`prix_m2`, `Valeur fonciere`)
- Selection of relevant original features

Outputs:
- `corr_with_target.csv`
- `rf_feature_importance.csv`
- `selected_features.csv`
- `data_model_ready_paris.csv`

---

## 🤖 Step 3 — Model Training

Script: `03_train_model.py`

### Why Target Encoding?

One-hot encoding for Paris postal codes:
- Creates sparse matrices
- Dilutes location signal
- Reduces model realism

**Solution**: Target Encoding for:
- Paris arrondissements (01 → 20)
- Property type (`Type local`)

This ensures:
- Strong location impact
- Stable predictions
- Realistic price gaps between districts

### Model Used
- **HistGradientBoostingRegressor**
- Well-suited for tabular data
- Fast and accurate
- Handles non-linear relationships

The trained pipeline is saved as:
```bash
models/model_prix_m2_paris_targetenc.pkl
```

---

## 🌐 Step 4 — Streamlit Application

Script: `app.py`

### App Features
- Select **Paris arrondissement (01–20)**
- Select property type
- Enter surface area and number of rooms
- Get:
  - Estimated €/m²
  - Estimated total price

The app uses the **same preprocessing and model pipeline** as training.

---

## 🚀 Run the Project Locally

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python notebooks/03_train_model.py
```

### 4️⃣ Launch the Streamlit app

```bash
streamlit run notebooks/app.py
```

## License

This project is licensed under the MIT License.
See the LICENSE file for more information.

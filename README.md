# PM2.5 Prediction from Meteorology — End-to-End MLOps Pipeline

Predicts **PM2.5 concentration** (fine particulate matter) from **meteorological factors**
using XGBoost, on the UCI Beijing Multi-Site Air Quality dataset (~420k hourly rows, 12 stations).
Task: **Regression** | Metric: **RMSE + R² + MAE**

> **Design note:** the dataset also contains co-pollutants (PM10, CO, NO2, SO2, O3). We **deliberately
> exclude them** as predictors — they are co-emitted symptoms of the same sources, not weather, and are
> unavailable at forecast time (using them is practical target leakage). The notebook's ablation section
> quantifies this (R² 0.59 → 0.94). The deployed model uses **weather only**.

---

## Project Structure

```
project_ml/
├── data/               # Raw data (tracked by DVC, not git)
├── models/             # Trained model artifacts (tracked by DVC)
├── notebooks/
│   └── 01_exploration.ipynb   # EDA + baseline + XGBoost + SHAP
├── src/
│   ├── data.py         # load, clean, split
│   ├── features.py     # preprocessing pipeline
│   ├── train.py        # Optuna + MLflow training entry point
│   └── predict.py      # inference helpers
├── serving/
│   ├── app.py          # FastAPI service
│   └── schemas.py      # Pydantic input/output schemas
├── config.yaml         # All hyperparameters and paths
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Get the data (UCI #501) into `data/beijing_air_quality.csv`:

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
ds = fetch_ucirepo(id=501)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
df.to_csv("data/beijing_air_quality.csv", index=False)
```

Or download the zip from the [UCI page](https://archive.ics.uci.edu/dataset/501/) and concatenate the 12
per-station CSVs into `data/beijing_air_quality.csv` (`dvc pull` also restores it once DVC is configured).

---

## Step 1 — Run Notebook (EDA + Experiments)

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Run all cells. The notebook:
- Explores data, checks missing values, analyzes distributions
- Trains baseline (Linear Regression) and improved model (XGBoost)
- Calculates SHAP feature importance
- Saves `models/feature_names.json`

---

## Step 2 — Production Training (Optuna + MLflow + DVC)

```bash
python -m src.train
```

This runs 50 Optuna trials to find the best XGBoost hyperparameters,  
logs all experiments to MLflow, and saves the final model.

**View MLflow UI:**
```bash
mlflow ui
```
Then open http://localhost:5000

---

## Step 3 — DVC Setup (artifact versioning on Google Drive)

```bash
dvc init
dvc remote add -d gdrive gdrive://<YOUR_GDRIVE_FOLDER_ID>
dvc add data/train.csv models/best_model.pkl models/preprocessor.pkl models/feature_names.json
git add .dvc data/.gitignore models/.gitignore
dvc push
```

Share your Google Drive folder with `aaaksenova2@gmail.com` (Viewer).

---

## Step 4 — Serve the Model

```bash
uvicorn serving.app:app --reload
```

Open **http://localhost:8000/docs** for the Swagger UI.

**Live-demo (vary ONLY wind speed → PM2.5 should drop as wind rises):**

Stagnant air (`WSPM: 0.5`) → high PM2.5:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"year":2016,"month":12,"day":15,"hour":8,"TEMP":-3.5,"PRES":1024.0,"DEWP":-12.0,"RAIN":0.0,"WSPM":0.5,"wd":"NW","station":"Dongsi"}'
```
Windy (`WSPM: 10.0`, everything else identical) → lower PM2.5:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"year":2016,"month":12,"day":15,"hour":8,"TEMP":-3.5,"PRES":1024.0,"DEWP":-12.0,"RAIN":0.0,"WSPM":10.0,"wd":"NW","station":"Dongsi"}'
```
This pair demonstrates the core physics: **wind disperses particles**. Change one variable at a
time in the demo — changing several at once mixes effects and can mask the trend.

**Final metrics (honest time-based split, meteorology only):** RMSE ≈ 60, MAE ≈ 41, **R² ≈ 0.48**.

---

## Reproduce Best Run

```bash
git clone <repo>
pip install -r requirements.txt
dvc pull
python -m src.train
uvicorn serving.app:app --reload
```

---

## Key Design Decisions

| Choice | Why |
|--------|-----|
| Target = PM2.5 | Most health-critical pollutant; physically driven by weather |
| Exclude co-pollutants | Co-emitted symptoms, not weather; unavailable at forecast time (leakage). Ablation: R² 0.59→0.94 |
| RMSE as primary metric | Penalises large errors more — important for public health alerts |
| XGBoost over Linear Regression | Non-linear weather→PM2.5 relationships, skewed target, feature interactions |
| Optuna for HP tuning | Bayesian optimisation finds better params than grid search in fewer trials |
| MLflow local tracking | Full experiment history without external infrastructure |
| DVC + GDrive | Model artifacts versioned separately from code — reproducibility without bloating git |
| FastAPI + Pydantic | Typed, auto-documented API; Swagger UI for instant demo |

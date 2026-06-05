import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from serving.schemas import MeteoInput, PM25Prediction, pm25_to_category
from src.predict import load_model, load_feature_names, predict

app = FastAPI(
    title="PM2.5 Prediction API",
    description="Predicts PM2.5 concentration from meteorological factors (Beijing Air Quality model).",
    version="1.0.0",
)

MODEL = None
FEATURE_NAMES = None


@app.on_event("startup")
def startup():
    global MODEL, FEATURE_NAMES
    MODEL = load_model("models/best_model.pkl")
    FEATURE_NAMES = load_feature_names("models/feature_names.json")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/feature-info")
def feature_info():
    return {"features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}


def build_feature_vector(payload: MeteoInput) -> dict:
    vector = {name: 0.0 for name in FEATURE_NAMES}

    numeric = {
        "year": payload.year, "month": payload.month, "day": payload.day,
        "hour": payload.hour, "TEMP": payload.TEMP, "PRES": payload.PRES,
        "DEWP": payload.DEWP, "RAIN": payload.RAIN, "WSPM": payload.WSPM,
    }
    for k, v in numeric.items():
        if k in vector:
            vector[k] = float(v)

    wd_col = f"wd_{payload.wd}"
    if wd_col in vector:
        vector[wd_col] = 1.0

    station_col = f"station_{payload.station}"
    if station_col in vector:
        vector[station_col] = 1.0

    return vector


@app.post("/predict", response_model=PM25Prediction)
def predict_pm25(payload: MeteoInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_feature_vector(payload)
    pm25_value = predict(MODEL, features)
    pm25_value = max(0.0, round(pm25_value, 2))

    return PM25Prediction(
        pm25=pm25_value,
        category=pm25_to_category(pm25_value),
    )

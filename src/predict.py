import json
import joblib
import pandas as pd


def load_model(model_path: str = "models/best_model.pkl"):
    return joblib.load(model_path)


def load_feature_names(path: str = "models/feature_names.json") -> list:
    with open(path) as f:
        return json.load(f)


def predict(model, features: dict) -> float:
    df = pd.DataFrame([features])
    return float(model.predict(df)[0])

import json
import joblib
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data import load_config, load_data, clean_data, split_data
from src.features import build_preprocessor, get_feature_names

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def objective(trial, X_train, y_train, X_val, y_val, preprocessor):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "random_state": 42,
    }
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(**params)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, preds)))


def train(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    df = load_data(cfg["data"]["raw_path"])
    print(f"Loaded data: {df.shape}")

    df = clean_data(
        df,
        target_col=cfg["data"]["target_column"],
        drop_cols=cfg["features"]["drop_columns"],
    )
    print(f"After cleaning: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col=cfg["data"]["target_column"],
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
        strategy=cfg["data"].get("split_strategy", "random"),
        time_columns=cfg["data"].get("time_columns"),
    )
    print(f"Split strategy: {cfg['data'].get('split_strategy', 'random')}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)
    feature_names = X_train.columns.tolist()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    print(f"Running Optuna ({cfg['optuna']['n_trials']} trials)...")
    study = optuna.create_study(direction=cfg["optuna"]["direction"])
    study.optimize(
        lambda t: objective(t, X_train, y_train, X_val, y_val, preprocessor),
        n_trials=cfg["optuna"]["n_trials"],
    )

    best_params = {**study.best_params, "tree_method": "hist", "random_state": 42}
    print(f"Best val RMSE: {study.best_value:.4f}")

    with mlflow.start_run(run_name="best_xgboost"):
        mlflow.log_params(best_params)
        mlflow.log_param("n_optuna_trials", cfg["optuna"]["n_trials"])
        mlflow.log_param("n_features", len(feature_names))

        final_model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(**best_params)),
        ])

        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        final_model.fit(X_trainval, y_trainval)

        metrics = evaluate(y_test, final_model.predict(X_test))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_model, "model")

        print("\n=== Test Set Results ===")
        for k, v in metrics.items():
            print(f"  {k.upper():4s}: {v:.4f}")

    Path(cfg["artifacts"]["model_path"]).parent.mkdir(exist_ok=True)
    joblib.dump(final_model, cfg["artifacts"]["model_path"])
    joblib.dump(preprocessor, cfg["artifacts"]["preprocessor_path"])
    with open(cfg["artifacts"]["feature_names_path"], "w") as f:
        json.dump(feature_names, f)

    print(f"\nModel saved -> {cfg['artifacts']['model_path']}")
    return final_model, metrics


if __name__ == "__main__":
    train()

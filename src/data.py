import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame, target_col: str, drop_cols: list) -> pd.DataFrame:
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")

    for col in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df[f"{col}_quarter"] = parsed.dt.quarter
            df = df.drop(columns=[col])
        except Exception:
            pass

    df = df.dropna(subset=[target_col])

    num_cols = df.select_dtypes(include=[np.number]).columns.difference([target_col])
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
    strategy: str = "random",
    time_columns: list = None,
):
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    if strategy == "time":
        cols = [c for c in (time_columns or []) if c in X.columns]
        order = X.sort_values(cols, kind="mergesort").index
        X = X.loc[order]
        y = y.loc[order]

        n = len(X)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val

        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_val, y_val = X.iloc[n_train:n_train + n_val], y.iloc[n_train:n_train + n_val]
        X_test, y_test = X.iloc[n_train + n_val:], y.iloc[n_train + n_val:]
        return X_train, X_val, X_test, y_train, y_val, y_test

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

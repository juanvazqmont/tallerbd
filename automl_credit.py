import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGB = False
    XGBRegressor = None


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from path."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transformer with imputers, scalers and encoders."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
    return preprocessor


def get_models() -> dict:
    """Return candidate models with their parameter distributions."""
    models = {
        "ridge": (
            Ridge(),
            {
                "model__alpha": np.logspace(-3, 3, 100),
            },
        ),
        "lasso": (
            Lasso(max_iter=10000),
            {
                "model__alpha": np.logspace(-3, 3, 100),
            },
        ),
        "rf": (
            RandomForestRegressor(random_state=42),
            {
                "model__n_estimators": np.arange(50, 501, 50),
                "model__max_depth": [None] + list(range(2, 11)),
            },
        ),
    }
    if HAS_XGB:
        models["xgb"] = (
            XGBRegressor(objective="reg:squarederror", random_state=42),
            {
                "model__n_estimators": np.arange(50, 501, 50),
                "model__max_depth": range(2, 11),
                "model__learning_rate": np.linspace(0.01, 0.3, 30),
            },
        )
    return models


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


RMSE_SCORER = make_scorer(rmse, greater_is_better=False)
R2_SCORER = make_scorer(r2_score)


def evaluate_model(X, y, model, param_distributions, cv):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring={"rmse": RMSE_SCORER, "r2": R2_SCORER},
        refit="rmse",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search


def main():
    parser = argparse.ArgumentParser(description="AutoML for UCI Credit Card dataset")
    parser.add_argument(
        "--data",
        default="UCI_Credit_Card.csv",
        help="Path to dataset CSV (defaults to UCI_Credit_Card.csv)",
    )
    parser.add_argument(
        "--output-model",
        default="best_model.joblib",
        help="File to store the trained model",
    )
    parser.add_argument(
        "--output-results",
        default="results.json",
        help="File to store the evaluation results",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = load_data(args.data)

    if "default.payment.next.month" in df.columns:
        y = df["default.payment.next.month"]
        X = df.drop(columns=["default.payment.next.month"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    preprocessor = build_preprocessor(df)
    models = get_models()

    results = {}
    best_score = float("inf")
    best_search = None
    for name, (model, params) in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        search = evaluate_model(X, y, pipeline, params, args.cv)
        rmse_score = -search.best_score_
        r2 = search.cv_results_["mean_test_r2"][search.best_index_]
        results[name] = {
            "best_params": search.best_params_,
            "rmse": rmse_score,
            "r2": r2,
        }
        if rmse_score < best_score:
            best_score = rmse_score
            best_search = search

    if best_search is None:
        raise RuntimeError("No model was successfully trained")

    with open(args.output_results, "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_search.best_estimator_, args.output_model)

    print("Best model:", best_search.best_estimator_)
    print("Best parameters:", best_search.best_params_)
    print("RMSE:", -best_search.best_score_)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from world_cup_predictor.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODELS_DIR,
    TARGET_MAP,
)


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict


def _multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    classes = np.arange(probabilities.shape[1])
    truth = (y_true[:, None] == classes).astype(float)
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def _build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def prepare_training_frame(model_df: pd.DataFrame) -> pd.DataFrame:
    training = model_df.copy()
    training["target"] = training["result_label"].map(TARGET_MAP)
    return training


def split_train_test(model_df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = model_df[model_df["date"] < pd.Timestamp(cutoff_date)].copy()
    test_df = model_df[model_df["date"] >= pd.Timestamp(cutoff_date)].copy()
    if train_df.empty or test_df.empty:
        split_index = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:split_index].copy()
        test_df = model_df.iloc[split_index:].copy()
    return train_df, test_df


def _fit_model(estimator, train_df: pd.DataFrame, test_df: pd.DataFrame, name: str) -> ModelResult:
    pipeline = Pipeline(steps=[("preprocess", _build_preprocessor()), ("model", estimator)])
    x_train = train_df[FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    y_train = train_df["target"]
    x_test = test_df[FEATURE_COLUMNS + CATEGORICAL_FEATURES]
    y_test = test_df["target"]

    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)
    predictions = pipeline.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "log_loss": log_loss(y_test, probabilities, labels=np.array([0, 1, 2])),
        "multiclass_brier": _multiclass_brier_score(y_test.to_numpy(), probabilities),
    }
    return ModelResult(name=name, pipeline=pipeline, metrics=metrics)


def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[list[ModelResult], ModelResult]:
    logistic = LogisticRegression(max_iter=2000, random_state=42)
    forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
    )
    results = [
        _fit_model(logistic, train_df, test_df, "logistic_regression"),
        _fit_model(forest, train_df, test_df, "random_forest"),
    ]
    best = min(results, key=lambda result: result.metrics["log_loss"])
    return results, best


def save_models(results: list[ModelResult]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for result in results:
        joblib.dump(result.pipeline, MODELS_DIR / f"{result.name}.joblib")


def predict_match_probabilities(model: Pipeline, fixtures: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(fixtures[FEATURE_COLUMNS + CATEGORICAL_FEATURES])
    output = fixtures.copy()
    output["home_win_prob"] = probabilities[:, 0]
    output["draw_prob"] = probabilities[:, 1]
    output["away_win_prob"] = probabilities[:, 2]
    output["home_advancement_prob"] = output["home_win_prob"] + output["draw_prob"] * (
        output["home_win_prob"] / np.clip(output["home_win_prob"] + output["away_win_prob"], 1e-9, None)
    )
    output["away_advancement_prob"] = 1.0 - output["home_advancement_prob"]
    return output

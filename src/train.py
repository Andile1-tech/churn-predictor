from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data import run_pipeline


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_pipeline(preprocessor: ColumnTransformer, classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def print_report(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def train(save: bool = True) -> dict:
    print("[train] Loading cleaned data ...")
    df = run_pipeline()
    print(f"[train] Data shape: {df.shape}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_cols, categorical_cols = split_feature_types(df)
    print(f"[train] Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[train] Categorical features ({len(categorical_cols)}): {categorical_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[train] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    candidates = {
        "logistic_regression": build_pipeline(
            preprocessor,
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest": build_pipeline(
            preprocessor,
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }

    print("\n[train] Cross-validated ROC-AUC (5 folds):")
    cv_scores: dict[str, float] = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        cv_scores[name] = float(scores.mean())
        print(f"  {name:20s}  {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\n[train] Best by CV ROC-AUC: {best_name}")

    if best_name == "logistic_regression":
        param_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__l1_ratio": [0],
            "classifier__solver": ["liblinear", "lbfgs"],
        }

    else:
        param_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 5],
        }

    print(f"\n[train] Grid searching {best_name} ...")
    grid = GridSearchCV(
        candidates[best_name],
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"[train] Best params: {grid.best_params_}")
    print(f"[train] Best CV ROC-AUC: {grid.best_score_:.4f}")

    metrics = evaluate_model(best_model, X_test, y_test)
    print("\n[train] Held-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print_report(best_model, X_test, y_test)

    if save:
        joblib.dump(best_model, MODEL_PATH)
        print(f"\n[train] Saved model -> {MODEL_PATH}")

        metrics_path = MODEL_PATH.with_name("metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": best_name,
                    "cv_roc_auc": cv_scores[best_name],
                    "test_metrics": metrics,
                    "best_params": {k: str(v) for k, v in grid.best_params_.items()},
                },
                f,
                indent=2,
            )
        print(f"[train] Saved metrics -> {metrics_path}")

    return {"model": best_name, "metrics": metrics}


if __name__ == "__main__":
    train()

from __future__ import annotations

import functools
from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH


@functools.lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_one(customer: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    df = pd.DataFrame([customer])
    proba = float(model.predict_proba(df)[:, 1][0])
    label = int(proba >= 0.5)
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": label,
        "risk_label": "High risk" if label == 1 else "Low risk",
    }


def predict_batch(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = load_model()
    df = pd.DataFrame(customers)
    probs = model.predict_proba(df)[:, 1]
    return [
        {
            "churn_probability": round(float(p), 4),
            "churn_prediction": int(p >= 0.5),
            "risk_label": "High risk" if p >= 0.5 else "Low risk",
        }
        for p in probs
    ]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for classification. Install it or use the recommended environment."
        ) from exc
    return {
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "LinearSVC": LinearSVC,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


@dataclass
class EvaluationResult:
    accuracy: float
    confusion_matrix: np.ndarray


def train_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    classifier_name: str = "linear_svc",
    random_state: int = 0,
):
    sk = _import_sklearn()
    make_pipeline = sk["make_pipeline"]
    StandardScaler = sk["StandardScaler"]
    LinearSVC = sk["LinearSVC"]
    LogisticRegression = sk["LogisticRegression"]

    if classifier_name == "linear_svc":
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LinearSVC(random_state=random_state, dual=True, max_iter=5000),
        )
    elif classifier_name == "logreg":
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                multi_class="auto",
            ),
        )
    else:
        raise ValueError("classifier_name must be 'linear_svc' or 'logreg'.")

    model.fit(train_features, train_labels)
    return model


def evaluate_classifier(model, features: np.ndarray, labels: np.ndarray) -> EvaluationResult:
    sk = _import_sklearn()
    predictions = model.predict(features)
    accuracy = float(sk["accuracy_score"](labels, predictions))
    cm = sk["confusion_matrix"](labels, predictions)
    return EvaluationResult(accuracy=accuracy, confusion_matrix=cm)

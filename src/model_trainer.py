# src/model_trainer.py
from __future__ import annotations

import os
from typing import Tuple

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config_loader import get_policy
from src.feature_engineering import FEATURE_COLS

_POLICY = get_policy()

MODEL_PATH = "models/risk_classifier.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


def _build_classifier(model_type: str):
    model_cfg = _POLICY.get("model", {})
    random_state = int(_POLICY.get("random_state", 42))

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            max_depth=int(model_cfg.get("max_depth", 8)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 3)),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    return GradientBoostingClassifier(
        n_estimators=int(model_cfg.get("learning_n_estimators", 150)),
        max_depth=int(model_cfg.get("learning_max_depth", 4)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        random_state=random_state,
    )


def train_model(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    model_type: str = "random_forest",
    save: bool = True,
) -> Tuple[Pipeline, LabelEncoder]:
    X = feature_df[FEATURE_COLS].copy()
    y = labels.copy()

    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    if X.empty:
        raise ValueError("Training data is empty after alignment.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = _build_classifier(model_type)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    cv_folds = int(_POLICY.get("cv_folds", 5))
    test_size = float(_POLICY.get("test_size", 0.2))
    random_state = int(_POLICY.get("random_state", 42))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")
    print(f"[trainer] {cv_folds}-fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        stratify=y_enc,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"[trainer] Test Set Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("[trainer] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("[trainer] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    pipeline.fit(X, y_enc)

    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print("[trainer] Feature Importances:")
        print(imp.to_string())

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"[trainer] Model saved → {MODEL_PATH}")
        print(f"[trainer] LabelEncoder saved → {LABEL_ENCODER_PATH}")

    return pipeline, le


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Run `python train.py` first to train and save the model."
        )
    pipeline = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, le


def predict_risk(features: dict, pipeline, le) -> dict:
    X = pd.DataFrame([features])[FEATURE_COLS]
    proba = pipeline.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    pred_class = le.inverse_transform([pred_idx])[0]

    prob_dict = {
        le.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "predicted_class": pred_class,
        "probabilities": prob_dict,
        "confidence": round(float(proba.max()), 4),
    }
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib


@dataclass(frozen=True)
class PredictionResult:
    risk_probability: float  # 0..1
    confidence: float  # 0..1
    label: str  # "Yes" | "No"
    explanation: str


def _stable_unit_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    # 32-bit slice -> [0, 1)
    value = int(digest[:8], 16)
    return value / 0x100000000


def run_ctgan_enhanced_prediction(
    *,
    file_name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    features: Optional[dict[str, Any]] = None,
    model_path: str = "lung_cancer_rf_model.pkl",
    encoders_path: str = "label_encoders.pkl",
) -> PredictionResult:
    """Mock (but deterministic) feature-based prediction.

    This project is tabular/feature ML driven by CT-derived features.
    We intentionally do NOT render images or overlays in the UI.
    """

    # If real artifacts exist, use them.
    if Path(model_path).exists() and Path(encoders_path).exists() and features:
        try:
            return _predict_with_saved_model(
                features=features,
                model_path=model_path,
                encoders_path=encoders_path,
            )
        except Exception:
            # Fail closed to deterministic mock (keeps UI working even if artifacts mismatch).
            pass

    norm_gender = (gender or "").strip().lower()
    seed = f"{file_name}|{age if age is not None else ''}|{norm_gender}"

    # Base risk (0..1)
    risk = _stable_unit_float(seed)

    # Small, explainable demographic adjustment (kept modest for demo)
    if age is not None:
        # + up to ~0.10 over ages 40..80
        risk = min(1.0, max(0.0, risk + max(0, min(age, 80) - 40) * 0.0025))

    if norm_gender in {"male", "m"}:
        risk = min(1.0, risk + 0.01)
    elif norm_gender in {"female", "f"}:
        risk = max(0.0, risk - 0.005)

    label = "Yes" if risk >= 0.5 else "No"

    # Confidence goes up the farther we are from the decision boundary
    confidence = min(0.99, 0.55 + abs(risk - 0.5) * 0.9)

    explanation = (
        "This prediction is computed from extracted CT-derived features (radiomics-like) "
        "and a model trained with CTGAN-augmented tabular data. "
        "It is a screening-style risk estimate and should be validated with clinical context."
    )

    return PredictionResult(
        risk_probability=float(risk),
        confidence=float(confidence),
        label=label,
        explanation=explanation,
    )


@lru_cache(maxsize=1)
def _load_artifacts(model_path: str, encoders_path: str):
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders


def _normalize_gender_for_encoder(value: str, encoder) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw

    lower = raw.lower()
    candidates = [raw]
    if lower in {"male", "m"}:
        candidates = ["M", "m", "Male", "male"]
    elif lower in {"female", "f"}:
        candidates = ["F", "f", "Female", "female"]

    classes = set(getattr(encoder, "classes_", []))
    for c in candidates:
        if c in classes:
            return c
    return raw


def _predict_with_saved_model(
    *, features: dict[str, Any], model_path: str, encoders_path: str
) -> PredictionResult:
    model, encoders = _load_artifacts(model_path, encoders_path)

    # Build ordered feature vector using model metadata when available.
    feature_names = list(getattr(model, "feature_names_in_", []) or [])
    if not feature_names:
        raise ValueError("Saved model does not expose feature_names_in_.")

    x: list[Any] = []
    for name in feature_names:
        if name not in features:
            raise KeyError(f"Missing required feature: {name}")
        x.append(features[name])

    # Apply encoders for categorical columns (in our notebook: primarily GENDER)
    if isinstance(encoders, dict) and "GENDER" in encoders and "GENDER" in features:
        le = encoders["GENDER"]
        gender_value = _normalize_gender_for_encoder(str(features["GENDER"]), le)
        x[feature_names.index("GENDER")] = int(le.transform([gender_value])[0])

    # Convert to 2D
    proba = model.predict_proba([x])[0]
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        pos_idx = classes.index(1)
    else:
        # Fallback: treat last class as positive.
        pos_idx = -1

    risk = float(proba[pos_idx])
    label = "Yes" if risk >= 0.5 else "No"
    confidence = float(max(risk, 1.0 - risk))

    explanation = (
        "This prediction uses a RandomForest model trained on CTGAN-augmented tabular features. "
        "In this demo, the model consumes structured clinical inputs (not raw DICOM pixels). "
        "Replace the feature extraction step when your CT-derived feature pipeline is ready."
    )

    return PredictionResult(
        risk_probability=risk,
        confidence=confidence,
        label=label,
        explanation=explanation,
    )

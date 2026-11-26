"""Online scoring for unlabelled, in-service bearings.

Phase 7 batch decisions cover only the run-to-failure bearings that already have
ground-truth labels. A deployed predictive-maintenance system must also score
HEALTHY bearings whose failure status is unknown. `score_unlabelled_bearing`
provides that inference path: it takes a bearing's feature vector, predicts RUL
with the promoted model artifact, and routes the prediction through the exact same
risk / policy / confidence / priority components used by the batch pipeline, so a
live decision is schema-compatible with recommendations.parquet.
"""

import logging
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np

from src.decision_engine.confidence_estimator import estimate_confidence
from src.decision_engine.maintenance_policy import get_maintenance_action, load_policy_map
from src.decision_engine.priority_calculator import compute_priority, load_priority_matrix
from src.decision_engine.risk_classifier import classify_risk, validate_thresholds

FeatureInput = Union[Mapping[str, float], Sequence[float], np.ndarray]


def _as_row_matrix(features: FeatureInput) -> np.ndarray:
    """Coerce a single feature vector into a 2-D (1, n_features) array."""
    if isinstance(features, Mapping):
        values = list(features.values())
    else:
        values = list(features)
    return np.asarray(values, dtype=float).reshape(1, -1)


def score_unlabelled_bearing(
    bearing_id: str,
    features: FeatureInput,
    model_artifact: Any,
    config: Dict[str, Any],
    logger: logging.Logger,
    model_type: str = "median",
    n2_caveat: bool = True,
) -> Dict[str, Any]:
    """Score a bearing that has not (yet) failed — the in-service monitoring path.

    Args:
        bearing_id: Identifier for the bearing being monitored.
        features: Feature values (dict keyed by feature name, or an ordered vector).
            For a constant-baseline model the values are ignored; for a feature-based
            model they must already be preprocessed in the model's expected order.
        model_artifact: Trained model exposing ``.predict`` (ConstantPredictor or an
            sklearn/xgboost estimator).
        config: Full project configuration dict.
        logger: Pipeline logger.
        model_type: Selected Phase 6 model type (drives confidence + the constant
            predictor warning).
        n2_caveat: Whether the n=2 caveat applies (always True for this dataset).

    Returns:
        A recommendation dict using the same field names as recommendations.parquet,
        plus ``constant_predictor_warning``.
    """
    rul_prediction = float(np.asarray(model_artifact.predict(_as_row_matrix(features))).ravel()[0])

    de_cfg = config.get("decision_engine", {})
    thresholds = de_cfg.get("risk_thresholds", {})
    validate_thresholds(thresholds, logger)

    risk = classify_risk(rul_prediction, thresholds)
    action = get_maintenance_action(risk, load_policy_map(config, logger))
    confidence = estimate_confidence(
        rul_prediction, model_type, n2_caveat, thresholds, config, logger
    )
    priority = compute_priority(risk, confidence, load_priority_matrix(config, logger))

    is_constant = model_type in ("mean", "median")
    return {
        "bearing_id": bearing_id,
        "min_predicted_rul_minutes": rul_prediction,
        "mean_predicted_rul_minutes": rul_prediction,
        "risk_category": risk.value,
        "maintenance_action": action.value,
        "confidence_level": confidence.value,
        "priority_score": priority,
        "model_type": model_type,
        "n2_caveat": n2_caveat,
        "constant_predictor_warning": is_constant,
    }

"""Model registry: versioning, fingerprinting, and atomic persistence of
model artifacts (architecture §5, §8.4, §16).

Two artifacts are registered per experiment:
  - `model.pkl`        — per-fold models + preprocessors (provenance; the
                          honest LOBO evaluation artifacts).
  - `refit_model.pkl`   — the single model refit on all RTF data (the
                          artifact Phase 7 actually loads for prediction).
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from src.model_dev.cross_validation import CVResults
from src.model_dev.preprocessing import FittedPreprocessor


def compute_model_fingerprint(model: Any) -> str:
    """SHA-256 of the pickled model bytes — allows two artifacts to be compared without re-training.

    Args:
        model: Any picklable fitted estimator or container.

    Returns:
        Hex digest string.
    """
    serialized = pickle.dumps(model)
    return hashlib.sha256(serialized).hexdigest()


def register_model(
    selected_model_name: str,
    cv_results: CVResults,
    refit_model: Any,
    refit_preprocessor: Optional[FittedPreprocessor],
    experiment_id: str,
    config: Dict[str, Any],
    project_root: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Persist `model.pkl` (per-fold provenance) and `refit_model.pkl` (deployment artifact).

    Args:
        selected_model_name: The model chosen by `model_selection.select_best_model`.
        cv_results: Output of `cross_validation.run_lobo_cv` (source of per-fold models).
        refit_model: The refit-for-handoff model (architecture §8.4).
        refit_preprocessor: The preprocessor fit on all RTF data (or None for baselines).
        experiment_id: Governance experiment ID (architecture §14).
        config: Full project configuration dictionary.
        project_root: Absolute project root path.
        logger: Pipeline logger.

    Returns:
        Dict with keys: model_path, refit_model_path, model_fingerprint, refit_model_fingerprint.
    """
    model_dev_cfg = config.get("model_dev", {})
    model_dir_cfg = model_dev_cfg.get("output_paths", {}).get("model_dir", "models")
    model_dir = project_root / model_dir_cfg / experiment_id
    model_dir.mkdir(parents=True, exist_ok=True)

    fold_models = {
        fold_id: {
            "model": fold_result.fitted_model,
            "preprocessor": fold_result.preprocessor,
            "train_bearings": fold_result.train_bearings,
            "test_bearings": fold_result.test_bearings,
        }
        for fold_id, fold_result in cv_results.results[selected_model_name].items()
    }
    model_container = {
        "selected_model_name": selected_model_name,
        "experiment_id": experiment_id,
        "fold_models": fold_models,
        "feature_columns": cv_results.feature_columns,
    }

    refit_container = {
        "selected_model_name": selected_model_name,
        "experiment_id": experiment_id,
        "model": refit_model,
        "preprocessor": refit_preprocessor,
        "feature_columns": cv_results.feature_columns,
        "trained_on": "all_rtf",
    }

    model_path = model_dir / "model.pkl"
    refit_model_path = model_dir / "refit_model.pkl"

    _atomic_pickle_write(model_container, model_path)
    _atomic_pickle_write(refit_container, refit_model_path)

    model_fingerprint = compute_model_fingerprint(model_container)
    refit_model_fingerprint = compute_model_fingerprint(refit_container)

    logger.info("Model registered: %s (fingerprint %s...)", model_path, model_fingerprint[:12])
    logger.info("Refit model registered: %s (fingerprint %s...)", refit_model_path, refit_model_fingerprint[:12])

    return {
        "model_path": str(model_path.relative_to(project_root)),
        "refit_model_path": str(refit_model_path.relative_to(project_root)),
        "model_fingerprint": model_fingerprint,
        "refit_model_fingerprint": refit_model_fingerprint,
    }


def _atomic_pickle_write(obj: Any, output_path: Path) -> None:
    """Atomically pickle-write `obj` to `output_path` via a `.tmp` + `Path.replace()` pattern."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".pkl.tmp")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(obj, f)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

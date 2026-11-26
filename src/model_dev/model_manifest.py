"""Selected-model provenance + environment capture (architecture §5, §16).

Extends the Phase 3→4→5 lineage chain with Phase 6's own version,
fingerprints, seed, and library-version environment capture.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import sklearn
import xgboost
import shap

from src.model_dev.evaluation import EvaluationResult
from src.model_dev.model_selection import SelectionResult


def get_git_commit() -> str:
    """Retrieve the current short git commit SHA, or 'unknown' if unavailable.

    Reuses the pattern established in `src/signal_analysis/gating_contract.py`.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def capture_environment() -> Dict[str, str]:
    """Capture library versions for reproducibility bounding (architecture §16)."""
    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_model_manifest(
    experiment_id: str,
    config_fingerprint: str,
    seed: int,
    selected_model: str,
    model_fingerprint: str,
    refit_model_fingerprint: str,
    evaluation_result: EvaluationResult,
    selection_result: SelectionResult,
    phase5_contract: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the `model_manifest.json` payload.

    Args:
        experiment_id: Governance experiment ID.
        config_fingerprint: SHA-256 of the model_dev config.
        seed: Global reproducibility seed.
        selected_model: The model chosen by model_selection.
        model_fingerprint: Fingerprint of the per-fold model container.
        refit_model_fingerprint: Fingerprint of the refit-for-handoff artifact.
        evaluation_result: Full evaluation results.
        selection_result: Selection hierarchy outcome.
        phase5_contract: Parsed Phase 5 contract (for upstream lineage).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Dict representing model_manifest.json payload.
    """
    manifest = {
        "phase": "6",
        "model_version": "1.0",
        "experiment_id": experiment_id,
        "config_fingerprint": config_fingerprint,
        "random_seed": seed,
        "git_commit": get_git_commit(),
        "environment": capture_environment(),
        "selected_model": selected_model,
        "selection_resolved_at_priority": selection_result.resolved_at_priority,
        "selection_resolved_by": selection_result.resolved_by,
        "model_fingerprint": model_fingerprint,
        "refit_model_fingerprint": refit_model_fingerprint,
        "model_evaluations_summary": {
            name: {
                "mean_mae": ev.mean_mae,
                "mae_spread": ev.mae_spread,
                "stability_ratio": ev.stability_ratio,
            }
            for name, ev in evaluation_result.model_evaluations.items()
        },
        "upstream_lineage": {
            "phase5_contract_version": phase5_contract.get("contract_version"),
            "phase5_dataset_version": phase5_contract.get("dataset_version"),
            "phase5_config_fingerprint": phase5_contract.get("config_fingerprint"),
        },
        "statistical_limitations": phase5_contract.get("statistical_limitations", {
            "n_rtf_units": len(phase5_contract.get("artifacts", {}).get("training_dataset", {}).get("units", [])),
            "caveat": (
                "Performance estimates cannot carry classical confidence intervals. "
                "Results are methodology validation, not fleet-level guarantees."
            ),
        }),
    }
    logger.info("Model manifest built for experiment %s", experiment_id)
    return manifest


def save_model_manifest(manifest: Dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `model_manifest.json`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Model manifest saved: %s", output_path)

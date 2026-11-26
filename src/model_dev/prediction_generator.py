"""Emit held-out LOBO predictions for the selected model, grain-aligned
to (unit_id, channel, timestamp) — architecture §5, §12.
"""

import logging
from pathlib import Path

import pandas as pd

from src.model_dev.cross_validation import CVResults


def generate_predictions(cv_results: CVResults, selected_model_name: str, logger: logging.Logger) -> pd.DataFrame:
    """Assemble `model_predictions.parquet`: every RTF row's out-of-sample prediction.

    Each bearing is the held-out test unit in exactly one LOBO fold, so the
    combined output covers the full RTF training dataset with true
    out-of-sample predictions (never a row predicted by a model trained on
    that same row's bearing).

    Args:
        cv_results: Output of `cross_validation.run_lobo_cv`.
        selected_model_name: The model chosen by `model_selection.select_best_model`.
        logger: Pipeline logger.

    Returns:
        DataFrame with grain columns + fold_id + actual_rul_minutes + predicted_rul_minutes.
    """
    frames = []
    for fold_id, fold_result in cv_results.results[selected_model_name].items():
        df = fold_result.grain_df.copy()
        df["fold_id"] = fold_id
        df["actual_rul_minutes"] = fold_result.y_test_true
        df["predicted_rul_minutes"] = fold_result.y_test_pred
        frames.append(df)

    predictions = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("unit_id", "channel", "timestamp") if c in predictions.columns]
    predictions = predictions.sort_values(sort_cols).reset_index(drop=True)

    unit_col = "unit_id" if "unit_id" in predictions.columns else "bearing_id"
    logger.info(
        "Predictions generated: %d rows (model=%s) covering units %s",
        len(predictions), selected_model_name, sorted(predictions[unit_col].unique().tolist()),
    )
    return predictions


def save_predictions(predictions: pd.DataFrame, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write `model_predictions.parquet`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    try:
        predictions.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Predictions saved: %s (%d rows)", output_path, len(predictions))

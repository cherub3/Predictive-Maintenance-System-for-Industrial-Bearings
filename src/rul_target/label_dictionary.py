"""Label dictionary generator for Phase 5 artifacts.

Produces label_dictionary.json — a machine-readable schema for every label
column emitted in rul_labels.parquet and training_dataset.parquet.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import get_censored_units, get_rtf_units


def build_label_dictionary(
    rul_labels: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the label dictionary metadata object.

    Args:
        rul_labels: Labels DataFrame from label_generator.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Dictionary representing the label_dictionary.json payload.
    """
    rul_cfg = config.get("rul_target", {})
    rtf_units: List[str] = get_rtf_units(config)
    censored_units: List[str] = get_censored_units(config)
    emit_rul_cycles: bool = rul_cfg.get("emit_rul_cycles", True)
    emit_rul_frac: bool = rul_cfg.get("emit_rul_frac", False)

    rtf_labels = rul_labels[~rul_labels["is_censored"]]

    label_columns: List[Dict[str, Any]] = []

    # rul_minutes — always present
    label_columns.append({
        "name": "rul_minutes",
        "dtype": "float64",
        "unit": "minutes",
        "description": (
            "Remaining useful life in minutes. "
            "RUL(t) = (t_fail - t) in minutes; RUL(t_fail) = 0. "
            "NaN for censored (right-censored survivor) units."
        ),
        "null_for_units": censored_units,
        "rtf_range": _float_range(rtf_labels, "rul_minutes"),
        "rtf_at_failure": 0.0,
    })

    # rul_cycles — optional
    if emit_rul_cycles and "rul_cycles" in rul_labels.columns:
        label_columns.append({
            "name": "rul_cycles",
            "dtype": "float64",
            "unit": "acquisition_cycles",
            "description": (
                "Remaining cycles (count of unique future timestamps for this unit). "
                "Complementary to rul_minutes. NaN for censored units."
            ),
            "null_for_units": censored_units,
            "rtf_range": _float_range(rtf_labels, "rul_cycles"),
            "rtf_at_failure": 0,
        })

    # rul_frac — optional
    if emit_rul_frac and "rul_frac" in rul_labels.columns:
        label_columns.append({
            "name": "rul_frac",
            "dtype": "float64",
            "unit": "fraction",
            "description": (
                "Fractional remaining life (rul_minutes / total_life_minutes). "
                "Normalized per-unit — never pooled across units. "
                "Range [0, 1]. NaN for censored units."
            ),
            "null_for_units": censored_units,
            "rtf_range": _float_range(rtf_labels, "rul_frac"),
            "rtf_at_failure": 0.0,
        })

    # is_censored
    label_columns.append({
        "name": "is_censored",
        "dtype": "bool",
        "unit": None,
        "description": (
            "True for right-censored units (surviving bearings without failure event). "
            "These rows have NULL ground-truth RUL and must not be used as supervision."
        ),
        "null_for_units": [],
        "true_for_units": censored_units,
        "false_for_units": rtf_units,
    })

    dictionary = {
        "schema_version": "1.0",
        "label_version": rul_cfg.get("label_version", "1.0"),
        "rul_definition": {
            "formula": "RUL(t) = (t_fail - t).total_seconds() / 60",
            "t_fail_definition": "max(timestamp) of that unit's valid acquisition files",
            "failure_model": "instantaneous at final acquisition timestamp",
            "primary_unit": rul_cfg.get("rul_primary_unit", "minutes"),
        },
        "unit_classification": {
            "rtf_units": rtf_units,
            "censored_units": censored_units,
        },
        "label_columns": label_columns,
        "per_unit_statistics": _per_unit_statistics(rul_labels, rtf_units),
    }

    logger.info("Label dictionary built: %d label columns", len(label_columns))
    return dictionary


def _float_range(df: pd.DataFrame, col: str) -> Optional[Dict[str, float]]:
    """Compute min/max for a float column; return None if column absent or all NaN."""
    if col not in df.columns:
        return None
    valid = df[col].dropna()
    if valid.empty:
        return None
    return {"min": float(valid.min()), "max": float(valid.max())}


def _per_unit_statistics(
    rul_labels: pd.DataFrame, rtf_units: List[str]
) -> Dict[str, Any]:
    """Compute per-unit label statistics for RTF units."""
    stats = {}
    for unit_id in rtf_units:
        unit_rows = rul_labels[rul_labels["unit_id"] == unit_id]
        # Take one channel to avoid double-counting (both channels have identical timestamps)
        first_ch = sorted(unit_rows["channel"].unique())[0]
        ch_rows = unit_rows[unit_rows["channel"] == first_ch]
        if "rul_minutes" not in ch_rows.columns or ch_rows["rul_minutes"].isna().all():
            stats[unit_id] = {"error": "no rul_minutes data"}
            continue
        stats[unit_id] = {
            "total_life_minutes": float(ch_rows["rul_minutes"].max()),
            "total_cycles": int(ch_rows["rul_minutes"].notna().sum()),
            "min_rul_minutes": float(ch_rows["rul_minutes"].min()),
            "max_rul_minutes": float(ch_rows["rul_minutes"].max()),
        }
    return stats


def save_label_dictionary(
    dictionary: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write label dictionary to JSON.

    Args:
        dictionary: Label dictionary payload.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(
            json.dumps(dictionary, indent=2, default=str), encoding="utf-8"
        )
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Label dictionary saved: %s", output_path)

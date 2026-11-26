"""Dataset manifest generator for Phase 5 artifacts.

Produces dataset_manifest.json — a machine-readable provenance record for the
training dataset. Contains row counts, column counts, file sizes, SHA-256
fingerprints, config fingerprint, and unit-level decomposition.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import get_censored_units, get_rtf_units
from src.rul_target.split_generator import get_fold_ids


def build_dataset_manifest(
    training_dataset: pd.DataFrame,
    rul_labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    output_paths: Dict[str, Path],
    config_fingerprint: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the dataset manifest metadata object.

    Args:
        training_dataset: The joined RTF training dataset.
        rul_labels: Labels DataFrame (all units).
        fold_assignments: Fold assignments DataFrame (RTF units).
        config: Full project configuration dictionary.
        output_paths: Dict mapping artifact names to their resolved Path objects.
        config_fingerprint: SHA-256 of serialized config (from Phase 3/4 pattern).
        logger: Pipeline logger.

    Returns:
        Dictionary representing the dataset_manifest.json payload.
    """
    rul_cfg = config.get("rul_target", {})
    rtf_units: List[str] = get_rtf_units(config)
    censored_units: List[str] = get_censored_units(config)
    fold_ids = get_fold_ids(fold_assignments)

    # Per-unit decomposition
    per_unit: Dict[str, Any] = {}
    for unit_id in rtf_units + censored_units:
        unit_rows = training_dataset[training_dataset["unit_id"] == unit_id] \
            if unit_id in rtf_units else pd.DataFrame()
        label_rows = rul_labels[rul_labels["unit_id"] == unit_id]
        is_rtf = unit_id in rtf_units
        per_unit[unit_id] = {
            "type": "rtf" if is_rtf else "censored",
            "training_rows": int(len(unit_rows)),
            "label_rows": int(len(label_rows)),
            **{
                f"{fold_id}_role": _get_unit_role(fold_assignments, unit_id, f"{fold_id}_role")
                for fold_id in fold_ids
            },
        }

    # Fold decomposition
    fold_summary: Dict[str, Any] = {}
    for fold_id in fold_ids:
        role_col = f"{fold_id}_role"
        fold_summary[fold_id] = {
            "train_rows": int((fold_assignments[role_col] == "train").sum()),
            "test_rows": int((fold_assignments[role_col] == "test").sum()),
            "train_units": sorted(
                fold_assignments[fold_assignments[role_col] == "train"]["unit_id"].unique().tolist()
            ),
            "test_units": sorted(
                fold_assignments[fold_assignments[role_col] == "test"]["unit_id"].unique().tolist()
            ),
        }

    # File fingerprints for already-written artifacts
    fingerprints: Dict[str, Optional[str]] = {}
    for name, path in output_paths.items():
        if path.exists():
            fingerprints[name] = _sha256_file(path)
        else:
            fingerprints[name] = None

    manifest = {
        "manifest_version": "1.0",
        "dataset_version": rul_cfg.get("dataset_version", "1.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint,
        "split_protocol": rul_cfg.get("split", {}).get("protocol", "lobo_cv"),
        "rtf_units": rtf_units,
        "censored_units": censored_units,
        "total_training_rows": int(len(training_dataset)),
        "total_training_columns": int(len(training_dataset.columns)),
        "feature_columns": sorted([
            c for c in training_dataset.columns
            if c.startswith(("td_", "fd_", "trend_", "env_", "cross_"))
        ]),
        "label_columns": sorted([
            c for c in training_dataset.columns
            if c in ("rul_minutes", "rul_cycles", "rul_frac", "is_censored")
            or c.endswith("_role")
        ]),
        "grain_columns": ["unit_id", "channel", "timestamp"],
        "per_unit": per_unit,
        "fold_summary": fold_summary,
        "artifact_fingerprints": fingerprints,
    }

    logger.info(
        "Dataset manifest built: %d training rows, %d columns, %d feature columns",
        len(training_dataset),
        len(training_dataset.columns),
        len(manifest["feature_columns"]),
    )
    return manifest


def _get_unit_role(
    fold_assignments: pd.DataFrame, unit_id: str, role_col: str
) -> Optional[str]:
    """Extract the fold role for a given unit_id."""
    unit_rows = fold_assignments[fold_assignments["unit_id"] == unit_id]
    if unit_rows.empty or role_col not in unit_rows.columns:
        return None
    roles = unit_rows[role_col].dropna().unique()
    if len(roles) == 0:
        return None
    if len(roles) == 1:
        return str(roles[0])
    return f"MULTIPLE:{sorted(roles)}"  # Should not happen; leakage validator catches this


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_dataset_manifest(
    manifest: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write dataset manifest to JSON.

    Args:
        manifest: Dataset manifest payload.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Dataset manifest saved: %s", output_path)

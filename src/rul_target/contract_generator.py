"""Phase 5 → Phase 6 contract generator.

Produces phase5_dataset_contract.yaml — machine-readable interface contract
that Phase 6 must consume without modification. This contract is the
authoritative handoff document between phases.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from src.config import get_censored_units, get_rtf_units
from src.rul_target.split_generator import get_fold_ids


def build_phase6_contract(
    training_dataset: pd.DataFrame,
    rul_labels: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    config_fingerprint: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build the Phase 5 → Phase 6 dataset contract.

    Args:
        training_dataset: The joined RTF training dataset.
        rul_labels: Labels DataFrame (all units).
        fold_assignments: Fold assignments DataFrame (RTF units).
        config: Full project configuration dictionary.
        config_fingerprint: SHA-256 of serialized config.
        logger: Pipeline logger.

    Returns:
        Dictionary representing phase5_dataset_contract.yaml payload.
    """
    rul_cfg = config.get("rul_target", {})
    output_cfg = rul_cfg.get("output_paths", {})

    rtf_units: List[str] = get_rtf_units(config)
    censored_units: List[str] = get_censored_units(config)

    # Determine which label columns exist
    label_cols_present = [
        c for c in ("rul_minutes", "rul_cycles", "rul_frac")
        if c in training_dataset.columns
    ]

    feature_cols = sorted([
        c for c in training_dataset.columns
        if c.startswith(("td_", "fd_", "trend_", "env_", "cross_"))
    ])

    fold_info: Dict[str, Any] = {}
    for fold_id in get_fold_ids(fold_assignments):
        role_col = f"{fold_id}_role"
        fold_info[fold_id] = {
            "train_units": sorted(
                fold_assignments[fold_assignments[role_col] == "train"]["unit_id"].unique().tolist()
            ),
            "test_units": sorted(
                fold_assignments[fold_assignments[role_col] == "test"]["unit_id"].unique().tolist()
            ),
        }

    contract = {
        "contract_version": "1.0",
        "phase": "5",
        "produced_by": "Phase 5 RUL Target Construction & Leakage Prevention",
        "consumed_by": "Phase 6 Feature Scaling, Model Training & Evaluation",
        "config_fingerprint": config_fingerprint,
        "dataset_version": rul_cfg.get("dataset_version", "1.0"),
        "label_version": rul_cfg.get("label_version", "1.0"),

        "artifacts": {
            "training_dataset": {
                "path": output_cfg.get("training_dataset", "data/processed/training_dataset.parquet"),
                "format": "parquet",
                "grain": ["unit_id", "channel", "timestamp"],
                "units": rtf_units,
                "total_rows": int(len(training_dataset)),
                "total_columns": int(len(training_dataset.columns)),
            },
            "rul_labels": {
                "path": output_cfg.get("rul_labels", "data/processed/rul_labels.parquet"),
                "format": "parquet",
                "units_included": rtf_units + censored_units,
                "rtf_units": rtf_units,
                "censored_units": censored_units,
            },
            "fold_assignments": {
                "path": output_cfg.get("fold_assignments", "data/processed/splits/fold_assignments.parquet"),
                "format": "parquet",
                "protocol": rul_cfg.get("split", {}).get("protocol", "lobo_cv"),
                "folds": fold_info,
            },
        },

        "columns": {
            "grain": ["unit_id", "channel", "timestamp"],
            "features": feature_cols,
            "labels": label_cols_present,
            "metadata": ["is_censored", "dataset_id", "bearing_id"],
            "fold_roles": [f"{fold_id}_role" for fold_id in fold_info.keys()],
        },

        "rul_specification": {
            "primary_target": "rul_minutes",
            "definition": "RUL(t) = (t_fail - t).total_seconds() / 60",
            "t_fail": "max(timestamp) of RTF unit's valid acquisition files",
            "value_at_failure": 0.0,
            "unit": "minutes",
            "monotone_constraint": "non_increasing",
        },

        "phase6_requirements": {
            "scaling": {
                "instruction": (
                    "Feature scaling MUST be fit on the training fold only within each LOBO fold. "
                    "Do NOT fit scaler on the full training_dataset. "
                    "Scaler must be applied identically to the test fold using the train-fold parameters."
                ),
                "forbidden": ["fit_on_full_dataset", "fit_on_test_fold"],
            },
            "label_handling": {
                "censored_rows": (
                    f"Censored units ({', '.join(censored_units)}) have NaN labels. "
                    "Do NOT impute, fabricate, or include them in supervised loss. "
                    "They may be used for unsupervised or semi-supervised methods "
                    "only with explicit architectural justification."
                ),
                "primary_supervision_signal": "rul_minutes",
            },
            "split_protocol": {
                "must_use": f"{', '.join(f'{fid}_role' for fid in fold_info.keys())} columns in training_dataset.parquet",
                "forbidden": [
                    "random_row_split",
                    "time_based_row_split",
                    "recomputing_splits_independently",
                ],
            },
            "leakage_prevention": {
                "inherited": (
                    "All L-01 through L-09 checks were verified by Phase 5 leakage_validator. "
                    "Phase 6 must not introduce new leakage vectors. "
                    "In particular: no global statistics on input features before fold-aware split."
                ),
            },
        },

        "statistical_limitations": {
            "n_rtf_units": len(rtf_units),
            "n_folds": len(fold_info),
            "caveat": (
                f"With only {len(rtf_units)} RTF unit{'s' if len(rtf_units) != 1 else ''}, "
                f"LOBO CV produces {len(fold_info)} evaluation data point{'s' if len(fold_info) != 1 else ''}. "
                "Performance estimates cannot carry classical confidence intervals. "
                "Results should be interpreted as methodology validation, not fleet-level guarantees."
            ),
        },
    }

    logger.info(
        "Phase 6 contract built: %d feature cols, labels=%s, %d folds",
        len(feature_cols),
        label_cols_present,
        len(fold_info),
    )
    return contract


def save_phase6_contract(
    contract: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write Phase 6 contract to YAML.

    Args:
        contract: Contract payload dictionary.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".yaml.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                contract,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=True,
            )
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Phase 6 contract saved: %s", output_path)

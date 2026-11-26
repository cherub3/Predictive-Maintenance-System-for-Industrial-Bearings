"""Load Phase 5 training dataset and fold assignments; defensively re-assert
unit-disjointness (architecture §4, §12).

Phase 6 never recomputes splits — it consumes `fold_assignments.parquet`
verbatim and only re-checks (never re-derives) unit-disjointness as a
defensive guard against upstream drift.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def load_training_dataset(config: Dict[str, Any], project_root: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load `training_dataset.parquet` produced by Phase 5.

    Args:
        config: Full project configuration dictionary.
        project_root: Absolute project root path.
        logger: Pipeline logger.

    Returns:
        Training dataset DataFrame.
    """
    rul_cfg = config.get("rul_target", {})
    path_cfg = rul_cfg.get("output_paths", {})
    path = project_root / path_cfg.get("training_dataset", "data/processed/training_dataset.parquet")

    if not path.exists():
        raise FileNotFoundError(f"training_dataset.parquet not found: {path}. Run Phase 5 first.")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Training dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def load_fold_assignments(config: Dict[str, Any], project_root: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load `fold_assignments.parquet` produced by Phase 5.

    Args:
        config: Full project configuration dictionary.
        project_root: Absolute project root path.
        logger: Pipeline logger.

    Returns:
        Fold assignments DataFrame.
    """
    rul_cfg = config.get("rul_target", {})
    path_cfg = rul_cfg.get("output_paths", {})
    path = project_root / path_cfg.get("fold_assignments", "data/processed/splits/fold_assignments.parquet")

    if not path.exists():
        raise FileNotFoundError(f"fold_assignments.parquet not found: {path}. Run Phase 5 first.")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info("Fold assignments loaded: %d rows", len(df))
    return df


def assert_unit_disjoint(fold_assignments: pd.DataFrame, fold_role_cols: List[str], logger: logging.Logger) -> None:
    """Defensively re-assert that no unit appears in both train and test for any fold.

    This does not recompute the split — it only guards against upstream drift
    (e.g. a corrupted or hand-edited fold_assignments.parquet).

    Keyed on `unit_id` (the qualified `dataset_id::bearing_id` string), NOT bare
    `bearing_id` — the same bearing_id string (e.g. "bearing_3") can refer to two
    physically unrelated bearings from different datasets that are BOTH present
    in the training dataset (e.g. Set 1's bearing_3 and Set 3's bearing_3 are
    both RTF units). Checking bare bearing_id would false-positive on this
    legitimate case, or worse, silently contaminate train with a test unit's
    rows if the isin() filter downstream used the same bare column.

    Args:
        fold_assignments: Fold assignments DataFrame, with a `unit_id` column.
        fold_role_cols: List of fold-role column names (e.g. ['fold_1_role', 'fold_2_role']).
        logger: Pipeline logger.

    Raises:
        ValueError: If any unit appears in both roles for the same fold.
    """
    for role_col in fold_role_cols:
        train_units = set(fold_assignments.loc[fold_assignments[role_col] == "train", "unit_id"].unique())
        test_units = set(fold_assignments.loc[fold_assignments[role_col] == "test", "unit_id"].unique())
        overlap = train_units & test_units
        if overlap:
            raise ValueError(
                f"Unit-disjointness violation in '{role_col}': unit(s) {sorted(overlap)} "
                "appear in BOTH train and test roles. This is a leakage-critical failure "
                "and must never occur with a correctly-generated Phase 5 fold_assignments.parquet."
            )
    logger.info("Unit-disjointness re-asserted for all folds ✓ (no unit in both roles)")


def get_fold_bearings(
    fold_assignments: pd.DataFrame, role_col: str
) -> Tuple[List[str], List[str]]:
    """Return (train_units, test_units) for a given fold-role column.

    Despite the name (kept for call-site compatibility), this returns qualified
    unit_ids, not bare bearing_ids — bearing_id alone is ambiguous across datasets.

    Args:
        fold_assignments: Fold assignments DataFrame, with a `unit_id` column.
        role_col: Fold-role column name (e.g. 'fold_1_role').

    Returns:
        Tuple of (sorted train unit_ids, sorted test unit_ids).
    """
    train_units = sorted(fold_assignments.loc[fold_assignments[role_col] == "train", "unit_id"].unique().tolist())
    test_units = sorted(fold_assignments.loc[fold_assignments[role_col] == "test", "unit_id"].unique().tolist())
    return train_units, test_units

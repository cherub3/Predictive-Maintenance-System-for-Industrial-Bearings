"""LOBO (Leave-One-Bearing-Out) fold assignment generator.

Implements the approved splitting strategy (Architecture §10), generalized to
however many RTF units are registered: one fold per RTF unit, where that unit
is held out as test and every other RTF unit trains. With 2 RTF units this is
the original 2-fold scheme; with N RTF units it produces N folds.

Only RTF units participate in the labelled split roles. Censored units
are never assigned a fold role. The assignment is deterministic and
enumerated — no randomness.

Output grain: (unit_id, channel, timestamp) with fold_1_role .. fold_N_role.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import get_censored_units, get_rtf_units


def get_fold_ids(fold_assignments: pd.DataFrame) -> List[str]:
    """Extracts the sorted list of fold_ids (e.g. ['fold_1', 'fold_2', ...]) present
    as `{fold_id}_role` columns in a fold_assignments/training_dataset DataFrame.

    Sorted numerically (fold_1, fold_2, ..., fold_10, not lexicographically), so
    downstream consumers iterate folds in a stable, human-expected order.
    """
    fold_ids = []
    for col in fold_assignments.columns:
        m = re.fullmatch(r"(fold_\d+)_role", col)
        if m:
            fold_ids.append(m.group(1))
    return sorted(fold_ids, key=lambda f: int(f.split("_")[1]))


def _build_lobo_folds(rtf_units: List[str]) -> List[Tuple[str, List[str], str]]:
    """Builds the LOBO fold definitions: one fold per RTF unit.

    Args:
        rtf_units: Sorted list of qualified RTF unit_ids.

    Returns:
        List of (fold_id, train_units, test_unit) tuples, fully enumerated.
    """
    folds = []
    for i, test_unit in enumerate(rtf_units, start=1):
        train_units = [u for u in rtf_units if u != test_unit]
        folds.append((f"fold_{i}", train_units, test_unit))
    return folds


def generate_fold_assignments(
    rul_labels: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Assign LOBO fold roles to every labelled RTF row.

    Censored rows are excluded from fold assignment — they never appear
    in any training or evaluation role.

    Args:
        rul_labels: Labels DataFrame from label_generator (includes censored rows).
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        DataFrame with grain (unit_id, channel, timestamp) and columns
        fold_1_role .. fold_N_role (N = number of RTF units), each taking
        values "train" or "test". Contains only RTF (non-censored) rows.
    """
    rul_cfg = config.get("rul_target", {})
    rtf_units: List[str] = get_rtf_units(config)
    protocol: str = rul_cfg.get("split", {}).get("protocol", "lobo_cv")

    logger.info(f"Generating fold assignments — protocol: {protocol}, RTF units: {rtf_units}")

    if protocol != "lobo_cv":
        raise ValueError(
            f"Unsupported split protocol '{protocol}'; only 'lobo_cv' is implemented"
        )

    folds = _build_lobo_folds(rtf_units)

    # Work only with RTF (labelled) rows
    labelled_rows = rul_labels[~rul_labels["is_censored"]].copy()

    if labelled_rows.empty:
        raise ValueError("No labelled (RTF) rows available for fold assignment")

    for fold_id, train_units, test_unit in folds:
        role_col = f"{fold_id}_role"
        labelled_rows[role_col] = pd.NA
        train_mask = labelled_rows["unit_id"].isin(train_units)
        test_mask = labelled_rows["unit_id"] == test_unit
        labelled_rows.loc[train_mask, role_col] = "train"
        labelled_rows.loc[test_mask, role_col] = "test"
        logger.info(
            f"  {fold_id}: train={train_units} ({train_mask.sum()} rows), "
            f"test={test_unit} ({test_mask.sum()} rows)"
        )

        unassigned = labelled_rows[role_col].isna().sum()
        if unassigned > 0:
            raise ValueError(
                f"Unassigned rows after fold generation for {fold_id}: {unassigned}. "
                "Check that RTF units in the datasets registry match unit_ids in the feature store."
            )

    # Retain only grain + descriptive + fold role columns
    role_cols = [f"{fold_id}_role" for fold_id, _, _ in folds]
    output = labelled_rows[
        ["unit_id", "dataset_id", "bearing_id", "channel", "timestamp"] + role_cols
    ].sort_values(["unit_id", "channel", "timestamp"]).reset_index(drop=True)

    logger.info(
        f"Fold assignment complete: {len(output)} RTF rows assigned across "
        f"{len(folds)} folds"
    )
    return output


def save_fold_assignments(
    fold_assignments: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Atomically write fold assignments to parquet.

    Args:
        fold_assignments: Fold assignment DataFrame.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    try:
        fold_assignments.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info(f"Fold assignments saved: {output_path}")


def generate_split_summary(
    fold_assignments: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Generate the split_summary.md document.

    Includes fold composition, row counts, the LOBO rationale, and the
    honesty caveat (n = number of RTF units) as required by Architecture §10.4.

    Args:
        fold_assignments: Fold assignment DataFrame.
        config: Full project configuration dictionary.
        logger: Pipeline logger.

    Returns:
        Markdown string for split_summary.md.
    """
    rtf_units: List[str] = get_rtf_units(config)
    censored_units: List[str] = get_censored_units(config)
    folds = _build_lobo_folds(rtf_units)
    n = len(rtf_units)

    lines = [
        "# Phase 5 — Split Summary",
        "",
        "**Protocol:** Leave-One-Bearing-Out (LOBO) Cross-Validation",
        "**RTF units:** " + ", ".join(rtf_units),
        "**Censored units (excluded from all folds):** " + ", ".join(censored_units),
        "",
        "---",
        "",
        f"## ⚠️ Honesty Caveat (n = {n})",
        "",
        f"This dataset contains **{n} run-to-failure bearing trajector{'y' if n == 1 else 'ies'}**.",
        f"LOBO CV therefore consists of exactly {n} fold{'s' if n != 1 else ''}, each training on",
        "the other trajectories and testing on one held-out trajectory. Any reported metric is",
        f"the result of {n} evaluation{'s' if n != 1 else ''}, one per held-out unit.",
        "",
        "Performance estimates from this evaluation:",
        f"- **{'Cannot' if n <= 2 else 'May not reliably'}** carry a classical confidence interval (n = {n} is small)",
        "- **Should be interpreted qualitatively** (does the model track the trend and",
        "  approximate magnitude of degradation?), not as a fleet-level accuracy guarantee",
        "- Represent a **methodology validation**, not a production performance claim",
        "",
        "---",
        "",
        "## Fold Definitions",
        "",
        "| Fold | Train units | Test unit |",
        "|------|-------------|-----------|",
    ]
    for fold_id, train_units, test_unit in folds:
        lines.append(f"| {fold_id} | {', '.join(train_units)} | {test_unit} |")
    lines.append("")

    # Fold composition row counts
    lines += ["## Fold Composition", ""]
    for fold_id, train_units, test_unit in folds:
        role_col = f"{fold_id}_role"
        train_count = (fold_assignments[role_col] == "train").sum()
        test_count = (fold_assignments[role_col] == "test").sum()
        train_seen = sorted(
            fold_assignments[fold_assignments[role_col] == "train"]["unit_id"].unique()
        )
        test_seen = sorted(
            fold_assignments[fold_assignments[role_col] == "test"]["unit_id"].unique()
        )
        overlap = set(train_seen) & set(test_seen)
        lines += [
            f"### {fold_id}",
            f"- Train rows: {train_count} (units: {train_seen})",
            f"- Test rows: {test_count} (units: {test_seen})",
            f"- Unit overlap: {sorted(overlap) if overlap else '∅ (none — correct)'}",
            "",
        ]

    lines += [
        "## Split Strategy Justification",
        "",
        "LOBO was selected because it is the only strategy that:",
        "1. Is leakage-free by construction (no unit appears in both train and test within a fold)",
        "2. Answers the correct generalization question: can a model trained on other failed",
        "   bearings predict the RUL of a completely unseen failed bearing?",
        "3. Uses every available failure mode as a test case across its folds",
        "",
        "All alternatives (random row split, time-based within-unit, last-20% holdout,",
        "expanding window, k-fold ignoring groups) were considered and rejected.",
        "See Architecture §10.3 for the full comparative justification.",
        "",
        "---",
        "_Generated by Phase 5 split_generator. Do not edit manually._",
    ]

    return "\n".join(lines)


def save_split_summary(summary_md: str, output_path: Path, logger: logging.Logger) -> None:
    """Atomically write split summary markdown.

    Args:
        summary_md: Markdown string.
        output_path: Final destination path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(summary_md, encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info(f"Split summary saved: {output_path}")

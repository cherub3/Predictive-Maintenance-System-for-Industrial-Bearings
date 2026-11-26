"""Phase 6 contract loader — validates Phase 6 artifacts at pipeline entry.

Architecture §5 and §7.1. Implements the entry gate: schema_version, n2_caveat,
prediction grain, and model.type must all be present before any decision logic
executes. Missing or invalid fields raise FATAL errors.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


@dataclass
class ContractValidationResult:
    """Result of Phase 6 contract entry-gate validation. Architecture §7.1."""

    passed: bool
    fatals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model_type: str = ""
    n2_caveat: bool = False
    n2_caveat_text: str = ""
    feature_columns: List[str] = field(default_factory=list)
    mean_mae_minutes: float = 0.0
    stability_ratio: float = 1.0
    experiment_id: str = ""
    phase6_config_fingerprint: str = ""
    phase6_model_fingerprint: str = ""
    top_permutation_ranked_features: List[str] = field(default_factory=list)
    expected_bearings: List[str] = field(default_factory=list)
    n_rtf_units: int = 0


def load_phase6_contract(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load phase6_model_contract.yaml. Architecture §5.

    Args:
        path: Path to the contract YAML file.
        logger: Pipeline logger.

    Returns:
        Parsed contract dict.

    Raises:
        FileNotFoundError: If the contract file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 6 contract not found at '{path}'. "
            "Phase 7 requires the approved Phase 6 contract to exist."
        )
    with path.open(encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    logger.info("Phase 6 contract loaded from: %s", path)
    return contract


def load_phase6_predictions(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load model_predictions.parquet. Architecture §5.

    Args:
        path: Path to the parquet file.
        logger: Pipeline logger.

    Returns:
        DataFrame at grain (unit_id, channel, timestamp).

    Raises:
        FileNotFoundError: If the predictions file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 6 predictions not found at '{path}'. "
            "Phase 7 requires approved Phase 6 predictions."
        )
    df = pd.read_parquet(path)
    logger.info(
        "Phase 6 predictions loaded: %d rows, columns=%s", len(df), df.columns.tolist()
    )
    return df


def load_phase6_manifest(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Load model_manifest.json for upstream lineage. Architecture §5.

    Args:
        path: Path to the JSON manifest.
        logger: Pipeline logger.

    Returns:
        Parsed manifest dict. Returns empty dict if file is missing (WARNING).
    """
    import json

    if not path.exists():
        logger.warning(
            "Phase 6 manifest not found at '%s'. Upstream lineage will be partial.", path
        )
        return {}
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info("Phase 6 manifest loaded from: %s", path)
    return manifest


def validate_phase6_contract(
    contract: Dict[str, Any],
    predictions: pd.DataFrame,
    logger: logging.Logger,
) -> ContractValidationResult:
    """Run Phase 6 contract entry-gate checks. Architecture §5c.

    FATAL conditions (pipeline must not proceed):
      - schema_version absent or != "1.0"
      - performance.n2_caveat absent or not True
      - prediction_schema.grain not matching expected columns
      - model.type absent

    Args:
        contract: Parsed Phase 6 contract dict.
        predictions: Loaded predictions DataFrame.
        logger: Pipeline logger.

    Returns:
        ContractValidationResult with passed=True only if all FATAL checks pass.
    """
    fatals: List[str] = []
    warnings: List[str] = []

    # Check schema_version
    schema_version = contract.get("schema_version")
    if schema_version != "1.0":
        fatals.append(
            f"schema_version is '{schema_version}' (expected '1.0'). "
            "Phase 7 contract entry gate failed."
        )

    # Check n2_caveat
    performance = contract.get("performance", {})
    n2_caveat = performance.get("n2_caveat")
    if n2_caveat is not True:
        fatals.append(
            f"performance.n2_caveat is '{n2_caveat}' (expected True). "
            "Phase 7 requires n2_caveat to be declared in Phase 6 contract."
        )

    # Check prediction grain
    expected_grain = ["unit_id", "channel", "timestamp"]
    declared_grain = contract.get("prediction_schema", {}).get("grain", [])
    if declared_grain != expected_grain:
        fatals.append(
            f"prediction_schema.grain is {declared_grain} "
            f"(expected {expected_grain})."
        )
    # Also verify actual DataFrame has these columns
    for col in expected_grain + ["predicted_rul_minutes"]:
        if col not in predictions.columns:
            fatals.append(
                f"predictions DataFrame is missing required column '{col}'."
            )

    # Check model.type
    model_type = contract.get("model", {}).get("type")
    if not model_type:
        fatals.append("model.type is absent from Phase 6 contract.")

    passed = len(fatals) == 0
    if passed:
        logger.info("Phase 6 contract entry gate: PASSED")
    else:
        logger.error("Phase 6 contract entry gate: FAILED — %d fatal(s)", len(fatals))
        for f in fatals:
            logger.error("  FATAL: %s", f)

    for w in warnings:
        logger.warning("  WARNING: %s", w)

    expected_bearings = (
        contract.get("model", {})
        .get("training_population", {})
        .get("rtf_units", [])
    )

    return ContractValidationResult(
        passed=passed,
        fatals=fatals,
        warnings=warnings,
        model_type=model_type or "",
        n2_caveat=bool(n2_caveat),
        n2_caveat_text=performance.get("n2_caveat_text", ""),
        feature_columns=contract.get("prediction_schema", {}).get("input_features", []),
        mean_mae_minutes=float(performance.get("mean_mae_minutes", 0.0)),
        stability_ratio=float(performance.get("stability_ratio", 1.0)),
        experiment_id=contract.get("experiment", {}).get("experiment_id", ""),
        phase6_config_fingerprint=contract.get("experiment", {}).get("config_fingerprint", ""),
        phase6_model_fingerprint=contract.get("model", {}).get("model_fingerprint", ""),
        top_permutation_ranked_features=contract.get("feature_importance", {}).get(
            "top_permutation_ranked_features", []
        ),
        expected_bearings=list(expected_bearings),
        n_rtf_units=int(contract.get("limitations", {}).get("n_rtf_units", len(expected_bearings))),
    )

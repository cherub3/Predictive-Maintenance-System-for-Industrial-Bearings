"""Phase 3 to Phase 4 machine-readable gating contract generator.

Generates reports/phase3_gating_decisions.yaml — the sole machine-readable
input consumed by Phase 4's feature_gate.py. Gate values are determined by
quantitative statistical tests on Phase 3 analysis results.

IMPORTANT: This file must never be manually edited. It is regenerated on every
Phase 3 execution. The contract schema version is "1.0".
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml


SCHEMA_VERSION = "1.0"
PRODUCER_PHASE = "phase_3_signal_analysis"


def _get_git_commit() -> str:
    """Retrieves the current short git commit SHA.

    Returns:
        str: 8-character short SHA, or 'unknown' if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_config_version(config: Dict[str, Any]) -> str:
    """Computes SHA256 fingerprint of the configuration dictionary.

    Args:
        config: Full project configuration dictionary.

    Returns:
        str: First 16 hex characters of SHA256(serialised_config).
    """
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]


def resolve_skewness_gate(
    health_df: pd.DataFrame,
    threshold: float,
    logger: logging.Logger,
) -> tuple:
    """Determines whether skewness provides statistically meaningful signal.

    Gate is True if mean absolute skewness across all successfully loaded
    rows exceeds the configured threshold.

    Args:
        health_df: Health indicators DataFrame.
        threshold: Absolute mean skewness threshold from config.
        logger: Pipeline logger.

    Returns:
        Tuple of (bool gate_value, str justification).
    """
    ok_rows = health_df[health_df["load_status"] == "OK"]
    if ok_rows.empty:
        return False, "No successfully loaded rows available for skewness analysis."

    mean_abs_skewness = float(ok_rows["skewness"].abs().mean())
    include = mean_abs_skewness > threshold
    justification = (
        f"Mean absolute skewness across all channels = {mean_abs_skewness:.4f}; "
        f"threshold = {threshold}. Gate set to {include}."
    )
    logger.info(f"Skewness gate: {include} (mean |skewness| = {mean_abs_skewness:.4f})")
    return include, justification


def resolve_rolling_variance_gate(
    health_df: pd.DataFrame,
    correlation_threshold: float,
    window: int,
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """Determines whether rolling variance adds information beyond RMS slope.

    Computes rolling RMS slope and rolling RMS variance for run-to-failure (RTF)
    bearings across every registered dataset. If their Pearson correlation is
    below the threshold, they carry independent information and the gate is True.

    Args:
        health_df: Health indicators DataFrame (with an `is_rtf` column).
        correlation_threshold: Pearson r threshold; gate True if r < threshold.
        window: Rolling window size (files).
        logger: Pipeline logger.

    Returns:
        Tuple of (bool gate_value, str justification).
    """
    failure_x = health_df[
        health_df["is_rtf"]
        & (health_df["axis"] == "X")
        & (health_df["load_status"] == "OK")
    ].sort_values("timestamp")

    if len(failure_x) < window * 2:
        msg = "Insufficient data for rolling variance gate analysis."
        if logger:
            logger.warning(msg)
        return False, msg

    # Rolling slope/variance computed WITHIN each dataset's own RTF timeline, then
    # concatenated — datasets are non-overlapping runs, so a rolling window must
    # never span a dataset boundary (that would compare, e.g., Set 1's last RTF
    # reading in Nov 2003 against Set 2's first RTF reading in Feb 2004).
    combined_parts = []
    for dataset_id, group in failure_x.groupby("dataset_id"):
        rms_series = group.groupby("timestamp")["rms"].mean()
        rolling_slope = rms_series.diff(window)
        rolling_var = rms_series.rolling(window).std()
        combined_parts.append(pd.DataFrame({"slope": rolling_slope, "variance": rolling_var}))

    combined = pd.concat(combined_parts).dropna() if combined_parts else pd.DataFrame(columns=["slope", "variance"])
    if len(combined) < 5:
        msg = "Insufficient data after rolling computation; gate defaulting to False."
        return False, msg

    correlation = float(combined["slope"].corr(combined["variance"]))
    include = abs(correlation) < correlation_threshold
    justification = (
        f"Pearson r(rolling_slope, rolling_variance) = {correlation:.4f}; "
        f"threshold = {correlation_threshold}. Gate set to {include}."
    )
    if logger:
        logger.info(f"Rolling variance gate: {include} (r = {correlation:.4f})")
    return include, justification


def resolve_envelope_gate(
    bearing_geometry_verified: bool,
    logger: logging.Logger,
) -> tuple:
    """Applies the Bearing Geometry Safety Rule for envelope features.

    Envelope features are excluded entirely when bearing geometry cannot be
    verified. This is a non-negotiable safety rule — geometry is never estimated.

    Args:
        bearing_geometry_verified: True if Phase 3 successfully loaded
            and confirmed bearing geometry parameters.
        logger: Pipeline logger.

    Returns:
        Tuple of (bool gate_value, str justification).
    """
    if bearing_geometry_verified:
        justification = (
            "Bearing geometry verified. Envelope features enabled pending "
            "signal characteristic analysis."
        )
        logger.info("Envelope gate: True (geometry verified)")
        return True, justification
    else:
        justification = (
            "Bearing Geometry Safety Rule applied: bearing geometry not verified. "
            "Envelope features excluded entirely. Geometry is never estimated."
        )
        logger.info("Envelope gate: False (Bearing Geometry Safety Rule)")
        return False, justification


def resolve_xy_energy_ratio_gate(
    health_df: pd.DataFrame,
    cv_threshold: float,
    logger: logging.Logger,
) -> tuple:
    """Determines whether X/Y energy asymmetry is statistically meaningful.

    Computes the coefficient of variation (CV) of the X-to-Y RMS ratio
    across the run timeline for failure bearings. High CV indicates sustained
    directional asymmetry that the cross-channel feature can exploit.

    Naturally scoped to paired_xy datasets only: single_axis datasets (Set 2/3)
    have no Y-channel rows in health_df at all, so the Y-side of the merge below
    is empty for them and they contribute nothing to the CV — the gate is
    structurally a paired_xy-only computation without needing an explicit filter.

    Args:
        health_df: Health indicators DataFrame.
        cv_threshold: CV threshold above which asymmetry is considered meaningful.
        logger: Pipeline logger.

    Returns:
        Tuple of (bool gate_value, str justification).
    """
    ok_rows = health_df[health_df["load_status"] == "OK"]

    x_rms = (
        ok_rows[ok_rows["axis"] == "X"]
        .groupby(["timestamp", "bearing_id"])["rms"]
        .mean()
        .reset_index()
        .rename(columns={"rms": "rms_x"})
    )
    y_rms = (
        ok_rows[ok_rows["axis"] == "Y"]
        .groupby(["timestamp", "bearing_id"])["rms"]
        .mean()
        .reset_index()
        .rename(columns={"rms": "rms_y"})
    )

    merged = x_rms.merge(y_rms, on=["timestamp", "bearing_id"])
    merged = merged[merged["rms_y"] > 0]

    if merged.empty:
        msg = "No valid X/Y pairs for energy ratio analysis; gate defaulting to False."
        logger.warning(msg)
        return False, msg

    merged["xy_ratio"] = merged["rms_x"] / merged["rms_y"]
    cv = float(merged["xy_ratio"].std() / merged["xy_ratio"].mean())
    include = cv > cv_threshold
    justification = (
        f"X/Y RMS ratio CV = {cv:.4f} across all bearings and timestamps; "
        f"threshold = {cv_threshold}. Gate set to {include}."
    )
    logger.info(f"XY energy ratio gate: {include} (CV = {cv:.4f})")
    return include, justification


def build_gating_contract(
    health_df: pd.DataFrame,
    config: Dict[str, Any],
    bearing_geometry_verified: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Builds the complete gating contract dictionary from analysis results.

    Args:
        health_df: Health indicators DataFrame.
        config: Full project configuration dictionary.
        bearing_geometry_verified: Whether bearing geometry was confirmed.
        logger: Pipeline logger.

    Returns:
        Dict: Complete gating contract in the schema_version 1.0 structure.
    """
    thresholds = config["signal_analysis"]["gate_thresholds"]

    skewness_include, skewness_just = resolve_skewness_gate(
        health_df,
        threshold=float(thresholds["skewness_abs_mean_threshold"]),
        logger=logger,
    )
    var_include, var_just = resolve_rolling_variance_gate(
        health_df,
        correlation_threshold=float(thresholds["rolling_variance_correlation_threshold"]),
        window=int(thresholds["rolling_variance_window_files"]),
        logger=logger,
    )
    env_include, env_just = resolve_envelope_gate(bearing_geometry_verified, logger)
    xy_include, xy_just = resolve_xy_energy_ratio_gate(
        health_df,
        cv_threshold=float(thresholds["xy_asymmetry_cv_threshold"]),
        logger=logger,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "phase": PRODUCER_PHASE,
            "module": "src/signal_analysis/main.py",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _get_git_commit(),
            "config_version": _compute_config_version(config),
        },
        "bearing_geometry": {
            "verified": bearing_geometry_verified,
            "source": "bearing_geometry.py" if bearing_geometry_verified else "not_loaded",
        },
        "gate_decisions": {
            "skewness": {
                "include": bool(skewness_include),
                "justification": skewness_just,
            },
            "rolling_variance": {
                "include": bool(var_include),
                "justification": var_just,
            },
            "envelope_features": {
                "include": bool(env_include),
                "justification": env_just,
            },
            "xy_energy_ratio": {
                "include": bool(xy_include),
                "justification": xy_just,
            },
        },
        "validation": {
            "all_gates_present": True,
        },
    }


def validate_contract(contract: Dict[str, Any]) -> bool:
    """Validates the gating contract against schema_version 1.0 requirements.

    Args:
        contract: Gating contract dictionary to validate.

    Returns:
        bool: True if all validation rules pass.

    Raises:
        ValueError: If a required field is missing or has an invalid type.
    """
    required_top = ["schema_version", "producer", "bearing_geometry", "gate_decisions", "validation"]
    for field in required_top:
        if field not in contract:
            raise ValueError(f"Missing required field: {field}")

    required_producer = ["phase", "module", "generated_at", "git_commit", "config_version"]
    for field in required_producer:
        if field not in contract["producer"]:
            raise ValueError(f"Missing producer field: {field}")

    if contract["producer"]["phase"] != PRODUCER_PHASE:
        raise ValueError(
            f"Producer phase mismatch: expected '{PRODUCER_PHASE}', "
            f"got '{contract['producer']['phase']}'"
        )

    required_gates = ["skewness", "rolling_variance", "envelope_features", "xy_energy_ratio"]
    for gate_name in required_gates:
        if gate_name not in contract["gate_decisions"]:
            raise ValueError(f"Missing gate decision: {gate_name}")
        gate = contract["gate_decisions"][gate_name]
        if not isinstance(gate.get("include"), bool):
            raise TypeError(
                f"Gate '{gate_name}' include must be bool, "
                f"got {type(gate.get('include'))}"
            )

    if not isinstance(contract["bearing_geometry"].get("verified"), bool):
        raise TypeError("bearing_geometry.verified must be bool")

    if contract["validation"].get("all_gates_present") is not True:
        raise ValueError("validation.all_gates_present must be True")

    return True


def write_gating_contract(
    contract: Dict[str, Any],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Writes the validated gating contract to disk atomically.

    Writes to a temporary file first, then renames to the final path.
    This prevents Phase 4 from reading a partial file on interruption.

    Args:
        contract: Validated gating contract dictionary.
        output_path: Final destination path for the YAML file.
        logger: Pipeline logger.

    Raises:
        ValueError: If contract fails pre-write validation.
    """
    validate_contract(contract)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".yaml.tmp")

    header = (
        "# phase3_gating_decisions.yaml\n"
        "# Machine-readable Phase 3 gate decisions for Phase 4 feature_gate.py.\n"
        "# DO NOT edit manually. Generated automatically by Phase 3 pipeline.\n"
        "# Human-readable findings: reports/engineering_findings.md\n\n"
    )

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(header)
            yaml.dump(
                contract,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        tmp_path.replace(output_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise exc

    logger.info(f"Gating contract written: {output_path}")
    gate_summary = {k: v["include"] for k, v in contract["gate_decisions"].items()}
    logger.info(f"Gate decisions: {gate_summary}")

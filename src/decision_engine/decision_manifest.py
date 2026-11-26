"""Decision manifest — governance artifact recording all Phase 7 run metadata.

Architecture §7.12 and §18a. Produced as decision_manifest.json. Contains:
  phase7 experiment_id, config_fingerprint, rule_version, git_commit,
  environment, phase6 upstream lineage, decision summary, statistical limitations.
"""

import importlib.metadata
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from src.utils import get_file_sha256

# Artifacts the dashboard loads and must verify against the manifest on boot.
# Paths are relative to the project root (the dashboard's working directory).
_DASHBOARD_ARTIFACTS = [
    "reports/recommendations.parquet",
    "reports/phase7_decision_contract.yaml",
    "reports/decision_report.md",
    "reports/split_summary.md",
    "reports/engineering_findings.md",
    "reports/evaluation_report.md",
    "reports/feature_importance_report.md",
    "reports/model_selection_report.md",
    "reports/decision_validation_report.md",
    "reports/leakage_report.md",
    "data/processed/feature_store.parquet",
]


def build_artifact_hashes(logger: logging.Logger) -> List[Dict[str, Any]]:
    """Compute SHA-256 hashes for every dashboard-consumed artifact.

    Recorded in the manifest so the dashboard can fail closed if any artifact is
    missing or does not match the hash captured at the end of the Phase 7 run.

    Args:
        logger: Pipeline logger.

    Returns:
        List of {"path": str, "sha256": str | None, "missing": bool} entries.
    """
    artifacts: List[Dict[str, Any]] = []
    for rel in _DASHBOARD_ARTIFACTS:
        p = Path(rel)
        if p.exists():
            artifacts.append({"path": rel, "sha256": get_file_sha256(p), "missing": False})
        else:
            logger.warning("Artifact missing while hashing manifest: %s", rel)
            artifacts.append({"path": rel, "sha256": None, "missing": True})
    logger.info("Manifest artifact hashes computed for %d artifact(s)", len(artifacts))
    return artifacts


def get_git_commit(logger: logging.Logger) -> str:
    """Return the current git commit SHA or 'unknown' if git is unavailable.

    Architecture §18a. This project has no .git repository; 'unknown' is the
    documented acceptable fallback (same convention as Phase 6 §18 manifest).

    Args:
        logger: Pipeline logger.

    Returns:
        Short git SHA string or 'unknown'.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            logger.info("Git commit: %s", commit)
            return commit
    except Exception:
        pass
    logger.info("Git commit: unknown (no .git repository or git unavailable)")
    return "unknown"


def capture_environment(logger: logging.Logger) -> Dict[str, str]:
    """Capture installed library versions. Architecture §18a.

    Args:
        logger: Pipeline logger.

    Returns:
        Dict mapping library name to version string.
    """
    libs = ["pandas", "numpy", "scikit-learn", "pyyaml", "pyarrow"]
    env: Dict[str, str] = {}
    for lib in libs:
        try:
            env[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            env[lib] = "unknown"
    logger.debug("Environment captured: %s", env)
    return env


def build_decision_manifest(
    experiment_id: str,
    config_fingerprint: str,
    rule_version: str,
    git_commit: str,
    environment: Dict[str, str],
    phase6_experiment_id: str,
    phase6_config_fingerprint: str,
    phase6_model_type: str,
    phase6_model_fingerprint: str,
    n2_caveat: bool,
    n_recommendations: int,
    n_bearings: int,
    risk_counts: Dict[str, int],
    override_counts: Dict[str, int],
    logger: logging.Logger,
    n_rtf_units: int = 2,
) -> Dict[str, Any]:
    """Build the complete decision_manifest.json payload. Architecture §18a.

    Args:
        experiment_id: Phase 7 experiment ID.
        config_fingerprint: SHA-256 of decision_engine config block.
        rule_version: Business rule set version.
        git_commit: Git SHA or 'unknown'.
        environment: Library versions dict.
        phase6_experiment_id: Phase 6 experiment ID.
        phase6_config_fingerprint: Phase 6 config fingerprint.
        phase6_model_type: Phase 6 selected model type.
        phase6_model_fingerprint: Phase 6 model fingerprint.
        n2_caveat: From Phase 6 contract.
        n_recommendations: Total recommendations produced.
        n_bearings: Number of distinct bearings in output.
        risk_counts: Count per risk category {str: int}.
        override_counts: Count per business rule ID {str: int}.
        logger: Pipeline logger.

    Returns:
        Dict representing the manifest payload.
    """
    from datetime import datetime, timezone

    manifest = {
        "phase": "7",
        "module": "decision_engine",
        "experiment_id": experiment_id,
        "config_fingerprint": config_fingerprint,
        "rule_version": rule_version,
        "git_commit": git_commit,
        "environment": environment,
        "phase6_upstream": {
            "experiment_id": phase6_experiment_id,
            "config_fingerprint": phase6_config_fingerprint,
            "model_type": phase6_model_type,
            "model_fingerprint": phase6_model_fingerprint,
        },
        "decision_summary": {
            "total_recommendations": n_recommendations,
            "bearing_count": n_bearings,
            "risk_category_counts": risk_counts,
            "business_rule_override_count": sum(override_counts.values()),
            "override_counts_by_rule": override_counts,
        },
        "statistical_limitations": {
            "n_rtf_units": n_rtf_units,
            "n2_caveat": n2_caveat,
            "model_selection_outcome": (
                "demotion_fallback" if phase6_model_type in ("mean", "median") else "promoted"
            ),
            "confidence_basis": "categorical_only_n2_insufficient_for_probabilities",
        },
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Decision manifest built: experiment_id=%s", experiment_id)
    return manifest


def save_decision_manifest(
    manifest: Dict[str, Any], output_path: Path, logger: logging.Logger
) -> None:
    """Atomically write decision_manifest.json. Architecture §18a.

    Args:
        manifest: Manifest payload dict.
        output_path: Target file path.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Decision manifest saved: %s", output_path)

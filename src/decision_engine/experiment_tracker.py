"""Experiment tracker — computes config fingerprint, experiment ID, and log record.

Architecture §7.13 and §18. Config fingerprint is SHA-256 of the decision_engine
config block (json.dumps, sort_keys=True). Experiment ID format:
  dec_{fp[:12]}_rv{rule_version}

No random seed: Phase 7 is fully deterministic (architecture §18c, §25 Q4).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict


def compute_config_fingerprint(config: Dict[str, Any]) -> str:
    """Compute SHA-256 of the decision_engine config block.

    Architecture §7.13, §18c. Any change to thresholds, priority matrix,
    rule parameters, or policy map produces a distinct fingerprint and thus
    a distinct experiment_id — preventing silent comparison of decisions
    produced under different configurations.

    Args:
        config: Full project configuration dict.

    Returns:
        Hex SHA-256 string of the serialised decision_engine block.
    """
    de_cfg = config.get("decision_engine", {})
    serialised = json.dumps(de_cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def generate_experiment_id(config_fingerprint: str, rule_version: str) -> str:
    """Generate a stable experiment ID. Architecture §7.13.

    Format: dec_{fp[:12]}_rv{rule_version}

    Args:
        config_fingerprint: Full SHA-256 hex string.
        rule_version: Business rule set version string (e.g. "1.0").

    Returns:
        Experiment ID string.
    """
    rv_slug = rule_version.replace(".", "")
    return f"dec_{config_fingerprint[:12]}_rv{rv_slug}"


def build_experiment_record(
    experiment_id: str,
    config_fingerprint: str,
    rule_version: str,
    phase6_experiment_id: str,
    phase6_config_fingerprint: str,
    n_recommendations: int,
    risk_counts: Dict[str, int],
    override_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Build the experiment record for decision_log.json. Architecture §7.13.

    Args:
        experiment_id: Phase 7 experiment ID.
        config_fingerprint: Phase 7 config fingerprint.
        rule_version: Business rule version.
        phase6_experiment_id: Phase 6 experiment ID (upstream lineage).
        phase6_config_fingerprint: Phase 6 config fingerprint (upstream lineage).
        n_recommendations: Total recommendations generated.
        risk_counts: Count per risk category.
        override_counts: Count per business rule ID.

    Returns:
        Dict record suitable for appending to decision_log.json.
    """
    from datetime import datetime, timezone

    return {
        "experiment_id": experiment_id,
        "config_fingerprint": config_fingerprint,
        "rule_version": rule_version,
        "phase6_upstream": {
            "experiment_id": phase6_experiment_id,
            "config_fingerprint": phase6_config_fingerprint,
        },
        "summary": {
            "total_recommendations": n_recommendations,
            "risk_category_counts": risk_counts,
            "business_rule_override_counts": override_counts,
        },
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }


def save_experiment_log(
    record: Dict[str, Any], output_path: Path, logger: logging.Logger
) -> None:
    """Atomically append an experiment record to decision_log.json.

    Architecture §7.13. Uses list-append semantics: reads existing list,
    appends new record, atomically rewrites the entire file.

    Args:
        record: Experiment record to append.
        output_path: Path to decision_log.json.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            existing = json.load(f)

    existing.append(record)
    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Decision log saved: %s (%d record(s))", output_path, len(existing))

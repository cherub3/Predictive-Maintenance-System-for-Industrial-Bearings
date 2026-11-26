"""Experiment governance: config fingerprint, experiment ID, and the
file-based (JSON) experiment log (architecture §14, §16).

Deliberately file-based, not a tracking server — right-sized for a single
engineer on a two-unit dataset (architecture §14).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List


def compute_config_fingerprint(config: Dict[str, Any]) -> str:
    """SHA-256 of the serialized `model_dev` config block — reuses the Phase 3-5 pattern."""
    model_dev_cfg = config.get("model_dev", {})
    serialized = json.dumps(model_dev_cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_experiment_id(config_fingerprint: str, seed: int) -> str:
    """Derive a stable experiment ID from the config fingerprint and seed.

    Re-running with an identical config + seed reproduces the same experiment
    ID (architecture §14 reproducibility).
    """
    return f"exp_{config_fingerprint[:12]}_seed{seed}"


def build_experiment_record(
    experiment_id: str,
    config_fingerprint: str,
    seed: int,
    git_commit: str,
    upstream_dataset_version: str,
    upstream_dataset_fingerprint: str,
    evaluation_summary: Dict[str, Any],
    selected_model: str,
    hp_search_summary: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Build one experiment-log record.

    Args:
        experiment_id: Stable experiment ID.
        config_fingerprint: SHA-256 of the model_dev config.
        seed: Global reproducibility seed used.
        git_commit: Short git commit SHA (or 'unknown').
        upstream_dataset_version: Phase 5 dataset_version.
        upstream_dataset_fingerprint: Phase 5 config_fingerprint (from its contract).
        evaluation_summary: Per-model mean MAE / spread summary.
        selected_model: The model chosen by model_selection.
        hp_search_summary: Whether/how hyperparameter search was used.
        logger: Pipeline logger.

    Returns:
        A JSON-serializable experiment record dict.
    """
    record = {
        "experiment_id": experiment_id,
        "config_fingerprint": config_fingerprint,
        "random_seed": seed,
        "git_commit": git_commit,
        "upstream_dataset_version": upstream_dataset_version,
        "upstream_dataset_fingerprint": upstream_dataset_fingerprint,
        "evaluation_summary": evaluation_summary,
        "selected_model": selected_model,
        "hyperparameter_search": hp_search_summary,
    }
    logger.info("Experiment record built: %s", experiment_id)
    return record


def save_experiment_log(record: Dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    """Append `record` to `experiment_log.json` (a JSON list), atomically rewriting the whole file.

    Args:
        record: A single experiment record from `build_experiment_record`.
        output_path: Path to `experiment_log.json`.
        logger: Pipeline logger.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[Dict[str, Any]] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.append(record)

    tmp_path = output_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    logger.info("Experiment log updated: %s (%d total records)", output_path, len(existing))

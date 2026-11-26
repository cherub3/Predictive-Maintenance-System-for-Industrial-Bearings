import os
import json
import hashlib
import yaml
import pandas as pd
import streamlit as st

# Paths to frozen artifacts
REPORTS_DIR = "reports"
DATA_DIR = "data/processed"

ARTIFACT_PATHS = {
    "decision_manifest": os.path.join(REPORTS_DIR, "decision_manifest.json"),
    "decision_contract": os.path.join(REPORTS_DIR, "phase7_decision_contract.yaml"),
    "decision_report": os.path.join(REPORTS_DIR, "decision_report.md"),
    "split_summary": os.path.join(REPORTS_DIR, "split_summary.md"),
    "recommendations": os.path.join(REPORTS_DIR, "recommendations.parquet"),
    "feature_store": os.path.join(DATA_DIR, "feature_store.parquet"),
    "engineering_findings": os.path.join(REPORTS_DIR, "engineering_findings.md"),
    "evaluation_report": os.path.join(REPORTS_DIR, "evaluation_report.md"),
    "feature_importance_report": os.path.join(REPORTS_DIR, "feature_importance_report.md"),
    "model_selection_report": os.path.join(REPORTS_DIR, "model_selection_report.md"),
    "decision_validation_report": os.path.join(REPORTS_DIR, "decision_validation_report.md"),
    "leakage_report": os.path.join(REPORTS_DIR, "leakage_report.md"),
}


def _sha256(path: str) -> str:
    """Compute the SHA-256 of a file, streaming in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(manifest: dict, base_dir: str = ".") -> list:
    """Verify every artifact recorded in the manifest against its SHA-256.

    Fail-closed: returns a list of human-readable problems. An EMPTY list means
    every recorded artifact exists on disk and its hash matches. A non-empty list
    means the dashboard must refuse to render (Degraded Mode) — it is safer to show
    no data than data that does not match the manifest.

    Args:
        manifest: Parsed decision_manifest.json (must contain an "artifacts" list).
        base_dir: Directory the artifact paths are relative to.

    Returns:
        List of problem strings (empty when the manifest verifies cleanly).
    """
    problems = []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return ["Manifest contains no 'artifacts' hash list — cannot verify integrity."]

    for entry in artifacts:
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel:
            problems.append(f"Malformed artifact entry (no path): {entry!r}")
            continue
        full = os.path.join(base_dir, rel)
        if entry.get("missing") or expected is None:
            problems.append(f"Artifact was missing when the manifest was written: {rel}")
            continue
        if not os.path.exists(full):
            problems.append(f"Artifact missing on disk: {rel}")
            continue
        actual = _sha256(full)
        if actual != expected:
            problems.append(
                f"Hash mismatch for {rel}: manifest={expected[:12]}… disk={actual[:12]}…"
            )
    return problems


@st.cache_data(show_spinner=False)
def validate_and_load_manifest():
    """Load the manifest and fail closed unless every artifact hash verifies.

    Raises:
        FileNotFoundError: If a required artifact path is absent.
        ValueError: If the manifest hash verification finds any mismatch/missing
            artifact (the dashboard treats this as Degraded Mode).
    """
    missing_files = [path for path in ARTIFACT_PATHS.values() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(f"Missing required artifacts: {missing_files}")

    with open(ARTIFACT_PATHS["decision_manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    problems = verify_manifest(manifest, base_dir=".")
    if problems:
        raise ValueError(
            "Artifact integrity check failed — entering Degraded Mode. "
            + "; ".join(problems)
        )

    return manifest


@st.cache_data(show_spinner=False)
def load_markdown_artifact(artifact_key: str):
    """Load a markdown artifact as UTF-8 text (None if absent)."""
    path = ARTIFACT_PATHS.get(artifact_key)
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_yaml_artifact(artifact_key: str):
    """Load a YAML artifact (None if absent)."""
    path = ARTIFACT_PATHS.get(artifact_key)
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_recommendations():
    """Load recommendations.parquet with a schema check.

    Raises:
        FileNotFoundError: If the recommendations artifact is absent.
        ValueError: If required columns are missing.
    """
    path = ARTIFACT_PATHS["recommendations"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Recommendations artifact not found: {path}")
    df = pd.read_parquet(path)
    expected_cols = [
        "bearing_id", "timestamp", "min_predicted_rul_minutes",
        "risk_category", "priority_score", "maintenance_action",
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema mismatch in {path}. Missing columns: {missing_cols}"
        )
    return df


@st.cache_data(show_spinner=False)
def load_feature_store():
    """Load the feature store parquet lazily when needed.

    Raises:
        FileNotFoundError: If the feature store artifact is absent.
    """
    path = ARTIFACT_PATHS["feature_store"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature store artifact not found: {path}")
    return pd.read_parquet(path)


def get_manifest():
    return validate_and_load_manifest()

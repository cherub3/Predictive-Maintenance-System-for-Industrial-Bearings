"""Global constants for the Bearing Remaining Useful Life Prediction project.

This module defines project-wide constants, file paths, and execution exit codes
to avoid hardcoding and maintain single source of truth configuration references.
"""

from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default Configuration Paths
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Default Encoding
DEFAULT_ENCODING = "utf-8"

# Pipeline Step Status Codes
EXIT_SUCCESS = 0
EXIT_ERROR_CONFIG = 10
EXIT_ERROR_DATA_INGESTION = 20
EXIT_ERROR_DATA_VALIDATION = 30
EXIT_ERROR_SIGNAL_ANALYSIS = 40
EXIT_ERROR_FEATURE_ENGINEERING = 50
EXIT_ERROR_RUL_TARGET = 60
EXIT_ERROR_MODEL_DEV = 70
EXIT_ERROR_DECISION_ENGINE = 80
EXIT_ERROR_UNKNOWN = 99

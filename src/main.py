"""Main entry point for the Predictive Maintenance & RUL Pipeline.

Initializes application configuration, sets up core logger handlers, and serves
as the orchestration layer for future data ingestion, validation, and preparation.
"""

import sys
from pathlib import Path

# Add project root to path if running directly to avoid import issues
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, ConfigurationError
from src.constants import EXIT_SUCCESS, EXIT_ERROR_CONFIG, EXIT_ERROR_UNKNOWN
from src.logger import setup_logger


def run_pipeline() -> int:
    """Orchestrates pipeline execution phases.

    Returns:
        int: Exit status code (0 for success).
    """
    # 1. Load Configurations
    try:
        config = load_config()
    except ConfigurationError as exc:
        print(f"CRITICAL: Configuration Initialization Failed. {exc}", file=sys.stderr)
        return EXIT_ERROR_CONFIG
    except Exception as exc:
        print(f"CRITICAL: Unexpected configuration failure. {exc}", file=sys.stderr)
        return EXIT_ERROR_UNKNOWN

    # 2. Set Up Logging Infrastructure
    logger = setup_logger(name="pipeline", config=config)
    logger.info("Initializing Remaining Useful Life Pipeline Phase 0 Verification...")

    # Log environment configuration details
    proj_name = config.get("project", {}).get("name", "Unknown Project")
    target_company = config.get("project", {}).get("target_company", "N/A")
    logger.info(f"Loaded Project Name: '{proj_name}' (Target: {target_company})")
    
    # 3. Establish Project Directories
    for key, path_val in config.get("paths", {}).items():
        dir_path = Path(path_val)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Verified directory existence for path config '{key}': {dir_path.resolve()}")

    logger.info("Phase 0 Validation & Boilerplate verification completed successfully.")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(run_pipeline())

"""Logging infrastructure for structured and reproducible process logging.

Configures concurrent stream and rotating file logging outputs with custom level
formatting and directory auto-creation based on workspace configuration.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "pipeline",
    config: Optional[Dict[str, Any]] = None,
    log_dir_override: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a structured logger instance.

    Args:
        name: Name of the logger namespace.
        config: Dictionary containing 'logging' parameters from config.yaml.
        log_dir_override: Optional explicit path to save logs, ignoring configuration paths.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Standard configuration defaults
    log_cfg = config.get("logging", {}) if config else {}
    log_level_str = log_cfg.get("level", "INFO").upper()
    log_format = log_cfg.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_dir_name = log_cfg.get("log_dir", "logs")
    log_file_name = log_cfg.get("log_file", "pipeline.log")
    max_bytes = log_cfg.get("max_bytes", 10485760)  # 10MB default
    backup_count = log_cfg.get("backup_count", 5)

    # Resolve log directory
    if log_dir_override:
        log_dir = log_dir_override
    else:
        # Default logs folder relative to execution directory
        log_dir = Path(log_dir_name)

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file_name

    # Set base levels
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # Formatter
    formatter = logging.Formatter(log_format)

    # Stream Handler (Stdout)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Rotating File Handler
    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:
        # Fallback if writing to file is not possible (e.g. read-only system)
        logger.warning(
            f"Failed to initialize rotating file log handler at {log_path}: {exc}"
        )

    return logger

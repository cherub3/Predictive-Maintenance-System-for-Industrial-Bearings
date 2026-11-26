"""Common helper utilities for path handling, file operations, and reproducibility.

Provides deterministic helpers used across ingest, validation, and analytics layers.
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Callable


def get_file_sha256(file_path: Path) -> str:
    """Computes the SHA256 checksum of a file to verify data integrity.

    Args:
        file_path: Path to the target file.

    Returns:
        str: Hexadecimal SHA256 string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in blocks to handle large datasets efficiently
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ensure_directory_exists(dir_path: Path) -> Path:
    """Ensures a directory exists, creating all parent directories if necessary.

    Args:
        dir_path: Path to check/create.

    Returns:
        Path: Resolved directory path.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def timer_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and print/log execution time of operations."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        # Use simple print as fallback if loggers aren't active yet, or custom logging inside wrappers
        print(f"Execution of '{func.__name__}' took {elapsed:.4f} seconds.")
        return result

    return wrapper

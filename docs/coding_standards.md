# Coding Standards and Guidelines

This document outlines the coding standards, patterns, and style guide rules to ensure consistent code quality, maintainability, and clean code principles.

## Style Guide & Formatting

- **PEP 8 Compliance**: Code must strictly comply with Python's PEP 8 style guide.
- **Formatter**: Code should be formatted with `black` (default line length: 88 characters).
- **Import Sorting**: Imports should be sorted using `isort` configured with the `black` profile.

## SOLID Principles & Clean Code

- **Single Responsibility Principle (SRP)**: Each class and module must have one responsibility. Data validation, configuration management, and log setups must remain decoupled.
- **No Hardcoding**:
  - All constants must be placed in `src/constants.py`.
  - All configurable variables must be loaded from `config/config.yaml`.
  - No magic numbers or hardcoded local paths are allowed in function bodies.
- **Type Hinting**: All function parameters, return types, and instance variables must include explicit type hints.
- **Exceptions**: Use custom exceptions (like `ConfigurationError`) instead of catching raw generic exceptions.

## Docstring Conventions

Use the **Google Style** for all module, class, and function-level docstrings.

```python
def example_function(param1: int, param2: str) -> bool:
    """A brief one-line summary of the function.

    A longer description explaining logic, constraints, and side-effects.

    Args:
        param1: Purpose and range of the first parameter.
        param2: Purpose and constraints of the second parameter.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: Under what circumstances this exception is raised.
    """
```

## Logging Standards

- Never use raw print statements for trace logging inside pipeline components. Always use the logger instantiated via `src.logger.setup_logger`.
- Use appropriate levels:
  - `DEBUG`: Highly detailed telemetry, loop iterations, file metadata.
  - `INFO`: Lifecycle milestones (e.g., "Ingestion Complete", "Validation Passed").
  - `WARNING`: Recoverable anomalies or non-blocking issues.
  - `ERROR`: Pipeline execution blocking errors.

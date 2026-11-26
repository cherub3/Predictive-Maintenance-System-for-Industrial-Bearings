# Contribution Guidelines

Thank you for contributing to this project. To maintain a clean history and high codebase standards, please follow the guidelines below.

## Git Workflow & Branching Strategy

1. **Branch Naming**:
   - `feature/feature-name` for new features or infrastructure additions.
   - `bugfix/issue-name` or `hotfix/issue-name` for bug fixes.
   - `docs/doc-updates` for changes to guides and readme files.

2. **Branching Process**:
   - Create all branches from the `main` branch.
   - Ensure your local `main` branch is up-to-date before branching.

## Commit Message Style

We follow the **Conventional Commits** specification:

- `feat: add data validator component`
- `fix: correct config path resolution on Windows`
- `docs: update setup guide with pytest instructions`
- `test: add unit test for SHA256 file utility`
- `refactor: clean up logging initialization parameters`

## Pull Request and Code Review

- **Self-Review**: Run `pytest` locally and check PEP8 compliance before pushing.
- **Pull Request Checklist**:
  - Code builds and tests pass.
  - Type hints and docstrings are added.
  - No new linter issues are introduced.
  - The implementation plan has been followed and updated if necessary.

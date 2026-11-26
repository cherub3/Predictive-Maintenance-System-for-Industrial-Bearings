# Developer Setup Guide

This guide describes how to set up your local development environment for the **Bearing Remaining Useful Life (RUL) Prediction & Maintenance Decision System**.

## Prerequisites

- **Python**: Version 3.10 or higher.
- **Git**: Installed and configured.

## Setting Up Virtual Environment

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd "Bearing Remaining-Useful-Life Prediction & Maintenance Decision System"
   ```

2. **Create virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   Install the baseline dependencies required for Phase 0 setup:
   ```bash
   pip install -r requirements.txt
   ```

## Verifying the Installation

1. **Verify pipeline bootstrap code**:
   Ensure you can run the primary application entry point without errors:
   ```bash
   python src/main.py
   ```
   You should see logging outputs verifying the execution of the pipeline setup:
   ```text
   2026-07-03 19:40:00,000 - pipeline - INFO - Initializing Remaining Useful Life Pipeline Phase 0 Verification...
   2026-07-03 19:40:00,005 - pipeline - INFO - Loaded Project Name: 'Bearing Remaining Useful Life Prediction & Maintenance Decision System' (Target: industrial bearing fleet)
   2026-07-03 19:40:00,010 - pipeline - INFO - Phase 0 Validation & Boilerplate verification completed successfully.
   ```

2. **Verify tests suite**:
   Verify pytest runs successfully:
   ```bash
   pytest tests/
   ```

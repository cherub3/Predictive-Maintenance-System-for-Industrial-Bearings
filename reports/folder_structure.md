# Folder Hierarchy and Structure

This document outlines the organization of the workspace directory.

```text
bearing_predictive_maintenance/
├── config/
│   └── config.yaml           # Global YAML configuration (datasets registry: set1/set2/set3)
├── src/
│   ├── config.py             # Config parser + datasets registry helpers
│   ├── constants.py          # Constants & exit codes
│   ├── logger.py             # Custom logging infrastructure
│   ├── utils.py              # File utilities
│   ├── main.py               # Application orchestrator
│   └── data_ingestion/       # Data ingestion & metadata extraction layer
├── data/
│   ├── raw/
│   │   ├── set1/             # NASA IMS Set 1 (2,156 files, 8 channels)
│   │   ├── set2/             # NASA IMS Set 2 (984 files, 4 channels)
│   │   └── set3/             # NASA IMS Set 3 (6,323 usable files, 4 channels)
│   ├── processed/            # Serialized metadata tables (Parquet, DuckDB)
│   └── external/             # Empty placeholder
├── reports/
│   ├── dataset_summary.md    # Summary statistics report (per-dataset breakdown)
│   ├── folder_structure.md   # Current document
│   ├── metadata_schema.md    # Metadata schema description
│   └── dataset_inventory.csv # Flat catalog inventory CSV (all datasets)
└── logs/
    └── pipeline.log          # Runtime log file
```

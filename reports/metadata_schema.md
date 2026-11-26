# Metadata Schema Specification

The metadata database tables (`metadata.parquet` and `metadata.duckdb`) and catalog (`dataset_inventory.csv`) use the following schema. One row per raw vibration file, spanning all registered datasets (Set 1/2/3).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `dataset_id` | String | Registry key of the source dataset (e.g. "set1", "set2", "set3"). |
| `experiment_name` | String | Display name of the dataset (e.g. "NASA IMS Set 1"). |
| `bearings_present` | String | Comma-separated list of bearings tracked in this dataset. |
| `channels_present` | String | Comma-separated list of channels recorded in this dataset. |
| `timestamp` | String (ISO) | Extracted date-time stamp from the file name. |
| `filename` | String | Base name of the ASCII data file. |
| `absolute_path` | String | Absolute local system path to the file. |
| `relative_path` | String | Workspace relative path to the file. |
| `signal_length` | Integer | Total sample lines (rows of vibration values). |
| `num_columns` | Integer | Column count found in the file. |
| `expected_channels_per_file` | Integer | Column count declared by the dataset's registry entry; a mismatch marks the file `parser_status="Failed"`. |
| `sampling_frequency_hz` | Float | Data acquisition sampling rate. |
| `file_sha256` | String | Calculated SHA256 checksum of raw data. |
| `parser_status` | String | Parser result indicator (`Success` or `Failed`). |
| `extraction_time` | String (ISO) | UTC timestamp when processing occurred. |
| `notes` | String | Channel mappings and failure descriptors, derived from the dataset registry. |

# 03. Raw Data Storage

## Overview
The raw data storage layer serves as the landing zone (or bronze layer) for incoming data ingestion. It stores raw datasets in their original, unmodified formats (e.g., CSV, JSON, Parquet, or log files) directly from source systems, APIs, or databases.

---

## Objectives
- **Immutability:** Keep raw data unaltered to preserve the original state and provide a single source of truth for auditing and debugging.
- **Schema-on-Read:** Avoid strict schema enforcement during ingestion to gracefully handle changes or inconsistencies in incoming data sources.
- **Traceability:** Maintain clear separation between raw files and processed or transformed feature data.

---

## Directory Structure & Configuration
- Raw data files are typically stored locally under `data/raw/` or uploaded to cloud object storage (e.g., AWS S3, Google Cloud Storage, or Azure Blob Storage).
- A `.gitkeep` file is used inside the raw directory to ensure empty folders are preserved within version control.

---

## Implementation Details
1. **Ingestion Script:** Pulls data from the source and writes it directly to the storage destination without modifications.
2. **Versioning & Validation:** Basic checks (such as file existence or row/column count sanity checks) can be applied before saving.

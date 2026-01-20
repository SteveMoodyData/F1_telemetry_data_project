# F1 Telemetry Analysis Platform

A Python-based platform for analyzing Formula 1 race telemetry using FastF1 and PySpark, with ML-powered predictive lap time analysis.

## Architecture

```
FastF1 API → Bronze Layer (Raw Telemetry) → Silver Layer (Validated) → Gold Layer (Analytics-Ready)
                                ↓
                          Data Quality Checks
                                ↓
                          ML Feature Store → Predictive Models
```

## Features

- **Data Ingestion**: Automated fetching of F1 telemetry data via FastF1
- **Distributed Processing**: PySpark for handling large-scale telemetry datasets
- **Data Quality**: Comprehensive validation for lap times, tire compounds, and track conditions
- **ML Analytics**: Predictive models for lap time forecasting based on historical patterns

## Setup

### Prerequisites
```bash
python 3.9+
Java 11+ (for PySpark)
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Create `.env` file:
```
SPARK_HOME=/path/to/spark
F1_CACHE_DIR=./data/cache
OUTPUT_DIR=./data/output
```

## Project Structure

```
f1-telemetry-analysis/
├── src/
│   ├── ingestion/          # FastF1 data fetching
│   ├── processing/         # PySpark transformations
│   ├── quality/            # Data validation rules
│   ├── ml/                 # ML models and features
│   └── utils/              # Shared utilities
├── notebooks/              # Exploratory analysis
├── tests/                  # Unit and integration tests
├── data/                   # Local data cache
└── config/                 # Configuration files
```

## Quick Start

```python
from src.ingestion.telemetry_loader import TelemetryLoader
from src.processing.spark_processor import SparkProcessor

# Load 2024 Bahrain GP data
loader = TelemetryLoader()
session = loader.load_session(2024, 'Bahrain', 'R')

# Process with PySpark
processor = SparkProcessor()
df = processor.process_telemetry(session)
```

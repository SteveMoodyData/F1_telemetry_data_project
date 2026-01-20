# F1 Telemetry Analysis Platform

A Python-based platform for analyzing Formula 1 race telemetry using FastF1 and PySpark, with ML-powered predictive lap time analysis.

## Features

- **Data Ingestion**: Load F1 telemetry data via FastF1 API
- **Distributed Processing**: PySpark with medallion architecture (Bronze → Silver → Gold)
- **Data Quality**: Comprehensive validation for lap times, tire compounds, and track conditions
- **ML Analytics**: Predictive models for lap time forecasting

## Quick Start

### Prerequisites

- Python 3.9-3.12 (3.11 recommended)
- Java 11 or 17 (for PySpark)
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/f1-telemetry-analysis.git
cd f1-telemetry-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
python quickstart.py
```

This will:
1. Download 2024 Bahrain GP data (~500MB, cached locally)
2. Process through PySpark medallion layers
3. Run data quality validation
4. Train ML models for lap time prediction
5. Display results and save best model

## Project Structure

```
f1-telemetry-analysis/
├── src/
│   ├── ingestion/          # FastF1 data loading
│   ├── processing/         # PySpark transformations
│   ├── quality/            # Data validation
│   └── ml/                 # ML models and features
├── notebooks/              # Jupyter notebooks
├── data/                   # Cached data and outputs
├── quickstart.py           # Demo script
└── requirements.txt        # Dependencies
```

## Usage Examples

### Load Race Data

```python
from src.ingestion.telemetry_loader import TelemetryLoader

loader = TelemetryLoader()
session = loader.load_session(2024, 'Monaco', 'R')
laps = loader.extract_lap_data(session)
```

### Process with PySpark

```python
from src.processing.spark_processor import SparkProcessor

processor = SparkProcessor()
bronze_df = processor.process_bronze_laps(laps)
silver_df = processor.process_silver_laps(bronze_df)
gold_stats = processor.process_gold_driver_stats(silver_df)
```

### Validate Data Quality

```python
from src.quality.telemetry_validator import TelemetryQualityValidator

validator = TelemetryQualityValidator()
passed, report = validator.validate_lap_data(laps)
```

### Train ML Model

```python
from src.ml.lap_time_predictor import LapTimeFeatureEngineer, LapTimePredictorModel

engineer = LapTimeFeatureEngineer()
df_features = engineer.engineer_lap_features(laps)
X, y, _ = engineer.prepare_model_data(df_features)

model = LapTimePredictorModel(model_type='xgboost')
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
F1_CACHE_DIR=./data/cache
OUTPUT_DIR=./data/output
MODEL_DIR=./data/models
```

## Development

### Run Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Jupyter Notebooks
```bash
jupyter notebook
```

Open `notebooks/complete_workflow.ipynb` for interactive analysis.

## Performance Tips

- First run downloads ~500MB per race (cached locally)
- Subsequent runs use cache (~2 seconds)
- For full season analysis, increase Spark memory:
  ```python
  processor.spark.conf.set("spark.driver.memory", "8g")
  ```

## Troubleshooting

### Common Issues

**Java not found:**
- Install Java 11: https://adoptium.net/temurin/releases/?version=11
- Set `JAVA_HOME` environment variable

**Package installation errors:**
- Use Python 3.11 (best compatibility)
- See `requirements.txt` for exact versions

**Memory errors:**
- Reduce dataset size or increase available RAM
- Process single races instead of full seasons

## License

MIT License - see LICENSE file for details

## Acknowledgments

- FastF1 library for F1 data access
- Apache Spark for distributed processing
- McLaren Racing for inspiring F1 passion 🏎️

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

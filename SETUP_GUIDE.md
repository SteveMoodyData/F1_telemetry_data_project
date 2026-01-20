# F1 Telemetry Analysis - Setup & Learning Guide

## Overview

This project demonstrates professional-grade data engineering practices applied to F1 telemetry analysis, combining your expertise in Databricks, data quality validation, and distributed processing with your passion for motorsports.

## Key Learning Objectives

### 1. **FastF1 Library Mastery**
- **What**: Official Python library for F1 data access
- **Why**: Provides race telemetry, lap times, tire data, weather conditions
- **Key Concepts**:
  - Session loading (FP1, FP2, FP3, Qualifying, Race)
  - Lap-level data extraction
  - Detailed telemetry (speed, throttle, brake, RPM)
  - Caching strategies for performance

### 2. **PySpark for Distributed Processing**
- **Medallion Architecture Implementation**:
  - **Bronze**: Raw data ingestion with minimal transformation
  - **Silver**: Cleaned, validated, enriched data
  - **Gold**: Business-level aggregations and analytics
- **Why PySpark**: Scales to full season analysis (20+ races, 20 drivers, 1000+ laps each)
- **Key Patterns**:
  - Window functions for rolling calculations
  - Partitioning strategies
  - Deduplication logic
  - Aggregation optimization

### 3. **Data Quality Validation**
- **Schema Validation**: Using Pandera for type checking
- **Business Rules**:
  - Sector time correlation (sum of sectors ≈ lap time)
  - Outlier detection (IQR method)
  - Tire life progression validation
  - Cross-compound performance consistency
- **Why Important**: F1 data has timing system quirks, missing values, and edge cases

### 4. **ML for Predictive Analytics**
- **Feature Engineering**:
  - Temporal features (session progress, stint age)
  - Performance features (delta to best, rolling averages)
  - Contextual features (tire compound, tire life, track evolution)
- **Model Selection**:
  - XGBoost: Best for structured data with complex interactions
  - LightGBM: Faster training, good for large datasets
  - Random Forest: Baseline comparison
- **Hyperparameter Optimization**: Optuna for automated tuning

## Installation & Setup

### Prerequisites

```bash
# System Requirements
- Python 3.9+
- Java 11+ (for PySpark)
- 8GB+ RAM recommended
```

### Step-by-Step Setup

```bash
# 1. Clone/Create project
cd f1-telemetry-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Verify setup
python -c "import fastf1; import pyspark; print('✓ Setup verified')"
```

### Java/Spark Setup

If you don't have Spark installed:

```bash
# macOS (Homebrew)
brew install apache-spark

# Ubuntu/Debian
sudo apt-get install default-jdk
wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar xvf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=$(pwd)/spark-3.5.0-bin-hadoop3
export PATH=$PATH:$SPARK_HOME/bin
```

## Quick Start

### Option 1: Run the Demo Script

```bash
python quickstart.py
```

This will:
1. Download 2024 Bahrain GP data (cached locally)
2. Process through PySpark layers
3. Run data quality validation
4. Train ML models
5. Display results and save best model

**First run takes 5-10 minutes** (data download + processing)  
**Subsequent runs take ~2 minutes** (using cached data)

### Option 2: Interactive Notebook

```bash
jupyter notebook notebooks/complete_workflow.ipynb
```

Step through each component interactively with visualizations.

## Project Architecture

```
Data Flow:
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastF1    │────▶│   PySpark    │────▶│  Data QA    │
│   Ingestion │     │  Processing  │     │ Validation  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ML Feature  │
                    │  Engineering │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Model Train/ │
                    │   Predict    │
                    └──────────────┘
```

## Key Components Deep Dive

### 1. TelemetryLoader (`src/ingestion/telemetry_loader.py`)

**Purpose**: Interface with FastF1 API

```python
from ingestion.telemetry_loader import TelemetryLoader

loader = TelemetryLoader()
session = loader.load_session(2024, 'Monaco', 'R')
laps = loader.extract_lap_data(session)
telemetry = loader.extract_telemetry(session, 'VER')  # Verstappen
```

**Key Methods**:
- `load_session()`: Load specific GP session
- `extract_lap_data()`: Get lap-level statistics
- `extract_telemetry()`: Get detailed telemetry (speed, throttle, etc.)
- `batch_load_season()`: Load entire season efficiently

### 2. SparkProcessor (`src/processing/spark_processor.py`)

**Purpose**: Distributed data processing with medallion architecture

```python
from processing.spark_processor import SparkProcessor

processor = SparkProcessor()

# Bronze: Raw data
bronze_df = processor.process_bronze_laps(laps_pdf)

# Silver: Validated & enriched
silver_df = processor.process_silver_laps(bronze_df)

# Gold: Analytics-ready
driver_stats = processor.process_gold_driver_stats(silver_df)
tire_stats = processor.process_gold_tire_analysis(silver_df)
```

**Key Transformations**:
- Type conversions and validation
- Delta calculations (to fastest lap, to session fastest)
- Rolling windows for performance trends
- Aggregations by driver, tire compound, circuit

### 3. TelemetryQualityValidator (`src/quality/telemetry_validator.py`)

**Purpose**: Comprehensive data quality checks

```python
from quality.telemetry_validator import TelemetryQualityValidator

validator = TelemetryQualityValidator()
passed, report = validator.validate_lap_data(laps_df)

if not passed:
    print(f"Issues: {report['issues']}")
    print(f"Warnings: {report['warnings']}")
```

**Validation Rules**:
1. **Schema Validation**: Types, ranges, required fields
2. **Sector Correlation**: Sector sum ≈ lap time (±0.5s tolerance)
3. **Outlier Detection**: IQR-based anomaly detection
4. **Tire Progression**: Sequential tire life in stints
5. **Cross-Compound Analysis**: Expected performance ordering

### 4. ML Pipeline (`src/ml/lap_time_predictor.py`)

**Purpose**: Feature engineering and predictive modeling

```python
from ml.lap_time_predictor import LapTimeFeatureEngineer, LapTimePredictorModel

# Feature engineering
engineer = LapTimeFeatureEngineer()
df_features = engineer.engineer_lap_features(laps_df)
X, y, feature_names = engineer.prepare_model_data(df_features)

# Train model
model = LapTimePredictorModel(model_type='xgboost')
model.train(X_train, y_train, optimize=True)  # optimize=True runs Optuna

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"MAE: {metrics['mae']:.3f}s, R²: {metrics['r2']:.4f}")
```

**Engineered Features**:
- Tire age bins (Fresh, Early, Mid, Late, VeryOld)
- Session progression percentile
- Rolling 3-lap average
- Delta to personal best
- Track evolution (session-wide pace trend)
- Sector time percentages

## Advanced Usage Examples

### Load Multiple Races

```python
# Load entire 2024 season
season_data = loader.batch_load_season(2024, session_type='R')

# Combine all races
all_laps = pd.concat(season_data.values(), ignore_index=True)
print(f"Total laps: {len(all_laps)} from {len(season_data)} races")
```

### Custom Data Quality Rules

```python
# Add custom validation
validator = TelemetryQualityValidator()

# Check for minimum tire stint length
def validate_stint_length(df):
    stint_lengths = df.groupby(['GrandPrix', 'Driver', 'Stint']).size()
    short_stints = stint_lengths[stint_lengths < 3]
    return len(short_stints) == 0

result = validate_stint_length(laps_df)
```

### Hyperparameter Optimization

```python
# Run automated hyperparameter search
model = LapTimePredictorModel(model_type='xgboost')
model.train(X_train, y_train, optimize=True)  # 50 Optuna trials

# View best parameters
print(model.model.get_params())
```

### Compare Multiple Compounds at Same Circuit

```python
# Analyze tire performance
tire_analysis = validator.validate_cross_compound_performance(laps_df)

for gp, data in tire_analysis.items():
    print(f"\n{gp}:")
    for compound_data in data['compound_performance']:
        print(f"  {compound_data['Compound']}: "
              f"{compound_data['median']:.2f}s "
              f"({compound_data['count']} laps)")
```

## Performance Considerations

### Memory Optimization

For large datasets (full seasons):

```python
# Process in batches
processor = SparkProcessor()
processor.spark.conf.set("spark.sql.shuffle.partitions", "200")
processor.spark.conf.set("spark.driver.memory", "8g")

# Write to Delta Lake for incremental processing
processor.write_delta(
    df, 
    path='./data/delta/laps',
    mode='append',
    partition_by=['Year', 'GrandPrix']
)
```

### Caching Strategy

FastF1 caches data automatically:

```python
# First run downloads data
loader = TelemetryLoader(cache_dir='./data/cache')
session = loader.load_session(2024, 'Bahrain', 'R')  # ~30 seconds

# Subsequent runs use cache
session = loader.load_session(2024, 'Bahrain', 'R')  # ~2 seconds
```

## Common Issues & Solutions

### Issue: Java not found for PySpark

```bash
# Verify Java installation
java -version

# If not installed:
# macOS: brew install openjdk@11
# Ubuntu: sudo apt-get install default-jdk

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home)  # macOS
export JAVA_HOME=/usr/lib/jvm/default-java  # Ubuntu
```

### Issue: FastF1 data not loading

```bash
# Check internet connection
# FastF1 pulls from Ergast API and official F1 timing

# Clear cache if corrupted
rm -rf ./data/cache/*

# Verify with simple test
python -c "import fastf1; fastf1.Cache.enable_cache('./cache'); \
           session = fastf1.get_session(2023, 1, 'R'); \
           session.load(); print('✓ Data loaded')"
```

### Issue: Memory errors with PySpark

```python
# Reduce Spark memory if running locally
processor = SparkProcessor()
processor.spark.conf.set("spark.driver.memory", "2g")
processor.spark.conf.set("spark.executor.memory", "2g")

# Or process smaller subsets
laps_subset = laps_df.sample(frac=0.5)
```

## Next Steps & Extensions

### 1. Real-Time Race Strategy

Build a system that predicts optimal pit stop windows:
- Track tire degradation patterns
- Model fuel load effects
- Simulate strategy scenarios

### 2. Driver Performance Comparison

Analyze driver strengths:
- Sector-level comparison
- Qualifying vs race pace
- Wet weather performance

### 3. Circuit Characteristics

Cluster circuits by characteristics:
- High-speed vs technical
- Tire degradation patterns
- Overtaking opportunities

### 4. API Development

Deploy predictions as REST API:
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict_lap_time(features: dict):
    model = LapTimePredictorModel.load_model('./models/best.pkl')
    prediction = model.predict(features)
    return {"predicted_lap_time": prediction}
```

### 5. Dashboard Development

Create interactive visualizations with Plotly Dash or Streamlit:
- Live race simulation
- Strategy comparison
- Driver performance trends

## Resources

### F1 Data & Analysis
- FastF1 Documentation: https://docs.fastf1.dev/
- F1 Technical Regulations: https://www.fia.com/regulation/category/110
- Ergast API: http://ergast.com/mrd/

### Data Engineering
- PySpark Guide: https://spark.apache.org/docs/latest/api/python/
- Delta Lake: https://delta.io/
- Medallion Architecture: [Databricks resources]

### Machine Learning
- XGBoost: https://xgboost.readthedocs.io/
- Optuna: https://optuna.org/
- Scikit-learn: https://scikit-learn.org/

### McLaren & F1 Engineering
- McLaren Racing: https://www.mclaren.com/racing/
- F1 Technical Explained: Chain Bear F1 on YouTube
- Race Strategy Analysis: Peter Windsor on YouTube

## Contributing & Feedback

This is a learning project showcasing data engineering best practices. Feel free to:
- Extend the feature engineering
- Add new validation rules
- Experiment with different ML architectures
- Integrate with other data sources (weather, betting odds, etc.)

---

**Remember**: This combines your Databricks expertise (medallion architecture, data quality) with F1 passion. The patterns here scale directly to production data engineering work!

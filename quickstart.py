#!/usr/bin/env python3
"""
F1 Telemetry Analysis - Quick Start Demo

Demonstrates the complete pipeline:
1. Data ingestion from FastF1
2. PySpark processing
3. Data quality validation
4. ML model training and evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ingestion.telemetry_loader import TelemetryLoader
from processing.spark_processor import SparkProcessor
from quality.telemetry_validator import TelemetryQualityValidator
from ml.lap_time_predictor import LapTimeFeatureEngineer, LapTimePredictorModel

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def main():
    """Run the complete F1 telemetry analysis pipeline."""
    
    print("=" * 70)
    print("F1 TELEMETRY ANALYSIS - QUICK START DEMO")
    print("=" * 70)
    
    # ===== STEP 1: Data Ingestion =====
    print("\n[1/5] DATA INGESTION")
    print("-" * 70)
    
    loader = TelemetryLoader(cache_dir='./data/cache')
    
    # Load a recent race (2024 season)
    print("Loading 2024 Bahrain GP Race data...")
    try:
        session = loader.load_session(2024, 'Bahrain', 'R')
        laps_df = loader.extract_lap_data(session)
        print(f"✓ Loaded {len(laps_df)} laps from {laps_df['Driver'].nunique()} drivers")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        print("\nNote: This requires internet connection to download F1 data.")
        print("The data will be cached locally for future runs.")
        return
    
    # ===== STEP 2: PySpark Processing =====
    print("\n[2/5] DISTRIBUTED PROCESSING WITH PYSPARK")
    print("-" * 70)
    
    processor = SparkProcessor(app_name="F1-QuickStart")
    
    print("Processing through medallion architecture...")
    bronze_df = processor.process_bronze_laps(laps_df)
    print(f"  Bronze layer: {bronze_df.count()} records")
    
    silver_df = processor.process_silver_laps(bronze_df)
    print(f"  Silver layer: {silver_df.count()} validated records")
    
    gold_drivers = processor.process_gold_driver_stats(silver_df)
    print(f"  Gold layer: {gold_drivers.count()} driver statistics")
    
    print("\nTop 5 Fastest Drivers:")
    gold_drivers.select(
        'Driver', 'FastestLap', 'AvgLapTime', 'ConsistencyScore'
    ).orderBy('FastestLap').show(5)
    
    # Convert to pandas for ML
    laps_validated = silver_df.toPandas()
    
    # ===== STEP 3: Data Quality Validation =====
    print("\n[3/5] DATA QUALITY VALIDATION")
    print("-" * 70)
    
    validator = TelemetryQualityValidator()
    passed, report = validator.validate_lap_data(laps_validated)
    
    print(f"Validation Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Total Records: {report['total_rows']}")
    print(f"Schema Valid: {'✓' if report['schema_valid'] else '✗'}")
    print(f"Issues: {len(report['issues'])}")
    print(f"Warnings: {len(report['warnings'])}")
    
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings'][:3]:  # Show first 3
            print(f"  - {warning}")
    
    # ===== STEP 4: Feature Engineering =====
    print("\n[4/5] FEATURE ENGINEERING FOR ML")
    print("-" * 70)
    
    engineer = LapTimeFeatureEngineer()
    df_features = engineer.engineer_lap_features(laps_validated)
    
    print(f"Original columns: {len(laps_validated.columns)}")
    print(f"Engineered columns: {len(df_features.columns)}")
    print(f"New features created: {len(df_features.columns) - len(laps_validated.columns)}")
    
    # Prepare for ML
    X, y, feature_names = engineer.prepare_model_data(df_features)
    print(f"\nModel dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target range: {y.min():.2f}s to {y.max():.2f}s")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ===== STEP 5: ML Model Training =====
    print("\n[5/5] MACHINE LEARNING MODEL TRAINING")
    print("-" * 70)
    
    results = {}
    
    for model_type in ['xgboost', 'lightgbm']:
        print(f"\nTraining {model_type.upper()} model...")
        
        model = LapTimePredictorModel(model_type=model_type)
        model.train(X_train, y_train, optimize=False)
        
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"  MAE:  {metrics['mae']:.3f} seconds")
        print(f"  RMSE: {metrics['rmse']:.3f} seconds")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Compare models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    import pandas as pd
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())
    
    # Best model
    best_model = results_df['r2'].idxmax()
    print(f"\n✓ Best Model: {best_model.upper()} (R² = {results_df.loc[best_model, 'r2']:.4f})")
    
    # Feature importance
    best_model_obj = LapTimePredictorModel(model_type=best_model)
    best_model_obj.train(X_train, y_train, optimize=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(best_model_obj.feature_importance.head(10).to_string(index=False))
    
    # Save model
    model_path = f'./data/models/{best_model}_lap_predictor.pkl'
    best_model_obj.save_model(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Cleanup
    processor.stop()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Open notebooks/complete_workflow.ipynb for detailed analysis")
    print("  2. Load more races with loader.batch_load_season(2024)")
    print("  3. Run hyperparameter optimization with optimize=True")
    print("  4. Explore telemetry-level predictions with detailed features")
    print("\nFor questions or issues, check the README.md")


if __name__ == "__main__":
    main()

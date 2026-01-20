"""
PySpark Telemetry Processor

Handles distributed processing of F1 telemetry data using PySpark.
Implements medallion architecture (Bronze -> Silver -> Gold layers).
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType, BooleanType
)
from pyspark.sql.window import Window
import pandas as pd
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkProcessor:
    """Distributed processing of F1 telemetry data using PySpark."""
    
    def __init__(self, app_name: str = "F1-Telemetry-Analysis"):
        """Initialize Spark session with optimized configuration."""
        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
        )
        logger.info(f"Spark session initialized: {app_name}")
    
    def pandas_to_spark(self, pdf: pd.DataFrame) -> DataFrame:
        """Convert Pandas DataFrame to Spark DataFrame."""
        return self.spark.createDataFrame(pdf)
    
    def process_bronze_laps(self, laps_pdf: pd.DataFrame) -> DataFrame:
        """
        Bronze layer: Raw lap data with minimal transformations.
        
        Args:
            laps_pdf: Pandas DataFrame from FastF1
        
        Returns:
            Spark DataFrame with typed columns
        """
        logger.info("Processing Bronze layer - Raw laps")
        
        # Convert timedelta columns to seconds
        time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in time_cols:
            if col in laps_pdf.columns:
                laps_pdf[col] = laps_pdf[col].dt.total_seconds()
        
        # Convert to Spark
        df = self.pandas_to_spark(laps_pdf)
        
        # Add ingestion metadata
        df = df.withColumn("ingestion_timestamp", F.current_timestamp())
        df = df.withColumn("record_source", F.lit("fastf1_api"))
        
        logger.info(f"Bronze layer created: {df.count()} records")
        return df
    
    def process_silver_laps(self, bronze_df: DataFrame) -> DataFrame:
        """
        Silver layer: Cleaned and validated lap data.
        
        Applies:
        - Data quality filters
        - Type conversions
        - Derived columns
        - Deduplication
        
        Args:
            bronze_df: Bronze layer DataFrame
        
        Returns:
            Cleaned Spark DataFrame
        """
        logger.info("Processing Silver layer - Validated laps")
        
        df = bronze_df
        
        # Remove invalid laps
        df = df.filter(
            (F.col("LapTime").isNotNull()) &
            (F.col("LapTime") > 0) &
            (F.col("IsAccurate") == True)
        )
        
        # Calculate total lap time from sectors (for validation)
        df = df.withColumn(
            "SectorSum",
            F.col("Sector1Time") + F.col("Sector2Time") + F.col("Sector3Time")
        )
        
        # Flag potential data quality issues
        df = df.withColumn(
            "TimeMismatch",
            F.when(
                F.abs(F.col("LapTime") - F.col("SectorSum")) > 0.1,
                True
            ).otherwise(False)
        )
        
        # Add compound standardization
        df = df.withColumn(
            "CompoundStd",
            F.when(F.col("Compound") == "SOFT", "SOFT")
            .when(F.col("Compound") == "MEDIUM", "MEDIUM")
            .when(F.col("Compound") == "HARD", "HARD")
            .when(F.col("Compound") == "INTERMEDIATE", "INTERMEDIATE")
            .when(F.col("Compound") == "WET", "WET")
            .otherwise("UNKNOWN")
        )
        
        # Calculate lap time delta to fastest
        window_spec = Window.partitionBy("GrandPrix", "Session")
        
        df = df.withColumn(
            "FastestLapTime",
            F.min("LapTime").over(window_spec)
        )
        
        df = df.withColumn(
            "DeltaToFastest",
            F.col("LapTime") - F.col("FastestLapTime")
        )
        
        # Deduplicate based on key columns
        df = df.dropDuplicates([
            "GrandPrix", "Session", "Driver", "LapNumber"
        ])
        
        logger.info(f"Silver layer created: {df.count()} records")
        return df
    
    def process_gold_driver_stats(self, silver_df: DataFrame) -> DataFrame:
        """
        Gold layer: Driver performance aggregations.
        
        Args:
            silver_df: Silver layer DataFrame
        
        Returns:
            Aggregated driver statistics
        """
        logger.info("Processing Gold layer - Driver statistics")
        
        driver_stats = silver_df.groupBy(
            "GrandPrix", "Session", "Driver", "DriverNumber"
        ).agg(
            F.count("LapNumber").alias("TotalLaps"),
            F.min("LapTime").alias("FastestLap"),
            F.avg("LapTime").alias("AvgLapTime"),
            F.stddev("LapTime").alias("StdDevLapTime"),
            F.avg("DeltaToFastest").alias("AvgDeltaToFastest"),
            F.countDistinct("CompoundStd").alias("TireCompoundsUsed"),
            F.sum(
                F.when(F.col("PitInTime").isNotNull(), 1).otherwise(0)
            ).alias("PitStops")
        )
        
        # Add consistency metric (lower is better)
        driver_stats = driver_stats.withColumn(
            "ConsistencyScore",
            F.col("StdDevLapTime") / F.col("AvgLapTime")
        )
        
        logger.info(f"Gold layer created: {driver_stats.count()} driver stats")
        return driver_stats
    
    def process_gold_tire_analysis(self, silver_df: DataFrame) -> DataFrame:
        """
        Gold layer: Tire compound performance analysis.
        
        Args:
            silver_df: Silver layer DataFrame
        
        Returns:
            Tire compound statistics
        """
        logger.info("Processing Gold layer - Tire analysis")
        
        tire_stats = silver_df.filter(
            F.col("CompoundStd") != "UNKNOWN"
        ).groupBy(
            "GrandPrix", "Circuit", "CompoundStd"
        ).agg(
            F.count("LapNumber").alias("LapCount"),
            F.avg("LapTime").alias("AvgLapTime"),
            F.min("LapTime").alias("BestLapTime"),
            F.avg("TyreLife").alias("AvgTyreLife"),
            F.max("TyreLife").alias("MaxTyreLife"),
            F.stddev("LapTime").alias("StdDevLapTime")
        )
        
        # Calculate degradation rate (lap time increase per lap on tire)
        # This is simplified - proper analysis would use window functions per stint
        tire_stats = tire_stats.withColumn(
            "EstimatedDegradation",
            F.col("StdDevLapTime") / F.col("AvgTyreLife")
        )
        
        logger.info(f"Gold layer created: {tire_stats.count()} tire stats")
        return tire_stats
    
    def add_track_conditions_features(
        self, 
        laps_df: DataFrame,
        weather_df: Optional[DataFrame] = None
    ) -> DataFrame:
        """
        Enrich lap data with track condition features.
        
        Args:
            laps_df: Lap DataFrame
            weather_df: Optional weather DataFrame
        
        Returns:
            Enriched DataFrame with track condition features
        """
        if weather_df is None:
            logger.info("No weather data provided, using session-level features only")
            return laps_df
        
        # Join weather data based on timestamp proximity
        # This is a simplified join - production would use interval joins
        enriched = laps_df.alias("laps").join(
            weather_df.alias("weather"),
            (F.col("laps.GrandPrix") == F.col("weather.GrandPrix")) &
            (F.col("laps.Session") == F.col("weather.Session")),
            "left"
        )
        
        logger.info("Track conditions features added")
        return enriched
    
    def write_delta(
        self, 
        df: DataFrame, 
        path: str, 
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None
    ):
        """
        Write DataFrame to Delta format for lakehouse storage.
        
        Args:
            df: DataFrame to write
            path: Output path
            mode: Write mode (overwrite, append)
            partition_by: Optional partition columns
        """
        writer = df.write.format("delta").mode(mode)
        
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        
        writer.save(path)
        logger.info(f"Data written to {path}")
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()
        logger.info("Spark session stopped")


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize processor
    processor = SparkProcessor()
    
    # Simulate loading data (in practice, use TelemetryLoader)
    # For demo, create sample data
    sample_data = {
        'Driver': ['VER', 'VER', 'HAM', 'HAM'],
        'LapNumber': [1, 2, 1, 2],
        'LapTime': [90.5, 89.8, 91.2, 90.1],
        'Sector1Time': [30.1, 29.8, 30.5, 30.0],
        'Sector2Time': [30.2, 30.0, 30.4, 30.1],
        'Sector3Time': [30.2, 30.0, 30.3, 30.0],
        'Compound': ['SOFT', 'SOFT', 'MEDIUM', 'MEDIUM'],
        'TyreLife': [1, 2, 1, 2],
        'IsAccurate': [True, True, True, True],
        'GrandPrix': ['Bahrain GP'] * 4,
        'Session': ['Race'] * 4,
        'Circuit': ['Bahrain'] * 4
    }
    
    sample_pdf = pd.DataFrame(sample_data)
    
    # Process through medallion layers
    bronze_df = processor.process_bronze_laps(sample_pdf)
    silver_df = processor.process_silver_laps(bronze_df)
    gold_drivers = processor.process_gold_driver_stats(silver_df)
    
    print("\nGold Layer - Driver Stats:")
    gold_drivers.show(truncate=False)
    
    processor.stop()

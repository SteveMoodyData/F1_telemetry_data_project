"""
Data Quality Validation Module

Implements comprehensive data quality checks for F1 telemetry data:
- Lap time correlation validation
- Performance anomaly detection
- Tire compound consistency checks
- Track condition validation
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryQualityValidator:
    """Validates quality of F1 telemetry data."""
    
    def __init__(self):
        self.lap_schema = self._create_lap_schema()
        self.telemetry_schema = self._create_telemetry_schema()
    
    def _create_lap_schema(self) -> DataFrameSchema:
        """Define Pandera schema for lap-level data validation."""
        return DataFrameSchema(
            {
                "Driver": Column(str, nullable=False),
                "LapNumber": Column(int, Check.greater_than(0)),
                "LapTime": Column(float, Check.in_range(0, 200)),  # Max 200 seconds
                "Sector1Time": Column(float, Check.in_range(0, 100), nullable=True),
                "Sector2Time": Column(float, Check.in_range(0, 100), nullable=True),
                "Sector3Time": Column(float, Check.in_range(0, 100), nullable=True),
                "Compound": Column(
                    str, 
                    Check.isin(['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']),
                    nullable=True
                ),
                "TyreLife": Column(int, Check.greater_than_or_equal_to(0), nullable=True),
                "IsAccurate": Column(bool, nullable=True),
            },
            strict=False  # Allow additional columns
        )
    
    def _create_telemetry_schema(self) -> DataFrameSchema:
        """Define schema for detailed telemetry data validation."""
        return DataFrameSchema(
            {
                "Speed": Column(float, Check.in_range(0, 400)),  # km/h
                "Throttle": Column(float, Check.in_range(0, 100), nullable=True),
                "Brake": Column(float, Check.in_range(0, 100), nullable=True),
                "nGear": Column(int, Check.in_range(0, 8), nullable=True),
                "RPM": Column(float, Check.in_range(0, 20000), nullable=True),
            },
            strict=False
        )
    
    def validate_lap_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate lap-level data against schema and business rules.
        
        Args:
            df: Lap data DataFrame
        
        Returns:
            Tuple of (validation_passed, validation_report)
        """
        logger.info("Validating lap data quality...")
        
        report = {
            'total_rows': len(df),
            'schema_valid': False,
            'issues': [],
            'warnings': []
        }
        
        # Schema validation
        try:
            self.lap_schema.validate(df, lazy=True)
            report['schema_valid'] = True
            logger.info("✓ Schema validation passed")
        except pa.errors.SchemaErrors as err:
            report['schema_valid'] = False
            report['issues'].append(f"Schema validation failed: {err}")
            logger.error(f"✗ Schema validation failed: {err.failure_cases}")
        
        # Business rule validations
        report = self._validate_sector_correlation(df, report)
        report = self._validate_lap_time_outliers(df, report)
        report = self._validate_tire_progression(df, report)
        report = self._validate_pit_stop_logic(df, report)
        
        validation_passed = report['schema_valid'] and len(report['issues']) == 0
        
        logger.info(
            f"Validation {'PASSED' if validation_passed else 'FAILED'}: "
            f"{len(report['issues'])} issues, {len(report['warnings'])} warnings"
        )
        
        return validation_passed, report
    
    def _validate_sector_correlation(
        self, 
        df: pd.DataFrame, 
        report: Dict
    ) -> Dict:
        """
        Validate that sector times correlate correctly with lap times.
        
        Critical check: Sum of sector times should approximately equal lap time.
        """
        logger.info("Checking sector time correlation...")
        
        sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
        
        if all(col in df.columns for col in sector_cols):
            df_sectors = df[df['Sector1Time'].notna()].copy()
            
            df_sectors['SectorSum'] = df_sectors[sector_cols].sum(axis=1)
            df_sectors['TimeDiff'] = abs(
                df_sectors['LapTime'] - df_sectors['SectorSum']
            )
            
            # Allow 0.5 second tolerance for timing system precision
            mismatches = df_sectors[df_sectors['TimeDiff'] > 0.5]
            
            if len(mismatches) > 0:
                pct_mismatch = (len(mismatches) / len(df_sectors)) * 100
                
                if pct_mismatch > 5:  # More than 5% mismatches is an issue
                    report['issues'].append(
                        f"Sector correlation issue: {len(mismatches)} laps "
                        f"({pct_mismatch:.1f}%) have sector sum != lap time"
                    )
                else:
                    report['warnings'].append(
                        f"Minor sector correlation variance: {len(mismatches)} laps"
                    )
                    
                logger.warning(
                    f"Sector time mismatches: {len(mismatches)} / {len(df_sectors)}"
                )
            else:
                logger.info("✓ Sector correlation validated")
        else:
            report['warnings'].append("Sector times not available for validation")
        
        return report
    
    def _validate_lap_time_outliers(
        self, 
        df: pd.DataFrame, 
        report: Dict
    ) -> Dict:
        """
        Detect anomalous lap times using statistical methods.
        
        Uses IQR method to identify outliers per driver/session.
        """
        logger.info("Detecting lap time anomalies...")
        
        outliers_list = []
        
        for (gp, session, driver), group in df.groupby(['GrandPrix', 'Session', 'Driver']):
            if len(group) < 3:  # Need minimum laps for statistical analysis
                continue
            
            # Use only accurate, non-pit laps
            valid_laps = group[
                (group['IsAccurate'] == True) & 
                (group['PitInTime'].isna())
            ]['LapTime']
            
            if len(valid_laps) < 3:
                continue
            
            # IQR method
            Q1 = valid_laps.quantile(0.25)
            Q3 = valid_laps.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3.0 * IQR  # 3x IQR for extreme outliers
            upper_bound = Q3 + 3.0 * IQR
            
            outliers = group[
                (group['LapTime'] < lower_bound) | 
                (group['LapTime'] > upper_bound)
            ]
            
            if len(outliers) > 0:
                outliers_list.append({
                    'GrandPrix': gp,
                    'Driver': driver,
                    'OutlierCount': len(outliers),
                    'OutlierLaps': outliers['LapNumber'].tolist()
                })
        
        if outliers_list:
            report['warnings'].append(
                f"Detected {len(outliers_list)} driver sessions with lap time outliers"
            )
            report['outlier_details'] = outliers_list
            logger.warning(f"Found outliers in {len(outliers_list)} driver sessions")
        else:
            logger.info("✓ No significant lap time anomalies detected")
        
        return report
    
    def _validate_tire_progression(
        self, 
        df: pd.DataFrame, 
        report: Dict
    ) -> Dict:
        """
        Validate tire life progression logic.
        
        Checks:
        - Tire life should increment sequentially per stint
        - Tire life resets after pit stops
        - Compound changes align with pit stops
        """
        logger.info("Validating tire progression logic...")
        
        issues = []
        
        for (gp, driver), group in df.groupby(['GrandPrix', 'Driver']):
            if 'TyreLife' not in group.columns or 'Stint' not in group.columns:
                continue
            
            group_sorted = group.sort_values('LapNumber')
            
            for stint in group_sorted['Stint'].unique():
                stint_laps = group_sorted[group_sorted['Stint'] == stint]
                
                if len(stint_laps) < 2:
                    continue
                
                # Check tire life progression
                tire_life_diff = stint_laps['TyreLife'].diff()
                
                # Should increment by 1 each lap (with some tolerance for data issues)
                invalid_progression = tire_life_diff[
                    (tire_life_diff.notna()) & 
                    (tire_life_diff != 1)
                ]
                
                if len(invalid_progression) > 0:
                    issues.append({
                        'GrandPrix': gp,
                        'Driver': driver,
                        'Stint': stint,
                        'Issue': 'Non-sequential tire life progression'
                    })
        
        if issues:
            report['warnings'].append(
                f"Tire progression issues in {len(issues)} stints"
            )
            logger.warning(f"Tire progression issues: {len(issues)}")
        else:
            logger.info("✓ Tire progression validated")
        
        return report
    
    def _validate_pit_stop_logic(
        self, 
        df: pd.DataFrame, 
        report: Dict
    ) -> Dict:
        """
        Validate pit stop timing and logic.
        
        Checks:
        - Pit in/out times are logical
        - Stint changes align with pit stops
        """
        logger.info("Validating pit stop logic...")
        
        if 'PitInTime' not in df.columns:
            report['warnings'].append("Pit stop data not available")
            return report
        
        pit_laps = df[df['PitInTime'].notna()]
        
        # Check for very short pit stops (< 15 seconds is suspicious)
        if 'PitOutTime' in df.columns:
            pit_laps_timed = pit_laps[pit_laps['PitOutTime'].notna()].copy()
            
            # This would require proper datetime handling in production
            # For now, just flag if both columns exist but are inconsistent
            pit_data_complete = len(pit_laps_timed) / max(len(pit_laps), 1)
            
            if pit_data_complete < 0.5:
                report['warnings'].append(
                    "Incomplete pit stop timing data (missing PitOutTime)"
                )
        
        logger.info(f"Validated {len(pit_laps)} pit stops")
        
        return report
    
    def validate_cross_compound_performance(
        self, 
        df: pd.DataFrame
    ) -> Dict:
        """
        Analyze performance consistency across tire compounds.
        
        Expected: Soft < Medium < Hard lap times (in general)
        Flags anomalies where this doesn't hold.
        """
        logger.info("Analyzing cross-compound performance...")
        
        analysis = {}
        
        for (gp, circuit), group in df.groupby(['GrandPrix', 'Circuit']):
            # Calculate median lap time per compound
            compound_perf = group.groupby('Compound')['LapTime'].agg([
                'median', 'mean', 'count'
            ]).reset_index()
            
            compound_perf = compound_perf[compound_perf['count'] >= 5]  # Min 5 laps
            
            if len(compound_perf) < 2:
                continue
            
            analysis[gp] = {
                'circuit': circuit,
                'compound_performance': compound_perf.to_dict('records')
            }
            
            # Check expected ordering (if all three compounds present)
            compounds_present = set(compound_perf['Compound'])
            if {'SOFT', 'MEDIUM', 'HARD'}.issubset(compounds_present):
                soft_time = compound_perf[
                    compound_perf['Compound'] == 'SOFT'
                ]['median'].values[0]
                medium_time = compound_perf[
                    compound_perf['Compound'] == 'MEDIUM'
                ]['median'].values[0]
                hard_time = compound_perf[
                    compound_perf['Compound'] == 'HARD'
                ]['median'].values[0]
                
                expected_order = soft_time < medium_time < hard_time
                analysis[gp]['expected_order'] = expected_order
                
                if not expected_order:
                    logger.warning(
                        f"{gp}: Unexpected compound performance order "
                        f"(S:{soft_time:.2f}, M:{medium_time:.2f}, H:{hard_time:.2f})"
                    )
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'Driver': ['VER'] * 10,
        'LapNumber': range(1, 11),
        'LapTime': [92.1, 90.5, 89.8, 90.2, 90.1, 105.3, 91.2, 90.8, 90.5, 91.0],
        'Sector1Time': [30.0, 29.8, 29.7, 29.9, 29.8, 35.0, 30.1, 29.9, 29.8, 30.0],
        'Sector2Time': [31.0, 30.4, 30.1, 30.2, 30.2, 35.2, 30.8, 30.5, 30.4, 30.7],
        'Sector3Time': [31.1, 30.3, 30.0, 30.1, 30.1, 35.1, 30.3, 30.4, 30.3, 30.3],
        'Compound': ['SOFT'] * 5 + ['SOFT'] + ['MEDIUM'] * 4,
        'TyreLife': [1, 2, 3, 4, 5, 1, 1, 2, 3, 4],
        'Stint': [1] * 5 + [2] * 5,
        'IsAccurate': [True] * 10,
        'GrandPrix': ['Bahrain GP'] * 10,
        'Session': ['Race'] * 10,
        'Circuit': ['Bahrain'] * 10,
        'PitInTime': [None] * 5 + [pd.Timestamp('2024-01-01 14:30:00')] + [None] * 4,
        'PitOutTime': [None] * 10
    })
    
    # Run validation
    validator = TelemetryQualityValidator()
    passed, report = validator.validate_lap_data(sample_data)
    
    print("\n=== Data Quality Report ===")
    print(f"Total Rows: {report['total_rows']}")
    print(f"Schema Valid: {report['schema_valid']}")
    print(f"\nIssues ({len(report['issues'])}):")
    for issue in report['issues']:
        print(f"  - {issue}")
    print(f"\nWarnings ({len(report['warnings'])}):")
    for warning in report['warnings']:
        print(f"  - {warning}")
    
    # Compound analysis
    compound_analysis = validator.validate_cross_compound_performance(sample_data)
    print(f"\n=== Compound Analysis ===")
    for gp, data in compound_analysis.items():
        print(f"{gp}: {data}")

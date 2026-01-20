"""
FastF1 Telemetry Data Ingestion Module

Handles fetching and initial processing of F1 telemetry data from the FastF1 API.
Implements caching strategies and data extraction for downstream processing.
"""

import fastf1
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryLoader:
    """Loads F1 telemetry data using FastF1 library."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize telemetry loader with caching.
        
        Args:
            cache_dir: Directory for FastF1 cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        logger.info(f"FastF1 cache enabled at {self.cache_dir}")
    
    def load_session(
        self, 
        year: int, 
        grand_prix: str, 
        session: str = 'R'
    ) -> fastf1.core.Session:
        """
        Load a specific F1 session.
        
        Args:
            year: Season year (e.g., 2024)
            grand_prix: Grand Prix name (e.g., 'Bahrain', 'Monaco')
            session: Session identifier ('FP1', 'FP2', 'FP3', 'Q', 'R')
        
        Returns:
            FastF1 Session object with loaded data
        """
        logger.info(f"Loading {year} {grand_prix} GP - Session: {session}")
        
        try:
            session_obj = fastf1.get_session(year, grand_prix, session)
            session_obj.load()
            logger.info(f"Session loaded successfully: {session_obj.event['EventName']}")
            return session_obj
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            raise
    
    def extract_lap_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract lap-level data from session.
        
        Args:
            session: Loaded FastF1 session
        
        Returns:
            DataFrame with lap-level information
        """
        laps = session.laps
        
        # Select relevant columns
        lap_columns = [
            'Time', 'Driver', 'DriverNumber', 'LapTime', 'LapNumber',
            'Stint', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time', 
            'Sector3Time', 'Compound', 'TyreLife', 'FreshTyre',
            'TrackStatus', 'IsAccurate', 'LapStartTime'
        ]
        
        # Filter to available columns
        available_cols = [col for col in lap_columns if col in laps.columns]
        lap_df = laps[available_cols].copy()
        
        # Add metadata
        lap_df['Year'] = session.event['EventDate'].year
        lap_df['GrandPrix'] = session.event['EventName']
        lap_df['Session'] = session.name
        lap_df['Circuit'] = session.event['Location']
        
        logger.info(f"Extracted {len(lap_df)} laps from session")
        return lap_df
    
    def extract_telemetry(
        self, 
        session: fastf1.core.Session,
        driver: str,
        lap_number: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract detailed telemetry data for specific driver/lap.
        
        Args:
            session: Loaded FastF1 session
            driver: Driver identifier (e.g., 'VER', 'HAM')
            lap_number: Specific lap number (None for fastest lap)
        
        Returns:
            DataFrame with telemetry data (Speed, Throttle, Brake, etc.)
        """
        driver_laps = session.laps.pick_driver(driver)
        
        if lap_number:
            lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
        else:
            lap = driver_laps.pick_fastest()
        
        telemetry = lap.get_telemetry()
        
        # Add lap context
        telemetry['Driver'] = driver
        telemetry['LapNumber'] = lap['LapNumber']
        telemetry['LapTime'] = lap['LapTime'].total_seconds()
        telemetry['Compound'] = lap['Compound']
        
        logger.info(f"Extracted telemetry for {driver} - Lap {lap['LapNumber']}")
        return telemetry
    
    def extract_weather_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract weather and track condition data.
        
        Args:
            session: Loaded FastF1 session
        
        Returns:
            DataFrame with weather conditions
        """
        weather = session.weather_data
        
        if weather is not None:
            weather['GrandPrix'] = session.event['EventName']
            weather['Session'] = session.name
            logger.info(f"Extracted {len(weather)} weather data points")
            return weather
        else:
            logger.warning("No weather data available for session")
            return pd.DataFrame()
    
    def batch_load_season(
        self, 
        year: int, 
        session_type: str = 'R',
        exclude_gp: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all race sessions for a complete season.
        
        Args:
            year: Season year
            session_type: Session type to load for each GP
            exclude_gp: List of GP names to exclude
        
        Returns:
            Dictionary mapping GP names to lap DataFrames
        """
        exclude_gp = exclude_gp or []
        schedule = fastf1.get_event_schedule(year)
        
        season_data = {}
        
        for idx, event in schedule.iterrows():
            gp_name = event['EventName']
            
            if gp_name in exclude_gp:
                logger.info(f"Skipping {gp_name}")
                continue
            
            try:
                session = self.load_session(year, gp_name, session_type)
                lap_data = self.extract_lap_data(session)
                season_data[gp_name] = lap_data
                logger.info(f"✓ Loaded {gp_name}: {len(lap_data)} laps")
            except Exception as e:
                logger.error(f"✗ Failed to load {gp_name}: {e}")
                continue
        
        logger.info(f"Season batch load complete: {len(season_data)} GPs loaded")
        return season_data


# Example usage
if __name__ == "__main__":
    loader = TelemetryLoader()
    
    # Load 2024 Bahrain GP Race
    session = loader.load_session(2024, 'Bahrain', 'R')
    
    # Extract lap data
    laps = loader.extract_lap_data(session)
    print(f"\nLap Data Shape: {laps.shape}")
    print(laps.head())
    
    # Extract telemetry for fastest Verstappen lap
    telemetry = loader.extract_telemetry(session, 'VER')
    print(f"\nTelemetry Data Shape: {telemetry.shape}")
    print(telemetry.head())

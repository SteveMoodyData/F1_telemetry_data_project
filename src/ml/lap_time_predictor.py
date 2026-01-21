"""
Machine Learning Module for Predictive Lap Time Analysis

Implements:
- Feature engineering from telemetry data
- Multiple model architectures (XGBoost, LightGBM, Random Forest)
- Hyperparameter optimization with Optuna
- Model evaluation and performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LapTimeFeatureEngineer:
    """Creates ML features from F1 telemetry data."""
    
    def __init__(self):
        self.label_encoders = {}
    
    def engineer_lap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for lap time prediction.
        
        Features include:
        - Driver characteristics
        - Tire compound and age
        - Track conditions
        - Historical performance
        - Session progression
        """
        logger.info("Engineering lap time prediction features...")
        
        df = df.copy()
        
        # === Tire Features ===
        df['TyreLifeBin'] = pd.cut(
            df['TyreLife'], 
            bins=[0, 5, 10, 15, 20, 100],
            labels=['Fresh', 'Early', 'Mid', 'Late', 'VeryOld']
        )
        
        df['IsFreshTyre'] = (df['TyreLife'] <= 2).astype(int)
        
        # === Session Progression Features ===
        df['SessionProgress'] = df['LapNumber'] / df.groupby(
            ['GrandPrix', 'Session', 'Driver']
        )['LapNumber'].transform('max')
        
        df['LapNumberBin'] = pd.cut(
            df['SessionProgress'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Early', 'EarlyMid', 'LateMid', 'Late']
        )
        
        # === Historical Performance Features ===
        # Rolling average of last 3 laps
        df['LapTime_Rolling3'] = df.groupby(
            ['GrandPrix', 'Session', 'Driver']
        )['LapTime'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Best lap so far in session
        df['BestLapSoFar'] = df.groupby(
            ['GrandPrix', 'Session', 'Driver']
        )['LapTime'].transform('cummin')
        
        # Delta to own best
        df['DeltaToBest'] = df['LapTime'] - df['BestLapSoFar']
        
        # === Driver Performance Features ===
        # Average pace vs field
        session_fastest = df.groupby(
            ['GrandPrix', 'Session']
        )['LapTime'].transform('min')
        
        df['DeltaToSessionFastest'] = df['LapTime'] - session_fastest
        
        # === Sector Features ===
        if all(col in df.columns for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']):
            # Sector as percentage of lap time
            df['Sector1Pct'] = df['Sector1Time'] / df['LapTime'] * 100
            df['Sector2Pct'] = df['Sector2Time'] / df['LapTime'] * 100
            df['Sector3Pct'] = df['Sector3Time'] / df['LapTime'] * 100
        
        # === Stint Features ===
        df['LapsIntoStint'] = df.groupby(
            ['GrandPrix', 'Session', 'Driver', 'Stint']
        ).cumcount() + 1
        
        # === Track Evolution Features ===
        # Track typically gets faster over session
        df['TrackEvolution'] = df.groupby(
            ['GrandPrix', 'Session']
        )['LapTime'].transform(
            lambda x: x.expanding().mean()
        )
        
        logger.info(f"Feature engineering complete: {len(df.columns)} features")
        
        return df
    
    def prepare_model_data(
        self, 
        df: pd.DataFrame,
        target_col: str = 'LapTime',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for ML model training.
        
        Args:
            df: DataFrame with engineered features
            target_col: Target variable column name
            exclude_cols: Columns to exclude from features
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing model training data...")
        
        if exclude_cols is None:
            exclude_cols = [
                'LapTime', 'Time', 'LapStartTime', 'PitInTime', 'PitOutTime',
                'TrackStatus', 'IsAccurate', 'Sector1Time', 'Sector2Time', 
                'Sector3Time', 'FreshTyre', 'ingestion_timestamp', 'record_source'
            ]
        
        # Separate target
        y = df[target_col].copy()
        
        # Select feature columns
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and col != target_col
        ]
        
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(
                    X[col].astype(str)
                )
            else:
                X[col] = self.label_encoders[col].transform(
                    X[col].astype(str)
                )
        
        # Handle missing values
        X = X.fillna(X.median())
        
        logger.info(
            f"Model data prepared: {X.shape[0]} samples, "
            f"{X.shape[1]} features, {len(categorical_cols)} categorical"
        )
        
        return X, y, X.columns.tolist()


class LapTimePredictorModel:
    """ML model for predicting F1 lap times."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize predictor model.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        optimize: bool = False
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            optimize: Whether to run hyperparameter optimization
        """
        logger.info(f"Training {self.model_type} model...")
        
        if optimize:
            logger.info("Running hyperparameter optimization with Optuna...")
            best_params = self._optimize_hyperparameters(X_train, y_train)
            params = best_params
        else:
            params = self._get_default_params()
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**params, random_state=42)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**params, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("Model training complete")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(
            f"Performance: MAE={metrics['mae']:.3f}s, "
            f"RMSE={metrics['rmse']:.3f}s, R²={metrics['r2']:.3f}"
        )
        
        return metrics
    
    def _get_default_params(self) -> Dict:
        """Get default hyperparameters for each model type."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
    
    def _optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_trials: int = 50
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Training features
            y: Training target
            n_trials: Number of optimization trials
        
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = xgb.XGBRegressor(**params, random_state=42)
            
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            
            # Cross-validation
            scores = cross_val_score(
                model, X, y, cv=5, 
                scoring='neg_mean_absolute_error'
            )
            
            return -scores.mean()  # Optuna minimizes
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best MAE: {study.best_value:.3f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importance = data['feature_importance']
        self.model_type = data['model_type']
        logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'Driver': np.random.choice(['VER', 'HAM', 'LEC'], n_samples),
        'LapNumber': np.random.randint(1, 60, n_samples),
        'Compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], n_samples),
        'TyreLife': np.random.randint(1, 30, n_samples),
        'Stint': np.random.randint(1, 4, n_samples),
        'LapTime': 90 + np.random.randn(n_samples) * 2,
        'GrandPrix': 'Bahrain GP',
        'Session': 'Race',
        'Circuit': 'Bahrain'
    })
    
    # Feature engineering
    engineer = LapTimeFeatureEngineer()
    df_features = engineer.engineer_lap_features(sample_df)
    
    # Prepare data
    X, y, feature_names = engineer.prepare_model_data(df_features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LapTimePredictorModel(model_type='xgboost')
    model.train(X_train, y_train, optimize=False)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\n=== Model Performance ===")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Feature importance
    print("\n=== Top 10 Features ===")
    print(model.feature_importance.head(10))

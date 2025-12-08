"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging
from datetime import datetime, timedelta
import json
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

logger = logging.getLogger(__name__)


class BacktestResults:
    """Store and manage backtest results"""
    
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        
    def add_result(self, pred: float, actual: float, timestamp: datetime):
        self.predictions.append(pred)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)
        
    def get_metrics(self) -> Dict:
        """Calculate backtest metrics"""
        if len(self.predictions) < 2:
            return {}
            
        preds = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        mse = mean_squared_error(actuals, preds)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - actuals))
        mape = mean_absolute_percentage_error(actuals, preds)
        
        # Direction accuracy (did we get the direction right?)
        direction_correct = 0
        for i in range(1, len(preds)):
            pred_direction = 1 if preds[i] > preds[i-1] else -1
            actual_direction = 1 if actuals[i] > actuals[i-1] else -1
            if pred_direction == actual_direction:
                direction_correct += 1
        
        direction_accuracy = (direction_correct / (len(preds) - 1)) * 100 if len(preds) > 1 else 0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'num_predictions': len(self.predictions)
        }


class ConfidenceIntervalCalculator:
    """Calculate prediction confidence intervals using multiple methods"""
    
    def __init__(self, lookback_periods: int = 30):
        self.lookback_periods = lookback_periods
        self.historical_errors = []
        
    def add_error(self, error: float):
        """Track historical prediction errors"""
        self.historical_errors.append(abs(error))
        # Keep only recent errors
        if len(self.historical_errors) > self.lookback_periods:
            self.historical_errors.pop(0)
    
    def calculate_ci(self, prediction: float, confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for prediction
        
        Returns:
            Tuple of (lower_bound, upper_bound, margin_of_error)
        """
        if not self.historical_errors:
            # Default: ±5% margin if no history
            margin = prediction * 0.05
            return prediction - margin, prediction + margin, margin
        
        # Calculate margin based on historical error distribution
        historical_std = np.std(self.historical_errors)
        historical_mean = np.mean(self.historical_errors)
        
        # Use mean absolute error as base margin
        margin = historical_mean * 1.96  # ~95% confidence
        
        lower = prediction - margin
        upper = prediction + margin
        
        return float(lower), float(upper), float(margin)


class EnhancedHybridPredictor:
    """Enhanced hybrid model with confidence intervals and backtesting"""
    
    def __init__(
        self,
        window_size: int = 30,
        lstm_units: List[int] = [64, 32],
        dropout_rates: List[float] = [0.15, 0.10],
        xgb_params: dict = None,
        enable_attention: bool = True
    ):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        self.enable_attention = enable_attention
        
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.price_scaler = None
        self.confidence_calculator = ConfidenceIntervalCalculator()
        self.backtest_results = BacktestResults()
        
        # Model accuracy tracking
        self.model_accuracy_history = {
            'lstm': [],
            'xgboost': [],
            'ensemble': []
        }
        
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Enhanced feature engineering with additional indicators"""
        df = pd.DataFrame({'price': price_series.astype(float)})
        
        # Log returns
        df['returns'] = np.log(df['price']).diff()
        
        # Moving averages
        df['ma7'] = df['price'].rolling(7, min_periods=1).mean()
        df['ma14'] = df['price'].rolling(14, min_periods=1).mean()
        df['ma30'] = df['price'].rolling(30, min_periods=1).mean()
        
        # MA distances (normalized)
        df['ma7_dist'] = (df['price'] - df['ma7']) / df['price']
        df['ma14_dist'] = (df['price'] - df['ma14']) / df['price']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['price'], period=14)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        
        # Momentum
        df['momentum'] = df['price'].pct_change(periods=14)
        
        # Price range (volatility proxy)
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - \
                           df['price'].rolling(7, min_periods=1).min()
        
        # NEW: Bollinger Band position
        sma = df['price'].rolling(20, min_periods=1).mean()
        std = df['price'].rolling(20, min_periods=1).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        df['bb_position'] = (df['price'] - lower_band) / (upper_band - lower_band)
        df['bb_position'] = df['bb_position'].clip(0, 1)  # Normalize to 0-1
        
        # NEW: Rate of change
        df['roc'] = df['price'].pct_change(periods=5)
        
        # NEW: Volatility trend
        df['volatility_trend'] = df['volatility'].rolling(5, min_periods=1).mean()
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM with optional attention mechanism"""
        model = keras.Sequential()
        
        # Add LSTM layers
        for i, (units, dropout) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            return_sequences = i < len(self.lstm_units) - 1 or self.enable_attention
            
            model.add(layers.LSTM(
                units,
                input_shape=input_shape if i == 0 else None,
                return_sequences=return_sequences
            ))
            
            # Add attention if enabled and not last layer
            if self.enable_attention and return_sequences:
                model.add(layers.MultiHeadAttention(
                    num_heads=4,
                    key_dim=min(32, units // 2),
                    dropout=dropout
                ))
            
            model.add(layers.Dropout(dropout))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm(
        self,
        price_series: pd.Series,
        epochs: int = 20,
        batch_size: int = 16,
        validation_split: float = 0.1
    ) -> dict:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        df = self.prepare_features(price_series)
        
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 
                       'volatility', 'momentum', 'bb_position', 'roc']
        features = df[feature_cols].values
        target = df['returns'].shift(-1).fillna(0).values
        
        # Scale features
        self.feature_scaler = MinMaxScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.window_size, len(features_scaled)):
            X.append(features_scaled[i-self.window_size:i])
            y.append(target[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.lstm_model = self.build_lstm_model((self.window_size, X.shape[2]))
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Reduce verbosity for Streamlit compatibility
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Calculate validation metrics
        val_predictions = self.lstm_model.predict(X_val, verbose=0).flatten()
        val_mae = np.mean(np.abs(val_predictions - y_val))
        
        logger.info(f"LSTM training complete. Val MAE: {val_mae:.6f}")
        
        return {
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_mae': float(val_mae),
            'epochs_trained': len(history.history['loss'])
        }
    
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        df = self.prepare_features(price_series)
        
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]
        y = df['returns'].shift(-1).dropna().values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )
        
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        train_score = self.xgb_model.score(X_train, y_train)
        test_score = self.xgb_model.score(X_test, y_test)
        
        logger.info(f"XGBoost training complete. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return {
            'train_r2': float(train_score),
            'test_r2': float(test_score)
        }
    
    def predict_lstm(
        self,
        price_series: pd.Series,
        horizon: int = 7
    ) -> List[float]:
        """Predict future prices using LSTM"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
        
        df = self.prepare_features(price_series)
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 
                       'volatility', 'momentum', 'bb_position', 'roc']
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        current_window = features_scaled[-self.window_size:]
        predictions = []
        current_price = float(price_series.iloc[-1])
        
        for _ in range(horizon):
            X = current_window.reshape(1, self.window_size, -1)
            next_return = self.lstm_model.predict(X, verbose=0)[0, 0]
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            
            new_features = current_window[-1].copy()
            new_features[0] = next_return
            current_window = np.vstack([current_window[1:], new_features])
            current_price = next_price
        
        return predictions
    
    def predict_xgboost(
        self,
        price_series: pd.Series,
        horizon: int = 7
    ) -> List[float]:
        """Predict future prices using XGBoost"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
        
        df = self.prepare_features(price_series)
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        
        predictions = []
        current_price = float(price_series.iloc[-1])
        current_features = df[feature_cols].iloc[-1:].values
        
        for _ in range(horizon):
            next_return = self.xgb_model.predict(current_features)[0]
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            current_price = next_price
        
        return predictions
    
    def predict_ensemble_with_ci(
        self,
        price_series: pd.Series,
        horizon: int = 7,
        lstm_weight: float = 0.7,
        xgb_weight: float = 0.3
    ) -> Dict:
        """Generate ensemble predictions with confidence intervals"""
        logger.info(f"Generating {horizon}-day forecast with confidence intervals...")
        
        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)
        
        ensemble = [
            lstm_weight * l + xgb_weight * x
            for l, x in zip(lstm_preds, xgb_preds)
        ]
        
        # Calculate confidence intervals
        predictions_with_ci = []
        for i, pred in enumerate(ensemble):
            lower, upper, margin = self.confidence_calculator.calculate_ci(pred)
            predictions_with_ci.append({
                'price': float(pred),
                'lower_ci': float(lower),
                'upper_ci': float(upper),
                'margin': float(margin),
                'day': i + 1
            })
        
        logger.info("Ensemble forecast with confidence intervals complete")
        
        return {
            'ensemble': ensemble,
            'lstm': lstm_preds,
            'xgboost': xgb_preds,
            'predictions_with_ci': predictions_with_ci
        }
    
    def backtest_on_historical(
        self,
        price_series: pd.Series,
        test_periods: int = 5
    ) -> Dict:
        """
        Perform walk-forward backtesting
        
        Args:
            price_series: Historical prices
            test_periods: Number of periods to test
            
        Returns:
            Backtest metrics and results
        """
        logger.info(f"Running walk-forward backtest with {test_periods} periods...")
        
        backtest = BacktestResults()
        
        # Use last N periods for testing
        test_start_idx = len(price_series) - test_periods - 7
        
        for test_idx in range(test_start_idx, len(price_series) - 7):
            # Train on data up to test_idx
            train_data = price_series.iloc[:test_idx]
            
            if len(train_data) < self.window_size + 10:
                continue
            
            try:
                # Train models
                self.train_lstm(train_data, epochs=10)
                self.train_xgboost(train_data)
                
                # Predict next 7 days
                results = self.predict_ensemble_with_ci(train_data, horizon=7)
                next_pred = results['ensemble'][0]
                
                # Actual price 1 day ahead
                actual_price = price_series.iloc[test_idx + 1]
                
                backtest.add_result(next_pred, actual_price, price_series.index[test_idx])
                
            except Exception as e:
                logger.warning(f"Backtest period failed: {e}")
                continue
        
        metrics = backtest.get_metrics()
        
        return {
            'metrics': metrics,
            'backtest_data': backtest
        }
    
    def get_model_accuracy(self) -> Dict:
        """Get historical model accuracy"""
        return {
            'lstm_accuracy': float(np.mean(self.model_accuracy_history['lstm'])) if self.model_accuracy_history['lstm'] else 0.0,
            'xgb_accuracy': float(np.mean(self.model_accuracy_history['xgboost'])) if self.model_accuracy_history['xgboost'] else 0.0,
            'ensemble_accuracy': float(np.mean(self.model_accuracy_history['ensemble'])) if self.model_accuracy_history['ensemble'] else 0.0,
        }


def train_and_predict_enhanced(
    price_series: pd.Series,
    horizon: int = 7,
    enable_backtest: bool = False,
    **kwargs
) -> dict:
    """Enhanced training and prediction with backtesting option"""
    predictor = EnhancedHybridPredictor(**kwargs)
    
    # Train models
    lstm_history = predictor.train_lstm(price_series, epochs=15)
    xgb_history = predictor.train_xgboost(price_series)
    
    # Generate predictions with confidence intervals
    results = predictor.predict_ensemble_with_ci(price_series, horizon)
    
    # Optional: Run backtesting (disabled by default for free tier)
    backtest_results = None
    if enable_backtest and len(price_series) > 100:
        try:
            backtest_results = predictor.backtest_on_historical(price_series, test_periods=5)
        except Exception as e:
            logger.warning(f"Backtesting failed: {e}")
    
    return {
        'lstm': results['lstm'],
        'xgboost': results['xgboost'],
        'ensemble': results['ensemble'],
        'predictions_with_ci': results['predictions_with_ci'],
        'lstm_history': lstm_history,
        'xgb_history': xgb_history,
        'backtest': backtest_results,
        'predictor': predictor
    }
    }

"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging

# ML imports
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class HybridPredictor:
    """Hybrid model combining LSTM and XGBoost for price prediction"""
    
    def __init__(
        self,
        window_size: int = 30,
        lstm_units: List[int] = [64, 32],
        dropout_rates: List[float] = [0.15, 0.10],
        xgb_params: dict = None
    ):
        """
        Initialize hybrid predictor
        """
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        
        # XGBoost default parameters
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
        # We need a dedicated scaler for price to inverse transform predictions
        self.price_scaler = MinMaxScaler() 
        
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Prepare features from price series"""
        df = pd.DataFrame({'price': price_series.astype(float)})
        
        # Log returns
        df['returns'] = np.log(df['price']).diff()
        
        # Moving averages
        df['ma7'] = df['price'].rolling(7, min_periods=1).mean()
        df['ma14'] = df['price'].rolling(14, min_periods=1).mean()
        
        # MA distances (normalized)
        df['ma7_dist'] = (df['price'] - df['ma7']) / df['price']
        df['ma14_dist'] = (df['price'] - df['ma14']) / df['price']
        
        # RSI (Simple implementation to avoid circular dependency)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        
        # Momentum
        df['momentum'] = df['price'].pct_change(periods=14)
        
        return df.bfill().fillna(0)

    def train_lstm(self, price_series: pd.Series, epochs: int = 15, batch_size: int = 16):
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        df = self.prepare_features(price_series)
        
        # Features to use
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 'volatility', 'momentum']
        data = df[feature_cols].values
        
        # Scale features
        self.feature_scaler = MinMaxScaler()
        data_scaled = self.feature_scaler.fit_transform(data)
        
        # Prepare Price Scaler for later inversion (fitting on price directly)
        self.price_scaler.fit(df[['price']])
        
        # Create sequences
        X, y = [], []
        # Target is next day's return
        target = df['returns'].shift(-1).fillna(0).values
        
        for i in range(self.window_size, len(data_scaled)):
            X.append(data_scaled[i-self.window_size:i])
            y.append(target[i])
            
        X, y = np.array(X), np.array(y)
        
        # Build Model
        model = keras.Sequential()
        model.add(layers.LSTM(self.lstm_units[0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(layers.Dropout(self.dropout_rates[0]))
        model.add(layers.LSTM(self.lstm_units[1], return_sequences=False))
        model.add(layers.Dropout(self.dropout_rates[1]))
        model.add(layers.Dense(1))
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.lstm_model = model

    def train_xgboost(self, price_series: pd.Series):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        df = self.prepare_features(price_series)
        
        # Features
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 'volatility', 'momentum']
        X = df[feature_cols].values
        # Target: next day return
        y = df['returns'].shift(-1).fillna(0).values
        
        # Remove last row (no target)
        X = X[:-1]
        y = y[:-1]
        
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y)

    def predict_lstm(self, price_series: pd.Series, horizon: int) -> List[float]:
        """Generate LSTM predictions"""
        df = self.prepare_features(price_series)
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 'volatility', 'momentum']
        
        # Initial sequence
        last_sequence = df[feature_cols].iloc[-self.window_size:].values
        last_sequence_scaled = self.feature_scaler.transform(last_sequence)
        current_seq = last_sequence_scaled.reshape(1, self.window_size, len(feature_cols))
        
        current_price = float(price_series.iloc[-1])
        predictions = []
        
        for _ in range(horizon):
            # Predict log return
            pred_return = self.lstm_model.predict(current_seq, verbose=0)[0][0]
            
            # Convert to price
            next_price = current_price * np.exp(pred_return)
            predictions.append(next_price)
            
            # Update sequence (simplified: roll and append new return, keep other features static/last known)
            # In a real PRO system, you'd re-calculate technicals based on new price. 
            # For speed, we just update the return feature.
            new_step = current_seq[0, -1, :].copy()
            new_step[0] = pred_return # Update returns column
            
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_step
            current_price = next_price
            
        return predictions

    def predict_xgboost(self, price_series: pd.Series, horizon: int) -> List[float]:
        """Generate XGBoost predictions"""
        df = self.prepare_features(price_series)
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 'volatility', 'momentum']
        
        last_row = df[feature_cols].iloc[-1:].values
        current_price = float(price_series.iloc[-1])
        predictions = []
        
        for _ in range(horizon):
            pred_return = self.xgb_model.predict(last_row)[0]
            next_price = current_price * np.exp(pred_return)
            predictions.append(next_price)
            
            # Simple update
            last_row[0, 0] = pred_return
            current_price = next_price
            
        return predictions

    def predict_ensemble(
        self, 
        price_series: pd.Series, 
        horizon: int = 7, 
        lstm_weight: float = 0.7, 
        xgb_weight: float = 0.3
    ) -> Dict:
        """
        Generate ensemble forecast with Confidence Intervals
        """
        logger.info(f"Generating {horizon}-day ensemble forecast with CI...")
        
        lstm_preds = np.array(self.predict_lstm(price_series, horizon))
        xgb_preds = np.array(self.predict_xgboost(price_series, horizon))
        
        # Weighted Ensemble
        ensemble_mean = (lstm_preds * lstm_weight) + (xgb_preds * xgb_weight)
        
        # Calculate Disagreement (Model Uncertainty)
        # We use the absolute difference between models as a proxy for uncertainty
        model_disagreement = np.abs(lstm_preds - xgb_preds)
        
        # Base volatility factor from recent history (last 30 days)
        recent_vol = price_series.pct_change().std()
        if np.isnan(recent_vol): recent_vol = 0.02
        
        # Construct CI: Widens with time and model disagreement
        lower_bound = []
        upper_bound = []
        
        for i, price in enumerate(ensemble_mean):
            # Uncertainty grows with sqrt of time (t)
            # Factor 1.96 is for 95% CI roughly
            time_factor = np.sqrt(i + 1)
            
            # Combine historical volatility and model disagreement
            sigma = (recent_vol * time_factor * price) + (model_disagreement[i] * 0.5)
            
            lower_bound.append(float(price - (1.96 * sigma)))
            upper_bound.append(float(price + (1.96 * sigma)))
            
        return {
            "mean": ensemble_mean.tolist(),
            "lower": lower_bound,
            "upper": upper_bound,
            "disagreement_index": float(np.mean(model_disagreement) / np.mean(ensemble_mean)),
            "lstm_preds": lstm_preds.tolist(),
            "xgb_preds": xgb_preds.tolist()
        }


def train_and_predict(price_series: pd.Series, horizon: int = 7, **kwargs) -> dict:
    """
    Train models and generate predictions in one call
    """
    predictor = HybridPredictor(**kwargs)
    predictor.train_lstm(price_series)
    predictor.train_xgboost(price_series)
    
    ensemble_data = predictor.predict_ensemble(price_series, horizon)
    
    return {
        'lstm': ensemble_data['lstm_preds'],
        'xgboost': ensemble_data['xgb_preds'],
        'ensemble': ensemble_data['mean'],
        'confidence_intervals': {
            'lower': ensemble_data['lower'],
            'upper': ensemble_data['upper']
        },
        'risk_metrics': {
            'model_disagreement': ensemble_data['disagreement_index']
        }
    }

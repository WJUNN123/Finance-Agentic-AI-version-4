"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

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
        
        Args:
            window_size: Number of days to use for prediction
            lstm_units: List of LSTM layer units
            dropout_rates: Dropout rates for each LSTM layer
            xgb_params: XGBoost parameters
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
        self.price_scaler = None
        
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Prepare features from price series
        
        Args:
            price_series: Pandas Series of prices
            
        Returns:
            DataFrame with engineered features
        """
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
        
        # Volume-like proxy (using price volatility)
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - \
                           df['price'].rolling(7, min_periods=1).min()
        
        return df.bfill().fillna(0)
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM neural network"""
        model = keras.Sequential()
        
        # Add LSTM layers
        for i, (units, dropout) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(layers.LSTM(
                units,
                input_shape=input_shape if i == 0 else None,
                return_sequences=return_sequences
            ))
            model.add(layers.Dropout(dropout))
            
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile
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
        """
        Train LSTM model
        
        Args:
            price_series: Historical prices
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Training history
        """
        logger.info("Training LSTM model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Select features for LSTM
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 
                       'volatility', 'momentum']
        features = df[feature_cols].values
        
        # Target: next day's return
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
        
        # Build and train model
        self.lstm_model = self.build_lstm_model((self.window_size, X.shape[2]))
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        logger.info(f"LSTM training complete. Final loss: {history.history['loss'][-1]:.6f}")
        
        return history.history
        
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """
        Train XGBoost model
        
        Args:
            price_series: Historical prices
            
        Returns:
            Model metrics
        """
        logger.info("Training XGBoost model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Use all features
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]  # Exclude last row (no target)
        y = df['returns'].shift(-1).dropna().values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )
        
        # Train model
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Calculate metrics
        train_score = self.xgb_model.score(X_train, y_train)
        test_score = self.xgb_model.score(X_test, y_test)
        
        logger.info(f"XGBoost training complete. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score
        }
        
    def predict_lstm(
        self,
        price_series: pd.Series,
        horizon: int = 7
    ) -> List[float]:
        """Predict future prices using LSTM"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained. Call train_lstm() first.")
            
        df = self.prepare_features(price_series)
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'rsi', 
                       'volatility', 'momentum']
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        # Start with last window
        current_window = features_scaled[-self.window_size:]
        predictions = []
        current_price = float(price_series.iloc[-1])
        
        for _ in range(horizon):
            # Predict next return
            X = current_window.reshape(1, self.window_size, -1)
            next_return = self.lstm_model.predict(X, verbose=0)[0, 0]
            
            # Convert return to price
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            
            # Update window (simplified - just repeat last features with new return)
            new_features = current_window[-1].copy()
            new_features[0] = next_return  # Update return
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
            raise ValueError("XGBoost model not trained. Call train_xgboost() first.")
            
        df = self.prepare_features(price_series)
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        
        predictions = []
        current_price = float(price_series.iloc[-1])
        current_features = df[feature_cols].iloc[-1:].values
        
        for _ in range(horizon):
            # Predict next return
            next_return = self.xgb_model.predict(current_features)[0]
            
            # Convert to price
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            
            # Update features (simplified)
            current_price = next_price
            
        return predictions
        
    def predict_ensemble(
        self,
        price_series: pd.Series,
        horizon: int = 7,
        lstm_weight: float = 0.7,
        xgb_weight: float = 0.3
    ) -> List[float]:
        """
        Generate ensemble prediction combining LSTM and XGBoost
        
        Args:
            price_series: Historical prices
            horizon: Days to forecast
            lstm_weight: Weight for LSTM predictions
            xgb_weight: Weight for XGBoost predictions
            
        Returns:
            List of predicted prices
        """
        logger.info(f"Generating {horizon}-day ensemble forecast...")
        
        # Get predictions from both models
        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)
        
        # Combine predictions
        ensemble = [
            lstm_weight * lstm + xgb_weight * xgb
            for lstm, xgb in zip(lstm_preds, xgb_preds)
        ]
        
        logger.info("Ensemble forecast complete")
        return ensemble


# Convenience function
def train_and_predict(
    price_series: pd.Series,
    horizon: int = 7,
    **kwargs
) -> dict:
    """
    Train models and generate predictions in one call
    
    Returns:
        Dictionary with lstm_preds, xgb_preds, ensemble_preds
    """
    predictor = HybridPredictor(**kwargs)
    
    # Train both models
    predictor.train_lstm(price_series)
    predictor.train_xgboost(price_series)
    
    # Generate predictions
    return {
        'lstm': predictor.predict_lstm(price_series, horizon),
        'xgboost': predictor.predict_xgboost(price_series, horizon),
        'ensemble': predictor.predict_ensemble(price_series, horizon)
    }

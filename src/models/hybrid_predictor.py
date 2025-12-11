"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging
import pickle
import hashlib
from pathlib import Path
import functools
import time

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

logger = logging.getLogger(__name__)


def log_performance(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"🚀 Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"✅ {func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


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
    
    def _get_model_hash(self, price_series: pd.Series) -> str:
        """Generate hash for model caching based on data characteristics"""
        # Create hash from data length and recent price points
        data_str = f"{len(price_series)}_{price_series.iloc[-1]:.2f}_{price_series.iloc[0]:.2f}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    def save_models(self, coin_id: str, price_series: pd.Series):
        """
        Save trained models to disk for reuse
        
        Args:
            coin_id: Cryptocurrency identifier
            price_series: Price series used for training
        """
        model_dir = Path('models_cache')
        model_dir.mkdir(exist_ok=True)
        
        model_hash = self._get_model_hash(price_series)
        
        try:
            # Save LSTM model
            if self.lstm_model:
                lstm_path = model_dir / f'{coin_id}_lstm_{model_hash}.h5'
                self.lstm_model.save(lstm_path)
                logger.info(f"💾 Saved LSTM model: {lstm_path.name}")
            
            # Save XGBoost model and scalers
            model_data = {
                'xgb_model': self.xgb_model,
                'feature_scaler': self.feature_scaler,
                'window_size': self.window_size,
                'timestamp': time.time()
            }
            xgb_path = model_dir / f'{coin_id}_xgb_{model_hash}.pkl'
            with open(xgb_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"💾 Saved XGBoost model: {xgb_path.name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save models: {e}")
    
    def load_models(self, coin_id: str, price_series: pd.Series) -> bool:
        """
        Load trained models from disk
        
        Args:
            coin_id: Cryptocurrency identifier
            price_series: Current price series
            
        Returns:
            True if models loaded successfully, False otherwise
        """
        model_dir = Path('models_cache')
        if not model_dir.exists():
            return False
        
        model_hash = self._get_model_hash(price_series)
        lstm_path = model_dir / f'{coin_id}_lstm_{model_hash}.h5'
        xgb_path = model_dir / f'{coin_id}_xgb_{model_hash}.pkl'
        
        try:
            # Load LSTM model
            if lstm_path.exists():
                self.lstm_model = keras.models.load_model(lstm_path)
                logger.info(f"✅ Loaded LSTM model from cache")
            
            # Load XGBoost model and scalers
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.xgb_model = model_data['xgb_model']
                    self.feature_scaler = model_data['feature_scaler']
                    self.window_size = model_data['window_size']
                    
                # Check if model is recent (< 7 days old)
                model_age_days = (time.time() - model_data.get('timestamp', 0)) / 86400
                if model_age_days > 7:
                    logger.warning(f"⚠️ Cached model is {model_age_days:.1f} days old, might need retraining")
                else:
                    logger.info(f"✅ Loaded XGBoost model from cache (age: {model_age_days:.1f} days)")
            
            return self.lstm_model is not None and self.xgb_model is not None
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """
        Prepare enhanced features from price series (20+ features)
        
        Args:
            price_series: Pandas Series of prices
            
        Returns:
            DataFrame with engineered features
        """
        df = pd.DataFrame({'price': price_series.astype(float)})
        
        # Log returns
        df['returns'] = np.log(df['price']).diff()
        
        # Moving averages (multiple periods)
        df['ma7'] = df['price'].rolling(7, min_periods=1).mean()
        df['ma14'] = df['price'].rolling(14, min_periods=1).mean()
        df['ma30'] = df['price'].rolling(30, min_periods=1).mean()
        df['ma50'] = df['price'].rolling(50, min_periods=1).mean()
        
        # MA distances (normalized)
        df['ma7_dist'] = (df['price'] - df['ma7']) / df['price']
        df['ma14_dist'] = (df['price'] - df['ma14']) / df['price']
        df['ma30_dist'] = (df['price'] - df['ma30']) / df['price']
        
        # MA crossovers (binary signals)
        df['ma7_ma30_cross'] = (df['ma7'] > df['ma30']).astype(int)
        df['ma14_ma50_cross'] = (df['ma14'] > df['ma50']).astype(int)
        
        # RSI (multiple periods)
        df['rsi_7'] = self._calculate_rsi(df['price'], period=7)
        df['rsi_14'] = self._calculate_rsi(df['price'], period=14)
        df['rsi_21'] = self._calculate_rsi(df['price'], period=21)
        
        # Volatility (multiple windows)
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        df['volatility_short'] = df['returns'].rolling(7, min_periods=1).std()
        df['volatility_long'] = df['returns'].rolling(50, min_periods=1).std()
        
        # Momentum (multiple periods)
        df['momentum_7'] = df['price'].pct_change(periods=7)
        df['momentum_14'] = df['price'].pct_change(periods=14)
        df['momentum_30'] = df['price'].pct_change(periods=30)
        
        # Price range features
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - \
                           df['price'].rolling(7, min_periods=1).min()
        df['price_range_norm'] = df['price_range'] / df['price']
        
        # Bollinger Bands position
        rolling_mean = df['price'].rolling(20, min_periods=1).mean()
        rolling_std = df['price'].rolling(20, min_periods=1).std()
        df['bb_position'] = (df['price'] - rolling_mean) / (2 * rolling_std + 1e-10)
        
        # Rate of change
        df['roc_10'] = df['price'].pct_change(periods=10) * 100
        df['roc_20'] = df['price'].pct_change(periods=20) * 100
        
        # Price acceleration (change in returns)
        df['acceleration'] = df['returns'].diff()
        
        # Distance from highs/lows
        df['dist_from_high'] = (df['price'].rolling(30, min_periods=1).max() - df['price']) / df['price']
        df['dist_from_low'] = (df['price'] - df['price'].rolling(30, min_periods=1).min()) / df['price']
        
        # Exponential moving averages
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
        
        # MACD components
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        logger.debug(f"📊 Prepared {len(df.columns)} features for training")
        
        return df.bfill().fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
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
    
    @log_performance
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
        logger.info("🧠 Training LSTM model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Select features for LSTM (subset of all features)
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'ma30_dist',
                       'rsi_7', 'rsi_14', 'volatility', 'volatility_short',
                       'momentum_7', 'momentum_14', 'bb_position', 
                       'macd_histogram', 'acceleration']
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
        
        if len(X) < 10:
            logger.warning("⚠️ Insufficient data for LSTM training")
            return {'loss': [0], 'val_loss': [0]}
        
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
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        logger.info(f"✅ LSTM training complete. Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")
        
        return history.history
    
    @log_performance
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """
        Train XGBoost model
        
        Args:
            price_series: Historical prices
            
        Returns:
            Model metrics
        """
        logger.info("🌲 Training XGBoost model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Use all features except price and returns
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]  # Exclude last row (no target)
        y = df['returns'].shift(-1).dropna().values
        
        if len(X) < 10:
            logger.warning("⚠️ Insufficient data for XGBoost training")
            return {'train_score': 0, 'test_score': 0}
        
        # Split data (time-based, no shuffle)
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
        
        logger.info(f"✅ XGBoost training complete. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(feature_cols, self.xgb_model.feature_importances_))
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
        feature_cols = ['returns', 'ma7_dist', 'ma14_dist', 'ma30_dist',
                       'rsi_7', 'rsi_14', 'volatility', 'volatility_short',
                       'momentum_7', 'momentum_14', 'bb_position', 
                       'macd_histogram', 'acceleration']
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
            
            # Update window (simplified - use last features with new return)
            new_features = current_window[-1].copy()
            new_features[0] = next_return  # Update return feature
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
            
            # Update for next iteration
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
            lstm_weight: Weight for LSTM predictions (default 0.7)
            xgb_weight: Weight for XGBoost predictions (default 0.3)
            
        Returns:
            List of predicted prices
        """
        logger.info(f"🎯 Generating {horizon}-day ensemble forecast...")
        
        # Get predictions from both models
        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)
        
        # Combine predictions with weights
        ensemble = [
            lstm_weight * lstm + xgb_weight * xgb
            for lstm, xgb in zip(lstm_preds, xgb_preds)
        ]
        
        logger.info(f"✅ Ensemble forecast complete. Final: ${ensemble[-1]:,.2f}")
        
        return ensemble


# Convenience function with caching support
def train_and_predict(
    price_series: pd.Series,
    horizon: int = 7,
    coin_id: str = 'unknown',
    use_cache: bool = True,
    **kwargs
) -> dict:
    """
    Train models and generate predictions with caching
    
    Args:
        price_series: Historical price series
        horizon: Forecast horizon in days
        coin_id: Cryptocurrency identifier for caching
        use_cache: Whether to use cached models
        **kwargs: Additional parameters for HybridPredictor
        
    Returns:
        Dictionary with lstm, xgboost, and ensemble predictions
    """
    predictor = HybridPredictor(**kwargs)
    
    # Try to load cached models
    models_loaded = False
    if use_cache:
        models_loaded = predictor.load_models(coin_id, price_series)
    
    if models_loaded:
        logger.info(f"⚡ Using cached models for {coin_id}")
    else:
        logger.info(f"🔄 Training new models for {coin_id}")
        predictor.train_lstm(price_series)
        predictor.train_xgboost(price_series)
        
        # Save models for future use
        if use_cache:
            predictor.save_models(coin_id, price_series)
    
    # Generate predictions
    return {
        'lstm': predictor.predict_lstm(price_series, horizon),
        'xgboost': predictor.predict_xgboost(price_series, horizon),
        'ensemble': predictor.predict_ensemble(price_series, horizon)
    }

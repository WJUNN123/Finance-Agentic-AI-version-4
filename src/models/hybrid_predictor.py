"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging
from datetime import datetime

# ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """Enhanced feature engineering with market correlation features"""
    
    @staticmethod
    def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical and statistical features"""
        
        # Existing basic features
        df['returns'] = np.log(df['price']).diff()
        df['ma7'] = df['price'].rolling(7, min_periods=1).mean()
        df['ma14'] = df['price'].rolling(14, min_periods=1).mean()
        df['ma30'] = df['price'].rolling(30, min_periods=1).mean()
        df['ma7_dist'] = (df['price'] - df['ma7']) / df['price']
        df['ma14_dist'] = (df['price'] - df['ma14']) / df['price']
        
        # NEW: EMA features
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # NEW: Multiple timeframe RSI
        df['rsi_7'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 7)
        df['rsi_14'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 14)
        df['rsi_21'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 21)
        
        # NEW: Bollinger Band features
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['price'].rolling(bb_period, min_periods=1).mean()
        bb_rolling_std = df['price'].rolling(bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * bb_rolling_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * bb_rolling_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # NEW: Volatility features
        df['volatility_7'] = df['returns'].rolling(7, min_periods=1).std()
        df['volatility_14'] = df['returns'].rolling(14, min_periods=1).std()
        df['volatility_30'] = df['returns'].rolling(30, min_periods=1).std()
        
        # NEW: Momentum features
        df['momentum_7'] = df['price'].pct_change(periods=7)
        df['momentum_14'] = df['price'].pct_change(periods=14)
        df['momentum_30'] = df['price'].pct_change(periods=30)
        
        # NEW: Rate of change
        df['roc_7'] = ((df['price'] - df['price'].shift(7)) / df['price'].shift(7)) * 100
        df['roc_14'] = ((df['price'] - df['price'].shift(14)) / df['price'].shift(14)) * 100
        
        # NEW: Average True Range (ATR) proxy
        df['high_proxy'] = df['price'].rolling(7, min_periods=1).max()
        df['low_proxy'] = df['price'].rolling(7, min_periods=1).min()
        df['atr'] = (df['high_proxy'] - df['low_proxy']).rolling(14, min_periods=1).mean()
        df['atr_ratio'] = df['atr'] / df['price']
        
        # NEW: Price acceleration
        df['price_accel'] = df['returns'].diff()
        
        # NEW: Volume proxy (using price range as volume indicator)
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - \
                           df['price'].rolling(7, min_periods=1).min()
        df['volume_proxy'] = df['price_range'] / df['price']
        
        # NEW: Trend strength
        df['trend_strength'] = abs(df['ma7'] - df['ma30']) / df['price']
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)


class ModelPerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.history = []
    
    def record_prediction(
        self,
        model_name: str,
        actual_price: float,
        predicted_price: float,
        timestamp: datetime
    ):
        """Record a prediction for tracking"""
        error = abs(actual_price - predicted_price)
        pct_error = (error / actual_price) * 100 if actual_price > 0 else 0
        
        self.history.append({
            'timestamp': timestamp,
            'model': model_name,
            'actual': actual_price,
            'predicted': predicted_price,
            'error': error,
            'pct_error': pct_error
        })
    
    def get_model_accuracy(self, model_name: str, last_n: int = 100) -> Dict:
        """Calculate model accuracy metrics"""
        model_records = [r for r in self.history if r['model'] == model_name][-last_n:]
        
        if not model_records:
            return {'mae': 0, 'mape': 0, 'rmse': 0, 'count': 0}
        
        errors = [r['error'] for r in model_records]
        pct_errors = [r['pct_error'] for r in model_records]
        
        mae = np.mean(errors)
        mape = np.mean(pct_errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'count': len(model_records)
        }


class HybridPredictor:
    """Enhanced hybrid model with advanced features and tracking"""
    
    def __init__(
        self,
        window_size: int = 30,
        lstm_units: List[int] = [128, 64, 32],
        dropout_rates: List[float] = [0.2, 0.15, 0.1],
        xgb_params: dict = None
    ):
        """
        Initialize enhanced hybrid predictor
        
        Args:
            window_size: Number of days to use for prediction
            lstm_units: List of LSTM layer units
            dropout_rates: Dropout rates for each LSTM layer
            xgb_params: XGBoost parameters
        """
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        
        # Enhanced XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.price_scaler = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.performance_tracker = ModelPerformanceTracker()
        
        # Track feature importance
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Prepare enhanced features from price series"""
        df = pd.DataFrame({'price': price_series.astype(float)})
        df = self.feature_engineer.add_advanced_features(df)
        return df
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build enhanced LSTM neural network with batch normalization"""
        model = keras.Sequential()
        
        # Add LSTM layers with batch normalization
        for i, (units, dropout) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(layers.LSTM(
                units,
                input_shape=input_shape if i == 0 else None,
                return_sequences=return_sequences
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Dense layers for better learning
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile with adaptive learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
        
    def train_lstm(
        self,
        price_series: pd.Series,
        epochs: int = 30,
        batch_size: int = 16,
        validation_split: float = 0.15
    ) -> dict:
        """Train enhanced LSTM model with better features"""
        logger.info("Training enhanced LSTM model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Select feature columns (now more features!)
        feature_cols = [
            'returns', 'ma7_dist', 'ma14_dist', 
            'macd', 'macd_hist',
            'rsi_7', 'rsi_14', 'rsi_21',
            'bb_width', 'bb_position',
            'volatility_7', 'volatility_14',
            'momentum_7', 'momentum_14',
            'roc_7', 'atr_ratio',
            'price_accel', 'volume_proxy', 'trend_strength'
        ]
        
        self.feature_names = feature_cols
        features = df[feature_cols].values
        
        # Target: next day's return
        target = df['returns'].shift(-1).fillna(0).values
        
        # Scale features with StandardScaler (better for LSTM)
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.window_size, len(features_scaled)):
            X.append(features_scaled[i-self.window_size:i])
            y.append(target[i])
            
        X = np.array(X)
        y = np.array(y)
        
        # Time series split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.lstm_model = self.build_lstm_model((self.window_size, X.shape[2]))
        
        # Enhanced callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Calculate metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info(f"LSTM training complete. Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'final_loss': final_loss,
            'final_val_loss': final_val_loss
        }
        
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """Train enhanced XGBoost model"""
        logger.info("Training enhanced XGBoost model...")
        
        # Prepare features
        df = self.prepare_features(price_series)
        
        # Use ALL features for XGBoost
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]
        y = df['returns'].shift(-1).dropna().values
        
        self.feature_names = feature_cols
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            score = model.score(X_val, y_val)
            scores.append(score)
        
        # Final training on all data
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y, verbose=False)
        
        # Get feature importance
        self.feature_importance = dict(zip(
            feature_cols,
            self.xgb_model.feature_importances_
        ))
        
        # Sort by importance
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        logger.info(f"XGBoost training complete. CV R²: {np.mean(scores):.4f}")
        logger.info(f"Top 5 features: {[f[0] for f in top_features]}")
        
        return {
            'cv_scores': scores,
            'mean_cv_score': np.mean(scores),
            'feature_importance': self.feature_importance
        }
        
    def predict_lstm(
        self,
        price_series: pd.Series,
        horizon: int = 7
    ) -> List[float]:
        """Predict with enhanced LSTM"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
            
        df = self.prepare_features(price_series)
        feature_cols = self.feature_names[:19]  # LSTM features
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        current_window = features_scaled[-self.window_size:]
        predictions = []
        current_price = float(price_series.iloc[-1])
        
        for _ in range(horizon):
            X = current_window.reshape(1, self.window_size, -1)
            next_return = self.lstm_model.predict(X, verbose=0)[0, 0]
            
            # Convert return to price
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            
            # Update window
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
        """Predict with enhanced XGBoost"""
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
        
    def predict_ensemble(
        self,
        price_series: pd.Series,
        horizon: int = 7,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        Enhanced ensemble prediction with confidence intervals
        
        Returns:
            Tuple of (predictions, components_dict)
        """
        logger.info(f"Generating {horizon}-day enhanced ensemble forecast...")
        
        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)
        
        # Weighted ensemble
        ensemble = [
            lstm_weight * lstm + xgb_weight * xgb
            for lstm, xgb in zip(lstm_preds, xgb_preds)
        ]
        
        # Calculate confidence intervals (simplified)
        volatility = price_series.pct_change().std()
        ci_multiplier = 1.96  # 95% confidence
        
        confidence_bands = []
        for pred in ensemble:
            margin = pred * volatility * ci_multiplier
            confidence_bands.append({
                'upper': pred + margin,
                'lower': max(0, pred - margin)
            })
        
        logger.info("Enhanced ensemble forecast complete")
        
        return ensemble, {
            'lstm': lstm_preds,
            'xgboost': xgb_preds,
            'confidence_bands': confidence_bands
        }


# Convenience function
def train_and_predict(
    price_series: pd.Series,
    horizon: int = 7,
    **kwargs
) -> dict:
    """
    Enhanced training and prediction
    
    Returns:
        Dictionary with predictions and metadata
    """
    predictor = HybridPredictor(**kwargs)
    
    # Train both models
    lstm_history = predictor.train_lstm(price_series)
    xgb_metrics = predictor.train_xgboost(price_series)
    
    # Generate predictions
    ensemble_preds, components = predictor.predict_ensemble(price_series, horizon)
    
    return {
        'lstm': components['lstm'],
        'xgboost': components['xgboost'],
        'ensemble': ensemble_preds,
        'confidence_bands': components['confidence_bands'],
        'lstm_metrics': lstm_history,
        'xgb_metrics': xgb_metrics,
        'feature_importance': predictor.feature_importance
    }

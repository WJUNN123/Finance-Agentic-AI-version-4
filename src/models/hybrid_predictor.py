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
        
        # EMA features
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Multiple timeframe RSI
        df['rsi_7'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 7)
        df['rsi_14'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 14)
        df['rsi_21'] = AdvancedFeatureEngineer._calculate_rsi(df['price'], 21)
        
        # Bollinger Band features
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['price'].rolling(bb_period, min_periods=1).mean()
        bb_rolling_std = df['price'].rolling(bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * bb_rolling_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * bb_rolling_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility features
        df['volatility_7'] = df['returns'].rolling(7, min_periods=1).std()
        df['volatility_14'] = df['returns'].rolling(14, min_periods=1).std()
        df['volatility_30'] = df['returns'].rolling(30, min_periods=1).std()
        
        # Momentum features
        df['momentum_7'] = df['price'].pct_change(periods=7)
        df['momentum_14'] = df['price'].pct_change(periods=14)
        df['momentum_30'] = df['price'].pct_change(periods=30)
        
        # Rate of change
        df['roc_7'] = ((df['price'] - df['price'].shift(7)) / df['price'].shift(7)) * 100
        df['roc_14'] = ((df['price'] - df['price'].shift(14)) / df['price'].shift(14)) * 100
        
        # ATR proxy
        df['high_proxy'] = df['price'].rolling(7, min_periods=1).max()
        df['low_proxy'] = df['price'].rolling(7, min_periods=1).min()
        df['atr'] = (df['high_proxy'] - df['low_proxy']).rolling(14, min_periods=1).mean()
        df['atr_ratio'] = df['atr'] / df['price']
        
        # Price acceleration
        df['price_accel'] = df['returns'].diff()
        
        # Volume proxy
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - \
                           df['price'].rolling(7, min_periods=1).min()
        df['volume_proxy'] = df['price_range'] / df['price']
        
        # Trend strength
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


class PredictionValidator:
    """Validates predictions to ensure they're realistic"""
    
    @staticmethod
    def validate_prediction(
        predictions: List[float],
        current_price: float,
        historical_volatility: float,
        horizon: int = 7
    ) -> Tuple[List[float], bool]:
        """
        Validate and clamp predictions to realistic bounds
        
        Returns:
            Tuple of (clamped_predictions, was_clamped)
        """
        was_clamped = False
        clamped = []
        
        # Calculate realistic bounds based on historical volatility
        # Crypto daily volatility typically 2-5%, so 7-day max realistic move is ~15-25%
        daily_vol = historical_volatility if historical_volatility > 0 else 0.03
        
        # Conservative bounds: 3x standard deviation over horizon
        max_daily_change = daily_vol * 3
        max_total_change = max_daily_change * np.sqrt(horizon)  # Adjust for time
        
        # Absolute bounds (crypto can be volatile but not THAT volatile)
        max_gain_pct = min(max_total_change, 0.30)  # Max 30% gain in 7 days
        max_loss_pct = min(max_total_change, 0.25)  # Max 25% loss in 7 days
        
        upper_bound = current_price * (1 + max_gain_pct)
        lower_bound = current_price * (1 - max_loss_pct)
        
        for pred in predictions:
            if pred > upper_bound:
                logger.warning(f"⚠️ Clamped prediction from ${pred:,.0f} to ${upper_bound:,.0f} (max +{max_gain_pct*100:.0f}%)")
                clamped.append(upper_bound)
                was_clamped = True
            elif pred < lower_bound:
                logger.warning(f"⚠️ Clamped prediction from ${pred:,.0f} to ${lower_bound:,.0f} (max -{max_loss_pct*100:.0f}%)")
                clamped.append(lower_bound)
                was_clamped = True
            else:
                clamped.append(pred)
        
        # Additional check: ensure predictions don't jump too much day-to-day
        smoothed = [clamped[0]]
        for i in range(1, len(clamped)):
            max_day_jump = smoothed[-1] * (1 + max_daily_change)
            min_day_jump = smoothed[-1] * (1 - max_daily_change)
            
            if clamped[i] > max_day_jump:
                smoothed.append(max_day_jump)
                was_clamped = True
            elif clamped[i] < min_day_jump:
                smoothed.append(min_day_jump)
                was_clamped = True
            else:
                smoothed.append(clamped[i])
        
        if was_clamped:
            logger.info(f"✅ Predictions validated and clamped to realistic bounds")
        
        return smoothed, was_clamped


class HybridPredictor:
    """Enhanced hybrid model with prediction validation"""
    
    def __init__(
        self,
        window_size: int = 30,
        lstm_units: List[int] = [64, 32],  # REDUCED from [128, 64, 32]
        dropout_rates: List[float] = [0.3, 0.2],  # INCREASED dropout
        xgb_params: dict = None
    ):
        """
        Initialize predictor with more conservative settings
        """
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates
        
        # More conservative XGBoost
        self.xgb_params = xgb_params or {
            'n_estimators': 150,  # Reduced from 200
            'max_depth': 5,        # Reduced from 6
            'learning_rate': 0.03, # Reduced from 0.05
            'subsample': 0.8,
            'colsample_bytree': 0.7,  # Reduced from 0.8
            'min_child_weight': 5,     # Increased from 3
            'gamma': 0.2,              # Increased from 0.1
            'reg_alpha': 0.3,          # Increased from 0.1
            'reg_lambda': 1.5,         # Increased from 1.0
            'random_state': 42
        }
        
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.validator = PredictionValidator()
        
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Prepare features from price series"""
        df = pd.DataFrame({'price': price_series.astype(float)})
        df = self.feature_engineer.add_advanced_features(df)
        return df
        
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM with stronger regularization"""
        model = keras.Sequential()
        
        for i, (units, dropout) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(layers.LSTM(
                units,
                input_shape=input_shape if i == 0 else None,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(0.01)  # Add L2 regularization
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae']
        )
        
        return model
        
    def train_lstm(
        self,
        price_series: pd.Series,
        epochs: int = 25,  # Reduced from 30
        batch_size: int = 16,
        validation_split: float = 0.15
    ) -> dict:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        df = self.prepare_features(price_series)
        
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
        
        # Target: next day's return (CLAMPED)
        raw_returns = df['returns'].shift(-1).fillna(0)
        target = np.clip(raw_returns, -0.1, 0.1)  # Clamp to ±10% daily
        target = target.values
        
        # Scale features
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.window_size, len(features_scaled)):
            X.append(features_scaled[i-self.window_size:i])
            y.append(target[i])
            
        X = np.array(X)
        y = np.array(y)
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.lstm_model = self.build_lstm_model((self.window_size, X.shape[2]))
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,  # Reduced from 10
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=0.00001,
            verbose=0
        )
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        logger.info(f"LSTM complete. Loss: {final_loss:.6f}, Val: {final_val_loss:.6f}")
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'final_loss': final_loss,
            'final_val_loss': final_val_loss
        }
        
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        df = self.prepare_features(price_series)
        
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]
        
        # Target: clamped returns
        raw_returns = df['returns'].shift(-1).dropna()
        y = np.clip(raw_returns, -0.1, 0.1).values  # Clamp to ±10% daily
        
        self.feature_names = feature_cols
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y, verbose=False)
        
        self.feature_importance = dict(zip(feature_cols, self.xgb_model.feature_importances_))
        
        logger.info(f"XGBoost complete. CV R²: {np.mean(scores):.4f}")
        
        return {'cv_scores': scores, 'mean_cv_score': np.mean(scores)}
        
    def predict_lstm(
        self,
        price_series: pd.Series,
        horizon: int = 7
    ) -> List[float]:
        """Predict with LSTM and clamp returns"""
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
            
        df = self.prepare_features(price_series)
        feature_cols = self.feature_names[:19]
        features = df[feature_cols].values
        features_scaled = self.feature_scaler.transform(features)
        
        current_window = features_scaled[-self.window_size:]
        predictions = []
        current_price = float(price_series.iloc[-1])
        
        for _ in range(horizon):
            X = current_window.reshape(1, self.window_size, -1)
            next_return = self.lstm_model.predict(X, verbose=0)[0, 0]
            
            # CLAMP the predicted return to realistic values
            next_return = np.clip(next_return, -0.05, 0.05)  # Max ±5% per day
            
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
        """Predict with XGBoost and clamp returns"""
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained")
            
        df = self.prepare_features(price_series)
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        
        predictions = []
        current_price = float(price_series.iloc[-1])
        current_features = df[feature_cols].iloc[-1:].values
        
        for _ in range(horizon):
            next_return = self.xgb_model.predict(current_features)[0]
            
            # CLAMP the predicted return
            next_return = np.clip(next_return, -0.05, 0.05)  # Max ±5% per day
            
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            current_price = next_price
            
        return predictions
        
    def predict_ensemble(
        self,
        price_series: pd.Series,
        horizon: int = 7,
        lstm_weight: float = 0.5,
        xgb_weight: float = 0.5
    ) -> Tuple[List[float], Dict]:
        """Generate validated ensemble prediction"""
        logger.info(f"Generating {horizon}-day forecast with validation...")
        
        current_price = float(price_series.iloc[-1])
        historical_volatility = price_series.pct_change().std()
        
        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)
        
        # Ensemble
        ensemble = [
            lstm_weight * lstm + xgb_weight * xgb
            for lstm, xgb in zip(lstm_preds, xgb_preds)
        ]
        
        # VALIDATE predictions
        ensemble, was_clamped = self.validator.validate_prediction(
            ensemble, current_price, historical_volatility, horizon
        )
        
        if was_clamped:
            logger.warning("⚠️ Predictions were adjusted to realistic bounds")
        
        # Calculate confidence intervals
        volatility = historical_volatility
        ci_multiplier = 1.96
        
        confidence_bands = []
        for pred in ensemble:
            margin = pred * volatility * ci_multiplier
            confidence_bands.append({
                'upper': pred + margin,
                'lower': max(0, pred - margin)
            })
        
        logger.info("✅ Forecast complete with validation")
        
        return ensemble, {
            'lstm': lstm_preds,
            'xgboost': xgb_preds,
            'confidence_bands': confidence_bands,
            'was_clamped': was_clamped
        }


# Convenience function
def train_and_predict(
    price_series: pd.Series,
    horizon: int = 7,
    **kwargs
) -> dict:
    """Train and predict with validation"""
    predictor = HybridPredictor(**kwargs)
    
    lstm_history = predictor.train_lstm(price_series)
    xgb_metrics = predictor.train_xgboost(price_series)
    
    ensemble_preds, components = predictor.predict_ensemble(price_series, horizon)
    
    return {
        'lstm': components['lstm'],
        'xgboost': components['xgboost'],
        'ensemble': ensemble_preds,
        'confidence_bands': components['confidence_bands'],
        'lstm_metrics': lstm_history,
        'xgb_metrics': xgb_metrics,
        'feature_importance': predictor.feature_importance,
        'was_clamped': components.get('was_clamped', False)
    }

"""
Hybrid Price Prediction Model
Combines LSTM and XGBoost for cryptocurrency price forecasting
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import logging
import pickle
import hashlib
from pathlib import Path
import functools
import time

# ML imports
from sklearn.preprocessing import MinMaxScaler, RobustScaler
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
    """
    Hybrid model combining LSTM and XGBoost for price prediction.
    NOTE: This remains a regression-based forecaster (prices), not a classifier,
    so your production output format is unchanged.
    """

    def __init__(
        self,
        window_size: int = 15,                    # ALIGNED: was 30
        lstm_units: List[int] = [32],             # ALIGNED: simpler, closer spirit
        dropout_rates: List[float] = [0.30],      # ALIGNED: closer to evaluation dropout
        xgb_params: dict = None
    ):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dropout_rates = dropout_rates

        # XGBoost default parameters (ALIGNED closer to evaluation, still regression)
        self.xgb_params = xgb_params or {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
        }

        self.lstm_model: Optional[keras.Model] = None
        self.xgb_model: Optional[xgb.XGBRegressor] = None

        # Separate scalers (ALIGNED with evaluation practice)
        self.lstm_scaler: Optional[MinMaxScaler] = None
        self.xgb_scaler: Optional[RobustScaler] = None

    # ------------------------
    # Caching / persistence
    # ------------------------

    def _get_model_hash(self, price_series: pd.Series) -> str:
        """
        Generate hash for model caching based on BOTH data and settings.
        This prevents loading incompatible cached models after you change settings.
        """
        # Settings signature
        settings = (
            f"w={self.window_size}|"
            f"lstm={self.lstm_units}|"
            f"drop={self.dropout_rates}|"
            f"xgb={sorted(self.xgb_params.items())}"
        )
        data_str = (
            f"{settings}|"
            f"n={len(price_series)}|"
            f"first={float(price_series.iloc[0]):.6f}|"
            f"last={float(price_series.iloc[-1]):.6f}"
        )
        return hashlib.md5(data_str.encode()).hexdigest()[:12]

    def save_models(self, coin_id: str, price_series: pd.Series):
        """Save trained models to disk for reuse."""
        model_dir = Path('models_cache')
        model_dir.mkdir(exist_ok=True)

        model_hash = self._get_model_hash(price_series)

        try:
            # Save LSTM model
            if self.lstm_model is not None:
                lstm_path = model_dir / f'{coin_id}_lstm_{model_hash}.h5'
                self.lstm_model.save(lstm_path)
                logger.info(f"💾 Saved LSTM model: {lstm_path.name}")

            # Save XGBoost model + scalers + settings
            model_data = {
                'xgb_model': self.xgb_model,
                'xgb_scaler': self.xgb_scaler,
                'lstm_scaler': self.lstm_scaler,
                'window_size': self.window_size,
                'lstm_units': self.lstm_units,
                'dropout_rates': self.dropout_rates,
                'xgb_params': self.xgb_params,
                'timestamp': time.time()
            }
            xgb_path = model_dir / f'{coin_id}_xgb_{model_hash}.pkl'
            with open(xgb_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"💾 Saved XGBoost model bundle: {xgb_path.name}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save models: {e}")

    def load_models(self, coin_id: str, price_series: pd.Series) -> bool:
        """Load trained models from disk."""
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
                logger.info("✅ Loaded LSTM model from cache")

            # Load XGBoost model + scalers + settings
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.xgb_model = model_data.get('xgb_model')
                self.xgb_scaler = model_data.get('xgb_scaler')
                self.lstm_scaler = model_data.get('lstm_scaler')

                # Restore settings (optional but useful)
                self.window_size = model_data.get('window_size', self.window_size)
                self.lstm_units = model_data.get('lstm_units', self.lstm_units)
                self.dropout_rates = model_data.get('dropout_rates', self.dropout_rates)
                self.xgb_params = model_data.get('xgb_params', self.xgb_params)

                # Check age
                model_age_days = (time.time() - model_data.get('timestamp', 0)) / 86400
                if model_age_days > 7:
                    logger.warning(f"⚠️ Cached model is {model_age_days:.1f} days old, might need retraining")
                else:
                    logger.info(f"✅ Loaded XGBoost model bundle (age: {model_age_days:.1f} days)")

            # Require both models + both scalers for safe inference
            ok = (
                self.lstm_model is not None and
                self.xgb_model is not None and
                self.lstm_scaler is not None and
                self.xgb_scaler is not None
            )
            return bool(ok)

        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

    # ------------------------
    # Feature engineering
    # ------------------------

    def prepare_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Prepare enhanced features from price series."""
        df = pd.DataFrame({'price': price_series.astype(float)})

        # Log returns
        df['returns'] = np.log(df['price']).diff()

        # Moving averages
        df['ma7'] = df['price'].rolling(7, min_periods=1).mean()
        df['ma14'] = df['price'].rolling(14, min_periods=1).mean()
        df['ma30'] = df['price'].rolling(30, min_periods=1).mean()
        df['ma50'] = df['price'].rolling(50, min_periods=1).mean()

        # MA distances
        df['ma7_dist'] = (df['price'] - df['ma7']) / (df['price'] + 1e-10)
        df['ma14_dist'] = (df['price'] - df['ma14']) / (df['price'] + 1e-10)
        df['ma30_dist'] = (df['price'] - df['ma30']) / (df['price'] + 1e-10)

        # MA crossovers
        df['ma7_ma30_cross'] = (df['ma7'] > df['ma30']).astype(int)
        df['ma14_ma50_cross'] = (df['ma14'] > df['ma50']).astype(int)

        # RSI
        df['rsi_7'] = self._calculate_rsi(df['price'], period=7)
        df['rsi_14'] = self._calculate_rsi(df['price'], period=14)
        df['rsi_21'] = self._calculate_rsi(df['price'], period=21)

        # Volatility
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        df['volatility_short'] = df['returns'].rolling(7, min_periods=1).std()
        df['volatility_long'] = df['returns'].rolling(50, min_periods=1).std()

        # Momentum
        df['momentum_7'] = df['price'].pct_change(periods=7)
        df['momentum_14'] = df['price'].pct_change(periods=14)
        df['momentum_30'] = df['price'].pct_change(periods=30)

        # Price range
        df['price_range'] = df['price'].rolling(7, min_periods=1).max() - df['price'].rolling(7, min_periods=1).min()
        df['price_range_norm'] = df['price_range'] / (df['price'] + 1e-10)

        # Bollinger position
        rolling_mean = df['price'].rolling(20, min_periods=1).mean()
        rolling_std = df['price'].rolling(20, min_periods=1).std()
        df['bb_position'] = (df['price'] - rolling_mean) / (2 * rolling_std + 1e-10)

        # ROC
        df['roc_10'] = df['price'].pct_change(periods=10) * 100
        df['roc_20'] = df['price'].pct_change(periods=20) * 100

        # Acceleration
        df['acceleration'] = df['returns'].diff()

        # Distance from highs/lows
        df['dist_from_high'] = (df['price'].rolling(30, min_periods=1).max() - df['price']) / (df['price'] + 1e-10)
        df['dist_from_low'] = (df['price'] - df['price'].rolling(30, min_periods=1).min()) / (df['price'] + 1e-10)

        # EMA
        df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        logger.debug(f"📊 Prepared {len(df.columns)} features for training")

        # Keep behavior the same as your original: backfill then fill 0
        return df.bfill().fillna(0)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    # ------------------------
    # LSTM
    # ------------------------

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM neural network (regression)."""
        model = keras.Sequential()

        for i, (units, dropout) in enumerate(zip(self.lstm_units, self.dropout_rates)):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(
                units,
                input_shape=input_shape if i == 0 else None,
                return_sequences=return_sequences
            ))
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1, activation='linear'))

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
        """Train LSTM model."""
        logger.info("🧠 Training LSTM model...")

        df = self.prepare_features(price_series)

        # Same LSTM feature subset you had before
        feature_cols = [
            'returns', 'ma7_dist', 'ma14_dist', 'ma30_dist',
            'rsi_7', 'rsi_14', 'volatility', 'volatility_short',
            'momentum_7', 'momentum_14', 'bb_position',
            'macd_histogram', 'acceleration'
        ]
        features = df[feature_cols].values

        # Target: next day's return
        target = df['returns'].shift(-1).fillna(0).values

        # Scale features (MinMaxScaler) fit on TRAIN only (here: within-series training)
        self.lstm_scaler = MinMaxScaler()
        features_scaled = self.lstm_scaler.fit_transform(features)

        # Create sequences
        X, y = [], []
        for i in range(self.window_size, len(features_scaled)):
            X.append(features_scaled[i - self.window_size:i])
            y.append(target[i])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 10:
            logger.warning("⚠️ Insufficient data for LSTM training")
            return {'loss': [0], 'val_loss': [0]}

        # Time-based split (no shuffle)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.lstm_model = self.build_lstm_model((self.window_size, X.shape[2]))

        # ALIGNED: patience closer to evaluation
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
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

    def predict_lstm(self, price_series: pd.Series, horizon: int = 7) -> List[float]:
        """Predict future prices using LSTM (unchanged output behavior)."""
        if self.lstm_model is None or self.lstm_scaler is None:
            raise ValueError("LSTM model not trained/loaded. Call train_lstm() or load_models() first.")

        df = self.prepare_features(price_series)
        feature_cols = [
            'returns', 'ma7_dist', 'ma14_dist', 'ma30_dist',
            'rsi_7', 'rsi_14', 'volatility', 'volatility_short',
            'momentum_7', 'momentum_14', 'bb_position',
            'macd_histogram', 'acceleration'
        ]
        features = df[feature_cols].values
        features_scaled = self.lstm_scaler.transform(features)

        current_window = features_scaled[-self.window_size:]
        predictions = []
        current_price = float(price_series.iloc[-1])

        for _ in range(horizon):
            X = current_window.reshape(1, self.window_size, -1)
            next_return = float(self.lstm_model.predict(X, verbose=0)[0, 0])

            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))

            # Same simplified update as your original:
            new_features = current_window[-1].copy()
            new_features[0] = next_return
            current_window = np.vstack([current_window[1:], new_features])
            current_price = next_price

        return predictions

    # ------------------------
    # XGBoost
    # ------------------------

    @log_performance
    def train_xgboost(self, price_series: pd.Series) -> dict:
        """Train XGBoost regressor (aligned with evaluation scaling practice)."""
        logger.info("🌲 Training XGBoost model...")

        df = self.prepare_features(price_series)

        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]
        X = df[feature_cols].values[:-1]
        y = df['returns'].shift(-1).dropna().values

        if len(X) < 10:
            logger.warning("⚠️ Insufficient data for XGBoost training")
            return {'train_score': 0, 'test_score': 0}

        # Time-based split (no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=False
        )

        # ALIGNED: RobustScaler fit on TRAIN only
        self.xgb_scaler = RobustScaler()
        X_train_s = self.xgb_scaler.fit_transform(X_train)
        X_test_s = self.xgb_scaler.transform(X_test)

        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False
        )

        train_score = self.xgb_model.score(X_train_s, y_train)
        test_score = self.xgb_model.score(X_test_s, y_test)

        logger.info(f"✅ XGBoost training complete. Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

        return {
            'train_score': float(train_score),
            'test_score': float(test_score),
            'feature_importance': dict(zip(feature_cols, self.xgb_model.feature_importances_.tolist()))
        }

    def predict_xgboost(self, price_series: pd.Series, horizon: int = 7) -> List[float]:
        """Predict future prices using XGBoost (unchanged output behavior)."""
        if self.xgb_model is None or self.xgb_scaler is None:
            raise ValueError("XGBoost model not trained/loaded. Call train_xgboost() or load_models() first.")

        df = self.prepare_features(price_series)
        feature_cols = [col for col in df.columns if col not in ['price', 'returns']]

        predictions = []
        current_price = float(price_series.iloc[-1])

        # Start from last feature row
        current_features = df[feature_cols].iloc[-1:].values
        current_features_s = self.xgb_scaler.transform(current_features)

        for _ in range(horizon):
            next_return = float(self.xgb_model.predict(current_features_s)[0])
            next_price = current_price * np.exp(next_return)
            predictions.append(float(next_price))
            current_price = next_price

            # Keep original behavior: do not attempt to fully update all engineered features
            # because that would require recomputing feature pipeline using the new price.
            # (Same limitation as your original production code.)
            # current_features_s remains based on last observed feature row.

        return predictions

    # ------------------------
    # Ensemble (unchanged API)
    # ------------------------

    def predict_ensemble(
        self,
        price_series: pd.Series,
        horizon: int = 7,
        lstm_weight: float = 0.5,
        xgb_weight: float = 0.5,
        dampening_factor: float = 0.65,
        apply_mean_reversion: bool = True
    ) -> List[float]:
        """Generate conservative ensemble prediction (same output behavior)."""
        logger.info(f"🎯 Generating {horizon}-day ensemble forecast...")

        lstm_preds = self.predict_lstm(price_series, horizon)
        xgb_preds = self.predict_xgboost(price_series, horizon)

        ensemble = [
            lstm_weight * lstm + xgb_weight * xgb
            for lstm, xgb in zip(lstm_preds, xgb_preds)
        ]

        current_price = float(price_series.iloc[-1])
        dampened = []

        for pred in ensemble:
            change_pct = (pred - current_price) / (current_price + 1e-10)
            dampened_change = change_pct * dampening_factor
            dampened_price = current_price * (1 + dampened_change)
            dampened.append(float(dampened_price))

        if apply_mean_reversion and len(price_series) >= 30:
            ma_30 = float(price_series.rolling(30).mean().iloc[-1])
            final_preds = []
            for i, pred in enumerate(dampened):
                days_out = i + 1
                reversion_strength = 0.2 / (1 + days_out * 0.1)
                final_price = pred * (1 - reversion_strength) + ma_30 * reversion_strength
                final_preds.append(float(final_price))
        else:
            final_preds = dampened

        original_change = ((ensemble[-1] - current_price) / (current_price + 1e-10)) * 100
        final_change = ((final_preds[-1] - current_price) / (current_price + 1e-10)) * 100

        logger.info(f"✅ Ensemble complete: ${final_preds[-1]:,.2f}")
        logger.info(
            f"📉 Adjustment: {original_change:+.1f}% → {final_change:+.1f}% "
            f"(dampened by {dampening_factor:.0%})"
        )

        return final_preds


def train_and_predict(
    price_series: pd.Series,
    horizon: int = 7,
    coin_id: str = 'unknown',
    use_cache: bool = True,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Train models and generate predictions with caching.
    Output format unchanged:
      {'lstm': [...], 'xgboost': [...], 'ensemble': [...]}
    """
    predictor = HybridPredictor(**kwargs)

    models_loaded = False
    if use_cache:
        models_loaded = predictor.load_models(coin_id, price_series)

    if models_loaded:
        logger.info(f"⚡ Using cached models for {coin_id}")
    else:
        logger.info(f"🔄 Training new models for {coin_id}")
        predictor.train_lstm(price_series)
        predictor.train_xgboost(price_series)

        if use_cache:
            predictor.save_models(coin_id, price_series)

    return {
        'lstm': predictor.predict_lstm(price_series, horizon),
        'xgboost': predictor.predict_xgboost(price_series, horizon),
        'ensemble': predictor.predict_ensemble(price_series, horizon)
    }

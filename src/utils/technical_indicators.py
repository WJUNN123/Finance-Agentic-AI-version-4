"""
Technical Indicators Utility
Calculate various technical indicators for crypto analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Price series
        period: RSI period (default 14)
        
    Returns:
        Current RSI value
    """
    if len(prices) < period + 1:
        return 50.0
        
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI series for entire price history"""
    if len(prices) < period + 1:
        return pd.Series(50.0, index=prices.index)
        
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate Upper, Middle, and Lower Bollinger Bands
    
    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band)
    """
    if len(prices) < period:
        val = float(prices.iloc[-1])
        return val * 1.05, val, val * 0.95
        
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    
    return float(upper.iloc[-1]), float(ma.iloc[-1]), float(lower.iloc[-1])

def calculate_roc(prices: pd.Series, period: int = 9) -> float:
    """
    Calculate Rate of Change (Percentage change over n periods)
    """
    if len(prices) < period:
        return 0.0
    return float(((prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]) * 100)

def calculate_volatility(prices: pd.Series, window: int = 30) -> float:
    """
    Calculate Annualized volatility
    """
    returns = np.log(prices / prices.shift(1))
    return float(returns.rolling(window=window).std().iloc[-1] * np.sqrt(365)) * 100

def identify_trend(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> str:
    """
    Identify trend based on Moving Average crossovers
    """
    if len(prices) < long_window:
        return "neutral"
        
    short_ma = prices.rolling(window=short_window).mean().iloc[-1]
    long_ma = prices.rolling(window=long_window).mean().iloc[-1]
    
    if short_ma > long_ma * 1.01: 
        return "uptrend"
    if short_ma < long_ma * 0.99: 
        return "downtrend"
    return "neutral"

def calculate_moving_averages(prices: pd.Series, periods: list = [7, 14, 30]) -> dict:
    """Calculate multiple moving averages"""
    mas = {}
    for period in periods:
        if len(prices) >= period:
            mas[f'ma{period}'] = float(prices.rolling(period).mean().iloc[-1])
        else:
            mas[f'ma{period}'] = float(prices.iloc[-1])
    return mas

def get_support_resistance(prices: pd.Series, window: int = 20) -> Tuple[float, float]:
    """Identify support and resistance levels"""
    if len(prices) < window:
        current = float(prices.iloc[-1])
        return current * 0.95, current * 1.05
    
    recent = prices.tail(window)
    return float(recent.min()), float(recent.max())

def get_all_indicators(prices: pd.Series, pct_24h: float = 0.0, pct_7d: float = 0.0) -> Dict:
    """
    Calculate all technical indicators including enhanced metrics for the Risk Engine
    """
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(prices)
    current_price = float(prices.iloc[-1])
    
    # BB Position (0 = lower band, 1 = upper band)
    bb_range = bb_upper - bb_lower
    bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
    
    # Base indicators
    indicators = {
        'rsi': calculate_rsi(prices),
        'volatility': calculate_volatility(prices),
        'roc': calculate_roc(prices),
        'trend': identify_trend(prices),
        'bb_upper': bb_upper,
        'bb_middle': bb_mid,
        'bb_lower': bb_lower,
        'bb_position': bb_position
    }
    
    # Add Moving Averages
    indicators.update(calculate_moving_averages(prices))
    
    # Add Support/Resistance
    sup, res = get_support_resistance(prices)
    indicators['support'] = sup
    indicators['resistance'] = res
    
    return indicators

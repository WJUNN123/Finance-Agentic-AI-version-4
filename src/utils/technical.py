"""
Technical Indicators Utility
Calculate various technical indicators for crypto analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
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
        return 50.0  # Neutral default
        
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


def calculate_moving_averages(
    prices: pd.Series,
    periods: list = [7, 14, 30]
) -> dict:
    """
    Calculate multiple moving averages
    
    Args:
        prices: Price series
        periods: List of MA periods
        
    Returns:
        Dictionary with MA values
    """
    mas = {}
    for period in periods:
        if len(prices) >= period:
            ma = prices.rolling(period, min_periods=1).mean()
            mas[f'ma{period}'] = float(ma.iloc[-1])
        else:
            mas[f'ma{period}'] = float(prices.iloc[-1])
            
    return mas


def calculate_volatility(
    prices: pd.Series,
    method: str = 'ewma',
    span: int = 20
) -> float:
    """
    Calculate price volatility
    
    Args:
        prices: Price series
        method: 'ewma' or 'std'
        span: Window size for calculation
        
    Returns:
        Volatility value
    """
    if len(prices) < 2:
        return 0.0
        
    # Calculate log returns
    log_returns = np.log(prices).diff().dropna()
    
    if len(log_returns) < 2:
        return 0.0
        
    if method == 'ewma':
        vol = log_returns.ewm(span=span, adjust=False).std().iloc[-1]
    else:
        vol = log_returns.std()
        
    return float(vol) if not np.isnan(vol) else 0.0


def calculate_momentum(
    prices: pd.Series,
    period: int = 14
) -> float:
    """
    Calculate price momentum
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        Momentum as percentage change
    """
    if len(prices) < period + 1:
        return 0.0
        
    momentum = ((prices.iloc[-1] - prices.iloc[-period-1]) / 
                prices.iloc[-period-1]) * 100
    
    return float(momentum) if not np.isnan(momentum) else 0.0


def get_support_resistance(
    prices: pd.Series,
    window: int = 20
) -> Tuple[float, float]:
    """
    Identify support and resistance levels
    
    Args:
        prices: Price series
        window: Window for finding local extrema
        
    Returns:
        Tuple of (support, resistance) levels
    """
    if len(prices) < window:
        current_price = float(prices.iloc[-1])
        return current_price * 0.95, current_price * 1.05
        
    recent_prices = prices.tail(window)
    support = float(recent_prices.min())
    resistance = float(recent_prices.max())
    
    return support, resistance


def identify_trend(
    prices: pd.Series,
    short_window: int = 7,
    long_window: int = 30
) -> str:
    """
    Identify price trend using moving averages
    
    Args:
        prices: Price series
        short_window: Short MA period
        long_window: Long MA period
        
    Returns:
        'uptrend', 'downtrend', or 'sideways'
    """
    if len(prices) < long_window:
        return "insufficient_data"
        
    ma_short = prices.rolling(short_window).mean().iloc[-1]
    ma_long = prices.rolling(long_window).mean().iloc[-1]
    
    diff_pct = ((ma_short - ma_long) / ma_long) * 100
    
    if diff_pct > 2:
        return "uptrend"
    elif diff_pct < -2:
        return "downtrend"
    else:
        return "sideways"


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Price series
        period: MA period
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period:
        current = float(prices.iloc[-1])
        return current * 1.05, current, current * 0.95
        
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    
    return (
        float(upper.iloc[-1]),
        float(ma.iloc[-1]),
        float(lower.iloc[-1])
    )


def get_all_indicators(
    prices: pd.Series,
    pct_24h: Optional[float] = None,
    pct_7d: Optional[float] = None
) -> dict:
    """
    Calculate all technical indicators at once
    
    Args:
        prices: Price series
        pct_24h: Optional 24h percentage change
        pct_7d: Optional 7d percentage change
        
    Returns:
        Dictionary with all indicators
    """
    indicators = {
        'rsi': calculate_rsi(prices),
        'volatility': calculate_volatility(prices),
        'momentum': calculate_momentum(prices),
        'trend': identify_trend(prices)
    }
    
    # Moving averages
    mas = calculate_moving_averages(prices)
    indicators.update(mas)
    
    # Support/Resistance
    support, resistance = get_support_resistance(prices)
    indicators['support'] = support
    indicators['resistance'] = resistance
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
    indicators['bb_upper'] = bb_upper
    indicators['bb_middle'] = bb_middle
    indicators['bb_lower'] = bb_lower
    
    # Add price changes if provided
    if pct_24h is not None:
        indicators['pct_24h'] = pct_24h
    if pct_7d is not None:
        indicators['pct_7d'] = pct_7d
        
    return indicators
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


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[float, float, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (macd, signal_line, histogram)
    """
    if len(prices) < slow:
        return 0.0, 0.0, 0.0
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return (
        float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0,
        float(signal_line.iloc[-1]) if not np.isnan(signal_line.iloc[-1]) else 0.0,
        float(histogram.iloc[-1]) if not np.isnan(histogram.iloc[-1]) else 0.0
    )


def calculate_stochastic(
    prices: pd.Series,
    period: int = 14,
    smooth_k: int = 3
) -> Tuple[float, float]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        prices: Price series
        period: Lookback period (default 14)
        smooth_k: Smoothing period for %K (default 3)
        
    Returns:
        Tuple of (%K, %D)
    """
    if len(prices) < period:
        return 50.0, 50.0
    
    # For prices only (no separate high/low), use rolling window
    low_min = prices.rolling(period).min()
    high_max = prices.rolling(period).max()
    
    # Calculate raw %K
    k_raw = 100 * ((prices - low_min) / (high_max - low_min))
    
    # Smooth %K
    k = k_raw.rolling(smooth_k).mean()
    
    # %D is moving average of %K
    d = k.rolling(3).mean()
    
    return (
        float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else 50.0,
        float(d.iloc[-1]) if not np.isnan(d.iloc[-1]) else 50.0
    )


def calculate_atr(
    prices: pd.Series,
    period: int = 14
) -> float:
    """
    Calculate Average True Range (ATR) - volatility indicator
    
    Args:
        prices: Price series
        period: ATR period
        
    Returns:
        ATR value
    """
    if len(prices) < period + 1:
        return 0.0
    
    # For single price series, use high-low approximation
    high = prices.rolling(2).max()
    low = prices.rolling(2).min()
    close = prices
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0


def calculate_obv(
    prices: pd.Series,
    volume: pd.Series = None
) -> float:
    """
    Calculate On-Balance Volume (OBV) trend
    
    Args:
        prices: Price series
        volume: Volume series (if None, uses price changes as proxy)
        
    Returns:
        OBV trend indicator (-1 to 1)
    """
    if len(prices) < 2:
        return 0.0
    
    if volume is None:
        # Use price changes as volume proxy
        volume = abs(prices.diff())
    
    # Calculate OBV
    obv = (np.sign(prices.diff()) * volume).fillna(0).cumsum()
    
    # Return trend (normalized)
    if len(obv) > 10:
        recent_trend = obv.iloc[-1] - obv.iloc[-10]
        max_change = abs(obv.diff()).max()
        if max_change > 0:
            return float(np.clip(recent_trend / max_change, -1, 1))
    
    return 0.0


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
    
    # MACD (NEW)
    macd, macd_signal, macd_hist = calculate_macd(prices)
    indicators['macd'] = macd
    indicators['macd_signal'] = macd_signal
    indicators['macd_histogram'] = macd_hist
    
    # Stochastic Oscillator (NEW)
    stoch_k, stoch_d = calculate_stochastic(prices)
    indicators['stochastic_k'] = stoch_k
    indicators['stochastic_d'] = stoch_d
    
    # ATR - Average True Range (NEW)
    indicators['atr'] = calculate_atr(prices)
    
    # OBV trend (NEW)
    indicators['obv_trend'] = calculate_obv(prices)
    
    # Add price changes if provided
    if pct_24h is not None:
        indicators['pct_24h'] = pct_24h
    if pct_7d is not None:
        indicators['pct_7d'] = pct_7d
    
    # Log indicator summary
    logger.debug(f"📊 Calculated {len(indicators)} technical indicators")
    logger.debug(f"   RSI: {indicators['rsi']:.1f}, MACD: {indicators['macd_histogram']:.4f}, Stoch: {indicators['stochastic_k']:.1f}")
        
    return indicators

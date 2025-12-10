"""
Technical Indicators Utility
Calculate various technical indicators for crypto analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# RELATIVE STRENGTH INDEX (RSI)
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI) for current price.
    
    RSI Definition:
    Momentum oscillator on 0-100 scale measuring the magnitude of recent
    price changes to evaluate overbought/oversold conditions.
    
    Formula:
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss (over period)
    
    Interpretation:
    - RSI > 70: OVERBOUGHT (potential pullback/reversal)
      - Sellers are in control
      - Price may consolidate or pullback
    - RSI < 30: OVERSOLD (potential bounce/reversal)
      - Buyers are in control
      - Price may consolidate or bounce
    - 30-70: NEUTRAL (normal activity)
    - Close to 50: Balanced (no directional pressure)
    
    Signals:
    - RSI above 50 + uptrend: Bullish
    - RSI below 50 + downtrend: Bearish
    - RSI divergence: Price makes new high/low but RSI doesn't (reversal warning)
    
    Args:
        prices (pd.Series): Historical price series
        period (int): Look-back period (default 14, common standard)
                     Higher period = smoother, slower response
                     Lower period = more sensitive, more whipsaws
    
    Returns:
        float: RSI value (0-100)
    
    Note:
        Returns 50.0 if insufficient data (neutral default)
    
    Example:
        >>> prices = pd.Series([100, 101, 102, 105, 103, 104])
        >>> rsi = calculate_rsi(prices, period=14)
        >>> if rsi > 70:
        ...     print("Overbought - consider taking profits")
        >>> elif rsi < 30:
        ...     print("Oversold - potential bounce")
    """
    if len(prices) < period + 1:
        logger.debug(f"Insufficient data for RSI: {len(prices)} < {period + 1}")
        return 50.0  # Neutral default
    
    try:
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        
        avg_gain = gains.rolling(window=period, min_periods=period).mean()
        avg_loss = losses.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
        logger.debug(f"RSI({period}): {current_rsi:.1f}")
        
        return current_rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return 50.0


def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI for entire price history (returns series).
    
    Args:
        prices (pd.Series): Price series
        period (int): Look-back period
    
    Returns:
        pd.Series: RSI values indexed by date
    
    Note:
        Use for visualization or multi-period analysis
    """
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


# ============================================================================
# MOVING AVERAGES
# ============================================================================

def calculate_moving_averages(
    prices: pd.Series,
    periods: list = [7, 14, 30]
) -> dict:
    """
    Calculate multiple moving averages (MA).
    
    MA Definition:
    Average of prices over N periods, smoothing out noise and identifying trends.
    
    Types:
    - SMA (Simple MA): Equal weight to all periods
    - EMA (Exponential MA): Heavier weight to recent periods (more responsive)
    
    Interpretation:
    - Price above MA: Uptrend
    - Price below MA: Downtrend
    - Price near MA: Consolidation/ranging
    
    Trend Strength:
    - Price far from MA: Strong trend
    - Price near MA: Weak trend
    
    Multiple MAs (Golden/Death Cross):
    - Fast MA crosses above slow MA: BULLISH (Golden Cross)
    - Fast MA crosses below slow MA: BEARISH (Death Cross)
    
    Args:
        prices (pd.Series): Price series
        periods (list): MA periods to calculate (default [7, 14, 30])
    
    Returns:
        dict: Current values for each MA
        Keys: 'ma7', 'ma14', 'ma30', etc.
    
    Example:
        >>> mas = calculate_moving_averages(prices)
        >>> if prices.iloc[-1] > mas['ma7'] > mas['ma14']:
        ...     print("Uptrend: price above fast MA above slow MA")
    """
    mas = {}
    
    for period in periods:
        try:
            if len(prices) >= period:
                ma = prices.rolling(period, min_periods=1).mean()
                mas[f'ma{period}'] = float(ma.iloc[-1])
            else:
                mas[f'ma{period}'] = float(prices.iloc[-1])
        except Exception as e:
            logger.error(f"Error calculating MA({period}): {e}")
            mas[f'ma{period}'] = float(prices.iloc[-1])
    
    logger.debug(f"Moving averages: {mas}")
    
    return mas


# ============================================================================
# VOLATILITY
# ============================================================================

def calculate_volatility(
    prices: pd.Series,
    method: str = 'std',
    span: int = 20
) -> float:
    """
    Calculate price volatility (variability).
    
    Volatility Definition:
    Measure of price fluctuation - how much price moves up and down.
    
    Methods:
    1. Standard Deviation: Dispersion of returns
    2. EWMA: Exponential-weighted volatility (recent periods matter more)
    
    Interpretation:
    - High volatility: Price swings widely
      - Wider stop-losses needed
      - Larger position sizing needed
      - Higher risk/reward potential
    
    - Low volatility: Price moves in narrow range
      - Tighter stops possible
      - Smaller moves to expect
      - Lower risk but lower reward
    
    Crypto Context:
    - Typical: 0.02-0.08 (2-8% daily moves)
    - High: >0.10 (>10% daily swings)
    - Low: <0.02 (<2% daily moves)
    
    Args:
        prices (pd.Series): Price series
        method (str): 'std' for standard deviation, 'ewma' for exponential
        span (int): Look-back window (default 20 days)
    
    Returns:
        float: Volatility measure (0.0-1.0+)
    
    Example:
        >>> vol = calculate_volatility(prices)
        >>> if vol > 0.10:
        ...     print("High volatility - use wider stops")
        >>> else:
        ...     print("Normal volatility")
    """
    if len(prices) < 2:
        logger.debug("Insufficient data for volatility calculation")
        return 0.0
    
    try:
        # Calculate log returns
        log_returns = np.log(prices).diff().dropna()
        
        if len(log_returns) < 2:
            return 0.0
        
        # Calculate volatility
        if method == 'ewma':
            vol = log_returns.ewm(span=span, adjust=False).std().iloc[-1]
        else:  # standard deviation
            vol = log_returns.std()
        
        vol = float(vol) if not np.isnan(vol) else 0.0
        logger.debug(f"Volatility({method}, span={span}): {vol:.4f}")
        
        return vol
    
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0


# ============================================================================
# MOMENTUM
# ============================================================================

def calculate_momentum(
    prices: pd.Series,
    period: int = 14
) -> float:
    """
    Calculate price momentum (rate of change).
    
    Momentum Definition:
    Rate at which price is changing - faster change = stronger momentum.
    
    Formula:
    Momentum = (Current Price - Price N days ago) / Price N days ago * 100%
    
    Interpretation:
    - Positive momentum: Price accelerating upward
    - Negative momentum: Price accelerating downward
    - Momentum divergence: Price makes new high but momentum doesn't (bearish)
    
    Signals:
    - Strong momentum + trend: Continuation likely
    - Weak/negative momentum + trend: Trend may be reversing
    - High momentum extremes: Pullback/reversal likely coming
    
    Args:
        prices (pd.Series): Price series
        period (int): Look-back period (default 14)
    
    Returns:
        float: Momentum as percentage change (-100 to +100+)
    
    Example:
        >>> momentum = calculate_momentum(prices)
        >>> if momentum > 5:
        ...     print("Strong upward momentum")
        >>> elif momentum < -5:
        ...     print("Strong downward momentum")
    """
    if len(prices) < period + 1:
        logger.debug(f"Insufficient data for momentum: {len(prices)} < {period + 1}")
        return 0.0
    
    try:
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-period-1]
        
        momentum = ((current_price - past_price) / past_price) * 100
        
        logger.debug(f"Momentum({period}): {momentum:.2f}%")
        
        return float(momentum) if not np.isnan(momentum) else 0.0
    
    except Exception as e:
        logger.error(f"Error calculating momentum: {e}")
        return 0.0


# ============================================================================
# SUPPORT & RESISTANCE
# ============================================================================

def get_support_resistance(
    prices: pd.Series,
    window: int = 20
) -> Tuple[float, float]:
    """
    Identify support and resistance price levels.
    
    Support/Resistance Definition:
    - Support: Price level where buyers tend to step in (floor)
    - Resistance: Price level where sellers tend to step in (ceiling)
    
    Method:
    Finds highest and lowest prices in recent window.
    
    Trading Signals:
    - Price bounces off support: Bullish
    - Price breaks above resistance: Bullish (breakout)
    - Price bounces off resistance: Bearish
    - Price breaks below support: Bearish (breakdown)
    
    Args:
        prices (pd.Series): Price series
        window (int): Look-back window (default 20 days)
    
    Returns:
        Tuple of (support_level, resistance_level)
    
    Example:
        >>> support, resistance = get_support_resistance(prices)
        >>> print(f"Buy near support: ${support:.2f}")
        >>> print(f"Sell near resistance: ${resistance:.2f}")
    """
    try:
        if len(prices) < window:
            current_price = float(prices.iloc[-1])
            return current_price * 0.95, current_price * 1.05
        
        recent_prices = prices.tail(window)
        support = float(recent_prices.min())
        resistance = float(recent_prices.max())
        
        logger.debug(f"Support: {support:.2f}, Resistance: {resistance:.2f}")
        
        return support, resistance
    
    except Exception as e:
        logger.error(f"Error calculating support/resistance: {e}")
        current_price = float(prices.iloc[-1])
        return current_price * 0.95, current_price * 1.05


# ============================================================================
# TREND IDENTIFICATION
# ============================================================================

def identify_trend(
    prices: pd.Series,
    short_window: int = 7,
    long_window: int = 30
) -> str:
    """
    Identify current market trend (uptrend, downtrend, or sideways).
    
    Method:
    Compares short-term MA with long-term MA:
    - Short MA > Long MA by >2%: Uptrend
    - Short MA < Long MA by >2%: Downtrend
    - Otherwise: Sideways
    
    Trend Definition:
    - Uptrend: Series of higher highs and higher lows
    - Downtrend: Series of lower highs and lower lows
    - Sideways: No clear direction
    
    Significance:
    - Uptrend: More weight to bullish signals
    - Downtrend: More weight to bearish signals
    - Sideways: Mean reversion strategies work better
    
    Args:
        prices (pd.Series): Price series
        short_window (int): Fast MA period (default 7)
        long_window (int): Slow MA period (default 30)
    
    Returns:
        str: 'uptrend', 'downtrend', 'sideways', or 'insufficient_data'
    
    Example:
        >>> trend = identify_trend(prices)
        >>> if trend == 'uptrend':
        ...     print("Bullish bias - favor long positions")
    """
    try:
        if len(prices) < long_window:
            logger.debug("Insufficient data for trend identification")
            return "insufficient_data"
        
        ma_short = prices.rolling(short_window).mean().iloc[-1]
        ma_long = prices.rolling(long_window).mean().iloc[-1]
        
        diff_pct = ((ma_short - ma_long) / ma_long) * 100
        
        if diff_pct > 2:
            trend = "uptrend"
        elif diff_pct < -2:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        logger.info(f"Trend: {trend} (diff: {diff_pct:.2f}%)")
        
        return trend
    
    except Exception as e:
        logger.error(f"Error identifying trend: {e}")
        return "sideways"


# ============================================================================
# BOLLINGER BANDS
# ============================================================================

def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands (volatility bands around MA).
    
    Bollinger Bands Definition:
    Envelope of volatility bands around moving average:
    - Upper Band: MA + (2 × std dev)
    - Middle Band: MA (20-day)
    - Lower Band: MA - (2 × std dev)
    
    Width (Upper - Lower):
    - Wide bands: High volatility
    - Narrow bands: Low volatility
    - Bands are "squeezing": Volatility about to increase
    - Bands are "expanding": Volatility is increasing
    
    Price Position Signals:
    - Price at upper band: Overbought (potential pullback)
    - Price at lower band: Oversold (potential bounce)
    - Price walks upper band: Strong uptrend (continuation)
    - Price walks lower band: Strong downtrend (continuation)
    
    Args:
        prices (pd.Series): Price series
        period (int): MA period (default 20)
        num_std (float): Number of std deviations (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    
    Example:
        >>> upper, middle, lower = calculate_bollinger_bands(prices)
        >>> current = prices.iloc[-1]
        >>> if current > upper:
        ...     print("Price at upper band - overbought signal")
        >>> elif current < lower:
        ...     print("Price at lower band - oversold signal")
    """
    try:
        if len(prices) < period:
            current = float(prices.iloc[-1])
            return current * 1.05, current, current * 0.95
        
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = ma + (num_std * std)
        lower = ma - (num_std * std)
        
        upper_val = float(upper.iloc[-1])
        middle_val = float(ma.iloc[-1])
        lower_val = float(lower.iloc[-1])
        
        logger.debug(f"Bollinger Bands: {lower_val:.2f} - {middle_val:.2f} - {upper_val:.2f}")
        
        return upper_val, middle_val, lower_val
    
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        current = float(prices.iloc[-1])
        return current * 1.05, current, current * 0.95


# ============================================================================
# COMPREHENSIVE INDICATOR FUNCTION
# ============================================================================

def get_all_indicators(
    prices: pd.Series,
    pct_24h: Optional[float] = None,
    pct_7d: Optional[float] = None
) -> dict:
    """
    Calculate all technical indicators at once.
    
    Convenience function that calculates complete technical analysis package:
    1. RSI (Relative Strength Index)
    2. Moving Averages (7, 14, 30-day)
    3. Volatility
    4. Momentum
    5. Support & Resistance
    6. Bollinger Bands
    7. Trend
    8. Price changes (24h, 7d)
    
    Args:
        prices (pd.Series): Historical price series
        pct_24h (Optional[float]): 24h percentage change (if available)
        pct_7d (Optional[float]): 7d percentage change (if available)
    
    Returns:
        Dict with all indicator values:
        - rsi: RSI(14)
        - volatility: Current volatility
        - momentum: Momentum(14)
        - trend: 'uptrend', 'downtrend', or 'sideways'
        - ma7, ma14, ma30: Moving averages
        - support, resistance: Price levels
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - pct_24h, pct_7d: Price changes (if provided)
    
    Example:
        >>> indicators = get_all_indicators(prices)
        >>> print(f"RSI: {indicators['rsi']:.1f}")
        >>> print(f"Trend: {indicators['trend']}")
        >>> print(f"Volatility: {indicators['volatility']:.4f}")
    """
    logger.info("Calculating comprehensive technical indicators")
    
    indicators = {}
    
    try:
        # Individual indicators
        indicators['rsi'] = calculate_rsi(prices)
        indicators['volatility'] = calculate_volatility(prices)
        indicators['momentum'] = calculate_momentum(prices)
        indicators['trend'] = identify_trend(prices)
        
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
        
        logger.info(f"Indicators calculated: {len(indicators)} metrics")
        
        return indicators
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return minimal indicators on error
        return {
            'rsi': 50,
            'volatility': 0.05,
            'momentum': 0,
            'trend': 'sideways',
            'support': float(prices.iloc[-1]) * 0.95,
            'resistance': float(prices.iloc[-1]) * 1.05
        }

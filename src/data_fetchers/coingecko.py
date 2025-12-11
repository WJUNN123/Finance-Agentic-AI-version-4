"""
CoinGecko Data Fetcher
Handles fetching live cryptocurrency market data from CoinGecko API
"""

import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import time
import logging
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=5):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        wait_time = (2 ** attempt) * base_delay
                        logger.warning(f"Rate limited. Waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (2 ** attempt) * base_delay
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        return wrapper
    return decorator


class DataValidator:
    """Validates cryptocurrency data for anomalies and quality issues"""
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame, coin_id: str) -> Tuple[bool, List[str]]:
        """
        Validate price data for quality issues
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if df.empty:
            return False, ["No data returned"]
        
        if 'price' not in df.columns:
            return False, ["Missing 'price' column"]
        
        prices = df['price'].dropna()
        
        if len(prices) == 0:
            return False, ["All prices are null"]
        
        # Check for zero/negative prices
        if (prices <= 0).any():
            issues.append(f"Found {(prices <= 0).sum()} zero/negative prices")
        
        # Check for extreme outliers (>10x spike or <0.1x drop)
        if len(prices) > 1:
            price_changes = prices.pct_change().abs()
            extreme_changes = price_changes[price_changes > 10.0]
            if len(extreme_changes) > 0:
                issues.append(f"Found {len(extreme_changes)} extreme price spikes (>1000%)")
        
        # Check data freshness (if timestamp exists)
        if hasattr(df.index, 'tz'):
            latest = df.index[-1]
            age_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
            if age_hours > 24:
                issues.append(f"Data is {age_hours:.1f} hours old")
        
        # Check for gaps in data
        if len(df) > 1:
            expected_points = (df.index[-1] - df.index[0]).days + 1
            actual_points = len(df)
            if actual_points < expected_points * 0.9:  # Allow 10% gap
                issues.append(f"Missing {expected_points - actual_points} data points")
        
        is_valid = len(issues) == 0
        
        if issues:
            logger.warning(f"Data validation issues for {coin_id}: {', '.join(issues)}")
        
        return is_valid, issues
    
    @staticmethod
    def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data by removing/fixing anomalies"""
        if df.empty or 'price' not in df.columns:
            return df
        
        df = df.copy()
        
        # Remove negative/zero prices
        df = df[df['price'] > 0]
        
        # Remove extreme outliers using IQR method
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        before_count = len(df)
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        after_count = len(df)
        
        if before_count > after_count:
            logger.info(f"Removed {before_count - after_count} outlier data points")
        
        # Fill small gaps with interpolation
        df = df.sort_index()
        if len(df) > 2:
            df['price'] = df['price'].interpolate(method='linear', limit=2)
        
        return df


class BinanceFallback:
    """Fallback data source using Binance public API"""
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Map common coin symbols to Binance trading pairs
    SYMBOL_MAP = {
        'bitcoin': 'BTCUSDT',
        'ethereum': 'ETHUSDT',
        'binancecoin': 'BNBUSDT',
        'ripple': 'XRPUSDT',
        'solana': 'SOLUSDT',
        'cardano': 'ADAUSDT',
        'dogecoin': 'DOGEUSDT',
    }
    
    @staticmethod
    @retry_with_backoff(max_retries=2)
    def get_current_price(coin_id: str) -> Optional[float]:
        """Get current price from Binance"""
        symbol = BinanceFallback.SYMBOL_MAP.get(coin_id.lower())
        if not symbol:
            logger.warning(f"No Binance mapping for {coin_id}")
            return None
        
        try:
            url = f"{BinanceFallback.BASE_URL}/ticker/price"
            response = requests.get(url, params={'symbol': symbol}, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data['price'])
            logger.info(f"✅ Binance fallback: {coin_id} = ${price:,.2f}")
            return price
            
        except Exception as e:
            logger.error(f"Binance fallback failed for {coin_id}: {e}")
            return None
    
    @staticmethod
    @retry_with_backoff(max_retries=2)
    def get_historical_klines(coin_id: str, days: int = 180) -> Optional[pd.DataFrame]:
        """Get historical price data from Binance"""
        symbol = BinanceFallback.SYMBOL_MAP.get(coin_id.lower())
        if not symbol:
            return None
        
        try:
            url = f"{BinanceFallback.BASE_URL}/klines"
            
            # Binance limits: 1000 klines per request
            interval = '1d' if days > 90 else '4h'
            limit = min(days * (6 if interval == '4h' else 1), 1000)
            
            response = requests.get(url, params={
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }, timeout=20)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse kline data: [timestamp, open, high, low, close, volume, ...]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.set_index('timestamp')
            df['price'] = df['close'].astype(float)
            
            # Keep only price column
            df = df[['price']]
            
            logger.info(f"✅ Binance historical data: {len(df)} points for {coin_id}")
            return df
            
        except Exception as e:
            logger.error(f"Binance historical fallback failed: {e}")
            return None


class CoinGeckoFetcher:
    """Fetches cryptocurrency data from CoinGecko API with fallback and validation"""
    
    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3", 
                 timeout: int = 20):
        """
        Initialize CoinGecko fetcher
        
        Args:
            base_url: Base URL for CoinGecko API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.last_request_time = 0
        self.min_request_interval = 1.2
        self.validator = DataValidator()
        self.binance = BinanceFallback()
        
        # Circuit breaker pattern
        self.failure_count = 0
        self.circuit_open = False
        self.circuit_open_time = None
        self.circuit_timeout = 60  # seconds
        
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should open/close"""
        if self.circuit_open:
            elapsed = time.time() - self.circuit_open_time
            if elapsed > self.circuit_timeout:
                logger.info("Circuit breaker closing, retrying CoinGecko")
                self.circuit_open = False
                self.failure_count = 0
            else:
                logger.warning(f"Circuit breaker OPEN. Using fallback sources.")
                return True
        return False
    
    def _record_failure(self):
        """Record API failure for circuit breaker"""
        self.failure_count += 1
        if self.failure_count >= 3:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.error("Circuit breaker OPENED due to repeated failures")
    
    def _record_success(self):
        """Record API success"""
        self.failure_count = 0
        if self.circuit_open:
            self.circuit_open = False
            logger.info("Circuit breaker CLOSED after successful request")
        
    @retry_with_backoff(max_retries=3)
    def get_market_data(self, coin_ids: List[str]) -> pd.DataFrame:
        """Fetch market data with fallback support"""
        
        if self._check_circuit_breaker():
            # Use Binance fallback
            return self._get_market_data_fallback(coin_ids)
        
        self._rate_limit()
        
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": max(1, len(coin_ids)),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        
        try:
            logger.info(f"Fetching market data for: {', '.join(coin_ids)}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"No data returned from CoinGecko, trying fallback")
                return self._get_market_data_fallback(coin_ids)
            
            df = pd.DataFrame(data)
            self._record_success()
            
            logger.info(f"✅ Successfully fetched data for {len(df)} coins")
            return df
            
        except Exception as e:
            logger.error(f"CoinGecko market data failed: {e}")
            self._record_failure()
            return self._get_market_data_fallback(coin_ids)
    
    def _get_market_data_fallback(self, coin_ids: List[str]) -> pd.DataFrame:
        """Fallback market data using Binance"""
        data = []
        
        for coin_id in coin_ids:
            price = self.binance.get_current_price(coin_id)
            if price:
                data.append({
                    'id': coin_id,
                    'current_price': price,
                    'market_cap': 0,  # Not available from Binance
                    'total_volume': 0,
                    'price_change_percentage_24h': 0,
                    'price_change_percentage_7d_in_currency': 0
                })
        
        if not data:
            logger.error("All fallback sources failed")
            return pd.DataFrame()
        
        logger.info(f"✅ Fallback data retrieved for {len(data)} coins")
        return pd.DataFrame(data)
            
    @retry_with_backoff(max_retries=3)
    def get_historical_data(self, coin_id: str, days: int = 180) -> pd.DataFrame:
        """
        Fetch historical price data with validation and fallback
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data
            
        Returns:
            DataFrame with timestamp and price columns
        """
        
        if self._check_circuit_breaker():
            return self._get_historical_fallback(coin_id, days)
        
        self._rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 90 else "hourly"
        }
        
        try:
            logger.info(f"Fetching {days} days of historical data for {coin_id}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get("prices", [])
            
            if not prices:
                logger.warning(f"No historical data from CoinGecko, trying fallback")
                return self._get_historical_fallback(coin_id, days)
                
            df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.set_index("timestamp").drop(columns=["timestamp_ms"])
            
            # Validate data quality
            is_valid, issues = self.validator.validate_price_data(df, coin_id)
            
            if not is_valid:
                logger.warning(f"Data validation failed: {issues}")
                # Try to clean the data
                df = self.validator.clean_price_data(df)
                
                # If still not enough data, try fallback
                if len(df) < days * 0.5:
                    logger.warning("Insufficient data after cleaning, trying fallback")
                    return self._get_historical_fallback(coin_id, days)
            
            self._record_success()
            logger.info(f"✅ Successfully fetched {len(df)} historical data points")
            return df
            
        except Exception as e:
            logger.error(f"CoinGecko historical data failed: {e}")
            self._record_failure()
            return self._get_historical_fallback(coin_id, days)
    
    def _get_historical_fallback(self, coin_id: str, days: int) -> pd.DataFrame:
        """Fallback historical data using Binance"""
        df = self.binance.get_historical_klines(coin_id, days)
        
        if df is not None and not df.empty:
            # Validate and clean
            is_valid, issues = self.validator.validate_price_data(df, coin_id)
            if not is_valid:
                df = self.validator.clean_price_data(df)
            return df
        
        logger.error("All historical data sources failed")
        return pd.DataFrame(columns=["price"])
            
    def get_coin_info(self, coin_id: str) -> Dict:
        """Fetch detailed coin information"""
        self._rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false"
        }
        
        try:
            logger.info(f"Fetching coin info for {coin_id}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error fetching coin info: {e}")
            raise
            
    def validate_coin_id(self, coin_id: str) -> bool:
        """Check if a coin ID is valid"""
        try:
            data = self.get_market_data([coin_id])
            return not data.empty
        except Exception:
            return False


# Singleton instance
_fetcher = None

def get_fetcher() -> CoinGeckoFetcher:
    """Get or create singleton CoinGecko fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = CoinGeckoFetcher()
    return _fetcher

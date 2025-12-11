"""
CoinGecko Data Fetcher
Handles fetching live cryptocurrency market data from CoinGecko API
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timezone
import time
import logging
import functools

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.Timeout as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached for {func.__name__}: Timeout")
                        raise
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"⏱️ Timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        if attempt == max_retries - 1:
                            raise
                        wait_time = base_delay * (2 ** attempt) * 5  # Longer wait for rate limits
                        logger.warning(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached for {func.__name__}: {e}")
                        raise
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"🔄 Request failed on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


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


class CoinGeckoFetcher:
    """Fetches cryptocurrency data from CoinGecko API"""
    
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
        self.min_request_interval = 1.2  # Rate limiting: ~50 requests/minute
        
    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    @log_performance
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_market_data(self, coin_ids: List[str]) -> pd.DataFrame:
        """
        Fetch market data with retry logic and exponential backoff
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            
        Returns:
            DataFrame with market data
            
        Raises:
            requests.RequestException: If all retries fail
        """
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
        
        logger.info(f"📊 Fetching market data for: {', '.join(coin_ids)}")
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            logger.warning(f"⚠️ No data returned for: {coin_ids}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Successfully fetched data for {len(df)} coins")
        return df
    
    @log_performance
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_historical_data(self, coin_id: str, days: int = 180) -> pd.DataFrame:
        """
        Fetch historical price data for a coin
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            days: Number of days of historical data (max 365 for free tier)
            
        Returns:
            DataFrame with timestamp and price columns
            
        Raises:
            requests.RequestException: If API request fails
        """
        self._rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 90 else "hourly"
        }
        
        logger.info(f"📈 Fetching {days} days of historical data for {coin_id}")
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        prices = data.get("prices", [])
        
        if not prices:
            logger.warning(f"⚠️ No historical data returned for {coin_id}")
            return pd.DataFrame(columns=["timestamp", "price"])
            
        df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.set_index("timestamp").drop(columns=["timestamp_ms"])
        
        logger.info(f"✅ Successfully fetched {len(df)} historical data points")
        return df
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_coin_info(self, coin_id: str) -> Dict:
        """
        Fetch detailed information about a specific coin
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Dictionary with coin information
        """
        self._rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false"
        }
        
        logger.info(f"ℹ️ Fetching coin info for {coin_id}")
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def validate_coin_id(self, coin_id: str) -> bool:
        """
        Check if a coin ID is valid
        
        Args:
            coin_id: CoinGecko coin ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            data = self.get_market_data([coin_id])
            return not data.empty
        except Exception as e:
            logger.error(f"❌ Validation failed for {coin_id}: {e}")
            return False


# Singleton instance for easy import
_fetcher = None

def get_fetcher() -> CoinGeckoFetcher:
    """Get or create singleton CoinGecko fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = CoinGeckoFetcher()
    return _fetcher


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

logger = logging.getLogger(__name__)


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
        
    def get_market_data(self, coin_ids: List[str], max_retries: int = 3) -> pd.DataFrame:
        """Fetch market data with retry logic and exponential backoff"""
        
        for attempt in range(max_retries):
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
                logger.info(f"Fetching market data for: {', '.join(coin_ids)} (attempt {attempt + 1})")
                response = requests.get(url, params=params, timeout=self.timeout)
                
                # Handle 429 specifically
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s
                    logger.warning(f"⚠️ Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    logger.warning(f"No data returned for: {coin_ids}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                logger.info(f"✅ Successfully fetched data for {len(df)} coins")
                return df
                
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                logger.error(f"❌ Failed after {max_retries} attempts: {e}")
                raise
                
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise
        
        return pd.DataFrame()
            
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
        
        try:
            logger.info(f"Fetching {days} days of historical data for {coin_id}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            prices = data.get("prices", [])
            
            if not prices:
                logger.warning(f"No historical data returned for {coin_id}")
                return pd.DataFrame(columns=["timestamp", "price"])
                
            df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.set_index("timestamp").drop(columns=["timestamp_ms"])
            
            logger.info(f"Successfully fetched {len(df)} historical data points")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
            
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
        
        try:
            logger.info(f"Fetching coin info for {coin_id}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Error fetching coin info: {e}")
            raise
            
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
        except Exception:
            return False


# Singleton instance for easy import
_fetcher = None

def get_fetcher() -> CoinGeckoFetcher:
    """Get or create singleton CoinGecko fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = CoinGeckoFetcher()
    return _fetcher

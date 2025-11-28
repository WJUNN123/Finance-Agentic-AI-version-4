"""Data fetching modules for cryptocurrency data and news"""

from .coingecko import CoinGeckoFetcher, get_fetcher
from .news import NewsFetcher, get_fetcher as get_news_fetcher

__all__ = [
    'CoinGeckoFetcher',
    'get_fetcher',
    'NewsFetcher',
    'get_news_fetcher'
]
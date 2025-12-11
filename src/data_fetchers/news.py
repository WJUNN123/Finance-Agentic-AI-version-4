"""
News Fetcher Module
Fetches cryptocurrency news from RSS feeds
"""

import feedparser
import time
from typing import List, Dict
from datetime import datetime
import logging
import functools

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ Max retries reached for {func.__name__}: {e}")
                        raise
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for {func.__name__} in {wait_time}s...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


class NewsFetcher:
    """Fetches cryptocurrency news from multiple RSS feeds"""
    
    DEFAULT_FEEDS = [
        {
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "name": "CoinDesk"
        },
        {
            "url": "https://cointelegraph.com/rss",
            "name": "Cointelegraph"
        },
        {
            "url": "https://news.google.com/rss/search?q=cryptocurrency&hl=en-US&gl=US&ceid=US:en",
            "name": "Google News"
        }
    ]
    
    def __init__(self, feeds: List[Dict] = None, max_articles_per_feed: int = 20):
        """
        Initialize news fetcher
        
        Args:
            feeds: List of feed dictionaries with 'url' and 'name' keys
            max_articles_per_feed: Maximum articles to fetch per feed
        """
        self.feeds = feeds or self.DEFAULT_FEEDS
        self.max_articles_per_feed = max_articles_per_feed
    
    def fetch_articles(self, keyword: str = None, max_total: int = 50) -> List[Dict]:
        """
        Fetch articles from all RSS feeds
        
        Args:
            keyword: Optional keyword to filter articles (case-insensitive)
            max_total: Maximum total articles to return
            
        Returns:
            List of article dictionaries with title, summary, link, published time
        """
        all_articles = []
        keyword_lower = keyword.lower() if keyword else None
        
        logger.info(f"📰 Fetching news articles (keyword: {keyword or 'all'})")
        
        for feed in self.feeds:
            try:
                articles = self._fetch_single_feed(
                    feed["url"], 
                    feed["name"],
                    keyword_lower
                )
                all_articles.extend(articles)
                logger.info(f"✅ Fetched {len(articles)} articles from {feed['name']}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching from {feed['name']}: {e}")
                continue
                
        # Remove duplicates based on title
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by published time (newest first)
        unique_articles.sort(key=lambda x: x["published_ts"], reverse=True)
        
        # Limit total articles
        result = unique_articles[:max_total]
        logger.info(f"📊 Returning {len(result)} total unique articles")
        
        return result
    
    @retry_with_backoff(max_retries=2, base_delay=1)
    def _fetch_single_feed(self, url: str, source_name: str, 
                          keyword: str = None) -> List[Dict]:
        """
        Fetch articles from a single RSS feed with retry logic
        
        Args:
            url: RSS feed URL
            source_name: Name of the source
            keyword: Optional keyword filter
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        logger.debug(f"🔍 Parsing feed: {source_name}")
        feed = feedparser.parse(url)
        
        if not feed.entries:
            logger.warning(f"⚠️ No entries found in feed: {source_name}")
            return articles
        
        for entry in feed.entries[:self.max_articles_per_feed]:
            try:
                # Extract article data
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                link = entry.get("link", "")
                
                # Parse published time
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    published_ts = time.mktime(published)
                else:
                    published_ts = time.time()
                
                # Filter by keyword if provided
                if keyword:
                    text_blob = f"{title} {summary}".lower()
                    if keyword not in text_blob:
                        continue
                
                # Create article dictionary
                article = {
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": source_name,
                    "published_ts": published_ts,
                    "published_date": self._format_timestamp(published_ts)
                }
                
                articles.append(article)
                
            except Exception as e:
                logger.debug(f"⚠️ Error parsing entry from {source_name}: {e}")
                continue
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title"""
        seen_titles = set()
        unique = []
        
        for article in articles:
            title = article["title"].strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique.append(article)
        
        logger.debug(f"🔄 Deduplicated: {len(articles)} → {len(unique)} articles")
        return unique
    
    def _format_timestamp(self, timestamp: float) -> str:
        """Format Unix timestamp to human-readable date"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return "Unknown"
    
    def get_headlines(self, articles: List[Dict]) -> List[str]:
        """
        Extract just the headlines from articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of headline strings
        """
        return [article["title"] for article in articles if article.get("title")]


# Singleton instance
_fetcher = None

def get_fetcher() -> NewsFetcher:
    """Get or create singleton news fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = NewsFetcher()
    return _fetcher

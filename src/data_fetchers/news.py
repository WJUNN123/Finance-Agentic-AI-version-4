"""
News Fetcher Module
Fetches cryptocurrency news from RSS feeds
"""

import feedparser
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class NewsCache:
    """Simple in-memory cache for news articles"""
    
    def __init__(self, ttl_minutes: int = 30):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[List[Dict]]:
        """Get cached articles if still valid"""
        if key in self.cache:
            articles, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.info(f"✅ Cache HIT for key: {key}")
                return articles
            else:
                logger.info(f"⏰ Cache EXPIRED for key: {key}")
                del self.cache[key]
        return None
    
    def set(self, key: str, articles: List[Dict]):
        """Cache articles"""
        self.cache[key] = (articles, datetime.now())
        logger.info(f"💾 Cached {len(articles)} articles for key: {key}")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("🗑️ Cache cleared")


class NewsFetcher:
    """Enhanced news fetcher with multiple sources and fallbacks"""
    
    # Extended feed list with more sources
    DEFAULT_FEEDS = [
        {
            "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "name": "CoinDesk",
            "priority": 1
        },
        {
            "url": "https://cointelegraph.com/rss",
            "name": "Cointelegraph",
            "priority": 1
        },
        {
            "url": "https://news.google.com/rss/search?q=cryptocurrency&hl=en-US&gl=US&ceid=US:en",
            "name": "Google News - Crypto",
            "priority": 2
        },
        {
            "url": "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en",
            "name": "Google News - Bitcoin",
            "priority": 2
        },
        {
            "url": "https://cryptonews.com/news/feed/",
            "name": "CryptoNews",
            "priority": 3
        }
    ]
    
    def __init__(
        self, 
        feeds: List[Dict] = None, 
        max_articles_per_feed: int = 25,
        enable_cache: bool = True,
        cache_ttl_minutes: int = 30
    ):
        """
        Initialize enhanced news fetcher
        
        Args:
            feeds: List of feed dictionaries with 'url', 'name', 'priority'
            max_articles_per_feed: Maximum articles to fetch per feed
            enable_cache: Enable caching of articles
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.feeds = feeds or self.DEFAULT_FEEDS
        self.max_articles_per_feed = max_articles_per_feed
        self.enable_cache = enable_cache
        
        # Initialize cache
        self.cache = NewsCache(ttl_minutes=cache_ttl_minutes) if enable_cache else None
        
        # Track feed health
        self.feed_health = {feed['name']: {'success': 0, 'failure': 0} for feed in self.feeds}
        
    def _get_cache_key(self, keyword: Optional[str], max_total: int) -> str:
        """Generate cache key based on parameters"""
        key_string = f"{keyword}_{max_total}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def fetch_articles(
        self, 
        keyword: str = None, 
        max_total: int = 50,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Fetch articles from all RSS feeds with caching
        
        Args:
            keyword: Optional keyword to filter articles (case-insensitive)
            max_total: Maximum total articles to return
            use_cache: Whether to use cached results
            
        Returns:
            List of article dictionaries
        """
        # Check cache first
        if use_cache and self.cache:
            cache_key = self._get_cache_key(keyword, max_total)
            cached_articles = self.cache.get(cache_key)
            if cached_articles:
                return cached_articles
        
        all_articles = []
        keyword_lower = keyword.lower() if keyword else None
        successful_feeds = 0
        
        # Sort feeds by priority (lower number = higher priority)
        sorted_feeds = sorted(self.feeds, key=lambda x: x.get('priority', 99))
        
        for feed in sorted_feeds:
            try:
                articles = self._fetch_single_feed(
                    feed["url"], 
                    feed["name"],
                    keyword_lower
                )
                
                if articles:
                    all_articles.extend(articles)
                    self.feed_health[feed['name']]['success'] += 1
                    successful_feeds += 1
                    logger.info(f"✅ Fetched {len(articles)} articles from {feed['name']}")
                else:
                    logger.warning(f"⚠️ No articles from {feed['name']}")
                    
            except Exception as e:
                self.feed_health[feed['name']]['failure'] += 1
                logger.error(f"❌ Error fetching from {feed['name']}: {e}")
                continue
        
        # Log feed health status
        if successful_feeds == 0:
            logger.error("🚨 All news feeds failed! Check internet connection.")
        else:
            logger.info(f"📰 Successfully fetched from {successful_feeds}/{len(sorted_feeds)} feeds")
        
        # Remove duplicates
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Sort by published time (newest first)
        unique_articles.sort(key=lambda x: x["published_ts"], reverse=True)
        
        # Limit total articles
        result = unique_articles[:max_total]
        
        # Cache the results
        if use_cache and self.cache and result:
            cache_key = self._get_cache_key(keyword, max_total)
            self.cache.set(cache_key, result)
        
        logger.info(f"📊 Returning {len(result)} unique articles")
        return result
        
    def _fetch_single_feed(
        self, 
        url: str, 
        source_name: str, 
        keyword: str = None,
        timeout: int = 15
    ) -> List[Dict]:
        """
        Fetch articles from a single RSS feed with timeout
        
        Args:
            url: RSS feed URL
            source_name: Name of the source
            keyword: Optional keyword filter
            timeout: Request timeout in seconds
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            # Parse feed with timeout
            feed = feedparser.parse(url, timeout=timeout)
            
            # Check for feed parsing errors
            if hasattr(feed, 'bozo') and feed.bozo:
                logger.warning(f"⚠️ Feed parsing issue for {source_name}: {feed.bozo_exception}")
            
            # Check if feed has entries
            if not hasattr(feed, 'entries') or not feed.entries:
                logger.warning(f"⚠️ No entries found in {source_name}")
                return articles
            
            for entry in feed.entries[:self.max_articles_per_feed]:
                # Extract article data with better error handling
                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                link = entry.get("link", "").strip()
                
                # Skip if no title or link
                if not title or not link:
                    continue
                
                # Parse published time with fallback
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    try:
                        published_ts = time.mktime(published)
                    except (ValueError, OverflowError):
                        published_ts = time.time()
                else:
                    published_ts = time.time()
                    
                # Filter by keyword if provided
                if keyword:
                    text_blob = f"{title} {summary}".lower()
                    if keyword not in text_blob:
                        continue
                
                # Extract author if available
                author = entry.get("author", "Unknown")
                
                # Create article dictionary
                article = {
                    "title": title,
                    "summary": summary[:500] if summary else "",  # Limit summary length
                    "link": link,
                    "source": source_name,
                    "author": author,
                    "published_ts": published_ts,
                    "published_date": self._format_timestamp(published_ts)
                }
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"❌ Error parsing feed {url}: {type(e).__name__} - {str(e)}")
            
        return articles
        
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate articles based on title similarity
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Deduplicated list
        """
        seen_titles = set()
        unique = []
        
        for article in articles:
            # Normalize title for comparison
            title = article["title"].lower().strip()
            
            # Simple deduplication by exact title match
            if title not in seen_titles:
                seen_titles.add(title)
                unique.append(article)
        
        duplicates_removed = len(articles) - len(unique)
        if duplicates_removed > 0:
            logger.info(f"🔄 Removed {duplicates_removed} duplicate articles")
        
        return unique
        
    def _format_timestamp(self, timestamp: float) -> str:
        """
        Format Unix timestamp to human-readable date
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted date string
        """
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, OSError):
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
    
    def get_feed_health_report(self) -> Dict[str, Dict]:
        """
        Get health report for all feeds
        
        Returns:
            Dictionary with feed names and their success/failure counts
        """
        report = {}
        for feed_name, stats in self.feed_health.items():
            total = stats['success'] + stats['failure']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0
            report[feed_name] = {
                'success': stats['success'],
                'failure': stats['failure'],
                'success_rate': f"{success_rate:.1f}%"
            }
        return report
    
    def clear_cache(self):
        """Clear the article cache"""
        if self.cache:
            self.cache.clear()


# Singleton instance
_fetcher = None

def get_fetcher() -> NewsFetcher:
    """Get or create singleton news fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = NewsFetcher()
    return _fetcher

"""
Sentiment Analysis Module
Uses Twitter-RoBERTa model for cryptocurrency news sentiment analysis
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# SENTIMENT TREND ANALYZER
# ============================================================================

class SentimentTrendAnalyzer:
    """
    Analyzes sentiment trends over time to detect improving or deteriorating conditions.
    
    Key Insight:
    Sentiment CHANGE often signals turning points before price moves.
    
    Example:
    - Sentiment improving from 0.2 to 0.5: Accumulation phase (BULLISH)
    - Sentiment deteriorating from 0.7 to 0.4: Distribution phase (BEARISH)
    
    Attributes:
        window_size (int): Number of periods to analyze for trend
        sentiment_history (List[Dict]): Historical sentiment scores with timestamps
    """
    
    def __init__(self, window_size: int = 7):
        """
        Initialize trend analyzer.
        
        Args:
            window_size (int): Days to look back for trend (default 7 = 1 week)
        """
        self.window_size = window_size
        self.sentiment_history = []
        logger.info(f"SentimentTrendAnalyzer initialized with {window_size}-period window")
    
    def add_sentiment_point(self, timestamp: datetime, score: float):
        """
        Add a sentiment data point to history.
        
        Args:
            timestamp (datetime): When the sentiment score was calculated
            score (float): Sentiment score (-1.0 to +1.0)
        """
        self.sentiment_history.append({
            'timestamp': timestamp,
            'score': score
        })
        logger.debug(f"Added sentiment point: {score:.2f} at {timestamp}")
    
    def get_trend(self) -> Dict:
        """
        Analyze sentiment trend using linear regression.
        
        Calculates:
        - Trend direction: improving, deteriorating, or stable
        - Trend strength: How fast is sentiment changing (slope magnitude)
        - Current score: Most recent sentiment score
        - Previous average: Average of earlier scores for comparison
        
        Method:
        Uses numpy.polyfit to fit a line to recent sentiment scores.
        Slope > 0.01 = improving
        Slope < -0.01 = deteriorating
        Otherwise = stable
        
        Returns:
            Dict with keys:
            - trend: 'improving', 'deteriorating', or 'insufficient_data'
            - direction: Same as trend (for consistency)
            - strength: Magnitude of slope (0.0 to 1.0)
            - slope: Actual regression slope
            - current_score: Latest sentiment score
            - previous_avg: Average of earlier scores
            
        Example:
            >>> trend_data = analyzer.get_trend()
            >>> if trend_data['direction'] == 'improving':
            ...     print("Sentiment improving - bullish signal")
        """
        if len(self.sentiment_history) < 2:
            logger.debug("Insufficient sentiment history for trend analysis")
            return {
                'trend': 'insufficient_data',
                'direction': 'neutral',
                'strength': 0.0
            }
        
        # Get recent scores
        recent = self.sentiment_history[-self.window_size:]
        scores = [s['score'] for s in recent]
        
        # Fit trend line using linear regression
        if len(scores) >= 2:
            x = np.arange(len(scores))
            coefficients = np.polyfit(x, scores, 1)  # Degree 1 = linear
            slope = coefficients[0]
            
            # Classify trend
            if slope > 0.01:
                direction = 'improving'
            elif slope < -0.01:
                direction = 'deteriorating'
            else:
                direction = 'stable'
            
            strength = abs(slope)
            current = float(scores[-1])
            previous_avg = float(np.mean(scores[:-1])) if len(scores) > 1 else current
            
            logger.info(f"Sentiment trend: {direction} (slope={slope:.4f}, strength={strength:.4f})")
            
            return {
                'trend': direction,
                'direction': direction,
                'strength': float(strength),
                'slope': float(slope),
                'current_score': current,
                'previous_avg': previous_avg
            }
        
        return {
            'trend': 'insufficient_data',
            'direction': 'neutral',
            'strength': 0.0
        }


# ============================================================================
# SOURCE CREDIBILITY WEIGHTER
# ============================================================================

class SourceCredibilityWeighter:
    """
    Weights news sources by their credibility and reliability.
    
    Not all news sources are equally reliable. This module assigns weights
    based on known credibility tiers in crypto journalism.
    
    Weight Scale:
    - 1.0: Top tier (CoinDesk, Cointelegraph)
    - 0.8-0.9: Mid tier (Crypto.com, Messari)
    - 0.5: Unknown sources
    
    This prevents rumor mills and low-quality blogs from skewing sentiment.
    
    Attributes:
        SOURCE_WEIGHTS (Dict[str, float]): Credibility weight for each source
    """
    
    # Curated list of crypto news sources ranked by credibility
    SOURCE_WEIGHTS = {
        'coindesk': 1.0,          # Tier 1: Institutional news
        'cointelegraph': 0.95,    # Tier 1: Established media
        'crypto.com': 0.85,       # Tier 2: Exchange/platform news
        'messari': 0.9,           # Tier 1: Crypto research
        'glassnode': 0.88,        # Tier 2: On-chain analytics
        'the block': 0.87,        # Tier 2: Crypto research
        'decrypt': 0.82,          # Tier 2: Tech journalism
        'coinmetrics': 0.89,      # Tier 2: On-chain metrics
        'defipulse': 0.85,        # Tier 2: DeFi analytics
        'google news': 0.8,       # Tier 2: Aggregator
        'twitter': 0.4,           # Tier 3: User-generated (noisy)
        'reddit': 0.3,            # Tier 4: Community (very noisy)
    }
    
    def get_source_weight(self, source: str) -> float:
        """
        Get credibility weight for a news source.
        
        Args:
            source (str): Source name (e.g., "CoinDesk")
        
        Returns:
            float: Weight between 0.0 (unreliable) and 1.0 (most reliable)
            Returns 0.5 for unknown sources (neutral assumption)
        
        Note:
            Matching is case-insensitive and partial (substring match).
            This allows "Google News" to match "google news".
        """
        source_lower = source.lower()
        
        # Check if source matches any known source
        for known_source, weight in self.SOURCE_WEIGHTS.items():
            if known_source in source_lower:
                logger.debug(f"Source '{source}' matched '{known_source}' → weight {weight}")
                return weight
        
        # Unknown source: neutral weight
        logger.debug(f"Unknown source '{source}' → default weight 0.5")
        return 0.5
    
    def weight_sentiments(self, sentiments: List[Dict]) -> List[Dict]:
        """
        Apply source credibility weights to sentiment scores.
        
        Takes raw sentiment scores and adjusts them based on source reliability.
        
        Formula:
        weighted_score = raw_sentiment_score * source_weight
        
        Example:
        - CoinDesk positive (0.9 confidence) × 1.0 weight = 0.9 adjusted
        - Unknown blog positive (0.9 confidence) × 0.5 weight = 0.45 adjusted
        
        Args:
            sentiments (List[Dict]): Raw sentiment analysis results
        
        Returns:
            List[Dict]: Sentiments with added fields:
            - source_weight: Credibility weight for this source
            - weighted_score: Confidence adjusted for source credibility
        
        Note:
            Weighted scores are used in aggregate sentiment calculation,
            so high-credibility sources have more influence.
        """
        weighted = []
        
        for sentiment in sentiments:
            source = sentiment.get('source', 'unknown')
            weight = self.get_source_weight(source)
            
            # Adjust confidence score based on source credibility
            adjusted_score = sentiment['score'] * weight
            
            weighted.append({
                **sentiment,
                'source_weight': weight,
                'weighted_score': adjusted_score
            })
        
        return weighted


# ============================================================================
# ENHANCED SENTIMENT ANALYZER
# ============================================================================

class SentimentAnalyzer:
    """
    Enhanced sentiment analyzer with trend detection and source weighting.
    
    This module analyzes cryptocurrency news sentiment at scale:
    1. Uses transformer-based NLP model (Twitter-RoBERTa)
    2. Detects sentiment trends (improving vs deteriorating)
    3. Weights sources by credibility
    4. Identifies extreme sentiment conditions
    5. Provides comprehensive insights
    
    Key Improvements over Basic Sentiment:
    - Single score: "Positive" (0.5)
    - Enhanced output: "Positive, Improving, 72% confidence, Source-weighted"
    
    This provides much richer signal for decision-making.
    
    Attributes:
        model_name (str): HuggingFace model identifier
        device (int): -1 for CPU, 0+ for GPU
        batch_size (int): Process N texts simultaneously for efficiency
        pipeline: Loaded transformers pipeline
        trend_analyzer: SentimentTrendAnalyzer instance
        source_weighter: SourceCredibilityWeighter instance
        is_extreme_sentiment (bool): Is sentiment >75% one direction?
        sentiment_percentile (float): % of most common sentiment
    
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> headlines = ["Bitcoin surges 20%!", "Crypto crash warning"]
        >>> sources = ["CoinDesk", "unknown_blog"]
        >>> results = analyzer.analyze_texts(headlines, sources)
        >>> aggregate, df = analyzer.calculate_aggregate_sentiment(results)
        >>> insights = analyzer.get_sentiment_insights(results, breakdown)
        >>> print(f"Sentiment: {insights['consensus']} "
        ...       f"({insights['trend_direction']})")
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name (str): HuggingFace model to use
                Default: Twitter-RoBERTa (fine-tuned on Twitter, works well for crypto)
            device (int): -1 for CPU (default), 0+ for GPU
                Use GPU (0) if available for faster processing
            batch_size (int): Texts to process together (default 32)
                Higher = faster but uses more memory
            max_length (int): Max tokens per text (default 512)
                Longer texts are truncated
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pipeline = None
        
        # Label mapping: different models use different label formats
        self.label_mapping = {
            # Common label names
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
            # Alternative format (some models)
            "LABEL_0": -1.0,   # Negative
            "LABEL_1": 0.0,    # Neutral
            "LABEL_2": 1.0     # Positive
        }
        
        # Initialize helpers
        self.trend_analyzer = SentimentTrendAnalyzer()
        self.source_weighter = SourceCredibilityWeighter()
        
        # Sentiment extremity tracking
        self.is_extreme_sentiment = False
        self.sentiment_percentile = 50.0
        
        logger.info(f"SentimentAnalyzer initialized: model={model_name}, device={device}")
    
    def load_model(self):
        """
        Load the sentiment analysis model (lazy loading).
        
        Models are large (~500MB) so only loaded when needed.
        This speeds up app startup and saves memory.
        
        Raises:
            Exception: If model fails to load (bad internet, disk space, etc.)
        """
        if self.pipeline is not None:
            return  # Already loaded
        
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                truncation=True,
                max_length=self.max_length
            )
            
            logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise
    
    def analyze_texts(
        self, 
        texts: List[str],
        sources: List[str] = None
    ) -> List[Dict]:
        """
        Analyze sentiment of multiple texts (typically news headlines).
        
        Process:
        1. Load model if not already loaded
        2. Run sentiment analysis on batch of texts
        3. Normalize labels (different models use different names)
        4. Attach source information
        5. Apply source credibility weighting
        
        Args:
            texts (List[str]): News headlines/articles to analyze
            sources (List[str]): Source of each text (e.g., "CoinDesk")
                Optional - if provided, used for credibility weighting
        
        Returns:
            List[Dict] with keys:
            - text: Original text
            - label: Sentiment label ('positive', 'negative', 'neutral')
            - score: Confidence (0.0 to 1.0)
            - value: Numeric value (-1.0, 0.0, or +1.0)
            - source: News source
            - published_ts: Timestamp
            - source_weight: Credibility of source (0.0 to 1.0)
            - weighted_score: Score adjusted for source credibility
        
        Raises:
            TransformerError: If model fails during inference
        
        Example:
            >>> texts = ["Bitcoin hits $100k", "Crypto crash expected"]
            >>> results = analyzer.analyze_texts(texts)
            >>> for r in results:
            ...     print(f"{r['text']}: {r['label']} "
            ...           f"({r['score']:.2f} confidence)")
        """
        if not texts:
            logger.warning("No texts provided for sentiment analysis")
            return []
        
        # Ensure model is loaded
        if self.pipeline is None:
            self.load_model()
        
        try:
            logger.info(f"Analyzing sentiment for {len(texts)} texts")
            
            # Run sentiment analysis in batches
            predictions = self.pipeline(
                texts,
                batch_size=self.batch_size,
                truncation=True,
                max_length=self.max_length
            )
            
            # Process results
            results = []
            for i, (text, pred) in enumerate(zip(texts, predictions)):
                # Normalize label name
                label = pred["label"].lower()
                score = float(pred["score"])
                
                # Map label to numeric value
                value = self.label_mapping.get(label, 0.0)
                
                # Get source if provided
                source = sources[i] if sources and i < len(sources) else "unknown"
                
                results.append({
                    "text": text,
                    "label": label,
                    "score": score,
                    "value": value,
                    "source": source,
                    "published_ts": datetime.now()
                })
            
            # Apply source weighting
            results = self.source_weighter.weight_sentiments(results)
            
            logger.info(f"Sentiment analysis complete: {len(results)} texts processed")
            return results
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
    
    def calculate_aggregate_sentiment(
        self, 
        analyses: List[Dict],
        use_weighting: bool = True
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate weighted aggregate sentiment score from individual analyses.
        
        Two modes:
        1. Weighted (default): High-credibility sources have more influence
        2. Unweighted: All sources treated equally
        
        Formula (Weighted):
        aggregate = Σ(sentiment_value × confidence × source_weight) / Σ(confidence × source_weight)
        
        This gives:
        - More weight to high-confidence sentiment
        - More weight to high-credibility sources
        - Less weight to low-confidence or low-credibility sentiment
        
        Args:
            analyses (List[Dict]): Sentiment analysis results from analyze_texts()
            use_weighting (bool): Apply source credibility weighting (default True)
        
        Returns:
            Tuple of:
            - aggregate_score (float): -1.0 (very negative) to +1.0 (very positive)
            - dataframe (pd.DataFrame): All sentiment data for further analysis
        
        Example:
            >>> score, df = analyzer.calculate_aggregate_sentiment(results)
            >>> print(f"Market sentiment: {score:.2f}")
            >>> if score > 0.5:
            ...     print("Strong positive sentiment")
            >>> if score < -0.5:
            ...     print("Strong negative sentiment")
        """
        if not analyses:
            logger.warning("No analyses provided for aggregation")
            return 0.0, pd.DataFrame()
        
        # Determine which scores to use
        if use_weighting and 'weighted_score' in analyses[0]:
            scores = [a['weighted_score'] for a in analyses]
            weights = [a['score'] for a in analyses]  # Use original confidence as weight
            logger.info("Using source-weighted sentiment calculation")
        else:
            scores = [a['value'] for a in analyses]
            weights = [a['score'] for a in analyses]  # Use confidence as weight
            logger.info("Using unweighted sentiment calculation")
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            aggregate_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            aggregate_score = 0.0
        
        # Clamp to [-1, 1] range
        aggregate_score = max(-1.0, min(1.0, aggregate_score))
        
        # Update trend
        self.trend_analyzer.add_sentiment_point(datetime.now(), aggregate_score)
        
        # Create DataFrame
        df = pd.DataFrame(analyses)
        
        logger.info(f"Aggregate sentiment: {aggregate_score:.3f}")
        
        return float(aggregate_score), df
    
    def get_sentiment_breakdown(self, analyses: List[Dict]) -> Dict[str, float]:
        """
        Get percentage breakdown of positive/neutral/negative sentiment.
        
        Classifies each analysis into one of three categories and
        calculates percentage distribution.
        
        Args:
            analyses (List[Dict]): Sentiment analysis results
        
        Returns:
            Dict with keys:
            - positive: % of positive sentiment (0-100)
            - neutral: % of neutral sentiment (0-100)
            - negative: % of negative sentiment (0-100)
            
        Total always sums to 100%
        
        Example:
            >>> breakdown = analyzer.get_sentiment_breakdown(results)
            >>> print(f"Sentiment: {breakdown['positive']:.0f}% pos, "
            ...       f"{breakdown['neutral']:.0f}% neu, "
            ...       f"{breakdown['negative']:.0f}% neg")
        """
        if not analyses:
            logger.debug("No analyses for breakdown")
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        total = len(analyses)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for analysis in analyses:
            label = analysis["label"].lower()
            # Classify label
            if label in ["positive", "label_2"]:
                counts["positive"] += 1
            elif label in ["negative", "label_0"]:
                counts["negative"] += 1
            else:
                counts["neutral"] += 1
        
        # Calculate percentages
        percentages = {
            key: (count / total) * 100.0 
            for key, count in counts.items()
        }
        
        # Detect extreme sentiment
        max_pct = max(percentages.values())
        self.is_extreme_sentiment = max_pct > 75  # More than 75% one direction
        self.sentiment_percentile = max_pct
        
        logger.info(f"Sentiment breakdown: "
                   f"{percentages['positive']:.1f}% pos, "
                   f"{percentages['neutral']:.1f}% neu, "
                   f"{percentages['negative']:.1f}% neg")
        
        return percentages
    
    def get_sentiment_insights(
        self, 
        analyses: List[Dict], 
        breakdown: Dict
    ) -> Dict:
        """
        Generate comprehensive sentiment insights combining all metrics.
        
        Combines:
        - Overall consensus (which direction dominates)
        - Confidence in that consensus
        - Whether sentiment is improving or deteriorating
        - Whether sentiment is extreme (mean reversion risk)
        - Actionable warnings
        
        Args:
            analyses (List[Dict]): Individual sentiment analyses
            breakdown (Dict): Percentage breakdown (positive/neutral/negative)
        
        Returns:
            Dict with keys:
            - consensus: 'positive', 'negative', or 'neutral'
            - confidence: How strong the consensus (0.0 to 1.0)
            - trend_direction: 'improving', 'deteriorating', or 'stable'
            - trend_strength: How fast is it changing (0.0 to 1.0)
            - is_extreme: Boolean - is sentiment extreme?
            - extremeness_level: % of most common sentiment (0-100)
            - warning: String warning if extreme sentiment detected
            - trend_data: Full trend analysis data
        
        Example:
            >>> insights = analyzer.get_sentiment_insights(results, breakdown)
            >>> print(f"Consensus: {insights['consensus']}")
            >>> if insights['is_extreme']:
            ...     print(f"WARNING: {insights['warning']}")
        """
        if not analyses:
            logger.debug("No analyses for insights")
            return {
                'consensus': 'no_data',
                'confidence': 0.0,
                'trend_direction': 'unknown',
                'is_extreme': False,
                'warning': 'Insufficient sentiment data'
            }
        
        pos = breakdown.get('positive', 0)
        neg = breakdown.get('negative', 0)
        neu = breakdown.get('neutral', 0)
        
        # Determine consensus (which sentiment dominates)
        max_val = max(pos, neg, neu)
        if max_val == pos:
            consensus = 'positive'
        elif max_val == neg:
            consensus = 'negative'
        else:
            consensus = 'neutral'
        
        # Confidence = how dominant is the consensus
        confidence = max_val / 100.0
        
        # Get trend analysis
        trend_data = self.trend_analyzer.get_trend()
        trend_direction = trend_data.get('direction', 'unknown')
        
        # Generate warning for extreme sentiment
        warning = None
        if self.is_extreme_sentiment:
            if consensus == 'positive' and pos > 75:
                warning = (f"⚠️ EXTREME POSITIVE sentiment ({pos:.0f}%). "
                          f"Watch for mean reversion/pullback risk.")
            elif consensus == 'negative' and neg > 75:
                warning = (f"⚠️ EXTREME NEGATIVE sentiment ({neg:.0f}%). "
                          f"Potential capitulation/bounce opportunity.")
        
        insights = {
            'consensus': consensus,
            'confidence': float(confidence),
            'trend_direction': trend_direction,
            'trend_strength': float(trend_data.get('strength', 0.0)),
            'is_extreme': self.is_extreme_sentiment,
            'extremeness_level': float(self.sentiment_percentile),
            'warning': warning,
            'trend_data': trend_data
        }
        
        logger.info(f"Sentiment insights: consensus={consensus}, "
                   f"confidence={confidence:.2f}, "
                   f"trend={trend_direction}, "
                   f"extreme={self.is_extreme_sentiment}")
        
        return insights
    
    def get_recent_sentiment_trend(self, days: int = 7) -> List[Dict]:
        """
        Get sentiment trend over last N days.
        
        Useful for visualizing sentiment evolution over time.
        
        Args:
            days (int): Days to look back (default 7)
        
        Returns:
            List of dicts with 'timestamp' and 'score' keys
        """
        if not self.trend_analyzer.sentiment_history:
            return []
        
        recent = (self.trend_analyzer.sentiment_history[-days:] 
                 if days else self.trend_analyzer.sentiment_history)
        
        return [
            {
                'timestamp': item['timestamp'].isoformat(),
                'score': item['score']
            }
            for item in recent
        ]
    
    def interpret_sentiment(self, score: float) -> str:
        """
        Convert sentiment score to human-readable interpretation.
        
        Args:
            score (float): Sentiment score (-1.0 to +1.0)
        
        Returns:
            str: Human-readable interpretation
        
        Example:
            >>> text = analyzer.interpret_sentiment(0.7)
            >>> print(text)  # "Very Positive"
        """
        if score >= 0.5:
            return "Very Positive"
        elif score >= 0.2:
            return "Positive"
        elif score >= -0.2:
            return "Neutral"
        elif score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """
    Get or create singleton sentiment analyzer instance.
    
    Singleton pattern ensures only one model is loaded into memory,
    reducing memory usage and startup time.
    
    Returns:
        SentimentAnalyzer: Global analyzer instance
    
    Example:
        >>> analyzer = get_analyzer()
        >>> results = analyzer.analyze_texts(headlines)
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

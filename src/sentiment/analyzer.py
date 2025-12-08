"""
Sentiment Analysis Module
Uses Twitter-RoBERTa model for cryptocurrency news sentiment analysis
"""

from transformers import pipeline
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentTrendAnalyzer:
    """Analyze sentiment trends over time"""
    
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.sentiment_history = []
    
    def add_sentiment_point(self, timestamp: datetime, score: float):
        """Add sentiment data point"""
        self.sentiment_history.append({
            'timestamp': timestamp,
            'score': score
        })
    
    def get_trend(self) -> Dict:
        """Analyze sentiment trend"""
        if len(self.sentiment_history) < 2:
            return {'trend': 'insufficient_data', 'direction': 'neutral', 'strength': 0.0}
        
        # Get recent scores
        recent = self.sentiment_history[-self.window_size:]
        scores = [s['score'] for s in recent]
        
        # Calculate trend
        if len(scores) >= 2:
            # Linear regression slope
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            
            direction = 'improving' if slope > 0.01 else 'deteriorating' if slope < -0.01 else 'stable'
            strength = abs(slope)
            
            return {
                'trend': direction,
                'direction': direction,
                'strength': float(strength),
                'slope': float(slope),
                'current_score': float(scores[-1]),
                'previous_avg': float(np.mean(scores[:-1])) if len(scores) > 1 else scores[0]
            }
        
        return {'trend': 'insufficient_data', 'direction': 'neutral', 'strength': 0.0}


class SourceCredibilityWeighter:
    """Weight news sources by credibility"""
    
    SOURCE_WEIGHTS = {
        'coindesk': 1.0,
        'cointelegraph': 0.95,
        'google news': 0.8,
        'crypto.com': 0.85,
        'messari': 0.9,
        'glassnode': 0.88,
        'the block': 0.87,
        'decrypt': 0.82,
        'coinmetrics': 0.89,
        'defipulse': 0.85
    }
    
    def get_source_weight(self, source: str) -> float:
        """Get credibility weight for source (0.0 to 1.0)"""
        source_lower = source.lower()
        
        for known_source, weight in self.SOURCE_WEIGHTS.items():
            if known_source in source_lower:
                return weight
        
        # Default weight for unknown sources
        return 0.5
    
    def weight_sentiments(self, sentiments: List[Dict]) -> List[Dict]:
        """Apply source weights to sentiment scores"""
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


class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with trend and source analysis"""
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1,
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pipeline = None
        
        self.label_mapping = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
            "LABEL_0": -1.0,
            "LABEL_1": 0.0,
            "LABEL_2": 1.0
        }
        
        # Initialize helpers
        self.trend_analyzer = SentimentTrendAnalyzer()
        self.source_weighter = SourceCredibilityWeighter()
        
        # Cache for sentiment extremes
        self.is_extreme_sentiment = False
        self.sentiment_percentile = 50.0
    
    def load_model(self):
        """Load sentiment analysis model"""
        if self.pipeline is not None:
            return
        
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
    
    def analyze_texts(self, texts: List[str], sources: List[str] = None) -> List[Dict]:
        """Analyze sentiment with source information"""
        if not texts:
            return []
        
        if self.pipeline is None:
            self.load_model()
        
        try:
            logger.info(f"Analyzing sentiment for {len(texts)} texts")
            
            predictions = self.pipeline(
                texts,
                batch_size=self.batch_size,
                truncation=True,
                max_length=self.max_length
            )
            
            results = []
            for i, (text, pred) in enumerate(zip(texts, predictions)):
                label = pred["label"].lower()
                score = float(pred["score"])
                value = self.label_mapping.get(label, 0.0)
                
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
            
            logger.info(f"Sentiment analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
    
    def calculate_aggregate_sentiment(
        self, 
        analyses: List[Dict],
        use_weighting: bool = True
    ) -> Tuple[float, pd.DataFrame]:
        """Calculate weighted aggregate sentiment score"""
        if not analyses:
            return 0.0, pd.DataFrame()
        
        # Use weighted or simple scoring
        if use_weighting and 'weighted_score' in analyses[0]:
            scores = [a['weighted_score'] for a in analyses]
            weights = [a['score'] for a in analyses]  # Use original confidence as weight
        else:
            scores = [a['value'] for a in analyses]
            weights = [a['score'] for a in analyses]
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            aggregate_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            aggregate_score = 0.0
        
        # Update trend
        self.trend_analyzer.add_sentiment_point(datetime.now(), aggregate_score)
        
        # Create DataFrame
        df = pd.DataFrame(analyses)
        
        return float(aggregate_score), df
    
    def get_sentiment_breakdown(self, analyses: List[Dict]) -> Dict[str, float]:
        """Get percentage breakdown with extremeness detection"""
        if not analyses:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        
        total = len(analyses)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for analysis in analyses:
            label = analysis["label"]
            if label in ["positive", "label_2"]:
                counts["positive"] += 1
            elif label in ["negative", "label_0"]:
                counts["negative"] += 1
            else:
                counts["neutral"] += 1
        
        percentages = {
            key: (count / total) * 100.0 
            for key, count in counts.items()
        }
        
        # Detect extreme sentiment
        max_pct = max(percentages.values())
        self.is_extreme_sentiment = max_pct > 75  # More than 75% one direction
        self.sentiment_percentile = max_pct
        
        return percentages
    
    def get_sentiment_insights(self, analyses: List[Dict], breakdown: Dict) -> Dict:
        """Generate detailed sentiment insights"""
        if not analyses:
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
        
        # Determine consensus
        max_val = max(pos, neg, neu)
        if max_val == pos:
            consensus = 'positive'
        elif max_val == neg:
            consensus = 'negative'
        else:
            consensus = 'neutral'
        
        # Calculate confidence as percentage of consensus
        confidence = max_val / 100.0
        
        # Get trend
        trend_data = self.trend_analyzer.get_trend()
        trend_direction = trend_data.get('direction', 'unknown')
        
        # Generate warning for extreme sentiment
        warning = None
        if self.is_extreme_sentiment:
            if consensus == 'positive' and pos > 75:
                warning = f"⚠️ Extreme positive sentiment ({pos:.0f}%). Watch for mean reversion risk."
            elif consensus == 'negative' and neg > 75:
                warning = f"⚠️ Extreme negative sentiment ({neg:.0f}%). Potential capitulation/bounce opportunity."
        
        return {
            'consensus': consensus,
            'confidence': float(confidence),
            'trend_direction': trend_direction,
            'trend_strength': float(trend_data.get('strength', 0.0)),
            'is_extreme': self.is_extreme_sentiment,
            'extremeness_level': float(self.sentiment_percentile),
            'warning': warning,
            'trend_data': trend_data
        }
    
    def get_recent_sentiment_trend(self, days: int = 7) -> List[Dict]:
        """Get sentiment trend over last N days"""
        if not self.trend_analyzer.sentiment_history:
            return []
        
        recent = self.trend_analyzer.sentiment_history[-days:] if days else self.trend_analyzer.sentiment_history
        
        return [
            {
                'timestamp': item['timestamp'].isoformat(),
                'score': item['score']
            }
            for item in recent
        ]
    
    def interpret_sentiment(self, score: float) -> str:
        """Interpret aggregate sentiment score"""
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


# Singleton instance
_analyzer = None

def get_analyzer() -> EnhancedSentimentAnalyzer:
    """Get or create singleton sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = EnhancedSentimentAnalyzer()
    return _analyzer

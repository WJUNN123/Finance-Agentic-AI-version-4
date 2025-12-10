"""
Sentiment Analysis Module
Uses Twitter-RoBERTa model for cryptocurrency news sentiment analysis
"""

from transformers import pipeline
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of cryptocurrency news using Twitter-RoBERTa"""
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: int = -1,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize sentiment analyzer
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pipeline = None
        
        # Sentiment label mapping
        self.label_mapping = {
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0,
            "LABEL_0": -1.0,
            "LABEL_1": 0.0,
            "LABEL_2": 1.0
        }
        
        # New: Credibility weights for known sources
        self.source_weights = {
            "CoinDesk": 1.2,
            "Cointelegraph": 1.1,
            "Reuters": 1.5,
            "Bloomberg": 1.5,
            "The Block": 1.2,
            "Decrypt": 1.0,
            "Twitter/X": 0.6,
            "Reddit": 0.5,
            "Unknown": 0.8
        }
        
    def load_model(self):
        """Load the sentiment analysis model (lazy loading)"""
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
                max_length=self.max_length,
                batch_size=self.batch_size
            )
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    def analyze_news(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment with source weighting and trend detection
        
        Args:
            articles: List of article dicts with 'title' and 'source' keys
            
        Returns:
            Dictionary with score, breakdown, and extreme flags
        """
        self.load_model()
        if not articles:
            return {
                "score": 0.0,
                "is_extreme": False,
                "breakdown": {"positive": 0, "neutral": 0, "negative": 0},
                "article_count": 0
            }

        analyses = []
        
        for article in articles:
            text = article.get("title", "")
            if not text:
                continue
                
            source = article.get("source", "Unknown")
            
            # Get base sentiment
            try:
                result = self.pipeline(text)[0]
                score = result['score'] # Confidence
                label = result['label']
                
                # Map label to -1, 0, 1
                val = self.label_mapping.get(label, 0.0)
                
                # Apply source weight
                # If source string contains a known key (e.g. "CoinDesk (RSS)"), use that weight
                weight = self.source_weights["Unknown"]
                for known_source, w in self.source_weights.items():
                    if known_source.lower() in str(source).lower():
                        weight = w
                        break
                
                analyses.append({
                    "text": text,
                    "raw_score": val,
                    "confidence": score,
                    "weight": weight,
                    "weighted_val": val * weight,
                    "label": label
                })
            except Exception as e:
                logger.warning(f"Error analyzing text '{text[:30]}...': {e}")
                continue
            
        # Calculate Weighted Average
        total_weight = sum(a['weight'] for a in analyses)
        weighted_sum = sum(a['weighted_val'] for a in analyses)
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Detect Extremes (if > 60% is strongly one direction)
        is_extreme = abs(final_score) > 0.6
        
        # Calculate Breakdown
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for a in analyses:
            if a["raw_score"] > 0: counts["positive"] += 1
            elif a["raw_score"] < 0: counts["negative"] += 1
            else: counts["neutral"] += 1
            
        total = len(analyses)
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in counts.items()}

        return {
            "score": final_score,
            "is_extreme": is_extreme,
            "interpretation": self.interpret_sentiment(final_score),
            "breakdown": percentages,
            "article_count": len(analyses),
            "details": analyses
        }
    
    # Legacy compatibility methods
    def analyze_texts(self, texts: List[str]) -> List[Dict]:
        """Legacy wrapper for simple text lists"""
        # Convert to dummy article format
        articles = [{"title": t, "source": "Unknown"} for t in texts]
        res = self.analyze_news(articles)
        # Return list structure expected by old code if needed, but we encourage using analyze_news
        return res["details"]

    def calculate_aggregate_sentiment(self, analyses: List[Dict]) -> Tuple[float, pd.DataFrame]:
        """Legacy wrapper"""
        if not analyses:
            return 0.0, pd.DataFrame()
        
        # Recalculate unweighted average for backward compatibility or extract from new logic
        total_val = sum(a['weighted_val'] for a in analyses)
        total_weight = sum(a['weight'] for a in analyses)
        score = total_val / total_weight if total_weight > 0 else 0.0
        
        df = pd.DataFrame(analyses)
        return float(score), df

    def get_sentiment_breakdown(self, analyses: List[Dict]) -> Dict:
        """Legacy wrapper"""
        if not analyses:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for a in analyses:
            if a["raw_score"] > 0: counts["positive"] += 1
            elif a["raw_score"] < 0: counts["negative"] += 1
            else: counts["neutral"] += 1
            
        total = len(analyses)
        return {k: (v / total * 100) for k, v in counts.items()}

    def interpret_sentiment(self, score: float) -> str:
        """Interpret aggregate sentiment score as text"""
        if score >= 0.5: return "Very Positive"
        elif score >= 0.2: return "Positive"
        elif score >= -0.2: return "Neutral"
        elif score >= -0.5: return "Negative"
        else: return "Very Negative"


# Singleton instance
_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """Get or create singleton sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

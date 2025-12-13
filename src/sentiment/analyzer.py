"""
Sentiment Analysis Module
Uses Twitter-RoBERTa model for cryptocurrency news sentiment analysis
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import logging
import torch

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
        
        Args:
            model_name: Hugging Face model identifier
            device: -1 for CPU, 0+ for GPU
            batch_size: Batch size for processing
            max_length: Maximum token length
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
            "LABEL_0": -1.0,  # Negative (some models use LABEL_X format)
            "LABEL_1": 0.0,   # Neutral
            "LABEL_2": 1.0    # Positive
        }
        
    def load_model(self):
        """Load the sentiment analysis model (lazy loading)"""
        if self.pipeline is not None:
            return
            
        try:
            logger.info(f"🤖 Loading sentiment model: {self.model_name}")
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                truncation=True,
                max_length=self.max_length
            )
            
            logger.info("✅ Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiment model: {e}")
            raise
            
    def analyze_texts(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries with 'text', 'label', 'score', 'value' keys
        """
        if not texts:
            return []
            
        # Ensure model is loaded
        if self.pipeline is None:
            self.load_model()
            
        try:
            logger.info(f"🔍 Analyzing sentiment for {len(texts)} texts")
            
            # Run sentiment analysis
            predictions = self.pipeline(
                texts,
                batch_size=self.batch_size,
                truncation=True,
                max_length=self.max_length
            )
            
            # Process results
            results = []
            for text, pred in zip(texts, predictions):
                label = pred["label"].lower()
                score = float(pred["score"])
                value = self.label_mapping.get(label, 0.0)
                
                results.append({
                    "text": text,
                    "label": label,
                    "score": score,
                    "value": value
                })
            
            # Log distribution
            pos_count = sum(1 for r in results if r['value'] > 0)
            neg_count = sum(1 for r in results if r['value'] < 0)
            neu_count = len(results) - pos_count - neg_count
            
            logger.info(f"✅ Sentiment analysis complete: {pos_count} pos, {neu_count} neu, {neg_count} neg")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during sentiment analysis: {e}")
            raise
    
    def calculate_aggregate_sentiment(
        self, 
        analyses: List[Dict],
        use_recency_bias: bool = True
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate weighted aggregate sentiment with recency bias
        
        Args:
            analyses: List of sentiment analysis results
            use_recency_bias: Weight recent news higher (default True)
            
        Returns:
            Tuple of (aggregate_score, dataframe)
            - aggregate_score: Weighted sentiment (-1 to +1)
            - dataframe: Detailed results with all analyses
        """
        if not analyses:
            return 0.0, pd.DataFrame(columns=["text", "label", "score", "value", "weight"])
        
        # Calculate weighted score with recency bias
        total_value = 0.0
        total_weight = 0.0
        
        for idx, analysis in enumerate(analyses):
            value = analysis["value"]
            confidence = analysis["score"]
            
            # Recency weight: newer articles get higher weight
            if use_recency_bias:
                # Linear decay from 1.0 to 0.5
                recency_weight = 1.0 - (idx / len(analyses)) * 0.5
            else:
                recency_weight = 1.0
            
            # Combined weight: confidence * recency
            weight = confidence * recency_weight
            
            total_value += value * weight
            total_weight += weight
            
            # Store weight in analysis for transparency
            analysis['weight'] = weight
            analysis['recency_weight'] = recency_weight
        
        aggregate_score = total_value / total_weight if total_weight > 0 else 0.0
        
        # Create DataFrame with weights
        df = pd.DataFrame(analyses)
        
        logger.info(f"📊 Aggregate sentiment: {aggregate_score:.3f} (recency_bias={use_recency_bias})")
        
        return float(aggregate_score), df
    
    def get_sentiment_confidence(self, analyses: List[Dict]) -> float:
        """
        Calculate confidence in sentiment prediction based on agreement
        FIXED: Penalize high neutral percentage (uninformative data)
        
        Args:
            analyses: List of sentiment analysis results
            
        Returns:
            Confidence score 0-1
        """
        if not analyses:
            return 0.0
        
        # Get sentiment values and model confidence scores
        values = [a['value'] for a in analyses]
        scores = [a['score'] for a in analyses]
        
        # NEW: Check for high neutral percentage (uninformative)
        neutral_count = sum(1 for v in values if abs(v) < 0.1)
        neutral_pct = neutral_count / len(values)
        
        # CRITICAL FIX: Heavily penalize high neutral percentage
        if neutral_pct > 0.8:  # 80%+ neutral = mostly uninformative
            logger.debug(f"⚠️ High neutral percentage ({neutral_pct:.0%}) - low confidence")
            return 0.30  # Very low confidence
        elif neutral_pct > 0.6:  # 60%+ neutral = somewhat uninformative
            logger.debug(f"⚠️ Moderate neutral percentage ({neutral_pct:.0%}) - reduced confidence")
            return 0.50  # Medium-low confidence
        
        # Average model confidence
        avg_model_confidence = np.mean(scores)
        
        # Check agreement among sources (low variance = high agreement)
        if len(values) > 1:
            variance = np.var(values)
            # Convert variance to agreement score (0 variance = 1.0 agreement)
            # Max variance for sentiment is 4.0 (all -1 or all +1 spread)
            agreement_score = 1.0 - min(variance / 4.0, 1.0)
        else:
            agreement_score = 1.0
        
        # Sample size factor (more samples = higher confidence)
        sample_size_factor = min(len(analyses) / 20.0, 1.0)  # Cap at 20 articles
        
        # Combined confidence: weighted average
        confidence = (
            avg_model_confidence * 0.4 +
            agreement_score * 0.4 +
            sample_size_factor * 0.2
        )
        
        logger.debug(f"🎯 Sentiment confidence: {confidence:.3f} (model: {avg_model_confidence:.2f}, "
                    f"agreement: {agreement_score:.2f}, samples: {len(analyses)}, neutral: {neutral_pct:.0%})")
        
        return float(confidence)
        
    def get_sentiment_breakdown(self, analyses: List[Dict]) -> Dict[str, float]:
        """
        Get percentage breakdown of positive/neutral/negative sentiment
        
        Args:
            analyses: List of sentiment analysis results
            
        Returns:
            Dictionary with percentage of each sentiment category
        """
        if not analyses:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            
        total = len(analyses)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for analysis in analyses:
            label = analysis["label"]
            # Normalize label
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
        
        return percentages
    
    def get_sentiment_strength(self, score: float) -> str:
        """
        Get textual description of sentiment strength
        
        Args:
            score: Aggregate sentiment score (-1 to +1)
            
        Returns:
            Strength descriptor
        """
        abs_score = abs(score)
        
        if abs_score >= 0.7:
            return "Very Strong"
        elif abs_score >= 0.5:
            return "Strong"
        elif abs_score >= 0.3:
            return "Moderate"
        elif abs_score >= 0.1:
            return "Weak"
        else:
            return "Neutral"
        
    def interpret_sentiment(self, score: float) -> str:
        """
        Interpret aggregate sentiment score as text
        
        Args:
            score: Sentiment score (-1 to +1)
            
        Returns:
            Textual interpretation
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


# Singleton instance
_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """Get or create singleton sentiment analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

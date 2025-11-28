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
            logger.info(f"Analyzing sentiment for {len(texts)} texts")
            
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
                
            logger.info(f"Sentiment analysis complete")
            return results
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
            
    def calculate_aggregate_sentiment(
        self, 
        analyses: List[Dict]
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate weighted aggregate sentiment score
        
        Args:
            analyses: List of sentiment analysis results
            
        Returns:
            Tuple of (aggregate_score, dataframe)
            - aggregate_score: Weighted sentiment (-1 to +1)
            - dataframe: Detailed results with all analyses
        """
        if not analyses:
            return 0.0, pd.DataFrame(columns=["text", "label", "score", "value"])
            
        # Calculate weighted score
        total_value = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            value = analysis["value"]
            weight = analysis["score"]
            total_value += value * weight
            total_weight += weight
            
        aggregate_score = total_value / total_weight if total_weight > 0 else 0.0
        
        # Create DataFrame
        df = pd.DataFrame(analyses)
        
        return float(aggregate_score), df
        
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
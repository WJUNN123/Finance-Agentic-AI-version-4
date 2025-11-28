"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class GeminiInsightGenerator:
    """Generates investment insights using Gemini 2.0 Flash"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """
        Initialize Gemini insight generator
        
        Args:
            api_key: Google Gemini API key
            model_name: Model identifier
            temperature: Generation temperature (0-1)
            max_tokens: Maximum output tokens
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini model initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
            
    def generate_insights(
        self,
        coin_id: str,
        coin_symbol: str,
        market_data: Dict,
        sentiment_score: float,
        technical_indicators: Dict,
        forecast_data: Optional[Dict] = None,
        top_headlines: Optional[List[str]] = None,
        risk_tolerance: str = "medium",
        horizon_days: int = 7
    ) -> Dict:
        """
        Generate comprehensive investment insights
        
        Args:
            coin_id: Cryptocurrency ID
            coin_symbol: Cryptocurrency symbol
            market_data: Current market data
            sentiment_score: Aggregated sentiment score (-1 to +1)
            technical_indicators: RSI, momentum, etc.
            forecast_data: Price forecast information
            top_headlines: Recent news headlines
            risk_tolerance: User's risk tolerance (low/medium/high)
            horizon_days: Investment time horizon
            
        Returns:
            Dictionary with recommendation, score, insight text, and metadata
        """
        # Build comprehensive prompt
        prompt = self._build_prompt(
            coin_id=coin_id,
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_score=sentiment_score,
            technical_indicators=technical_indicators,
            forecast_data=forecast_data,
            top_headlines=top_headlines,
            risk_tolerance=risk_tolerance,
            horizon_days=horizon_days
        )
        
        try:
            logger.info(f"Generating insights for {coin_symbol}...")
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                top_p=0.9,
                top_k=40
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            insight_text = response.text.strip()
            
            # Extract recommendation
            recommendation = self._extract_recommendation(insight_text)
            
            # Calculate confidence score
            score = self._calculate_score(
                insight_text=insight_text,
                sentiment_score=sentiment_score,
                technical_indicators=technical_indicators
            )
            
            logger.info(f"Generated insights successfully. Recommendation: {recommendation}")
            
            return {
                "recommendation": recommendation,
                "score": score,
                "insight": insight_text,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            # Return fallback
            return self._generate_fallback_insight(
                sentiment_score=sentiment_score,
                technical_indicators=technical_indicators
            )
            
    def _build_prompt(
        self,
        coin_id: str,
        coin_symbol: str,
        market_data: Dict,
        sentiment_score: float,
        technical_indicators: Dict,
        forecast_data: Optional[Dict],
        top_headlines: Optional[List[str]],
        risk_tolerance: str,
        horizon_days: int
    ) -> str:
        """Build comprehensive analysis prompt"""
        
        # Format market data
        price = market_data.get("price_usd", 0)
        pct_24h = market_data.get("pct_change_24h", 0)
        pct_7d = market_data.get("pct_change_7d", 0)
        market_cap = market_data.get("market_cap", 0)
        volume = market_data.get("volume_24h", 0)
        
        # Format technical indicators
        rsi = technical_indicators.get("rsi", 50)
        rsi_zone = self._get_rsi_zone(rsi)
        
        # Format headlines
        headlines_text = ""
        if top_headlines:
            headlines_text = "\n\nTop recent headlines:\n" + "\n".join(
                [f"- {h}" for h in top_headlines[:5]]
            )
            
        # Format forecast
        forecast_text = ""
        if forecast_data:
            last_pred = forecast_data.get("last_prediction")
            if last_pred:
                change_pct = ((last_pred - price) / price) * 100
                forecast_text = f"\n\n7-day forecast: ${last_pred:,.2f} ({change_pct:+.1f}%)"
                
        prompt = f"""You are an expert cryptocurrency analyst. Analyze the following data for {coin_id.upper()} ({coin_symbol.upper()}) and provide investment insights.

MARKET DATA:
- Current Price: ${price:,.2f}
- Market Cap: ${market_cap:,.0f}
- 24h Volume: ${volume:,.0f}
- 24h Change: {pct_24h:.2f}%
- 7d Change: {pct_7d:.2f}%
- RSI (14): {rsi:.1f} ({rsi_zone})

SENTIMENT ANALYSIS:
- News Sentiment Score: {sentiment_score:.3f} (range: -1 to +1, where +1 is very positive)

ANALYSIS PARAMETERS:
- Risk Tolerance: {risk_tolerance}
- Investment Horizon: {horizon_days} days{headlines_text}{forecast_text}

Please provide:
1. A clear BUY/SELL/HOLD recommendation with reasoning
2. Detailed insights covering:
   - Sentiment analysis interpretation
   - Technical momentum (24h and 7d trends)
   - RSI analysis and what it suggests
   - Risk factors to consider
   - Key catalysts to watch

Format your response as a structured analysis. Be specific about price levels, timeframes, and actionable advice. Consider the user's risk tolerance and investment horizon.

Keep the tone professional but accessible. Include appropriate disclaimers that this is educational content, not financial advice."""

        return prompt
        
    def _extract_recommendation(self, insight_text: str) -> str:
        """Extract BUY/SELL/HOLD recommendation from insight text"""
        text_lower = insight_text.lower()
        
        # Strong buy signals
        if any(phrase in text_lower for phrase in [
            "strong buy", "buy recommendation", "recommend buying"
        ]):
            return "BUY"
            
        # Buy signals
        if any(phrase in text_lower for phrase in [
            "buy", "accumulate", "long position", "enter position"
        ]):
            if "avoid" not in text_lower and "don't" not in text_lower:
                return "BUY"
                
        # Sell signals
        if any(phrase in text_lower for phrase in [
            "sell", "short", "avoid", "exit", "close position"
        ]):
            return "SELL / AVOID"
            
        # Hold signals
        if any(phrase in text_lower for phrase in [
            "hold", "wait", "neutral", "sideways", "consolidat"
        ]):
            return "HOLD / WAIT"
            
        # Default to hold
        return "HOLD / WAIT"
        
    def _calculate_score(
        self,
        insight_text: str,
        sentiment_score: float,
        technical_indicators: Dict
    ) -> float:
        """Calculate confidence score for recommendation"""
        
        text_lower = insight_text.lower()
        
        # Start with sentiment
        score = 0.4 * sentiment_score
        
        # Add technical momentum
        pct_24h = technical_indicators.get("pct_24h", 0)
        pct_7d = technical_indicators.get("pct_7d", 0)
        
        if pct_24h:
            momentum_24 = max(-1.0, min(1.0, pct_24h / 15.0))
            score += 0.2 * momentum_24
            
        if pct_7d:
            momentum_7 = max(-1.0, min(1.0, pct_7d / 40.0))
            score += 0.2 * momentum_7
            
        # Adjust based on RSI
        rsi = technical_indicators.get("rsi", 50)
        if rsi >= 70:
            score -= 0.15  # Overbought
        elif rsi <= 30:
            score += 0.15  # Oversold
            
        # Adjust based on LLM sentiment
        positive_words = [
            "bullish", "positive", "strong", "buy", "upward", 
            "growth", "opportunity", "momentum"
        ]
        negative_words = [
            "bearish", "negative", "weak", "sell", "downward", 
            "risk", "caution", "decline"
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            score += 0.1 * min(pos_count - neg_count, 3) / 3
        elif neg_count > pos_count:
            score -= 0.1 * min(neg_count - pos_count, 3) / 3
            
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))
        
    def _get_rsi_zone(self, rsi: float) -> str:
        """Get RSI zone description"""
        if rsi >= 70:
            return "Overbought"
        elif rsi <= 30:
            return "Oversold"
        else:
            return "Neutral"
            
    def _generate_fallback_insight(
        self,
        sentiment_score: float,
        technical_indicators: Dict
    ) -> Dict:
        """Generate rule-based insight when Gemini is unavailable"""
        
        pct_24h = technical_indicators.get("pct_24h", 0)
        pct_7d = technical_indicators.get("pct_7d", 0)
        rsi = technical_indicators.get("rsi", 50)
        
        # Determine recommendation
        if sentiment_score > 0.3 and pct_7d > 5 and rsi < 70:
            recommendation = "BUY"
            score = 0.6
        elif sentiment_score < -0.3 or rsi > 75:
            recommendation = "SELL / AVOID"
            score = -0.5
        else:
            recommendation = "HOLD / WAIT"
            score = 0.0
            
        insight = f"""**Recommendation: {recommendation}**

**Sentiment Analysis**: {'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'} market sentiment (score: {sentiment_score:.2f})

**Technical Momentum**: 
- 24h: {pct_24h:+.2f}%
- 7d: {pct_7d:+.2f}%

**RSI Analysis**: RSI is at {rsi:.1f}, indicating {self._get_rsi_zone(rsi).lower()} conditions.

**Note**: This is a rule-based analysis. AI-powered insights are temporarily unavailable.

**Disclaimer**: This is educational content only, not financial advice."""

        return {
            "recommendation": recommendation,
            "score": score,
            "insight": insight,
            "source": "fallback"
        }


# Convenience function
def generate_insights(api_key: str, **kwargs) -> Dict:
    """
    Generate insights with automatic error handling
    
    Args:
        api_key: Gemini API key
        **kwargs: Arguments for generate_insights method
        
    Returns:
        Insights dictionary
    """
    try:
        generator = GeminiInsightGenerator(api_key=api_key)
        return generator.generate_insights(**kwargs)
    except Exception as e:
        logger.error(f"Failed to generate insights: {e}")
        return {
            "recommendation": "HOLD / WAIT",
            "score": 0.0,
            "insight": "Unable to generate insights. Please try again later.",
            "source": "error"
        }

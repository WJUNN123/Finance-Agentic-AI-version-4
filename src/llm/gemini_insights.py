"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """Generates investment insights using Gemini 2.0 Flash"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 800  # Increased for more detailed reasoning
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
            
    def generate_insights(
        self,
        coin_symbol: str,
        market_data: Dict,
        sentiment_data: Dict, # Changed to accept full sentiment dict
        technical_indicators: Dict,
        prediction_data: Dict, # Changed to accept full prediction details
        top_headlines: List[str],
        horizon_days: int = 7
    ) -> Dict:
        """
        Generate comprehensive investment insights
        """
        # Build the data-rich prompt
        prompt = self._build_prompt(
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_data=sentiment_data,
            tech=technical_indicators,
            preds=prediction_data,
            headlines=top_headlines,
            horizon=horizon_days
        )
        
        try:
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            insight_text = response.text.strip()
            
            # Helper to extract a strict recommendation from the text
            rec = self._extract_recommendation(insight_text)
            
            # We trust the LLM's synthesis for the score now, 
            # rather than a manual calculation
            score = self._extract_confidence_score(insight_text)

            return {
                "recommendation": rec,
                "score": score,
                "insight": insight_text,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._get_fallback_response()

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        """Constructs a prompt that acts as a Data Science synthesiser"""
        
        # 1. Format Market Data
        curr_price = market_data.get('price_usd', 0)
        
        # 2. Format Predictions (Calculate ROI)
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble'):
            ensemble = preds['ensemble']
            final_pred = ensemble[-1]
            pred_roi = ((final_pred - curr_price) / curr_price) * 100
            
            # Analyze the shape of the curve (is it volatile?)
            trend_shape = "linear"
            if len(ensemble) > 2:
                mid_point = ensemble[len(ensemble)//2]
                if mid_point > curr_price and mid_point > final_pred:
                    trend_shape = "hump (rise then fall)"
                elif mid_point < curr_price and mid_point < final_pred:
                    trend_shape = "dip (fall then recover)"
            
            pred_text = (
                f"Hybrid LSTM+XGBoost Model Forecast ({horizon} days):\n"
                f"   - Predicted End Price: ${final_pred:,.2f}\n"
                f"   - Predicted ROI: {pred_roi:+.2f}%\n"
                f"   - Trend Shape: {trend_shape}"
            )

        # 3. Format Sentiment
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0})
        sent_text = (
            f"News Sentiment (RoBERTa Model):\n"
            f"   - Aggregate Score: {sent_score:.2f} (Scale: -1.0 to +1.0)\n"
            f"   - Breakdown: {breakdown.get('positive'):.1f}% Positive, {breakdown.get('negative'):.1f}% Negative"
        )

        # 4. Format Technicals
        rsi = tech.get('rsi', 50)
        bb_pos = "Neutral"
        if tech.get('bb_upper') and curr_price > tech['bb_upper']: bb_pos = "Above Upper Bollinger Band (Overextended)"
        elif tech.get('bb_lower') and curr_price < tech['bb_lower']: bb_pos = "Below Lower Bollinger Band (Oversold)"
        
        tech_text = (
            f"Technical Indicators:\n"
            f"   - RSI (14): {rsi:.1f} ({self._get_rsi_zone(rsi)})\n"
            f"   - Bollinger Bands: {bb_pos}\n"
            f"   - Trend: {tech.get('trend', 'Neutral')}\n"
            f"   - Volatility: {tech.get('volatility', 0):.4f}"
        )

        return f"""
You are a Senior Crypto Investment Analyst. Synthesize the following data sources to provide a final recommendation for {coin_symbol}.

### DATA SOURCES

1. {pred_text}
2. {sent_text}
3. {tech_text}
4. Recent Headlines:
   {chr(10).join(['- ' + h for h in headlines[:3]])}

### INSTRUCTIONS
Your goal is to weigh conflicting signals.
- If the AI Model predicts a price RISE, but Sentiment is NEGATIVE, be cautious.
- If Technicals are OVERBOUGHT (RSI > 70) but Model predicts a RISE, warn of a pullback.
- If all signals align (Model Up + Sentiment Positive + Technicals Bullish), this is a Strong Buy.

### OUTPUT FORMAT
Provide a response in the following format:

**Analysis Synthesis**
[A concise paragraph explaining how the model prediction aligns or conflicts with news sentiment and technicals.]

**Key Risks**
[Bullet points of specific risks based on the data (e.g., "Low liquidity," "RSI divergence," "Negative news flow")]

**Confidence Score**
[A number between 0 and 100 based on data alignment]

**Recommendation**
[BUY / SELL / HOLD]
"""

    def _extract_recommendation(self, text: str) -> str:
        if "Recommendation" in text:
            last_part = text.split("Recommendation")[-1].upper()
            if "BUY" in last_part: return "BUY"
            if "SELL" in last_part: return "SELL / AVOID"
        return "HOLD / WAIT"

    def _extract_confidence_score(self, text: str) -> float:
        import re
        # Look for "Confidence Score" followed by a number
        match = re.search(r"Confidence Score.*?(\d{1,3})", text, re.IGNORECASE | re.DOTALL)
        if match:
            return float(match.group(1)) / 100.0
        return 0.5

    def _get_rsi_zone(self, rsi):
        if rsi >= 70: return "Overbought"
        if rsi <= 30: return "Oversold"
        return "Neutral"

    def _get_fallback_response(self):
        return {
            "recommendation": "HOLD / WAIT",
            "score": 0.0,
            "insight": "AI unavailable. Based on standard rules: Check RSI and Trend manually.",
            "source": "fallback"
        }

# Singleton accessor
def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

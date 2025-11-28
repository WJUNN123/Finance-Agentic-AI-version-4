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
        model_name: str = "gemini-2.0-flash",
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
        llm,
        symbol: str,
        price: float,
        change_24h: float,
        change_7d: float,
        rsi: float,
        liquidity: float,
        volatility: float,
        trend: str,
        sentiment_score: float,
        sentiment_pos: float,
        sentiment_neu: float,
        sentiment_neg: float,
        headlines: list,
        prediction_pct: float,
        predicted_price: float,
        confidence: int
    ):
        """
        Generate insights using LLM with fallback rule-based logic.
        """
    
        # Convert headlines list into a readable string
        headline_text = "\n".join([f"- {h}" for h in headlines])
    
        llm_prompt = f"""
    You are a senior financial analyst specializing in cryptocurrency markets.
    You MUST return a JSON object following the exact required format.
    
    ### INPUT DATA
    Symbol: {symbol}
    Current Price: {price}
    24h Change: {change_24h}%
    7d Change: {change_7d}%
    RSI(14): {rsi}
    Liquidity Score: {liquidity}
    Volatility Score: {volatility}
    Trend Direction: {trend}
    AI Confidence Score: {confidence}
    
    ### SENTIMENT DATA
    Overall Sentiment Score: {sentiment_score}
    Positive News: {sentiment_pos}%
    Neutral News: {sentiment_neu}%
    Negative News: {sentiment_neg}%
    
    ### HEADLINES
    {headline_text}
    
    ### PRICE FORECAST
    7-Day Forecast % Change: {prediction_pct}%
    Expected Price in 7 Days: {predicted_price}
    
    ### TASK
    Based on ALL data above:
    1. Give BUY / SELL / HOLD recommendation
    2. Justify using sentiment + forecast + RSI + trend
    3. Provide a short insight (3–5 sentences)
    4. List major risks
    Return ONLY valid JSON.
    
    ### OUTPUT FORMAT (STRICT)
    {{
      "recommendation": "",
      "insight": "",
      "reasoning": "",
      "risk_factors": ""
    }}
    """
    
        try:
            response = llm.generate(llm_prompt)
    
            # Try to load JSON result
            import json
            result = json.loads(response)
    
            return {
                "source": "llm",
                "recommendation": result.get("recommendation", "HOLD"),
                "insight": result.get("insight", ""),
                "reasoning": result.get("reasoning", ""),
                "risk_factors": result.get("risk_factors", "")
            }
    
        except Exception as e:
            print("⚠️ LLM failed, using rule-based fallback:", e)
            return rule_based_fallback(
                symbol=symbol,
                change_24h=change_24h,
                change_7d=change_7d,
                rsi=rsi,
                sentiment_score=sentiment_score,
                prediction_pct=prediction_pct
            )

    def rule_based_fallback(symbol, change_24h, change_7d, rsi, sentiment_score, prediction_pct):
        # Trend logic
        trend = "uptrend" if change_7d > 0 else "downtrend"
    
        # Basic rules
        if prediction_pct > 5 and sentiment_score > 0:
            rec = "BUY"
        elif prediction_pct < -5 or sentiment_score < -0.1:
            rec = "SELL / AVOID"
        else:
            rec = "HOLD / WAIT"
    
        insight = f"{symbol} has a {trend} with mixed signals. Forecast suggests {prediction_pct:.2f}% change, and sentiment score is {sentiment_score}."
    
        return {
            "source": "rules",
            "recommendation": rec,
            "insight": insight,
            "reasoning": "Automatic rule-based model used due to LLM failure.",
            "risk_factors": "High volatility, sentiment uncertainty."
        }
        
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
        
    def extract_recommendation(text: str) -> str:
        """
        Extract only the explicit recommendation line coming from Gemini.
        Ignores all other occurrences of the words buy/sell/avoid/etc.
        """
    
        import re
        lines = text.splitlines()
    
        # Look for exact recommendation statement
        for line in lines:
            m = re.search(r"recommendation[:\-]\s*(.*)", line, re.I)
            if m:
                rec = m.group(1).strip().lower()
                # Normalize
                if rec.startswith("buy"):
                    return "BUY"
                if rec.startswith("sell"):
                    return "SELL"
                if rec.startswith("avoid"):
                    return "AVOID"
                if rec.startswith("hold"):
                    return "HOLD"
                return rec.upper()
    
        # If not found, safe fallback
        return "HOLD"
        
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

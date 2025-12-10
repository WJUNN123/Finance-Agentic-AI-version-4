"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re
import json

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """Generates investment insights using Gemini 2.0 Flash"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 1000
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
        sentiment_data: Dict,
        technical_indicators: Dict,
        prediction_data: Dict,
        top_headlines: List[str],
        horizon_days: int = 7
    ) -> Dict:
        """
        Generate comprehensive investment insights with consistent recommendations
        """
        # 1. Build structured prompt
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
            response_text = response.text.strip()
            
            # Parse structured JSON response
            parsed = self._parse_json_response(response_text)
            
            if parsed:
                return {
                    "recommendation": parsed.get("recommendation", "HOLD / WAIT"),
                    "score": parsed.get("confidence_score", 0.5),
                    "insight": parsed.get("analysis", ""),
                    "risks": parsed.get("risks", []),
                    "source": "gemini",
                    "model": self.model_name
                }
            else:
                # Fallback parsing if JSON fails
                logger.warning("Failed to parse JSON response, using fallback extraction")
                return self._fallback_parse(response_text)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._get_fallback_response()

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        """Constructs a structured prompt for consistent JSON output"""
        
        curr_price = market_data.get('price_usd', 0)
        
        # Process predictions
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble'):
            ensemble = preds['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                pred_roi = ((final_pred - curr_price) / curr_price) * 100
                
                # Analyze trajectory
                trend_shape = "neutral"
                if len(ensemble) > 2:
                    mid_point = ensemble[len(ensemble)//2]
                    if mid_point > curr_price and mid_point > final_pred:
                        trend_shape = "volatile (spike then decline)"
                    elif mid_point < curr_price and mid_point < final_pred:
                        trend_shape = "recovery (dip then rise)"
                    elif final_pred > curr_price:
                        trend_shape = "sustained upward"
                    else:
                        trend_shape = "sustained downward"
                
                pred_text = (
                    f"Predicted End Price: ${final_pred:,.2f} | "
                    f"Projected ROI: {pred_roi:+.2f}% | "
                    f"Trajectory: {trend_shape}"
                )

        # Process sentiment
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0, 'neutral': 0})
        sent_interpretation = self._interpret_sentiment_score(sent_score)
        sent_text = (
            f"Aggregate Score: {sent_score:.2f}/1.0 ({sent_interpretation}) | "
            f"Breakdown: {breakdown.get('positive', 0):.0f}% Positive, "
            f"{breakdown.get('neutral', 0):.0f}% Neutral, "
            f"{breakdown.get('negative', 0):.0f}% Negative"
        )

        # Process technicals
        rsi = tech.get('rsi', 50)
        volatility = tech.get('volatility', 0)
        trend = tech.get('trend', 'sideways')
        
        rsi_zone = self._get_rsi_zone(rsi)
        
        bb_status = "Within Bands"
        if tech.get('bb_upper') and curr_price > tech['bb_upper']: 
            bb_status = "Above Upper Band (Overbought)"
        elif tech.get('bb_lower') and curr_price < tech['bb_lower']: 
            bb_status = "Below Lower Band (Oversold)"
        
        tech_text = (
            f"RSI: {rsi:.1f} ({rsi_zone}) | "
            f"Bollinger Bands: {bb_status} | "
            f"Trend: {trend} | "
            f"Volatility: {volatility:.4f}"
        )

        # Headlines summary
        headlines_text = "\n  ".join(headlines[:3]) if headlines else "No recent headlines"

        return f"""You are a Professional Crypto Investment Analyst. Analyze the following data for {coin_symbol} and provide a clear, consistent recommendation.

=== MARKET DATA ===
Current Price: ${curr_price:,.2f}
Forecast ({horizon} days): {pred_text}

=== SENTIMENT ANALYSIS ===
News Sentiment: {sent_text}

=== TECHNICAL INDICATORS ===
{tech_text}

=== RECENT HEADLINES ===
  {headlines_text}

=== YOUR TASK ===
Provide a JSON response with EXACTLY this structure (no markdown, pure JSON):
{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "2-3 sentence professional summary explaining your recommendation",
  "risks": ["risk 1", "risk 2", "risk 3"],
  "reasoning": "Brief explanation of how you reconciled conflicting signals"
}}

=== DECISION LOGIC ===
1. Strong indicators (RSI, Trend, Price Prediction, Sentiment) must ALIGN for BUY or SELL
2. If signals conflict, lean toward HOLD with lower confidence
3. RSI < 30 = Oversold (watch for bounce, cautious on SELL)
4. RSI > 70 = Overbought (watch for pullback, cautious on BUY)
5. Price prediction > current by 5%+ AND positive sentiment = BUY candidate
6. Price prediction < current by 5%+ AND negative sentiment = SELL candidate
7. Otherwise = HOLD

Respond with ONLY valid JSON, no other text."""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Extract and parse JSON from response"""
        try:
            # Try direct JSON parsing
            parsed = json.loads(response_text)
            return self._validate_parsed_response(parsed)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^{}]*"recommendation"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return self._validate_parsed_response(parsed)
                except:
                    pass
        
        return None

    def _validate_parsed_response(self, parsed: Dict) -> Dict:
        """Validate and normalize parsed response"""
        # Normalize recommendation
        rec = parsed.get("recommendation", "HOLD").upper().strip()
        if "BUY" in rec:
            rec = "BUY"
        elif "SELL" in rec:
            rec = "SELL"
        else:
            rec = "HOLD"
        
        # Normalize confidence score
        score = parsed.get("confidence_score", 0.5)
        if isinstance(score, str):
            score = float(score.replace('%', '')) / 100.0
        score = max(0.0, min(1.0, float(score)))
        
        # Get analysis text
        analysis = parsed.get("analysis", "")
        if not analysis:
            analysis = parsed.get("reasoning", "")
        
        # Get risks
        risks = parsed.get("risks", [])
        if not isinstance(risks, list):
            risks = []
        
        return {
            "recommendation": rec,
            "confidence_score": score,
            "analysis": analysis,
            "risks": risks
        }

    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing if JSON extraction fails"""
        logger.warning("Using fallback text parsing")
        
        # Extract recommendation
        rec = "HOLD"
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            rec = "BUY"
        elif re.search(r'\bSELL\b', text, re.IGNORECASE):
            rec = "SELL"
        
        # Extract confidence score
        score = 0.5
        score_match = re.search(r'(?:confidence|score)[:\s]+(\d+)', text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1)) / 100.0
        
        return {
            "recommendation": rec,
            "score": score,
            "insight": text[:500],  # First 500 chars
            "risks": [],
            "source": "gemini",
            "model": self.model_name
        }

    def _interpret_sentiment_score(self, score: float) -> str:
        """Convert sentiment score to text"""
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

    def _get_rsi_zone(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi >= 70:
            return "Overbought"
        elif rsi <= 30:
            return "Oversold"
        else:
            return "Neutral"

    def _get_fallback_response(self) -> Dict:
        """Return fallback response on error"""
        return {
            "recommendation": "HOLD",
            "score": 0.5,
            "insight": "Unable to generate AI insights. Please check your API key and try again.",
            "risks": ["API unavailable"],
            "source": "fallback",
            "model": self.model_name
        }


def generate_insights(api_key: str, **kwargs) -> Dict:
    """Convenient function to generate insights"""
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

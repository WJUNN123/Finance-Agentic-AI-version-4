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
        max_tokens: int = 500
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
        Generate investment insights with minimal token usage.
        Optimized for free tier quotas.
        """
        # Build MINIMAL prompt (reduced from 800 to 200 tokens)
        prompt = self._build_minimal_prompt(
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_data=sentiment_data,
            technical=technical_indicators,
            predictions=prediction_data,
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
                    "source": "gemini",
                    "model": self.model_name
                }
            else:
                logger.warning("Failed to parse JSON response, using fallback extraction")
                return self._fallback_parse(response_text)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._get_fallback_response()

    def _build_minimal_prompt(self, coin_symbol, market_data, sentiment_data, technical, predictions, horizon):
        """
        Build MINIMAL prompt to reduce token usage.
        
        Original: 800+ tokens
        Optimized: ~200 tokens
        """
        
        curr_price = market_data.get('price_usd', 0)
        
        # Minimal prediction text
        pred_text = "Unknown"
        if predictions and predictions.get('ensemble') and len(predictions['ensemble']) > 0:
            final_pred = predictions['ensemble'][-1]
            pred_roi = ((final_pred - curr_price) / curr_price) * 100
            pred_text = f"${final_pred:,.0f} ({pred_roi:+.1f}%)"
        
        # Minimal sentiment
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {})
        pos = breakdown.get('positive', 0)
        
        # Minimal technical
        rsi = technical.get('rsi', 50)
        trend = technical.get('trend', 'sideways')
        vol = technical.get('volatility', 0)
        
        # COMPACT PROMPT (minimal tokens)
        return f"""Analyze {coin_symbol}. Give ONE JSON line.

Price: ${curr_price:,.2f}, Prediction: {pred_text}
RSI: {rsi:.0f}, Trend: {trend}, Vol: {vol:.3f}
Sentiment: {sent_score:.2f} ({pos:.0f}% positive)

Return ONLY:
{{"recommendation":"BUY"|"SELL"|"HOLD","confidence_score":0.0-1.0,"analysis":"1 sentence"}}"""

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
        rec = parsed.get("recommendation", "HOLD").upper().strip()
        if "BUY" in rec:
            rec = "BUY"
        elif "SELL" in rec:
            rec = "SELL"
        else:
            rec = "HOLD"
        
        score = parsed.get("confidence_score", 0.5)
        if isinstance(score, str):
            score = float(score.replace('%', '')) / 100.0
        score = max(0.0, min(1.0, float(score)))
        
        analysis = parsed.get("analysis", "")
        if not analysis:
            analysis = parsed.get("reasoning", "")
        
        return {
            "recommendation": rec,
            "confidence_score": score,
            "analysis": analysis
        }

    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing if JSON extraction fails"""
        logger.warning("Using fallback text parsing")
        
        rec = "HOLD"
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            rec = "BUY"
        elif re.search(r'\bSELL\b', text, re.IGNORECASE):
            rec = "SELL"
        
        score = 0.5
        score_match = re.search(r'(?:confidence|score)[:\s]+(\d+)', text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1)) / 100.0
        
        return {
            "recommendation": rec,
            "score": score,
            "insight": text[:300],
            "source": "gemini",
            "model": self.model_name
        }

    def _get_fallback_response(self) -> Dict:
        """Return fallback response on error"""
        return {
            "recommendation": "HOLD",
            "score": 0.5,
            "insight": "Unable to generate insights. Please try again.",
            "source": "fallback",
            "model": self.model_name
        }


def generate_insights(api_key: str, **kwargs) -> Dict:
    """Convenient function to generate insights"""
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re
import json
import time

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """Generates investment insights using Gemini 1.5 Flash"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",  # <--- CHANGED: Switched to 1.5 Flash
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
        Generate comprehensive investment insights with consistent recommendations
        Includes Retry Logic for 429 Rate Limits
        """
        # 1. Build optimized prompt
        prompt = self._build_prompt(
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_data=sentiment_data,
            tech=technical_indicators,
            preds=prediction_data,
            headlines=top_headlines,
            horizon=horizon_days
        )
        
        # 2. Retry Logic
        max_retries = 3
        base_wait = 60

        for attempt in range(max_retries):
            try:
                config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
                
                logger.info(f"Sending request to {self.model_name} (Attempt {attempt + 1}/{max_retries})...")
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
                    logger.warning("Failed to parse JSON response, using fallback extraction")
                    return self._fallback_parse(response_text)
            
            except Exception as e:
                error_str = str(e)
                # Check for Rate Limit (429) or Quota issues
                if ("429" in error_str or "quota" in error_str.lower()) and attempt < max_retries - 1:
                    logger.warning(f"⚠️ Quota exceeded. Retrying in {base_wait} seconds...")
                    time.sleep(base_wait)
                    continue
                
                logger.error(f"Error generating insights: {e}")
                # If 1.5 Flash also fails, return fallback immediately
                return self._get_fallback_response()
        
        return self._get_fallback_response()

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        """
        Constructs a HIGHLY OPTIMIZED prompt to minimize token usage.
        """
        curr_price = market_data.get('price_usd', 0)
        
        # Simplified Predictions
        pred_text = "None"
        if preds and preds.get('ensemble') and len(preds['ensemble']) > 0:
            final = preds['ensemble'][-1]
            roi = ((final - curr_price) / curr_price) * 100
            pred_text = f"${final:,.2f} ({roi:+.1f}%)"

        # Simplified Sentiment
        sent_score = sentiment_data.get('score', 0)
        
        # Simplified Technicals
        rsi = tech.get('rsi', 50)
        trend = tech.get('trend', 'sideways')

        # Limit headlines to just 2 to save tokens
        headlines_text = " | ".join(headlines[:2]) if headlines else "None"

        return f"""Act as Crypto Analyst.
Coin: {coin_symbol}
Price: ${curr_price:,.2f}
Forecast({horizon}d): {pred_text}
Sentiment: {sent_score:.2f} (-1 to 1)
RSI: {rsi:.1f}
Trend: {trend}
News: {headlines_text}

Task: Return valid JSON.
{{
"recommendation": "BUY"|"SELL"|"HOLD",
"confidence_score": 0.0-1.0,
"analysis": "Short 2 sentence summary.",
"risks": ["Risk 1", "Risk 2"]
}}"""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Extract and parse JSON from response"""
        try:
            # Clean markdown code blocks if present
            text = response_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return self._validate_parsed_response(parsed)
        except json.JSONDecodeError:
            # Regex fallback
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return self._validate_parsed_response(parsed)
                except:
                    pass
        return None

    def _validate_parsed_response(self, parsed: Dict) -> Dict:
        """Validate and normalize parsed response"""
        rec = parsed.get("recommendation", "HOLD").upper()
        if "BUY" in rec: rec = "BUY"
        elif "SELL" in rec: rec = "SELL"
        else: rec = "HOLD"
        
        score = parsed.get("confidence_score", 0.5)
        if isinstance(score, str):
            score = float(score.replace('%', '')) / 100
            
        return {
            "recommendation": rec,
            "confidence_score": float(score),
            "analysis": parsed.get("analysis", parsed.get("reasoning", "")),
            "risks": parsed.get("risks", [])
        }

    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing if JSON extraction fails"""
        rec = "HOLD"
        if "BUY" in text.upper(): rec = "BUY"
        elif "SELL" in text.upper(): rec = "SELL"
        
        return {
            "recommendation": rec,
            "score": 0.5,
            "insight": text[:300],
            "risks": [],
            "source": "gemini",
            "model": self.model_name
        }

    def _get_fallback_response(self) -> Dict:
        return {
            "recommendation": "HOLD",
            "score": 0.5,
            "insight": "AI Insights unavailable (Rate Limit/Error).",
            "risks": ["API Limit Reached"],
            "source": "fallback",
            "model": self.model_name
        }

def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

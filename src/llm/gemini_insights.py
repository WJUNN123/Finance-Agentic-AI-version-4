"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

from huggingface_hub import InferenceClient
from typing import Dict, List, Optional
import logging
import re
import json
import time
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# RETRY DECORATOR FOR API CALLS
# ============================================================================

def retry_with_backoff(max_retries=3, base_delay=2, max_delay=60):
    """
    Retry decorator with exponential backoff for API rate limits
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"❌ Max retries ({max_retries}) reached for {func.__name__}")
                            raise Exception(
                                f"API rate limit exceeded after {max_retries} retries. "
                                f"Please wait a few minutes or check your Hugging Face token."
                            )
                        
                        wait_time = min(delay * (2 ** (retries - 1)), max_delay)
                        logger.warning(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {retries}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Error in {func.__name__}: {error_str}")
                        raise
            
            raise Exception(f"Failed after {max_retries} retries")
        
        return wrapper
    return decorator


# ============================================================================
# LLAMA INSIGHT GENERATOR
# ============================================================================

class LlamaInsightGenerator:
    """Generates investment insights using Llama 3.1 via Hugging Face"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        try:
            self.client = InferenceClient(token=api_key)
            logger.info(f"✅ Llama model initialized: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Llama: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=5, max_delay=60)
    def _call_llama_api(self, prompt: str) -> str:
        """
        Call Llama API via Hugging Face with retry logic
        """
        logger.info(f"🤖 Calling Llama API...")
        
        try:
            # Use chat completion for Llama 3.1
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if not response or not response.choices:
                raise Exception("Empty response from Llama API")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ Llama API error: {error_str}")
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
        """Generate comprehensive investment insights with retry logic"""
        
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
            response_text = self._call_llama_api(prompt)
            parsed = self._parse_json_response(response_text)
            
            if parsed:
                logger.info(f"✅ Generated recommendation: {parsed.get('recommendation')} (confidence: {parsed.get('confidence_score', 0):.2f})")
                return {
                    "recommendation": parsed.get("recommendation", "HOLD"),
                    "score": parsed.get("confidence_score", 0.5),
                    "insight": parsed.get("analysis", ""),
                    "risks": parsed.get("risks", []),
                    "reasoning": parsed.get("reasoning", ""),
                    "key_factors": parsed.get("key_factors", []),
                    "source": "llama",
                    "model": self.model_name
                }
            else:
                logger.warning("⚠️ Failed to parse JSON response, using fallback extraction")
                return self._fallback_parse(response_text)
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error generating insights: {error_msg}")
            
            if "quota" in error_msg.lower() or "429" in error_msg or "rate limit" in error_msg.lower():
                raise Exception(
                    "⏱️ API rate limit exceeded. Please wait a few minutes and try again."
                )
            else:
                raise Exception(f"Failed to generate insights: {error_msg}")
    
    def _build_prompt(
        self,
        coin_symbol: str,
        market_data: Dict,
        sentiment_data: Dict,
        tech: Dict,
        preds: Dict,
        headlines: List[str],
        horizon: int
    ) -> str:
        """Build comprehensive prompt"""
        
        current_price = market_data.get('price_usd', 0)
        price_change_24h = market_data.get('pct_change_24h', 0)
        price_change_7d = market_data.get('pct_change_7d', 0)
        market_cap = market_data.get('market_cap', 0)
        volume_24h = market_data.get('volume_24h', 0)
        liquidity_ratio = (volume_24h / market_cap * 100) if market_cap > 0 else 0
        
        lstm_preds = preds.get('lstm', [])
        xgb_preds = preds.get('xgboost', [])
        ensemble_preds = preds.get('ensemble', [])
        
        pred_confidence = 0.5
        if lstm_preds and xgb_preds and ensemble_preds:
            lstm_final = lstm_preds[-1]
            xgb_final = xgb_preds[-1]
            ensemble_final = ensemble_preds[-1]
            
            if ensemble_final > 0:
                lstm_diff = abs((lstm_final - ensemble_final) / ensemble_final)
                xgb_diff = abs((xgb_final - ensemble_final) / ensemble_final)
                avg_diff = (lstm_diff + xgb_diff) / 2
                pred_confidence = max(0.3, min(1.0, 1.0 - avg_diff * 2))
        
        expected_roi = 0
        if ensemble_preds and current_price > 0:
            expected_roi = ((ensemble_preds[-1] - current_price) / current_price) * 100
        
        sentiment_score = sentiment_data.get('score', 0.0)
        sentiment_breakdown = sentiment_data.get('breakdown', {})
        sentiment_confidence = sentiment_data.get('confidence', 0.5)
        pos_pct = sentiment_breakdown.get('positive', 0)
        neg_pct = sentiment_breakdown.get('negative', 0)
        neu_pct = sentiment_breakdown.get('neutral', 0)
        
        headlines_text = "\n".join([f"- {h}" for h in headlines[:5]]) if headlines else "No recent headlines"
        
        rsi = tech.get('rsi', 50)
        trend = tech.get('trend', 'sideways')
        volatility = tech.get('volatility', 0.05)
        momentum = tech.get('momentum', 0)
        
        prompt = f"""You are an expert cryptocurrency analyst. Analyze {coin_symbol} and provide investment recommendation.

MARKET DATA:
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%
- Market Cap: ${market_cap/1e9:.2f}B
- Volume: ${volume_24h/1e9:.2f}B
- Liquidity: {liquidity_ratio:.2f}%

PREDICTIONS ({horizon}-day):
- LSTM: ${lstm_preds[-1]:,.2f} if lstm_preds else 'N/A'
- XGBoost: ${xgb_preds[-1]:,.2f} if xgb_preds else 'N/A'
- Ensemble: ${ensemble_preds[-1]:,.2f} if ensemble_preds else 'N/A'
- Expected ROI: {expected_roi:+.2f}%
- Confidence: {pred_confidence:.0%}

TECHNICAL:
- RSI: {rsi:.1f}
- Trend: {trend}
- Volatility: {volatility:.2%}
- Momentum: {momentum:+.2f}%

SENTIMENT:
- Score: {sentiment_score:+.2f}
- {pos_pct:.1f}% Pos, {neu_pct:.1f}% Neu, {neg_pct:.1f}% Neg
- Confidence: {sentiment_confidence:.0%}

HEADLINES:
{headlines_text}

Respond with VALID JSON ONLY:
{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "2-3 sentence summary",
  "risks": ["risk1", "risk2", "risk3"],
  "reasoning": "brief logic",
  "key_factors": ["factor1", "factor2", "factor3"]
}}

RULES:
- BUY: ROI > 5%, confidence > 60%, RSI < 70
- SELL: ROI < -3% or RSI > 75
- HOLD: ROI -3% to 5% or confidence < 50%

Return ONLY JSON (no markdown).
"""
        
        return prompt
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response"""
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            
            if "recommendation" in parsed and "confidence_score" in parsed:
                rec = parsed["recommendation"].upper()
                if rec not in ["BUY", "SELL", "HOLD"]:
                    rec = "HOLD"
                parsed["recommendation"] = rec
                
                conf = float(parsed.get("confidence_score", 0.5))
                parsed["confidence_score"] = max(0.0, min(1.0, conf))
                
                return parsed
            else:
                logger.warning("⚠️ JSON missing required fields")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing JSON: {e}")
            return None
    
    def _fallback_parse(self, response_text: str) -> Dict:
        """Fallback parser if JSON extraction fails"""
        logger.info("📝 Using fallback text parser")
        
        rec = "HOLD"
        if re.search(r'\bBUY\b', response_text, re.IGNORECASE):
            rec = "BUY"
        elif re.search(r'\bSELL\b', response_text, re.IGNORECASE):
            rec = "SELL"
        
        conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)[\%]?', response_text, re.IGNORECASE)
        confidence = float(conf_match.group(1)) / 100 if conf_match else 0.5
        if confidence > 1.0:
            confidence = confidence / 100
        
        return {
            "recommendation": rec,
            "score": confidence,
            "insight": response_text[:500],
            "risks": ["Unable to extract structured risks"],
            "reasoning": "Fallback parsing used",
            "key_factors": [],
            "source": "llama",
            "model": self.model_name
        }


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_insights(
    api_key: str,
    coin_symbol: str,
    market_data: Dict,
    sentiment_data: Dict,
    technical_indicators: Dict,
    prediction_data: Dict,
    top_headlines: List[str],
    horizon_days: int = 7
) -> Dict:
    """
    Generate investment insights using Llama 3.1
    
    Raises exceptions if API fails (no fallback).
    """
    try:
        generator = LlamaInsightGenerator(api_key=api_key)
        return generator.generate_insights(
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_data=sentiment_data,
            technical_indicators=technical_indicators,
            prediction_data=prediction_data,
            top_headlines=top_headlines,
            horizon_days=horizon_days
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate insights: {e}")
        raise
















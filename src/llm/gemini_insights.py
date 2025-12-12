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
import os

logger = logging.getLogger(__name__)

# ============================================================================
# RETRY DECORATOR FOR API CALLS
# ============================================================================

def retry_with_backoff(max_retries=3, base_delay=2, max_delay=60):
    """
    Retry decorator with exponential backoff for API rate limits
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
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
                    
                    # Check if it's a rate limit error (429)
                    if "429" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower():
                        retries += 1
                        if retries >= max_retries:
                            logger.error(f"❌ Max retries ({max_retries}) reached for {func.__name__}")
                            raise Exception(
                                f"DeepSeek API rate limit exceeded after {max_retries} retries. "
                                f"Please wait a few minutes or check your Hugging Face token."
                            )
                        
                        wait_time = min(delay * (2 ** (retries - 1)), max_delay)
                        logger.warning(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {retries}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    
                    # For non-rate-limit errors, raise immediately
                    else:
                        logger.error(f"❌ Error in {func.__name__}: {error_str}")
                        raise
            
            # Should not reach here
            raise Exception(f"Failed after {max_retries} retries")
        
        return wrapper
    return decorator


# ============================================================================
# DEEPSEEK INSIGHT GENERATOR
# ============================================================================

class DeepSeekInsightGenerator:
    """Generates investment insights using DeepSeek via Hugging Face"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        try:
            # Initialize Hugging Face Inference Client
            self.client = InferenceClient(token=api_key)
            logger.info(f"✅ DeepSeek model initialized: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DeepSeek: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=5, max_delay=60)
    def _call_deepseek_api(self, prompt: str) -> str:
        """
        Call DeepSeek API via Hugging Face with retry logic
        
        Args:
            prompt: The prompt to send to DeepSeek
            
        Returns:
            Response text from DeepSeek
            
        Raises:
            Exception: If API call fails after all retries
        """
        logger.info(f"🤖 Calling DeepSeek API...")
        
        try:
            # Call DeepSeek via Hugging Face Inference API
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                return_full_text=False
            )
            
            if not response:
                raise Exception("Empty response from DeepSeek API")
            
            return response.strip()
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"❌ DeepSeek API error: {error_str}")
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
        Generate comprehensive investment insights with retry logic
        
        Args:
            coin_symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            market_data: Market data dictionary
            sentiment_data: Sentiment analysis results
            technical_indicators: Technical indicators
            prediction_data: Price predictions
            top_headlines: Recent news headlines
            horizon_days: Forecast horizon
            
        Returns:
            Dictionary with recommendation, score, insight, risks, etc.
            
        Raises:
            Exception: If DeepSeek API fails after retries (no fallback)
        """
        # Build structured prompt
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
            # Call DeepSeek with retry logic
            response_text = self._call_deepseek_api(prompt)
            
            # Parse structured JSON response
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
                    "source": "deepseek",
                    "model": self.model_name
                }
            else:
                # Fallback parsing if JSON fails
                logger.warning("⚠️ Failed to parse JSON response, using fallback extraction")
                return self._fallback_parse(response_text)
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error generating insights: {error_msg}")
            
            # Check if it's a quota error
            if "quota" in error_msg.lower() or "429" in error_msg or "rate limit" in error_msg.lower():
                raise Exception(
                    "⏱️ DeepSeek API rate limit exceeded. Please wait a few minutes and try again. "
                    "Check your Hugging Face token at: https://huggingface.co/settings/tokens"
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
        """Build comprehensive prompt with all context"""
        
        # Extract key data
        current_price = market_data.get('price_usd', 0)
        price_change_24h = market_data.get('pct_change_24h', 0)
        price_change_7d = market_data.get('pct_change_7d', 0)
        market_cap = market_data.get('market_cap', 0)
        volume_24h = market_data.get('volume_24h', 0)
        
        # Calculate liquidity ratio
        liquidity_ratio = (volume_24h / market_cap * 100) if market_cap > 0 else 0
        
        # Get predictions
        lstm_preds = preds.get('lstm', [])
        xgb_preds = preds.get('xgboost', [])
        ensemble_preds = preds.get('ensemble', [])
        
        # Calculate prediction confidence
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
        
        # Calculate expected return
        expected_roi = 0
        if ensemble_preds and current_price > 0:
            expected_roi = ((ensemble_preds[-1] - current_price) / current_price) * 100
        
        # Get sentiment data
        sentiment_score = sentiment_data.get('score', 0.0)
        sentiment_breakdown = sentiment_data.get('breakdown', {})
        sentiment_confidence = sentiment_data.get('confidence', 0.5)
        pos_pct = sentiment_breakdown.get('positive', 0)
        neg_pct = sentiment_breakdown.get('negative', 0)
        neu_pct = sentiment_breakdown.get('neutral', 0)
        
        # Format headlines
        headlines_text = "\n".join([f"- {h}" for h in headlines[:5]]) if headlines else "No recent headlines"
        
        # Technical indicators
        rsi = tech.get('rsi', 50)
        trend = tech.get('trend', 'sideways')
        volatility = tech.get('volatility', 0.05)
        momentum = tech.get('momentum', 0)
        macd_histogram = tech.get('macd_histogram', 0)
        stoch_k = tech.get('stochastic_k', 50)
        bb_position = tech.get('bb_position', 0)
        
        # Build the comprehensive prompt
        prompt = f"""You are an expert cryptocurrency analyst. Analyze {coin_symbol} and provide a clear investment recommendation.

**MARKET CONTEXT:**
- Current Price: ${current_price:,.2f}
- 24h Change: {price_change_24h:+.2f}%
- 7d Change: {price_change_7d:+.2f}%
- Market Cap: ${market_cap/1e9:.2f}B
- 24h Volume: ${volume_24h/1e9:.2f}B
- Liquidity Ratio: {liquidity_ratio:.2f}%

**PRICE PREDICTIONS ({horizon}-day forecast):**
- LSTM: ${lstm_preds[-1]:,.2f} if lstm_preds else 'N/A'
- XGBoost: ${xgb_preds[-1]:,.2f} if xgb_preds else 'N/A'
- Ensemble: ${ensemble_preds[-1]:,.2f} if ensemble_preds else 'N/A'
- Expected ROI: {expected_roi:+.2f}%
- Confidence: {pred_confidence:.0%}

**TECHNICAL INDICATORS:**
- RSI: {rsi:.1f}
- Trend: {trend}
- Volatility: {volatility:.2%}
- Momentum: {momentum:+.2f}%
- MACD Histogram: {macd_histogram:.4f}
- Stochastic K: {stoch_k:.1f}
- BB Position: {bb_position:.2f}

**SENTIMENT:**
- Score: {sentiment_score:+.2f}
- Distribution: {pos_pct:.1f}% Pos, {neu_pct:.1f}% Neu, {neg_pct:.1f}% Neg
- Confidence: {sentiment_confidence:.0%}

**RECENT HEADLINES:**
{headlines_text}

**RESPOND WITH VALID JSON ONLY:**

{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "2-3 sentence analysis",
  "risks": ["risk 1", "risk 2", "risk 3"],
  "reasoning": "Brief decision logic",
  "key_factors": ["factor 1", "factor 2", "factor 3"]
}}

**DECISION RULES:**
- BUY: ROI > 5%, confidence > 60%, RSI < 70
- SELL: ROI < -3%, or RSI > 75 overbought
- HOLD: ROI -3% to 5%, or low confidence < 50%

Return ONLY valid JSON (no markdown, no explanation).
"""
        
        return prompt
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from DeepSeek"""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate required fields
            if "recommendation" in parsed and "confidence_score" in parsed:
                # Normalize recommendation
                rec = parsed["recommendation"].upper()
                if rec not in ["BUY", "SELL", "HOLD"]:
                    rec = "HOLD"
                parsed["recommendation"] = rec
                
                # Ensure confidence is 0-1
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
        
        # Try to extract recommendation
        rec = "HOLD"
        if re.search(r'\bBUY\b', response_text, re.IGNORECASE):
            rec = "BUY"
        elif re.search(r'\bSELL\b', response_text, re.IGNORECASE):
            rec = "SELL"
        
        # Extract confidence if present
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
            "source": "deepseek",
            "model": self.model_name
        }


# ============================================================================
# PUBLIC API (Keep same function name for compatibility)
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
    Generate investment insights using DeepSeek
    
    This is the main entry point for generating insights.
    Raises exceptions if DeepSeek API fails (no fallback).
    
    Args:
        api_key: Hugging Face API token (used for DeepSeek)
        coin_symbol: Cryptocurrency symbol
        market_data: Market data dict
        sentiment_data: Sentiment analysis results
        technical_indicators: Technical indicators
        prediction_data: Price predictions
        top_headlines: Recent news headlines
        horizon_days: Forecast horizon
        
    Returns:
        Dict with recommendation, score, insight, risks, etc.
        
    Raises:
        Exception: If DeepSeek API fails after retries
    """
    try:
        generator = DeepSeekInsightGenerator(api_key=api_key)
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
        raise  # Re-raise the exception (no fallback)


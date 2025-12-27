"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import json
import logging
import time
from typing import Dict, List, Optional

# Try new SDK first, fall back to old one
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_SDK = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_CONFIG = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.3,
    "max_output_tokens": 1024,
    "top_p": 0.9,
    "top_k": 40,
}

# Retry configuration for rate limits
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 2,  # seconds
    "max_delay": 10,
}


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an expert cryptocurrency analyst providing actionable investment insights.

CRITICAL RULES:
1. You MUST respond ONLY with valid JSON - no markdown, no explanations outside JSON
2. Base recommendations strictly on the data provided - do not invent numbers
3. Be specific about price levels, timeframes, and risk factors
4. Always include concrete reasoning tied to the metrics shown
5. If data quality is poor (low model agreement, high uncertainty), recommend HOLD

OUTPUT FORMAT (strict JSON):
{
    "recommendation": "BUY" | "SELL" | "HOLD",
    "score": 0.0-1.0,
    "insight": "2-3 sentence analysis with specific price targets and timeframe",
    "reasoning": "1-2 sentences explaining the key factors driving this recommendation",
    "risks": ["risk 1", "risk 2", "risk 3"],
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "entry_price": null or number,
    "target_price": null or number,
    "stop_loss": null or number
}

DECISION FRAMEWORK:
- BUY: Positive forecast (>5%), RSI < 65, model agreement > 70%, bullish trend
- SELL: Negative forecast (<-5%), RSI > 70, bearish trend, high risk signals
- HOLD: Mixed signals, low model agreement (<60%), minimal forecast (<3%), high uncertainty

SAFETY OVERRIDES (always force HOLD):
- Model agreement < 50%
- Extreme volatility (>15%) with low confidence
- Conflicting trend vs forecast signals
- Insufficient data quality"""


# ============================================================================
# GEMINI CLIENT
# ============================================================================

class GeminiInsightGenerator:
    """Generates investment insights using Gemini 2.0 Flash API"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google AI API key
        """
        self.api_key = api_key
        self.model = None
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            if USE_NEW_SDK:
                # New google.genai SDK
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"✅ Gemini client initialized (new SDK): {GEMINI_CONFIG['model']}")
            else:
                # Legacy google.generativeai SDK
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name=GEMINI_CONFIG["model"],
                    generation_config=genai.GenerationConfig(
                        temperature=GEMINI_CONFIG["temperature"],
                        max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
                        top_p=GEMINI_CONFIG["top_p"],
                        top_k=GEMINI_CONFIG["top_k"],
                    ),
                    system_instruction=SYSTEM_PROMPT
                )
                logger.info(f"✅ Gemini client initialized (legacy SDK): {GEMINI_CONFIG['model']}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    def _build_analysis_prompt(
        self,
        coin_symbol: str,
        market_data: Dict,
        sentiment_data: Dict,
        technical_indicators: Dict,
        prediction_data: Dict,
        top_headlines: List[str],
        horizon_days: int
    ) -> str:
        """Build the analysis prompt with all market data"""
        
        # Extract prediction data
        ensemble_preds = prediction_data.get('ensemble', [])
        lstm_preds = prediction_data.get('lstm', [])
        xgb_preds = prediction_data.get('xgboost', [])
        model_agreement = prediction_data.get('model_agreement', 0.5)
        
        # Calculate expected ROI
        current_price = market_data.get('price_usd', 0)
        predicted_price = ensemble_preds[-1] if ensemble_preds else current_price
        expected_roi = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Format predictions for display
        def format_preds(preds, label):
            if not preds:
                return f"{label}: No data"
            return f"{label}: ${preds[0]:,.2f} → ${preds[-1]:,.2f} ({((preds[-1]-preds[0])/preds[0]*100):+.1f}%)"
        
        prompt = f"""Analyze {coin_symbol} and provide investment recommendation.

=== MARKET DATA ===
Current Price: ${current_price:,.2f}
24h Change: {market_data.get('pct_change_24h', 0):+.2f}%
7d Change: {market_data.get('pct_change_7d', 0):+.2f}%
Market Cap: ${market_data.get('market_cap', 0):,.0f}
24h Volume: ${market_data.get('volume_24h', 0):,.0f}

=== {horizon_days}-DAY PRICE FORECAST ===
{format_preds(lstm_preds, 'LSTM Model')}
{format_preds(xgb_preds, 'XGBoost Model')}
{format_preds(ensemble_preds, 'Ensemble (Final)')}
Expected ROI: {expected_roi:+.2f}%
Model Agreement: {model_agreement:.0%}

=== TECHNICAL INDICATORS ===
RSI (14): {technical_indicators.get('rsi', 50):.1f}
Trend: {technical_indicators.get('trend', 'unknown')}
Volatility: {technical_indicators.get('volatility', 0):.2%}
Momentum (14d): {technical_indicators.get('momentum', 0):+.1f}%
MACD Histogram: {technical_indicators.get('macd_histogram', 0):.4f}
Stochastic %K: {technical_indicators.get('stochastic_k', 50):.1f}
Stochastic %D: {technical_indicators.get('stochastic_d', 50):.1f}
Bollinger Position: {technical_indicators.get('bb_position', 0.5):.2f} (0=lower, 1=upper)
Support: ${technical_indicators.get('support', current_price*0.95):,.2f}
Resistance: ${technical_indicators.get('resistance', current_price*1.05):,.2f}

=== SENTIMENT ANALYSIS ===
Overall Score: {sentiment_data.get('score', 0):.2f} (-1 bearish to +1 bullish)
Confidence: {sentiment_data.get('confidence', 0.5):.0%}
Breakdown: {sentiment_data.get('breakdown', {}).get('positive', 0):.0f}% positive, {sentiment_data.get('breakdown', {}).get('neutral', 0):.0f}% neutral, {sentiment_data.get('breakdown', {}).get('negative', 0):.0f}% negative

=== RECENT HEADLINES ===
{chr(10).join(['• ' + h for h in top_headlines[:5]]) if top_headlines else '• No recent headlines available'}

=== TASK ===
Provide your investment recommendation as JSON only. Consider:
1. Is the {expected_roi:+.1f}% forecast realistic given technicals?
2. Does {model_agreement:.0%} model agreement justify confidence?
3. What are the 3 biggest risks?
4. Specific entry, target, and stop-loss prices if recommending BUY/SELL

Respond with ONLY valid JSON, no other text."""

        return prompt
    
    def _call_gemini_with_retry(self, prompt: str) -> Optional[str]:
        """Call Gemini API with exponential backoff retry"""
        
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                logger.info(f"🤖 Calling Gemini API (attempt {attempt + 1})")
                
                if USE_NEW_SDK:
                    # New SDK call
                    response = self.client.models.generate_content(
                        model=GEMINI_CONFIG["model"],
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=GEMINI_CONFIG["temperature"],
                            max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
                            top_p=GEMINI_CONFIG["top_p"],
                            top_k=GEMINI_CONFIG["top_k"],
                            system_instruction=SYSTEM_PROMPT,
                        )
                    )
                    if response and response.text:
                        logger.info("✅ Gemini response received")
                        return response.text
                else:
                    # Legacy SDK call
                    response = self.model.generate_content(prompt)
                    if response and response.text:
                        logger.info("✅ Gemini response received")
                        return response.text
                
                logger.warning("⚠️ Empty response from Gemini")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if "429" in str(e) or "quota" in error_msg or "rate" in error_msg or "resource" in error_msg:
                    wait_time = min(
                        RETRY_CONFIG["base_delay"] * (2 ** attempt),
                        RETRY_CONFIG["max_delay"]
                    )
                    logger.warning(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Check for safety filter blocks
                if "safety" in error_msg or "blocked" in error_msg:
                    logger.warning("⚠️ Response blocked by safety filter")
                    return None
                
                # Other errors
                logger.error(f"❌ Gemini API error: {e}")
                
                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    wait_time = RETRY_CONFIG["base_delay"] * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    # Don't raise, return None to trigger fallback
                    return None
        
        return None
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON from Gemini response, handling common issues"""
        
        if not response_text:
            return None
        
        # Clean up the response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error: {e}")
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"❌ Could not parse response as JSON: {text[:200]}...")
            return None
    
    def _validate_response(self, parsed: Dict) -> Dict:
        """Validate and normalize the parsed response"""
        
        # Required fields with defaults
        # NOTE: Using 'score' to match app.py expectations (not 'confidence')
        defaults = {
            "recommendation": "HOLD",
            "score": 0.5,  # app.py expects 'score', not 'confidence'
            "insight": "Unable to generate detailed analysis.",
            "reasoning": "Insufficient data for confident recommendation.",
            "risks": [
                "Market volatility may impact predictions",
                "Model uncertainty affects reliability", 
                "External factors not captured in analysis"
            ],
            "key_factors": ["Technical indicators", "Price momentum", "Sentiment"],
            "entry_price": None,
            "target_price": None,
            "stop_loss": None
        }
        
        result = defaults.copy()
        
        # Update with parsed values
        for key in defaults:
            if key in parsed and parsed[key] is not None:
                result[key] = parsed[key]
        
        # Handle 'confidence' -> 'score' mapping (Gemini may return either)
        if "confidence" in parsed and parsed["confidence"] is not None:
            result["score"] = parsed["confidence"]
        
        # Validate recommendation
        valid_recs = ["BUY", "SELL", "HOLD"]
        if result["recommendation"].upper() not in valid_recs:
            result["recommendation"] = "HOLD"
        else:
            result["recommendation"] = result["recommendation"].upper()
        
        # Validate score (0.0 to 1.0)
        try:
            result["score"] = max(0.0, min(1.0, float(result["score"])))
        except (TypeError, ValueError):
            result["score"] = 0.5
        
        # Ensure risks is a list
        if not isinstance(result["risks"], list):
            result["risks"] = [str(result["risks"])]
        
        # Ensure key_factors is a list
        if not isinstance(result["key_factors"], list):
            result["key_factors"] = [str(result["key_factors"])]
        
        # Add source info
        result["source"] = "gemini_llm"
        result["model"] = GEMINI_CONFIG["model"]
        
        return result
    
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
        Generate investment insights using Gemini LLM
        
        Args:
            coin_symbol: Cryptocurrency symbol (e.g., "BTC")
            market_data: Current market metrics
            sentiment_data: News sentiment analysis
            technical_indicators: Technical analysis metrics
            prediction_data: ML model predictions
            top_headlines: Recent news headlines
            horizon_days: Forecast horizon
            
        Returns:
            Dictionary with recommendation, confidence, insights, risks
        """
        
        logger.info(f"🤖 Generating Gemini insights for {coin_symbol}...")
        
        try:
            # Build the prompt
            prompt = self._build_analysis_prompt(
                coin_symbol=coin_symbol,
                market_data=market_data,
                sentiment_data=sentiment_data,
                technical_indicators=technical_indicators,
                prediction_data=prediction_data,
                top_headlines=top_headlines,
                horizon_days=horizon_days
            )
            
            # Call Gemini API
            response_text = self._call_gemini_with_retry(prompt)
            
            if not response_text:
                logger.warning("⚠️ No response from Gemini, using fallback")
                return self._generate_fallback_response(
                    coin_symbol, market_data, prediction_data, technical_indicators
                )
            
            # Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            if not parsed:
                logger.warning("⚠️ Could not parse Gemini response, using fallback")
                return self._generate_fallback_response(
                    coin_symbol, market_data, prediction_data, technical_indicators
                )
            
            # Validate and normalize
            result = self._validate_response(parsed)
            
            logger.info(f"✅ Gemini insight: {result['recommendation']} "
                       f"(score: {result['score']:.0%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating Gemini insights: {e}")
            return self._generate_fallback_response(
                coin_symbol, market_data, prediction_data, technical_indicators
            )
    
    def _generate_fallback_response(
        self,
        coin_symbol: str,
        market_data: Dict,
        prediction_data: Dict,
        technical_indicators: Dict
    ) -> Dict:
        """Generate a safe fallback response when Gemini fails"""
        
        logger.info("📋 Using fallback rule-based response")
        
        # Simple rule-based fallback
        ensemble_preds = prediction_data.get('ensemble', [])
        current_price = market_data.get('price_usd', 0)
        model_agreement = prediction_data.get('model_agreement', 0.5)
        rsi = technical_indicators.get('rsi', 50)
        
        # Calculate expected ROI
        if ensemble_preds and current_price > 0:
            predicted_price = ensemble_preds[-1]
            expected_roi = ((predicted_price - current_price) / current_price) * 100
        else:
            expected_roi = 0
        
        # Simple decision logic
        if model_agreement < 0.6:
            recommendation = "HOLD"
            reasoning = f"Low model agreement ({model_agreement:.0%}) creates uncertainty"
        elif expected_roi > 5 and rsi < 65:
            recommendation = "BUY"
            reasoning = f"Positive {expected_roi:+.1f}% forecast with RSI {rsi:.0f}"
        elif expected_roi < -5 and rsi > 60:
            recommendation = "SELL"
            reasoning = f"Negative {expected_roi:+.1f}% forecast with RSI {rsi:.0f}"
        else:
            recommendation = "HOLD"
            reasoning = f"Mixed signals with {expected_roi:+.1f}% forecast"
        
        return {
            "recommendation": recommendation,
            "score": 0.50,  # Use 'score' to match app.py expectations
            "insight": f"{coin_symbol} analysis: {reasoning}. "
                      f"Model agreement: {model_agreement:.0%}. "
                      f"Consider waiting for clearer signals.",
            "reasoning": reasoning,
            "risks": [
                "Fallback analysis has limited depth",
                "Market conditions may change rapidly",
                "Always use proper risk management"
            ],
            "key_factors": [
                f"Forecast: {expected_roi:+.1f}%",
                f"RSI: {rsi:.0f}",
                f"Model Agreement: {model_agreement:.0%}"
            ],
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "source": "fallback_rules",
            "model": "rule_based_v1"
        }


# ============================================================================
# PUBLIC INTERFACE (Drop-in replacement for original)
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
    Generate investment insights using Gemini LLM
    
    This is a drop-in replacement for the original rule-based function.
    Same signature, enhanced output.
    
    Args:
        api_key: Google AI API key
        coin_symbol: Cryptocurrency symbol
        market_data: Current market metrics
        sentiment_data: News sentiment analysis
        technical_indicators: Technical analysis metrics
        prediction_data: ML model predictions
        top_headlines: Recent news headlines
        horizon_days: Forecast horizon
        
    Returns:
        Dictionary with recommendation, confidence, insights, risks
    """
    
    # Validate API key
    if not api_key or api_key.strip() == "":
        logger.warning("⚠️ No API key provided, using fallback")
        generator = GeminiInsightGenerator.__new__(GeminiInsightGenerator)
        return generator._generate_fallback_response(
            coin_symbol, market_data, prediction_data, technical_indicators
        )
    
    try:
        generator = GeminiInsightGenerator(api_key)
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
        logger.error(f"❌ Failed to initialize Gemini: {e}")
        # Return safe fallback
        return {
            "recommendation": "HOLD",
            "score": 0.40,  # Use 'score' to match app.py expectations
            "insight": f"Unable to analyze {coin_symbol} due to API error. "
                      f"Please verify your Gemini API key in Streamlit secrets.",
            "reasoning": f"API initialization failed: {str(e)[:100]}",
            "risks": [
                "Analysis unavailable - use caution",
                "Verify API key configuration",
                "Try again in a few minutes"
            ],
            "key_factors": ["API Error"],
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "source": "error",
            "model": "none"
        }


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

class RuleBasedInsightGenerator:
    """
    Legacy class for backwards compatibility.
    Now wraps the Gemini generator.
    """
    
    def __init__(self):
        logger.info("⚠️ RuleBasedInsightGenerator is deprecated. Using Gemini instead.")
        self._api_key = None
    
    def set_api_key(self, api_key: str):
        """Set API key for Gemini"""
        self._api_key = api_key
    
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
        """Generate insights (now uses Gemini if API key available)"""
        
        if self._api_key:
            return generate_insights(
                api_key=self._api_key,
                coin_symbol=coin_symbol,
                market_data=market_data,
                sentiment_data=sentiment_data,
                technical_indicators=technical_indicators,
                prediction_data=prediction_data,
                top_headlines=top_headlines,
                horizon_days=horizon_days
            )
        else:
            # Fallback to simple rules
            generator = GeminiInsightGenerator.__new__(GeminiInsightGenerator)
            return generator._generate_fallback_response(
                coin_symbol, market_data, prediction_data, technical_indicators
            )













































































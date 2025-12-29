"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

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

GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
]

GEMINI_CONFIG = {
    "model": GEMINI_MODELS[0],
    "fallback_models": GEMINI_MODELS[1:],
    "temperature": 0.3,
    "max_output_tokens": 1500,  # Slightly increased for reflection
    "top_p": 0.9,
    "top_k": 40,
}

RETRY_CONFIG = {
    "max_retries": 2,
    "base_delay": 1,
    "max_delay": 5,
}


# ============================================================================
# ENHANCED SYSTEM PROMPT WITH CONFLICT RESOLUTION + SELF-REFLECTION
# ============================================================================

ENHANCED_SYSTEM_PROMPT = """You are an expert cryptocurrency trader. You analyze data carefully, resolve conflicting signals, and self-check your recommendations before finalizing.

YOUR PROCESS (follow in order):
1. IDENTIFY CONFLICTS: Note any signals that contradict each other
2. RESOLVE CONFLICTS: Reason through which signals to trust and why
3. MAKE RECOMMENDATION: Based on resolved analysis
4. SELF-REFLECT: Critique your own recommendation - look for errors
5. FINALIZE: Adjust if your self-reflection found issues

SIGNAL INTERPRETATION RULES:
- RSI < 35: OVERSOLD = Bullish (buying opportunity)
- RSI > 65: OVERBOUGHT = Bearish (selling opportunity)
- RSI 35-65: Neutral zone
- Positive forecast (>3%): Bullish
- Negative forecast (<-3%): Bearish
- High model agreement (>75%): Trust the forecast more
- Low model agreement (<60%): Be skeptical of forecast

CONFLICT RESOLUTION PRIORITY (when signals disagree):
1. Model Agreement > 80% → Trust the ML forecast
2. RSI extreme (<30 or >70) → Trust RSI signal
3. Strong trend confirmed → Trust trend direction
4. High sentiment confidence (>70%) → Consider sentiment
5. When still unclear → Recommend HOLD

OUTPUT FORMAT (strict JSON):
{
    "conflicts_detected": [
        {"signal_1": "what it says", "signal_2": "what it says", "resolution": "which to trust and why"}
    ],
    "recommendation": "BUY" | "SELL" | "HOLD",
    "score": 0.0-1.0,
    "insight": "2-3 sentence analysis with specific reasoning",
    "reasoning": "1-2 sentences on key factors",
    "self_reflection": {
        "potential_issues": ["any concerns about this recommendation"],
        "confidence_adjustment": "none" | "reduced" | "increased",
        "final_check": "passed" | "adjusted"
    },
    "risks": ["risk 1", "risk 2", "risk 3"],
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "entry_price": number or null,
    "target_price": number or null,
    "stop_loss": number or null
}

CRITICAL: Respond with ONLY valid JSON. No other text."""


# ============================================================================
# CONFLICT DETECTION (Pre-processing before LLM call)
# ============================================================================

class ConflictDetector:
    """Detects conflicts between different signals before LLM analysis"""
    
    @staticmethod
    def detect_conflicts(
        market_data: Dict,
        technical_indicators: Dict,
        sentiment_data: Dict,
        prediction_data: Dict
    ) -> List[Dict]:
        """
        Detect conflicts between different signal sources.
        
        Returns:
            List of conflict dictionaries with signal details
        """
        conflicts = []
        
        # Extract values
        rsi = technical_indicators.get('rsi', 50)
        trend = technical_indicators.get('trend', 'sideways')
        macd_hist = technical_indicators.get('macd_histogram', 0)
        
        sentiment_score = sentiment_data.get('score', 0)
        sentiment_conf = sentiment_data.get('confidence', 0.5)
        
        ensemble = prediction_data.get('ensemble', [])
        model_agreement = prediction_data.get('model_agreement', 0.5)
        
        current_price = market_data.get('price_usd', 0)
        
        # Calculate forecast direction
        if ensemble and current_price > 0:
            forecast_change = ((ensemble[-1] - current_price) / current_price) * 100
        else:
            forecast_change = 0
        
        # === CONFLICT 1: RSI vs Forecast ===
        rsi_bullish = rsi < 40
        rsi_bearish = rsi > 60
        forecast_bullish = forecast_change > 3
        forecast_bearish = forecast_change < -3
        
        if rsi_bullish and forecast_bearish:
            conflicts.append({
                "type": "rsi_vs_forecast",
                "signal_1": f"RSI {rsi:.0f} (oversold = bullish)",
                "signal_2": f"Forecast {forecast_change:+.1f}% (bearish)",
                "severity": "high",
                "suggestion": "RSI may indicate short-term bounce despite bearish forecast"
            })
        elif rsi_bearish and forecast_bullish:
            conflicts.append({
                "type": "rsi_vs_forecast",
                "signal_1": f"RSI {rsi:.0f} (overbought = bearish)",
                "signal_2": f"Forecast {forecast_change:+.1f}% (bullish)",
                "severity": "high",
                "suggestion": "Price may pull back before continuing upward"
            })
        
        # === CONFLICT 2: Sentiment vs Technical ===
        sentiment_bullish = sentiment_score > 0.25 and sentiment_conf > 0.6
        sentiment_bearish = sentiment_score < -0.25 and sentiment_conf > 0.6
        technical_bullish = trend == 'uptrend' or (rsi < 40 and macd_hist > 0)
        technical_bearish = trend == 'downtrend' or (rsi > 60 and macd_hist < 0)
        
        if sentiment_bullish and technical_bearish:
            conflicts.append({
                "type": "sentiment_vs_technical",
                "signal_1": f"Sentiment {sentiment_score:+.2f} (bullish news)",
                "signal_2": f"Technical {trend}, RSI {rsi:.0f} (bearish)",
                "severity": "medium",
                "suggestion": "News hasn't reflected in price yet, or market is ignoring it"
            })
        elif sentiment_bearish and technical_bullish:
            conflicts.append({
                "type": "sentiment_vs_technical",
                "signal_1": f"Sentiment {sentiment_score:+.2f} (bearish news)",
                "signal_2": f"Technical {trend}, RSI {rsi:.0f} (bullish)",
                "severity": "medium",
                "suggestion": "Price resilient despite negative news - could be a strength signal"
            })
        
        # === CONFLICT 3: LSTM vs XGBoost ===
        lstm_preds = prediction_data.get('lstm', [])
        xgb_preds = prediction_data.get('xgboost', [])
        
        if lstm_preds and xgb_preds and current_price > 0:
            lstm_change = ((lstm_preds[-1] - current_price) / current_price) * 100
            xgb_change = ((xgb_preds[-1] - current_price) / current_price) * 100
            
            # Check if models disagree on direction
            if (lstm_change > 2 and xgb_change < -2) or (lstm_change < -2 and xgb_change > 2):
                conflicts.append({
                    "type": "model_disagreement",
                    "signal_1": f"LSTM: {lstm_change:+.1f}%",
                    "signal_2": f"XGBoost: {xgb_change:+.1f}%",
                    "severity": "high",
                    "suggestion": f"Model agreement only {model_agreement:.0%} - reduce confidence"
                })
        
        # === CONFLICT 4: Short-term vs Long-term ===
        pct_24h = market_data.get('pct_change_24h', 0)
        pct_7d = market_data.get('pct_change_7d', 0)
        
        if (pct_24h > 5 and pct_7d < -5) or (pct_24h < -5 and pct_7d > 5):
            conflicts.append({
                "type": "timeframe_conflict",
                "signal_1": f"24h: {pct_24h:+.1f}%",
                "signal_2": f"7d: {pct_7d:+.1f}%",
                "severity": "low",
                "suggestion": "Recent reversal - trend may be changing"
            })
        
        return conflicts
    
    @staticmethod
    def get_conflict_summary(conflicts: List[Dict]) -> str:
        """Format conflicts for the LLM prompt"""
        if not conflicts:
            return "No significant conflicts detected. Signals are aligned."
        
        lines = [f"⚠️ {len(conflicts)} CONFLICT(S) DETECTED:"]
        for i, c in enumerate(conflicts, 1):
            lines.append(f"\n{i}. {c['type'].upper()} [{c['severity']}]")
            lines.append(f"   • {c['signal_1']}")
            lines.append(f"   • {c['signal_2']}")
            lines.append(f"   💡 {c['suggestion']}")
        
        return "\n".join(lines)


# ============================================================================
# ENHANCED GEMINI CLIENT
# ============================================================================

class EnhancedGeminiInsightGenerator:
    """
    Enhanced Gemini client with:
    1. Pre-LLM conflict detection
    2. Conflict resolution in prompt
    3. Self-reflection in prompt
    4. All in a single API call
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.client = None
        self.current_model = GEMINI_CONFIG["model"]
        self.conflict_detector = ConflictDetector()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client"""
        try:
            if USE_NEW_SDK:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"✅ Enhanced Gemini client initialized: {self.current_model}")
            else:
                genai.configure(api_key=self.api_key)
                self._create_model(self.current_model)
                logger.info(f"✅ Enhanced Gemini client initialized: {self.current_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    def _create_model(self, model_name: str):
        """Create a GenerativeModel instance for legacy SDK"""
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_CONFIG["temperature"],
                max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
                top_p=GEMINI_CONFIG["top_p"],
                top_k=GEMINI_CONFIG["top_k"],
            ),
            system_instruction=ENHANCED_SYSTEM_PROMPT
        )
        self.current_model = model_name
    
    def _build_enhanced_prompt(
        self,
        coin_symbol: str,
        market_data: Dict,
        sentiment_data: Dict,
        technical_indicators: Dict,
        prediction_data: Dict,
        top_headlines: List[str],
        horizon_days: int,
        conflicts: List[Dict]
    ) -> str:
        """Build prompt with conflict info and self-reflection request"""
        
        # Extract data
        current_price = market_data.get('price_usd', 0)
        ensemble_preds = prediction_data.get('ensemble', [])
        lstm_preds = prediction_data.get('lstm', [])
        xgb_preds = prediction_data.get('xgboost', [])
        model_agreement = prediction_data.get('model_agreement', 0.5)
        
        predicted_price = ensemble_preds[-1] if ensemble_preds else current_price
        expected_roi = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Format predictions
        def format_preds(preds, label):
            if not preds:
                return f"{label}: No data"
            change = ((preds[-1] - preds[0]) / preds[0] * 100) if preds[0] > 0 else 0
            return f"{label}: ${preds[0]:,.2f} → ${preds[-1]:,.2f} ({change:+.1f}%)"
        
        # Get conflict summary
        conflict_summary = self.conflict_detector.get_conflict_summary(conflicts)
        
        prompt = f"""Analyze {coin_symbol} for a {horizon_days}-day investment decision.

=== MARKET DATA ===
Current Price: ${current_price:,.2f}
24h Change: {market_data.get('pct_change_24h', 0):+.2f}%
7d Change: {market_data.get('pct_change_7d', 0):+.2f}%
Market Cap: ${market_data.get('market_cap', 0):,.0f}
24h Volume: ${market_data.get('volume_24h', 0):,.0f}

=== {horizon_days}-DAY FORECAST ===
{format_preds(lstm_preds, 'LSTM')}
{format_preds(xgb_preds, 'XGBoost')}
{format_preds(ensemble_preds, 'Ensemble')}
Expected ROI: {expected_roi:+.2f}%
Model Agreement: {model_agreement:.0%} {"⚠️ LOW" if model_agreement < 0.6 else "✅ HIGH" if model_agreement > 0.8 else ""}

=== TECHNICAL INDICATORS ===
RSI (14): {technical_indicators.get('rsi', 50):.1f} {"📈 OVERSOLD" if technical_indicators.get('rsi', 50) < 35 else "📉 OVERBOUGHT" if technical_indicators.get('rsi', 50) > 65 else "(neutral)"}
Trend: {technical_indicators.get('trend', 'unknown')}
MACD Histogram: {technical_indicators.get('macd_histogram', 0):.4f} {"(bullish)" if technical_indicators.get('macd_histogram', 0) > 0 else "(bearish)"}
Volatility: {technical_indicators.get('volatility', 0):.2%}
Support: ${technical_indicators.get('support', current_price*0.95):,.2f}
Resistance: ${technical_indicators.get('resistance', current_price*1.05):,.2f}

=== SENTIMENT ===
Score: {sentiment_data.get('score', 0):.2f} (-1 to +1)
Confidence: {sentiment_data.get('confidence', 0.5):.0%}
Breakdown: {sentiment_data.get('breakdown', {}).get('positive', 0):.0f}% pos, {sentiment_data.get('breakdown', {}).get('neutral', 0):.0f}% neu, {sentiment_data.get('breakdown', {}).get('negative', 0):.0f}% neg

=== HEADLINES ===
{chr(10).join(['• ' + h for h in top_headlines[:5]]) if top_headlines else '• No headlines'}

=== ⚠️ CONFLICT ANALYSIS ===
{conflict_summary}

=== YOUR TASK ===
1. RESOLVE any conflicts listed above - explain which signal to trust and why
2. Make your BUY/SELL/HOLD recommendation
3. SELF-REFLECT: Before finalizing, ask yourself:
   - Does my recommendation match the data?
   - Am I interpreting RSI correctly? (low RSI = bullish, high RSI = bearish)
   - Is my confidence justified given the model agreement?
   - Are there any errors in my reasoning?
4. Adjust your recommendation if self-reflection found issues

Respond with ONLY valid JSON matching the required format."""

        return prompt
    
    def _call_gemini_with_retry(self, prompt: str) -> Optional[str]:
        """Call Gemini API with retry and model fallback"""
        
        models_to_try = [self.current_model] + [
            m for m in GEMINI_CONFIG.get("fallback_models", []) 
            if m != self.current_model
        ]
        
        for model_name in models_to_try:
            logger.info(f"🤖 Trying model: {model_name}")
            
            for attempt in range(RETRY_CONFIG["max_retries"]):
                try:
                    if USE_NEW_SDK:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=GEMINI_CONFIG["temperature"],
                                max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
                                top_p=GEMINI_CONFIG["top_p"],
                                top_k=GEMINI_CONFIG["top_k"],
                                system_instruction=ENHANCED_SYSTEM_PROMPT,
                            )
                        )
                        if response and response.text:
                            self.current_model = model_name
                            return response.text
                    else:
                        if self.current_model != model_name:
                            self._create_model(model_name)
                        response = self.model.generate_content(prompt)
                        if response and response.text:
                            return response.text
                            
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if "404" in str(e) or "not found" in error_msg:
                        logger.warning(f"⚠️ Model {model_name} not found, trying next...")
                        break
                    
                    is_rate_limit = any(x in error_msg for x in [
                        "429", "quota", "rate", "exhausted", "limit"
                    ])
                    
                    if is_rate_limit:
                        if attempt >= RETRY_CONFIG["max_retries"] - 1:
                            break
                        wait_time = RETRY_CONFIG["base_delay"] * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error(f"❌ Gemini error: {e}")
                    if attempt >= RETRY_CONFIG["max_retries"] - 1:
                        break
        
        return None
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON from response"""
        if not response_text:
            return None
        
        text = response_text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return None
    
    def _validate_response(self, parsed: Dict, conflicts: List[Dict]) -> Dict:
        """Validate and enhance the response"""
        
        defaults = {
            "recommendation": "HOLD",
            "score": 0.5,
            "insight": "Unable to generate analysis.",
            "reasoning": "Insufficient data.",
            "risks": ["Market volatility", "Model uncertainty", "External factors"],
            "key_factors": [],
            "entry_price": None,
            "target_price": None,
            "stop_loss": None,
            "conflicts_detected": [],
            "self_reflection": {
                "potential_issues": [],
                "confidence_adjustment": "none",
                "final_check": "passed"
            }
        }
        
        result = defaults.copy()
        
        for key in defaults:
            if key in parsed and parsed[key] is not None:
                result[key] = parsed[key]
        
        # Handle confidence -> score mapping
        if "confidence" in parsed:
            result["score"] = parsed["confidence"]
        
        # Validate recommendation
        rec = str(result["recommendation"]).upper()
        if rec not in ["BUY", "SELL", "HOLD"]:
            result["recommendation"] = "HOLD"
        else:
            result["recommendation"] = rec
        
        # Validate score
        try:
            result["score"] = max(0.0, min(1.0, float(result["score"])))
        except:
            result["score"] = 0.5
        
        # Add pre-detected conflicts if LLM didn't return them
        if not result["conflicts_detected"] and conflicts:
            result["conflicts_detected"] = conflicts
        
        # Adjust confidence based on conflicts
        if len(conflicts) >= 2 and result["score"] > 0.75:
            result["score"] = min(result["score"], 0.70)
            logger.info(f"📉 Reduced confidence due to {len(conflicts)} conflicts")
        
        # Add metadata
        result["source"] = "gemini_enhanced"
        result["model"] = self.current_model
        result["conflict_count"] = len(conflicts)
        
        return result
    
    def _generate_fallback_response(
        self,
        coin_symbol: str,
        market_data: Dict,
        prediction_data: Dict,
        technical_indicators: Dict,
        conflicts: List[Dict]
    ) -> Dict:
        """Generate rule-based fallback when Gemini fails"""
        
        current_price = market_data.get('price_usd', 0)
        ensemble = prediction_data.get('ensemble', [])
        model_agreement = prediction_data.get('model_agreement', 0.5)
        rsi = technical_indicators.get('rsi', 50)
        
        predicted_price = ensemble[-1] if ensemble else current_price
        expected_roi = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Count signals
        bullish_signals = 0
        bearish_signals = 0
        
        if expected_roi > 3:
            bullish_signals += 1
        elif expected_roi < -3:
            bearish_signals += 1
            
        if rsi < 40:
            bullish_signals += 1
        elif rsi > 60:
            bearish_signals += 1
            
        if model_agreement > 0.75:
            if expected_roi > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Reduce confidence if conflicts exist
        conflict_penalty = len(conflicts) * 0.05
        
        # Decision
        if bullish_signals >= 2 and bearish_signals == 0:
            recommendation = "BUY"
            confidence = min(0.80 - conflict_penalty, 0.85)
        elif bearish_signals >= 2 and bullish_signals == 0:
            recommendation = "SELL"
            confidence = min(0.75 - conflict_penalty, 0.80)
        else:
            recommendation = "HOLD"
            confidence = 0.55
        
        # Build conflict resolution text
        if conflicts:
            conflict_text = f" Detected {len(conflicts)} conflicting signals - confidence adjusted."
        else:
            conflict_text = ""
        
        return {
            "recommendation": recommendation,
            "score": confidence,
            "insight": f"{coin_symbol} shows {expected_roi:+.1f}% forecast with RSI at {rsi:.0f}.{conflict_text}",
            "reasoning": f"Based on {bullish_signals} bullish and {bearish_signals} bearish signals.",
            "risks": [
                "API fallback used - limited analysis",
                "Market volatility may impact predictions",
                "External factors not fully captured"
            ],
            "key_factors": [
                f"Forecast: {expected_roi:+.1f}%",
                f"RSI: {rsi:.0f}",
                f"Model Agreement: {model_agreement:.0%}"
            ],
            "conflicts_detected": conflicts,
            "self_reflection": {
                "potential_issues": ["Fallback mode - limited reasoning"],
                "confidence_adjustment": "reduced",
                "final_check": "fallback"
            },
            "entry_price": technical_indicators.get('support'),
            "target_price": predicted_price if recommendation == "BUY" else None,
            "stop_loss": technical_indicators.get('support', current_price * 0.95) * 0.95 if recommendation == "BUY" else None,
            "source": "enhanced_fallback",
            "model": "rule_based_v3",
            "conflict_count": len(conflicts)
        }
    
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
        Generate insights with conflict resolution and self-reflection.
        All done in a SINGLE Gemini call.
        """
        
        logger.info(f"🤖 Enhanced analysis for {coin_symbol}...")
        
        # Step 1: Detect conflicts BEFORE calling LLM
        conflicts = self.conflict_detector.detect_conflicts(
            market_data=market_data,
            technical_indicators=technical_indicators,
            sentiment_data=sentiment_data,
            prediction_data=prediction_data
        )
        
        if conflicts:
            logger.info(f"⚠️ Detected {len(conflicts)} signal conflicts")
            for c in conflicts:
                logger.info(f"   - {c['type']}: {c['severity']}")
        else:
            logger.info("✅ No signal conflicts detected")
        
        try:
            # Step 2: Build enhanced prompt with conflicts + self-reflection request
            prompt = self._build_enhanced_prompt(
                coin_symbol=coin_symbol,
                market_data=market_data,
                sentiment_data=sentiment_data,
                technical_indicators=technical_indicators,
                prediction_data=prediction_data,
                top_headlines=top_headlines,
                horizon_days=horizon_days,
                conflicts=conflicts
            )
            
            # Step 3: Single Gemini call (quota efficient!)
            response_text = self._call_gemini_with_retry(prompt)
            
            if not response_text:
                logger.warning("⚠️ No Gemini response, using fallback")
                return self._generate_fallback_response(
                    coin_symbol, market_data, prediction_data, 
                    technical_indicators, conflicts
                )
            
            # Step 4: Parse and validate
            parsed = self._parse_json_response(response_text)
            
            if not parsed:
                logger.warning("⚠️ Could not parse response, using fallback")
                return self._generate_fallback_response(
                    coin_symbol, market_data, prediction_data,
                    technical_indicators, conflicts
                )
            
            result = self._validate_response(parsed, conflicts)
            
            # Log self-reflection results
            reflection = result.get('self_reflection', {})
            if reflection.get('confidence_adjustment') == 'reduced':
                logger.info("📉 Self-reflection reduced confidence")
            elif reflection.get('confidence_adjustment') == 'increased':
                logger.info("📈 Self-reflection increased confidence")
            
            logger.info(f"✅ Enhanced insight: {result['recommendation']} "
                       f"(score: {result['score']:.0%}, conflicts: {len(conflicts)})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis error: {e}")
            return self._generate_fallback_response(
                coin_symbol, market_data, prediction_data,
                technical_indicators, conflicts
            )


# ============================================================================
# PUBLIC INTERFACE (Drop-in replacement)
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
    Generate investment insights with conflict resolution and self-reflection.
    
    Drop-in replacement for original generate_insights function.
    Same signature, enhanced output.
    """
    
    if not api_key or api_key.strip() == "":
        logger.warning("⚠️ No API key, using fallback")
        conflicts = ConflictDetector.detect_conflicts(
            market_data, technical_indicators, sentiment_data, prediction_data
        )
        generator = EnhancedGeminiInsightGenerator.__new__(EnhancedGeminiInsightGenerator)
        generator.conflict_detector = ConflictDetector()
        return generator._generate_fallback_response(
            coin_symbol, market_data, prediction_data, technical_indicators, conflicts
        )
    
    try:
        generator = EnhancedGeminiInsightGenerator(api_key)
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
        logger.error(f"❌ Failed to initialize enhanced Gemini: {e}")
        return {
            "recommendation": "HOLD",
            "score": 0.40,
            "insight": f"Analysis error: {str(e)[:100]}",
            "reasoning": "API error occurred",
            "risks": ["Analysis unavailable", "Verify API key", "Try again later"],
            "key_factors": ["Error"],
            "conflicts_detected": [],
            "self_reflection": {"final_check": "error"},
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

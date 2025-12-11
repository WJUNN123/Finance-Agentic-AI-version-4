"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """Enhanced Gemini integration with better prompt engineering"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.2,
        max_tokens: int = 1500
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Gemini {model_name} initialized")
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
        Generate comprehensive investment insights with multi-factor analysis
        """
        # Build enhanced structured prompt
        prompt = self._build_enhanced_prompt(
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
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json"  # Force JSON response
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            response_text = response.text.strip()
            
            # Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            if parsed:
                # Validate and enhance response
                validated = self._validate_and_enhance_response(parsed, market_data, prediction_data)
                return {
                    "recommendation": validated.get("recommendation", "HOLD"),
                    "score": validated.get("confidence_score", 0.5),
                    "insight": validated.get("analysis", ""),
                    "risks": validated.get("risks", []),
                    "opportunities": validated.get("opportunities", []),
                    "key_signals": validated.get("key_signals", {}),
                    "source": "gemini",
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.warning("Failed to parse JSON response, using fallback")
                return self._fallback_parse(response_text)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._get_fallback_response()

    def _build_enhanced_prompt(
        self, 
        coin_symbol: str, 
        market_data: Dict, 
        sentiment_data: Dict, 
        tech: Dict, 
        preds: Dict, 
        headlines: List[str], 
        horizon: int
    ) -> str:
        """Enhanced prompt with better structure and reasoning steps"""
        
        curr_price = market_data.get('price_usd', 0)
        pct_24h = market_data.get('pct_change_24h', 0)
        pct_7d = market_data.get('pct_change_7d', 0)
        
        # Process predictions with confidence bands if available
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble'):
            ensemble = preds['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                pred_roi = ((final_pred - curr_price) / curr_price) * 100
                
                # Analyze trajectory
                if len(ensemble) >= 3:
                    early = ensemble[0]
                    mid = ensemble[len(ensemble)//2]
                    final = ensemble[-1]
                    
                    if mid > early and mid > final:
                        trend_shape = "peak-and-decline"
                    elif mid < early and mid < final:
                        trend_shape = "dip-and-recovery"
                    elif final > curr_price * 1.02:
                        trend_shape = "steady-upward"
                    elif final < curr_price * 0.98:
                        trend_shape = "steady-downward"
                    else:
                        trend_shape = "sideways"
                else:
                    trend_shape = "insufficient-data"
                
                # Get confidence bands if available
                confidence_info = ""
                if 'confidence_bands' in preds and preds['confidence_bands']:
                    bands = preds['confidence_bands'][-1]  # Last day
                    upper = bands.get('upper', final_pred * 1.1)
                    lower = bands.get('lower', final_pred * 0.9)
                    confidence_range = ((upper - lower) / final_pred) * 100
                    confidence_info = f" | 95% CI Range: ±{confidence_range:.1f}%"
                
                pred_text = (
                    f"Predicted End Price: ${final_pred:,.2f} | "
                    f"Projected ROI: {pred_roi:+.2f}% | "
                    f"Trajectory: {trend_shape}{confidence_info}"
                )

        # Enhanced sentiment processing
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0, 'neutral': 0})
        
        pos_pct = breakdown.get('positive', 0)
        neg_pct = breakdown.get('negative', 0)
        neu_pct = breakdown.get('neutral', 0)
        
        # Calculate sentiment strength
        sent_strength = abs(pos_pct - neg_pct)
        if sent_strength > 30:
            sent_strength_label = "STRONG"
        elif sent_strength > 15:
            sent_strength_label = "MODERATE"
        else:
            sent_strength_label = "WEAK"
        
        sent_interpretation = self._interpret_sentiment_score(sent_score)
        sent_text = (
            f"Aggregate Score: {sent_score:.2f} ({sent_interpretation}) | "
            f"Distribution: {pos_pct:.0f}% Positive, {neu_pct:.0f}% Neutral, {neg_pct:.0f}% Negative | "
            f"Signal Strength: {sent_strength_label}"
        )

        # Enhanced technical analysis
        rsi = tech.get('rsi', 50)
        volatility = tech.get('volatility', 0)
        trend = tech.get('trend', 'sideways')
        momentum = tech.get('momentum', 0)
        
        rsi_zone = self._get_rsi_zone(rsi)
        rsi_signal = "BEARISH" if rsi > 70 else "BULLISH" if rsi < 30 else "NEUTRAL"
        
        # Bollinger Bands analysis
        bb_status = "Within Bands (Normal)"
        bb_signal = "NEUTRAL"
        if tech.get('bb_upper') and curr_price > tech['bb_upper']:
            bb_status = "Above Upper Band"
            bb_signal = "OVERBOUGHT"
        elif tech.get('bb_lower') and curr_price < tech['bb_lower']:
            bb_status = "Below Lower Band"
            bb_signal = "OVERSOLD"
        
        # Momentum analysis
        momentum_signal = "BULLISH" if momentum > 5 else "BEARISH" if momentum < -5 else "NEUTRAL"
        
        tech_text = (
            f"RSI: {rsi:.1f} ({rsi_zone}) → Signal: {rsi_signal} | "
            f"Bollinger Bands: {bb_status} → Signal: {bb_signal} | "
            f"Trend: {trend.upper()} | "
            f"Momentum: {momentum:.2f}% → Signal: {momentum_signal} | "
            f"Volatility: {volatility:.4f}"
        )

        # Price action summary
        price_action = (
            f"Current: ${curr_price:,.2f} | "
            f"24h: {pct_24h:+.2f}% | "
            f"7d: {pct_7d:+.2f}%"
        )

        # Recent headlines (top 5)
        headlines_text = "\n  ".join(headlines[:5]) if headlines else "No recent headlines"

        # Build the enhanced prompt
        return f"""You are an Expert Cryptocurrency Investment Analyst with 10+ years of experience in technical analysis, market psychology, and risk management. 

Analyze the following comprehensive data for {coin_symbol} and provide a professional investment recommendation.

═══════════════════════════════════════════════════════════════
📊 MARKET DATA
═══════════════════════════════════════════════════════════════
{price_action}

Forecast ({horizon} days): {pred_text}

═══════════════════════════════════════════════════════════════
💭 SENTIMENT ANALYSIS
═══════════════════════════════════════════════════════════════
News Sentiment: {sent_text}

═══════════════════════════════════════════════════════════════
📈 TECHNICAL INDICATORS
═══════════════════════════════════════════════════════════════
{tech_text}

═══════════════════════════════════════════════════════════════
📰 RECENT HEADLINES
═══════════════════════════════════════════════════════════════
  {headlines_text}

═══════════════════════════════════════════════════════════════
🎯 YOUR ANALYSIS TASK
═══════════════════════════════════════════════════════════════

Provide a JSON response with this EXACT structure:

{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "2-3 professional sentences explaining your recommendation based on multi-factor convergence/divergence",
  "key_signals": {{
    "technical": "BULLISH/BEARISH/MIXED with brief reason",
    "sentiment": "BULLISH/BEARISH/MIXED with brief reason",
    "prediction": "BULLISH/BEARISH/MIXED with brief reason"
  }},
  "risks": ["3-5 specific risk factors"],
  "opportunities": ["2-3 specific opportunity factors if applicable"],
  "reasoning": "Brief explanation of how you reconciled conflicting signals"
}}

═══════════════════════════════════════════════════════════════
🧠 DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════

**Step 1: Signal Alignment Analysis**
- Count BULLISH signals vs BEARISH signals across all factors
- Strong BUY: 70%+ signals aligned bullish + prediction shows >5% upside
- Strong SELL: 70%+ signals aligned bearish + prediction shows >5% downside
- HOLD: Mixed signals (<70% alignment) OR low conviction

**Step 2: Risk Assessment**
- RSI extremes (>70 or <30) = Contrarian signal (be cautious)
- High volatility = Increase uncertainty, lower confidence
- Weak sentiment strength = Lower conviction
- Prediction uncertainty (wide confidence bands) = Lower confidence

**Step 3: Confidence Scoring**
- 0.8-1.0: Strong signal alignment (>80%) + supportive fundamentals
- 0.6-0.8: Good alignment (70-80%) + reasonable conviction
- 0.4-0.6: Mixed signals, moderate uncertainty
- 0.2-0.4: Conflicting signals, high uncertainty
- 0.0-0.2: Very uncertain, lacking data

**Step 4: Risk-Adjusted Recommendation**
- Even with bullish signals, if RSI >75 or volatility >0.15, consider HOLD
- Even with bearish signals, if RSI <25 and sentiment improving, consider HOLD
- Always prioritize capital preservation in ambiguous situations

═══════════════════════════════════════════════════════════════

IMPORTANT:
- Be intellectually honest - don't force a BUY/SELL if signals are mixed
- Explain WHY you chose this recommendation despite any conflicting data
- Provide actionable, specific risks (not generic warnings)
- Keep analysis concise but insightful

Respond with ONLY valid JSON, no markdown formatting."""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Enhanced JSON parsing with better error handling"""
        try:
            # Try direct parsing first
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    return parsed
                except:
                    pass
            
            # Try to find JSON object
            json_match = re.search(r'\{[^{}]*"recommendation"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed
                except:
                    pass
        
        logger.error("Failed to parse JSON response")
        return None

    def _validate_and_enhance_response(self, parsed: Dict, market_data: Dict, prediction_data: Dict) -> Dict:
        """Validate and enhance the parsed response"""
        
        # Normalize recommendation
        rec = parsed.get("recommendation", "HOLD").upper().strip()
        if "BUY" in rec:
            rec = "BUY"
        elif "SELL" in rec:
            rec = "SELL"
        else:
            rec = "HOLD"
        
        # Validate confidence score
        score = parsed.get("confidence_score", 0.5)
        if isinstance(score, str):
            try:
                score = float(score.replace('%', '')) / 100.0
            except:
                score = 0.5
        score = max(0.0, min(1.0, float(score)))
        
        # Get analysis text
        analysis = parsed.get("analysis", "")
        if not analysis:
            analysis = parsed.get("reasoning", "Analysis not available")
        
        # Get risks
        risks = parsed.get("risks", [])
        if not isinstance(risks, list):
            risks = []
        
        # Get opportunities
        opportunities = parsed.get("opportunities", [])
        if not isinstance(opportunities, list):
            opportunities = []
        
        # Get key signals
        key_signals = parsed.get("key_signals", {})
        if not isinstance(key_signals, dict):
            key_signals = {}
        
        return {
            "recommendation": rec,
            "confidence_score": score,
            "analysis": analysis,
            "risks": risks[:5],  # Limit to top 5
            "opportunities": opportunities[:3],  # Limit to top 3
            "key_signals": key_signals,
            "reasoning": parsed.get("reasoning", "")
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
        
        # Extract confidence
        score = 0.5
        score_match = re.search(r'(?:confidence|score)[:\s]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if score > 1:
                    score = score / 100.0
            except:
                pass
        
        return {
            "recommendation": rec,
            "score": score,
            "insight": text[:600],
            "risks": ["Unable to extract detailed risks"],
            "opportunities": [],
            "key_signals": {},
            "source": "gemini_fallback",
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
            "insight": "AI insights temporarily unavailable. Please check your API key configuration.",
            "risks": ["API connectivity issue"],
            "opportunities": [],
            "key_signals": {},
            "source": "fallback",
            "model": self.model_name
        }


def generate_insights(api_key: str, **kwargs) -> Dict:
    """Convenient function to generate insights"""
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

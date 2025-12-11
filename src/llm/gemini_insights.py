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
    """Enhanced Gemini integration with prediction-aligned recommendations"""
    
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
        Generate insights with PREDICTION-FIRST recommendation logic
        """
        
        # CRITICAL FIX: Calculate predicted ROI FIRST
        curr_price = market_data.get('price_usd', 0)
        predicted_roi = 0.0
        final_pred = curr_price
        
        if prediction_data and prediction_data.get('ensemble'):
            ensemble = prediction_data['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                predicted_roi = ((final_pred - curr_price) / curr_price) * 100 if curr_price > 0 else 0
        
        # RULE-BASED FALLBACK (if Gemini fails or gives wrong recommendation)
        rule_based_rec = self._get_rule_based_recommendation(
            predicted_roi=predicted_roi,
            sentiment_data=sentiment_data,
            technical_indicators=technical_indicators,
            market_data=market_data
        )
        
        # Build enhanced prompt with EXPLICIT prediction emphasis
        prompt = self._build_prediction_focused_prompt(
            coin_symbol=coin_symbol,
            market_data=market_data,
            sentiment_data=sentiment_data,
            tech=technical_indicators,
            preds=prediction_data,
            headlines=top_headlines,
            horizon=horizon_days,
            predicted_roi=predicted_roi,
            final_pred=final_pred
        )
        
        try:
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json"
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            response_text = response.text.strip()
            
            # Parse JSON response
            parsed = self._parse_json_response(response_text)
            
            if parsed:
                # CRITICAL FIX: Validate recommendation matches prediction
                validated = self._validate_recommendation_matches_prediction(
                    parsed=parsed,
                    predicted_roi=predicted_roi,
                    rule_based_rec=rule_based_rec,
                    market_data=market_data
                )
                
                return {
                    "recommendation": validated.get("recommendation", "HOLD"),
                    "score": validated.get("confidence_score", 0.5),
                    "insight": validated.get("analysis", ""),
                    "risks": validated.get("risks", []),
                    "opportunities": validated.get("opportunities", []),
                    "key_signals": validated.get("key_signals", {}),
                    "source": "gemini",
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "predicted_roi": predicted_roi  # Include for debugging
                }
            else:
                logger.warning("Failed to parse JSON, using rule-based recommendation")
                return rule_based_rec
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return rule_based_rec

    def _get_rule_based_recommendation(
        self,
        predicted_roi: float,
        sentiment_data: Dict,
        technical_indicators: Dict,
        market_data: Dict
    ) -> Dict:
        """
        RULE-BASED fallback that ALWAYS aligns with prediction
        """
        
        # PRIMARY SIGNAL: Price prediction
        if predicted_roi <= -10:
            base_rec = "SELL"
            base_confidence = 0.75
            base_insight = f"Strong sell signal: Model predicts {predicted_roi:.1f}% decline over the forecast period."
        elif predicted_roi <= -5:
            base_rec = "SELL"
            base_confidence = 0.65
            base_insight = f"Bearish outlook: Model predicts {predicted_roi:.1f}% decline. Consider reducing exposure."
        elif predicted_roi < -2:
            base_rec = "HOLD"
            base_confidence = 0.55
            base_insight = f"Cautious stance: Model predicts {predicted_roi:.1f}% decline. Wait for better entry."
        elif predicted_roi < 2:
            base_rec = "HOLD"
            base_confidence = 0.50
            base_insight = f"Neutral outlook: Model predicts {predicted_roi:.1f}% movement. Consolidation expected."
        elif predicted_roi < 5:
            base_rec = "HOLD"
            base_confidence = 0.55
            base_insight = f"Slight upside: Model predicts {predicted_roi:.1f}% gain. Monitor for confirmation."
        elif predicted_roi < 10:
            base_rec = "BUY"
            base_confidence = 0.65
            base_insight = f"Bullish signal: Model predicts {predicted_roi:.1f}% upside. Favorable risk/reward."
        else:
            base_rec = "BUY"
            base_confidence = 0.75
            base_insight = f"Strong buy signal: Model predicts {predicted_roi:.1f}% upside over forecast period."
        
        # SECONDARY SIGNALS: Adjust confidence
        rsi = technical_indicators.get('rsi', 50)
        sentiment_score = sentiment_data.get('score', 0)
        pct_24h = market_data.get('pct_change_24h', 0)
        
        # RSI adjustment
        if rsi > 70 and base_rec == "BUY":
            base_confidence -= 0.15  # Overbought reduces buy confidence
        elif rsi < 30 and base_rec == "SELL":
            base_confidence -= 0.15  # Oversold reduces sell confidence
        
        # Sentiment adjustment
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0})
        if breakdown.get('positive', 0) > 60 and base_rec == "BUY":
            base_confidence += 0.05  # Strong positive sentiment boosts buy
        elif breakdown.get('negative', 0) > 60 and base_rec == "SELL":
            base_confidence += 0.05  # Strong negative sentiment boosts sell
        
        # Momentum alignment
        if pct_24h > 5 and base_rec == "BUY":
            base_confidence += 0.05
        elif pct_24h < -5 and base_rec == "SELL":
            base_confidence += 0.05
        
        # Cap confidence
        base_confidence = max(0.3, min(0.95, base_confidence))
        
        # Generate risks
        risks = []
        if abs(predicted_roi) > 15:
            risks.append("High predicted volatility - large price movement expected")
        if rsi > 70:
            risks.append("Overbought conditions (RSI > 70) may lead to pullback")
        elif rsi < 30:
            risks.append("Oversold conditions (RSI < 30) may indicate capitulation")
        if technical_indicators.get('volatility', 0) > 0.1:
            risks.append("High volatility increases price uncertainty")
        if base_rec == "SELL" and pct_24h > 0:
            risks.append("Recommendation contradicts recent upward momentum")
        
        return {
            "recommendation": base_rec,
            "score": base_confidence,
            "insight": base_insight,
            "risks": risks if risks else ["Market conditions remain uncertain"],
            "opportunities": [],
            "key_signals": {
                "prediction": f"{'BEARISH' if predicted_roi < -2 else 'BULLISH' if predicted_roi > 2 else 'NEUTRAL'} ({predicted_roi:+.1f}%)",
                "technical": f"RSI {rsi:.0f}, Volatility {technical_indicators.get('volatility', 0):.4f}",
                "sentiment": f"Score {sentiment_score:.2f}"
            },
            "source": "rule_based",
            "model": "fallback"
        }

    def _validate_recommendation_matches_prediction(
        self,
        parsed: Dict,
        predicted_roi: float,
        rule_based_rec: Dict,
        market_data: Dict
    ) -> Dict:
        """
        CRITICAL FIX: Validate AI recommendation matches prediction
        If mismatch detected, override with rule-based recommendation
        """
        
        ai_rec = parsed.get("recommendation", "HOLD").upper().strip()
        if "BUY" in ai_rec:
            ai_rec = "BUY"
        elif "SELL" in ai_rec:
            ai_rec = "SELL"
        else:
            ai_rec = "HOLD"
        
        # Check for CRITICAL mismatches
        mismatch = False
        
        # RULE 1: Don't recommend BUY if prediction shows >5% drop
        if ai_rec == "BUY" and predicted_roi < -5:
            logger.warning(f"⚠️ MISMATCH: AI says BUY but prediction is {predicted_roi:.1f}% DOWN")
            mismatch = True
        
        # RULE 2: Don't recommend SELL if prediction shows >5% gain
        if ai_rec == "SELL" and predicted_roi > 5:
            logger.warning(f"⚠️ MISMATCH: AI says SELL but prediction is {predicted_roi:.1f}% UP")
            mismatch = True
        
        # RULE 3: Strong drops (>10%) should NEVER be HOLD
        if predicted_roi < -10 and ai_rec != "SELL":
            logger.warning(f"⚠️ MISMATCH: Prediction shows {predicted_roi:.1f}% drop but AI says {ai_rec}")
            mismatch = True
        
        # RULE 4: Strong gains (>10%) should NEVER be SELL
        if predicted_roi > 10 and ai_rec == "SELL":
            logger.warning(f"⚠️ MISMATCH: Prediction shows {predicted_roi:.1f}% gain but AI says SELL")
            mismatch = True
        
        if mismatch:
            logger.info(f"✅ Overriding with rule-based recommendation: {rule_based_rec['recommendation']}")
            # Use rule-based recommendation but keep AI's analysis if good
            return {
                "recommendation": rule_based_rec["recommendation"],
                "confidence_score": rule_based_rec["score"],
                "analysis": parsed.get("analysis", rule_based_rec["insight"]),
                "risks": parsed.get("risks", rule_based_rec["risks"]),
                "opportunities": parsed.get("opportunities", []),
                "key_signals": rule_based_rec.get("key_signals", {}),
                "reasoning": f"Recommendation adjusted to align with {predicted_roi:+.1f}% predicted ROI"
            }
        
        # No mismatch - use AI recommendation
        score = parsed.get("confidence_score", 0.5)
        if isinstance(score, str):
            try:
                score = float(score.replace('%', '')) / 100.0
            except:
                score = 0.5
        score = max(0.0, min(1.0, float(score)))
        
        return {
            "recommendation": ai_rec,
            "confidence_score": score,
            "analysis": parsed.get("analysis", ""),
            "risks": parsed.get("risks", [])[:5],
            "opportunities": parsed.get("opportunities", [])[:3],
            "key_signals": parsed.get("key_signals", {}),
            "reasoning": parsed.get("reasoning", "")
        }

    def _build_prediction_focused_prompt(
        self,
        coin_symbol: str,
        market_data: Dict,
        sentiment_data: Dict,
        tech: Dict,
        preds: Dict,
        headlines: List[str],
        horizon: int,
        predicted_roi: float,
        final_pred: float
    ) -> str:
        """Build prompt that EMPHASIZES prediction alignment"""
        
        curr_price = market_data.get('price_usd', 0)
        pct_24h = market_data.get('pct_change_24h', 0)
        pct_7d = market_data.get('pct_change_7d', 0)
        
        # Determine trend from prediction
        if predicted_roi < -10:
            pred_interpretation = "⚠️ STRONG BEARISH - Major decline predicted"
        elif predicted_roi < -5:
            pred_interpretation = "📉 BEARISH - Downward movement expected"
        elif predicted_roi < -2:
            pred_interpretation = "↘️ SLIGHTLY BEARISH - Mild decline expected"
        elif predicted_roi < 2:
            pred_interpretation = "➡️ NEUTRAL - Sideways movement expected"
        elif predicted_roi < 5:
            pred_interpretation = "↗️ SLIGHTLY BULLISH - Mild gain expected"
        elif predicted_roi < 10:
            pred_interpretation = "📈 BULLISH - Upward movement expected"
        else:
            pred_interpretation = "🚀 STRONG BULLISH - Major gain predicted"
        
        # Sentiment
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0, 'neutral': 0})
        
        # Technical
        rsi = tech.get('rsi', 50)
        volatility = tech.get('volatility', 0)
        trend = tech.get('trend', 'sideways')
        
        headlines_text = "\n  ".join(headlines[:3]) if headlines else "No recent headlines"
        
        return f"""You are an Expert Cryptocurrency Investment Analyst. Your PRIMARY JOB is to make recommendations that ALIGN with price predictions.

═══════════════════════════════════════════════════════════════
🎯 PRIMARY SIGNAL: PRICE PREDICTION (MOST IMPORTANT!)
═══════════════════════════════════════════════════════════════
Current Price: ${curr_price:,.2f}
Predicted Price ({horizon} days): ${final_pred:,.2f}
Predicted ROI: {predicted_roi:+.2f}%

INTERPRETATION: {pred_interpretation}

⚠️ CRITICAL RULE: Your recommendation MUST align with this prediction!
- If predicted ROI < -5% → You MUST recommend SELL or HOLD (never BUY)
- If predicted ROI > +5% → You MUST recommend BUY or HOLD (never SELL)
- If predicted ROI between -5% and +5% → HOLD is acceptable

═══════════════════════════════════════════════════════════════
📊 SUPPORTING SIGNALS (Use to adjust confidence only)
═══════════════════════════════════════════════════════════════

MARKET DATA:
- 24h Change: {pct_24h:+.2f}%
- 7d Change: {pct_7d:+.2f}%

TECHNICAL:
- RSI: {rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})
- Trend: {trend.upper()}
- Volatility: {volatility:.4f}

SENTIMENT:
- News Sentiment Score: {sent_score:.2f}
- Distribution: {breakdown.get('positive', 0):.0f}% pos, {breakdown.get('negative', 0):.0f}% neg

HEADLINES:
  {headlines_text}

═══════════════════════════════════════════════════════════════
📋 YOUR TASK
═══════════════════════════════════════════════════════════════

Provide a JSON response:

{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "Explain your recommendation focusing on WHY the prediction shows {predicted_roi:+.1f}% movement",
  "key_signals": {{
    "prediction": "Primary driver of recommendation",
    "technical": "Supporting or conflicting signal",
    "sentiment": "Supporting or conflicting signal"
  }},
  "risks": ["List 3-5 specific risks"],
  "opportunities": ["List 1-3 opportunities if BUY, otherwise empty"],
  "reasoning": "Explain how prediction influenced your decision"
}}

═══════════════════════════════════════════════════════════════
🧠 DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════

STEP 1: Look at predicted ROI ({predicted_roi:+.2f}%)
- Is it <-5%? → Lean SELL
- Is it >+5%? → Lean BUY  
- Is it between -5% and +5%? → Lean HOLD

STEP 2: Check for extreme technical conflicts
- If RSI >75 and prediction is bullish → Lower confidence, might HOLD instead of BUY
- If RSI <25 and prediction is bearish → Lower confidence, might HOLD instead of SELL

STEP 3: Adjust confidence based on supporting signals
- Sentiment aligns with prediction → +10% confidence
- Momentum aligns with prediction → +10% confidence
- Conflicting signals → -20% confidence

STEP 4: Final recommendation
- Your recommendation MUST make sense given a {predicted_roi:+.1f}% predicted move
- If you recommend BUY, the prediction should show gains
- If you recommend SELL, the prediction should show losses
- HOLD is for unclear or small predicted moves

RESPOND WITH ONLY VALID JSON."""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Enhanced JSON parsing"""
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
            
            json_match = re.search(r'\{[^{}]*"recommendation"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        
        return None


def generate_insights(api_key: str, **kwargs) -> Dict:
    """Convenient function to generate insights"""
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

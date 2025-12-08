"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Scenario types for analysis"""
    BULL = "bull_case"
    BEAR = "bear_case"
    BASE = "base_case"


class EnhancedGeminiInsightGenerator:
    """Enhanced Gemini integration with multi-stage reasoning"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini {model_name} initialized successfully")
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
        risk_assessment: Dict,
        top_headlines: List[str],
        horizon_days: int = 7
    ) -> Dict:
        """Generate multi-stage insights"""
        
        logger.info(f"Generating {coin_symbol} insights with multi-stage reasoning...")
        
        try:
            # Stage 1: Fact extraction
            facts = self._extract_facts(
                market_data, sentiment_data, technical_indicators, prediction_data
            )
            
            # Stage 2: Build reasoning prompt
            prompt = self._build_reasoning_prompt(
                coin_symbol=coin_symbol,
                facts=facts,
                sentiment_data=sentiment_data,
                technical=technical_indicators,
                predictions=prediction_data,
                risks=risk_assessment,
                headlines=top_headlines,
                horizon=horizon_days
            )
            
            # Stage 3: Get Gemini response
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            response_text = response.text.strip()
            
            # Stage 4: Parse response
            parsed = self._parse_structured_response(response_text)
            
            if parsed:
                # Stage 5: Calibrate confidence
                calibrated = self._calibrate_confidence(parsed, risk_assessment)
                return calibrated
            else:
                return self._fallback_response()
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._fallback_response()
    
    def _extract_facts(
        self,
        market_data: Dict,
        sentiment_data: Dict,
        technical_indicators: Dict,
        prediction_data: Dict
    ) -> Dict:
        """Extract key facts for reasoning"""
        
        curr_price = market_data.get('price_usd', 0)
        
        # Prediction facts
        if prediction_data and prediction_data.get('ensemble'):
            ensemble = prediction_data['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                pred_roi = ((final_pred - curr_price) / curr_price) * 100
                
                # Analyze trajectory
                if len(ensemble) > 3:
                    first_third = np.mean(ensemble[:len(ensemble)//3])
                    last_third = np.mean(ensemble[2*len(ensemble)//3:])
                    
                    if last_third > first_third:
                        trajectory = "accelerating upward"
                    else:
                        trajectory = "decelerating"
                else:
                    trajectory = "insufficient data"
                
                prediction_facts = {
                    'final_price': final_pred,
                    'roi_percent': pred_roi,
                    'trajectory': trajectory
                }
            else:
                prediction_facts = {'status': 'no_prediction'}
        else:
            prediction_facts = {'status': 'no_prediction'}
        
        # Sentiment facts
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {})
        trend = sentiment_data.get('trend_direction', 'unknown')
        is_extreme = sentiment_data.get('is_extreme', False)
        
        # Technical facts
        rsi = technical_indicators.get('rsi', 50)
        volatility = technical_indicators.get('volatility', 0.05)
        trend_tech = technical_indicators.get('trend', 'sideways')
        support = technical_indicators.get('support', curr_price * 0.95)
        resistance = technical_indicators.get('resistance', curr_price * 1.05)
        
        return {
            'current_price': curr_price,
            'prediction': prediction_facts,
            'sentiment': {
                'score': sent_score,
                'breakdown': breakdown,
                'trend': trend,
                'is_extreme': is_extreme
            },
            'technical': {
                'rsi': rsi,
                'volatility': volatility,
                'trend': trend_tech,
                'support': support,
                'resistance': resistance
            }
        }
    
    def _build_reasoning_prompt(
        self,
        coin_symbol: str,
        facts: Dict,
        sentiment_data: Dict,
        technical: Dict,
        predictions: Dict,
        risks: Dict,
        headlines: List[str],
        horizon: int
    ) -> str:
        """Build multi-stage reasoning prompt"""
        
        pred_data = facts['prediction']
        if 'final_price' in pred_data:
            pred_text = f"${pred_data['final_price']:,.0f} ({pred_data['roi_percent']:+.1f}%)"
        else:
            pred_text = "No clear prediction"
        
        sentiment_interpretation = self._interpret_sentiment(facts['sentiment']['score'])
        tech_facts = facts['technical']
        
        headlines_text = "\n  ".join(headlines[:3]) if headlines else "No headlines available"
        
        # Risk summary
        risk_score = risks.get('overall_score', 0.5)
        regime = risks.get('regime', {}).get('value', 'unknown')
        top_risks = [r.description for r in risks.get('top_risks', [])]
        
        return f"""You are an expert crypto analyst. Analyze {coin_symbol} through MULTI-STAGE REASONING:

=== STAGE 1: FACT EXTRACTION ===
Current Price: ${facts['current_price']:,.2f}
Price Forecast ({horizon}d): {pred_text}
RSI: {tech_facts['rsi']:.1f}
Volatility: {tech_facts['volatility']:.4f}
Trend: {tech_facts['trend']}
Support: ${tech_facts['support']:,.2f} | Resistance: ${tech_facts['resistance']:,.2f}
Sentiment Score: {facts['sentiment']['score']:.2f} ({sentiment_interpretation})
Sentiment Trend: {facts['sentiment']['trend']}
Sentiment Breakdown: {facts['sentiment']['breakdown']}

=== STAGE 2: SIGNAL ALIGNMENT ANALYSIS ===
Analyze whether signals AGREE or CONFLICT:
- Does RSI align with trend? (Overbought in uptrend = warning)
- Does sentiment support price prediction?
- Is volatility consistent with predicted moves?
- Are support/resistance levels near predicted prices?

=== STAGE 3: SCENARIO EVALUATION ===
Build 3 scenarios with probabilities:

BULL CASE (IF validated):
- What conditions would make BUY thesis correct?
- Key support/resistance levels to watch
- Catalysts that could drive higher

BEAR CASE (IF validated):
- What conditions would invalidate BUY thesis?
- Where would price go if it breaks down?
- Key downside targets

BASE CASE (MOST LIKELY):
- Most probable outcome given current data
- Why is this more likely than others?
- Timeline for base case to resolve

=== STAGE 4: RISK-ADJUSTED RECOMMENDATION ===
Market Regime: {regime}
Overall Risk Level: {risk_score:.1f}/1.0
Top Risks: {', '.join(top_risks[:2]) if top_risks else 'Low risk'}

Extreme Sentiment: {"YES - High mean reversion risk" if facts['sentiment']['is_extreme'] else "No"}

=== STAGE 5: FINAL DECISION ===
Return ONLY this JSON (no markdown, pure JSON):
{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence_score": 0.0-1.0,
  "reasoning": "2-3 sentence explanation of top-level reasoning",
  "bull_case": {{
    "thesis": "Core bullish argument",
    "triggers": ["trigger1", "trigger2"],
    "target_price": target_or_null,
    "probability": 0.0-1.0
  }},
  "bear_case": {{
    "thesis": "Core bearish argument",
    "triggers": ["trigger1", "trigger2"],
    "stop_loss": level_or_null,
    "probability": 0.0-1.0
  }},
  "base_case": {{
    "thesis": "Most likely outcome",
    "timeline_days": number,
    "probability": 0.0-1.0
  }},
  "key_catalysts": ["catalyst1", "catalyst2"],
  "risks": ["risk1", "risk2", "risk3"],
  "action_items": ["monitor_metric1", "watch_for_signal1"]
}}

=== DECISION RULES ===
1. STRONG alignment = Higher confidence (70%+)
2. Mixed signals = HOLD or Lower confidence (40-60%)
3. Extreme sentiment = Always cap confidence at 60%
4. High volatility = Add 15% risk buffer
5. Model disagreement > 5% = Lower confidence
6. Support/resistance near prediction = Higher confidence

Recent Headlines:
{headlines_text}

Respond with ONLY valid JSON."""
    
    def _parse_structured_response(self, response_text: str) -> Optional[Dict]:
        """Parse structured JSON response"""
        try:
            # Try direct parsing
            parsed = json.loads(response_text)
            return self._validate_response(parsed)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(
                r'\{[\s\S]*"recommendation"[\s\S]*\}',
                response_text,
                re.DOTALL
            )
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return self._validate_response(parsed)
                except Exception as e:
                    logger.warning(f"JSON extraction failed: {e}")
        
        return None
    
    def _validate_response(self, parsed: Dict) -> Dict:
        """Validate and normalize response"""
        
        # Normalize recommendation
        rec = (parsed.get("recommendation", "HOLD") or "HOLD").upper().strip()
        if "BUY" in rec:
            rec = "BUY"
        elif "SELL" in rec:
            rec = "SELL"
        else:
            rec = "HOLD"
        
        # Normalize confidence
        confidence = parsed.get("confidence_score", 0.5)
        if isinstance(confidence, str):
            confidence = float(confidence.replace('%', '')) / 100.0
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Get scenarios
        bull_case = parsed.get('bull_case', {})
        bear_case = parsed.get('bear_case', {})
        base_case = parsed.get('base_case', {})
        
        # Get risks and catalysts
        risks = parsed.get('risks', [])
        catalysts = parsed.get('key_catalysts', [])
        action_items = parsed.get('action_items', [])
        
        return {
            'recommendation': rec,
            'confidence_score': confidence,
            'reasoning': parsed.get('reasoning', ''),
            'bull_case': bull_case,
            'bear_case': bear_case,
            'base_case': base_case,
            'scenarios': [
                {'type': 'bull', 'data': bull_case},
                {'type': 'bear', 'data': bear_case},
                {'type': 'base', 'data': base_case}
            ],
            'risks': risks[:5],  # Top 5
            'catalysts': catalysts[:3],  # Top 3
            'action_items': action_items[:3]
        }
    
    def _calibrate_confidence(
        self,
        parsed: Dict,
        risk_assessment: Dict
    ) -> Dict:
        """Calibrate confidence based on risk assessment"""
        
        base_confidence = parsed.get('confidence_score', 0.5)
        risk_score = risk_assessment.get('overall_score', 0.5)
        
        # Reduce confidence by risk
        adjustment = risk_score * 0.3  # Risk reduces confidence by up to 30%
        calibrated_confidence = base_confidence * (1.0 - adjustment)
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
        
        return {
            **parsed,
            'confidence_score': float(calibrated_confidence),
            'base_confidence': float(base_confidence),
            'risk_adjustment': float(adjustment),
            'source': 'gemini',
            'model': self.model_name
        }
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score"""
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
    
    def _fallback_response(self) -> Dict:
        """Fallback response on error"""
        return {
            'recommendation': 'HOLD',
            'confidence_score': 0.4,
            'reasoning': 'Unable to generate AI insights. Please check API key.',
            'bull_case': {},
            'bear_case': {},
            'base_case': {},
            'scenarios': [],
            'risks': ['API Error'],
            'catalysts': [],
            'action_items': [],
            'source': 'fallback',
            'model': self.model_name
        }


import numpy as np  # Import for array operations

def generate_insights_enhanced(api_key: str, **kwargs) -> Dict:
    """Convenient function for enhanced insights"""
    generator = EnhancedGeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

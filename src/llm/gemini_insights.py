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
            logger.info(f"✅ Gemini model initialized: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
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
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            
            logger.info(f"🤖 Generating insights for {coin_symbol}...")
            response = self.model.generate_content(prompt, generation_config=config)
            response_text = response.text.strip()
            
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
                    "source": "gemini",
                    "model": self.model_name
                }
            else:
                # Fallback parsing if JSON fails
                logger.warning("⚠️ Failed to parse JSON response, using fallback extraction")
                return self._fallback_parse(response_text)
            
        except Exception as e:
            logger.error(f"❌ Error generating insights: {e}")
            return self._get_fallback_response()
    
    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        """Constructs enhanced prompt with market context and structured reasoning"""
        
        curr_price = market_data.get('price_usd', 0)
        market_cap = market_data.get('market_cap', 0)
        volume_24h = market_data.get('volume_24h', 0)
        
        # === MARKET CONTEXT (ENHANCED) ===
        market_context = f"""
=== MARKET OVERVIEW ===
Asset: {coin_symbol}
Current Price: ${curr_price:,.2f}
Market Cap: ${market_cap:,.0f} ({self._format_market_cap(market_cap)})
24h Volume: ${volume_24h:,.0f}
Volume/MCap Ratio: {(volume_24h / market_cap * 100) if market_cap > 0 else 0:.2f}% (liquidity indicator)
Price Changes: 24h: {market_data.get('pct_24h', 0):+.2f}%, 7d: {market_data.get('pct_7d', 0):+.2f}%
"""
        
        # === PREDICTION ANALYSIS WITH CONFIDENCE ===
        pred_text = "No predictive models available."
        pred_confidence = 0.5
        
        if preds and preds.get('ensemble'):
            ensemble = preds['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                pred_roi = ((final_pred - curr_price) / curr_price) * 100
                
                # Calculate prediction confidence
                lstm_preds = preds.get('lstm', [])
                xgb_preds = preds.get('xgboost', [])
                
                if len(lstm_preds) > 0 and len(xgb_preds) > 0:
                    # Agreement between models
                    lstm_roi = ((lstm_preds[-1] - curr_price) / curr_price) * 100
                    xgb_roi = ((xgb_preds[-1] - curr_price) / curr_price) * 100
                    roi_diff = abs(lstm_roi - xgb_roi)
                    # High agreement = high confidence
                    pred_confidence = max(0.3, 1.0 - (roi_diff / 20.0))  # 0.3 to 1.0
                
                # Analyze trajectory
                trend_shape = "neutral"
                if len(ensemble) > 2:
                    mid_point = ensemble[len(ensemble)//2]
                    early_point = ensemble[0]
                    
                    if mid_point > curr_price and final_pred < mid_point:
                        trend_shape = "volatile (spike then decline)"
                    elif mid_point < curr_price and final_pred > mid_point:
                        trend_shape = "recovery (dip then rise)"
                    elif final_pred > curr_price > early_point:
                        trend_shape = "sustained upward momentum"
                    elif final_pred < curr_price < early_point:
                        trend_shape = "sustained downward momentum"
                    elif abs(final_pred - curr_price) / curr_price < 0.02:
                        trend_shape = "range-bound consolidation"
                
                pred_text = f"""
Predicted End Price: ${final_pred:,.2f} (Day {horizon})
Projected ROI: {pred_roi:+.2f}%
Price Trajectory: {trend_shape}
Model Agreement: {pred_confidence:.0%} confidence
Individual Models:
  - LSTM: ${lstm_preds[-1]:,.2f} ({((lstm_preds[-1] - curr_price) / curr_price * 100):+.2f}%)
  - XGBoost: ${xgb_preds[-1]:,.2f} ({((xgb_preds[-1] - curr_price) / curr_price * 100):+.2f}%)
"""
        
        # === SENTIMENT ANALYSIS WITH CONFIDENCE ===
        sent_score = sentiment_data.get('score', 0)
        sent_confidence = sentiment_data.get('confidence', 0.5)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0, 'neutral': 0})
        sent_interpretation = self._interpret_sentiment_score(sent_score)
        
        sent_text = f"""
Aggregate Score: {sent_score:.2f}/1.0 ({sent_interpretation})
Confidence Level: {sent_confidence:.0%}
Sentiment Distribution:
  - Positive: {breakdown.get('positive', 0):.0f}%
  - Neutral: {breakdown.get('neutral', 0):.0f}%
  - Negative: {breakdown.get('negative', 0):.0f}%
Sentiment Strength: {self._get_sentiment_strength(sent_score)}
"""
        
        # === ENHANCED TECHNICAL INDICATORS ===
        rsi = tech.get('rsi', 50)
        macd_hist = tech.get('macd_histogram', 0)
        stoch_k = tech.get('stochastic_k', 50)
        volatility = tech.get('volatility', 0)
        trend = tech.get('trend', 'sideways')
        momentum_14 = tech.get('momentum', 0)
        
        rsi_zone = self._get_rsi_zone(rsi)
        macd_signal = "Bullish crossover" if macd_hist > 0 else "Bearish crossover"
        stoch_signal = "Oversold zone" if stoch_k < 20 else ("Overbought zone" if stoch_k > 80 else "Neutral zone")
        
        bb_status = "Within Bands (normal)"
        bb_signal = "neutral"
        if tech.get('bb_upper') and curr_price > tech['bb_upper']:
            bb_status = "Above Upper Band (stretched)"
            bb_signal = "overbought warning"
        elif tech.get('bb_lower') and curr_price < tech['bb_lower']:
            bb_status = "Below Lower Band (compressed)"
            bb_signal = "oversold opportunity"
        
        tech_text = f"""
Price Action:
  - Trend: {trend.upper()}
  - Momentum (14d): {momentum_14:+.2f}%

Oscillators:
  - RSI (14): {rsi:.1f} - {rsi_zone}
  - MACD: {macd_signal} (histogram: {macd_hist:.4f})
  - Stochastic: {stoch_k:.1f} - {stoch_signal}
  - Bollinger Bands: {bb_status} → {bb_signal}

Risk Metrics:
  - Volatility: {volatility:.2%} ({self._volatility_level(volatility)})
  - Support Level: ${tech.get('support', 0):,.2f}
  - Resistance Level: ${tech.get('resistance', 0):,.2f}
"""
        
        # === NEWS HEADLINES ===
        headlines_text = "\n  • ".join(headlines[:5]) if headlines else "No recent headlines available"
        
        # === BUILD COMPLETE PROMPT ===
        return f"""You are a Professional Cryptocurrency Investment Analyst with expertise in:
- Technical Analysis (RSI, MACD, Bollinger Bands, trend analysis)
- Sentiment Analysis (news sentiment, market psychology)
- Quantitative Modeling (machine learning price predictions)
- Risk Management (volatility assessment, risk-reward analysis)

{market_context}

=== PRICE FORECAST ({horizon} days) ===
{pred_text}

=== SENTIMENT ANALYSIS ===
{sent_text}

=== TECHNICAL INDICATORS ===
{tech_text}

=== RECENT NEWS HEADLINES ===
  • {headlines_text}

=== ANALYSIS FRAMEWORK ===
Evaluate the following dimensions:

1. **TREND ALIGNMENT** (40% weight)
   - Do price prediction, technical trend, and momentum align?
   - Are we in an uptrend, downtrend, or consolidation?

2. **SENTIMENT CONCORDANCE** (25% weight)
   - Does news sentiment support the price prediction?
   - Is sentiment confidence high enough to act on?

3. **TECHNICAL SIGNALS** (25% weight)
   - Are oscillators (RSI, MACD, Stochastic) confirming or diverging?
   - Are we approaching key support/resistance levels?

4. **RISK/REWARD** (10% weight)
   - Does volatility level match the expected return?
   - What's the downside risk vs upside potential?

Signal Quality Scores:
- Prediction Confidence: {pred_confidence:.0%}
- Sentiment Confidence: {sent_confidence:.0%}
- Overall Data Quality: {(pred_confidence + sent_confidence) / 2:.0%}

=== YOUR TASK ===
Provide a JSON response with EXACTLY this structure (valid JSON only, no markdown):

{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence_score": 0.0 to 1.0,
  "analysis": "2-3 sentence professional summary with specific data points",
  "risks": ["specific risk 1", "specific risk 2", "specific risk 3"],
  "reasoning": "Explain how you weighted different signals and why",
  "key_factors": ["factor 1 supporting decision", "factor 2", "factor 3"]
}}

=== DECISION RULES ===

**BUY Criteria** (All must align):
- Predicted ROI > +5% with confidence > 60%
- RSI between 30-60 (not overbought)
- Positive sentiment (>0.2) with confidence > 50%
- Uptrend or recovery pattern
- MACD bullish crossover

**SELL Criteria** (All must align):
- Predicted ROI < -5% with confidence > 60%
- RSI > 70 (overbought) or trend breakdown
- Negative sentiment (<-0.2) with confidence > 50%
- Downtrend or distribution pattern
- MACD bearish crossover

**HOLD Criteria** (Default if signals conflict):
- Mixed signals (some bullish, some bearish)
- Low confidence in predictions or sentiment
- Sideways trend with no clear direction
- Predicted ROI between -5% to +5%

**Confidence Adjustment**:
- Reduce confidence by 20% if prediction confidence < 60%
- Reduce confidence by 15% if sentiment confidence < 50%
- Reduce confidence by 10% if volatility is High or Very High

Respond with ONLY valid JSON, no other text or markdown."""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Extract and parse JSON from response"""
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*|\s*```', '', response_text)
            cleaned = cleaned.strip()
            
            # Try direct JSON parsing
            parsed = json.loads(cleaned)
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
        rec = str(parsed.get("recommendation", "HOLD")).upper().strip()
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
            analysis = parsed.get("reasoning", "No detailed analysis provided.")
        
        # Get risks
        risks = parsed.get("risks", [])
        if not isinstance(risks, list):
            risks = []
        
        # Get key factors
        key_factors = parsed.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = []
        
        return {
            "recommendation": rec,
            "confidence_score": score,
            "analysis": analysis,
            "risks": risks[:5],  # Limit to 5 risks
            "reasoning": parsed.get("reasoning", ""),
            "key_factors": key_factors[:5]  # Limit to 5 factors
        }
    
    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing if JSON extraction fails"""
        logger.warning("⚠️ Using fallback text parsing")
        
        # Extract recommendation
        rec = "HOLD"
        if re.search(r'\bBUY\b', text, re.IGNORECASE):
            rec = "BUY"
        elif re.search(r'\bSELL\b', text, re.IGNORECASE):
            rec = "SELL"
        
        # Extract confidence score
        score = 0.5
        score_match = re.search(r'(?:confidence|score)[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            if score > 1:
                score = score / 100.0
        
        return {
            "recommendation": rec,
            "score": score,
            "insight": text[:500],  # First 500 chars
            "risks": ["Unable to parse detailed risks"],
            "reasoning": "Fallback parsing used",
            "key_factors": [],
            "source": "gemini",
            "model": self.model_name
        }
    
    def _format_market_cap(self, market_cap: float) -> str:
        """Format market cap with T/B/M suffix"""
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        else:
            return f"${market_cap:,.0f}"
    
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
    
    def _get_sentiment_strength(self, score: float) -> str:
        """Get sentiment strength descriptor"""
        abs_score = abs(score)
        if abs_score >= 0.7:
            return "Very Strong"
        elif abs_score >= 0.5:
            return "Strong"
        elif abs_score >= 0.3:
            return "Moderate"
        elif abs_score >= 0.1:
            return "Weak"
        else:
            return "Neutral"
    
    def _get_rsi_zone(self, rsi: float) -> str:
        """Interpret RSI value"""
        if rsi >= 70:
            return "Overbought (consider selling)"
        elif rsi <= 30:
            return "Oversold (potential buy opportunity)"
        elif rsi >= 60:
            return "Bullish territory"
        elif rsi <= 40:
            return "Bearish territory"
        else:
            return "Neutral zone"
    
    def _volatility_level(self, vol: float) -> str:
        """Classify volatility level"""
        if vol > 0.15:
            return "Very High Risk"
        elif vol > 0.10:
            return "High Risk"
        elif vol > 0.05:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def _get_fallback_response(self) -> Dict:
        """Return fallback response on error"""
        return {
            "recommendation": "HOLD",
            "score": 0.5,
            "insight": "Unable to generate AI insights at this time. Please check your API key or try again later.",
            "risks": ["AI analysis unavailable"],
            "reasoning": "Error in LLM generation",
            "key_factors": [],
            "source": "fallback",
            "model": self.model_name
        }


def generate_insights(api_key: str, **kwargs) -> Dict:
    """Convenient function to generate insights"""
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)


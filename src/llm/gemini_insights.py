"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    """Generates investment insights using Gemini 2.0 Flash"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 800  # Increased for detailed reasoning
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
        sentiment_data: Dict, # Changed to accept full sentiment details
        technical_indicators: Dict,
        prediction_data: Dict, # Changed to accept full prediction arrays
        top_headlines: List[str],
        horizon_days: int = 7
    ) -> Dict:
        """
        Generate comprehensive investment insights
        """
        # 1. Build the data-rich prompt
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
            
            # 2. Get Response
            response = self.model.generate_content(prompt, generation_config=config)
            insight_text = response.text.strip()
            
            # 3. Extract Structured Data from LLM Output
            rec = self._extract_recommendation(insight_text)
            
            # IMPORTANT: We extract the score FROM the LLM's text, 
            # ensuring it matches the written reasoning.
            score = self._extract_confidence_score(insight_text)

            return {
                "recommendation": rec,
                "score": score,
                "insight": insight_text,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._get_fallback_response()

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        """Constructs a prompt that forces the LLM to act as a Data Science Synthesizer"""
        
        # --- A. Process Market Data ---
        curr_price = market_data.get('price_usd', 0)
        
        # --- B. Process Predictions (Calculate ROI) ---
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble'):
            ensemble = preds['ensemble']
            if len(ensemble) > 0:
                final_pred = ensemble[-1]
                pred_roi = ((final_pred - curr_price) / curr_price) * 100
                
                # Analyze curve shape
                trend_shape = "linear"
                if len(ensemble) > 2:
                    mid_point = ensemble[len(ensemble)//2]
                    # Check for simple hump/dip patterns
                    if mid_point > curr_price and mid_point > final_pred:
                        trend_shape = "volatility (rise then fall)"
                    elif mid_point < curr_price and mid_point < final_pred:
                        trend_shape = "recovery (dip then rise)"
                
                pred_text = (
                    f"Hybrid LSTM+XGBoost Model Forecast ({horizon} days):\n"
                    f"   - Predicted End Price: ${final_pred:,.2f}\n"
                    f"   - Projected ROI: {pred_roi:+.2f}%\n"
                    f"   - Trajectory Shape: {trend_shape}"
                )

        # --- C. Process Sentiment ---
        sent_score = sentiment_data.get('score', 0)
        breakdown = sentiment_data.get('breakdown', {'positive': 0, 'negative': 0, 'neutral': 0})
        sent_text = (
            f"News Sentiment (RoBERTa Model):\n"
            f"   - Aggregate Score: {sent_score:.2f} (Scale: -1.0 to +1.0)\n"
            f"   - Breakdown: {breakdown.get('positive', 0):.1f}% Positive, "
            f"{breakdown.get('neutral', 0):.1f}% Neutral, "
            f"{breakdown.get('negative', 0):.1f}% Negative"
        )

        # --- D. Process Technicals ---
        rsi = tech.get('rsi', 50)
        
        # Interpret Bollinger Bands
        bb_status = "Within Bands"
        if tech.get('bb_upper') and curr_price > tech['bb_upper']: 
            bb_status = "ABOVE Upper Band (Statistically Overextended)"
        elif tech.get('bb_lower') and curr_price < tech['bb_lower']: 
            bb_status = "BELOW Lower Band (Statistically Oversold)"
        
        tech_text = (
            f"Technical Indicators:\n"
            f"   - RSI (14): {rsi:.1f} ({self._get_rsi_zone(rsi)})\n"
            f"   - Bollinger Bands: {bb_status}\n"
            f"   - Trend Status: {tech.get('trend', 'Neutral')}\n"
            f"   - Volatility Index: {tech.get('volatility', 0):.4f}"
        )

        # --- E. The Prompt ---
        return f"""
You are a Senior Crypto Investment Analyst. Synthesize the provided data to generate a final recommendation for {coin_symbol}.

### INPUT DATA

1. {pred_text}
2. {sent_text}
3. {tech_text}
4. Recent Headlines:
   {chr(10).join(['- ' + h for h in headlines[:3]])}

### ANALYSIS LOGIC
- **Conflict Resolution:** If the Predictive Model projects a massive drop (>5%) but Sentiment is Positive and RSI is Oversold, reduce confidence in the drop. The model might be reacting to lagging momentum.
- **Confirmation:** If Technical Trend is Down AND Model predicts Drop AND Sentiment is Neutral/Negative, high confidence in SELL.
- **RSI Check:** If RSI is < 30, be careful recommending SELL even if the model predicts a drop (potential bounce area).

### OUTPUT FORMAT
You MUST provide the output in this exact structure:

**Analysis Synthesis**
[A concise paragraph (~3 sentences) explaining how the model prediction aligns or conflicts with news sentiment and technicals. Be specific about the ROI %.]

**Key Risks**
[Bullet points of specific risks based on the data provided.]

**Confidence Score**
[A single number between 0 and 100 representing how much the data aligns. e.g. "75"]

**Recommendation**
[One word: BUY, SELL, or HOLD]
"""

    def _extract_recommendation(self, text: str) -> str:
        match = re.search(r"Recommendation\s*\n\**([A-Z\s/]+)\**", text, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip().upper()
            if "BUY" in cleaned: return "BUY"
            if "SELL" in cleaned: return "SELL / AVOID"
        return "HOLD / WAIT"

    def _extract_confidence_score(self, text: str) -> float:
        # Regex to find "Confidence Score" followed strictly by a number
        match = re.search(r"Confidence Score\**\s*\n\s*(\d{1,3})", text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(1.0, max(0.0, score / 100.0))
        return 0.5 # Default if parsing fails

    def _get_rsi_zone(self, rsi):
        if rsi >= 70: return "Overbought"
        if rsi <= 30: return "Oversold"
        return "Neutral"

    def _get_fallback_response(self):
        return {
            "recommendation": "HOLD / WAIT",
            "score": 0.0,
            "insight": "AI unavailable. Based on standard rules: Check RSI and Trend manually.",
            "source": "fallback"
        }

# Singleton accessor
def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

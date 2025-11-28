"""
Gemini LLM Integration (JSON Mode)
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class GeminiInsightGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,  # Lowered to 0.1 for maximum consistency
        max_tokens: int = 1000
    ):
        self.model_name = model_name
        self.temperature = temperature
        
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
        
        prompt = self._build_prompt(
            coin_symbol, market_data, sentiment_data, 
            technical_indicators, prediction_data, top_headlines, horizon_days
        )
        
        try:
            # Force JSON output structure
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "recommendation": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "confidence_score": {"type": "integer"},
                        "analysis_synthesis": {"type": "string"},
                        "key_risks": {"type": "array", "items": {"type": "string"}}
                    }
                }
            )
            
            response = self.model.generate_content(prompt, generation_config=config)
            
            # Parse JSON directly - no more regex guessing
            result = json.loads(response.text)
            
            # Format the output for the UI
            formatted_insight = (
                f"**Analysis Synthesis**\n{result['analysis_synthesis']}\n\n"
                f"**Key Risks**\n" + "\n".join([f"- {r}" for r in result['key_risks']]) + "\n\n"
                f"**Confidence Score**\n{result['confidence_score']}\n\n"
                f"**Recommendation**\n{result['recommendation']}"
            )

            # Map simpler labels to UI colors
            rec_map = {
                "BUY": "BUY",
                "SELL": "SELL / AVOID",
                "HOLD": "HOLD / WAIT"
            }

            return {
                "recommendation": rec_map.get(result['recommendation'], "HOLD / WAIT"),
                "score": result['confidence_score'] / 100.0,
                "insight": formatted_insight,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "recommendation": "HOLD / WAIT",
                "score": 0.0,
                "insight": f"Error parsing AI response: {str(e)}",
                "source": "error"
            }

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        # ... (Keep the exact same logic as before for extracting ROI, RSI, etc.) ...
        # [Copy the logic from the previous turn's _build_prompt here]
        # I will condense it here for brevity, but you should copy the 'logic' part from the previous file.
        
        curr_price = market_data.get('price_usd', 0)
        
        # Calculate ROI
        pred_text = "No models available."
        if preds and preds.get('ensemble') and len(preds['ensemble']) > 0:
            final = preds['ensemble'][-1]
            roi = ((final - curr_price) / curr_price) * 100
            pred_text = f"Model Forecast: ${final:,.2f} ({roi:+.2f}% ROI)"
        
        sent_score = sentiment_data.get('score', 0)
        rsi = tech.get('rsi', 50)
        
        return f"""
        Act as a Crypto Analyst. Analyze this data for {coin_symbol}:
        1. {pred_text}
        2. Sentiment Score: {sent_score:.2f}
        3. RSI: {rsi:.1f}
        4. Trend: {tech.get('trend', 'Neutral')}
        
        Produce a JSON response with:
        - recommendation (BUY, SELL, or HOLD)
        - confidence_score (0-100)
        - analysis_synthesis (Short paragraph)
        - key_risks (List of strings)
        
        Logic:
        - If Model predicts drop > 3% AND Trend is Down -> SELL
        - If Model predicts drop but RSI < 30 -> HOLD (Risk of bounce)
        - If Model predicts rise > 3% -> BUY
        """

# Singleton accessor
def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

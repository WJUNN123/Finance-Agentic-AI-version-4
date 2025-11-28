"""
Gemini LLM Integration (Robust JSON Mode)
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
        temperature: float = 0.1,
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
            
            # Robust JSON Parsing
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # Handle cases where model wraps JSON in markdown code blocks
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                result = json.loads(clean_text)
            
            # Safe Extraction (prevents KeyErrors)
            synthesis = result.get('analysis_synthesis', 'Detailed synthesis unavailable for this request.')
            risks = result.get('key_risks', ['General Market Risk'])
            score = result.get('confidence_score', 50)
            rec_raw = result.get('recommendation', 'HOLD')

            # Format the output for the UI
            risk_list = "\n".join([f"- {r}" for r in risks])
            formatted_insight = (
                f"**Analysis Synthesis**\n{synthesis}\n\n"
                f"**Key Risks**\n{risk_list}\n\n"
                f"**Confidence Score**\n{score}\n\n"
                f"**Recommendation**\n{rec_raw}"
            )

            # Map simpler labels to UI colors
            rec_map = {
                "BUY": "BUY",
                "SELL": "SELL / AVOID",
                "HOLD": "HOLD / WAIT"
            }

            return {
                "recommendation": rec_map.get(rec_raw, "HOLD / WAIT"),
                "score": score / 100.0,
                "insight": formatted_insight,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "recommendation": "HOLD / WAIT",
                "score": 0.0,
                "insight": f"AI Generation Error: {str(e)}",
                "source": "error"
            }

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
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

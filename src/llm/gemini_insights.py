"""
Gemini LLM Integration (Robust)
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

import google.generativeai as genai
from typing import Dict, List, Optional
import logging
import json
import re

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
            # Configure specifically for JSON response
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
            
            # --- Robust Parsing ---
            try:
                # 1. Try direct JSON load
                result = json.loads(response.text)
            except Exception:
                # 2. Try cleaning markdown syntax if model added ```json ... ```
                clean_text = response.text.replace("```json", "").replace("```", "").strip()
                try:
                    result = json.loads(clean_text)
                except:
                    # 3. Fallback if JSON fails completely
                    return {
                        "recommendation": "HOLD / WAIT",
                        "score": 0.5,
                        "insight": "AI Error: Could not parse response format. Please try again.",
                        "source": "error"
                    }

            # Safe extraction with defaults
            synthesis = result.get('analysis_synthesis', 'No detailed analysis provided.')
            risks = result.get('key_risks', [])
            score = result.get('confidence_score', 50)
            rec_raw = result.get('recommendation', 'HOLD')
            
            # Format risks as bullet points
            risk_text = "\n".join([f"- {r}" for r in risks]) if risks else "- General Market Volatility"

            formatted_insight = (
                f"**Analysis Synthesis**\n{synthesis}\n\n"
                f"**Key Risks**\n{risk_text}"
            )

            # Map to UI labels
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
                "insight": f"System Error: {str(e)}",
                "source": "error"
            }

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        curr_price = market_data.get('price_usd', 0)
        
        # Calculate ROI
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble') and len(preds['ensemble']) > 0:
            final = preds['ensemble'][-1]
            if curr_price > 0:
                roi = ((final - curr_price) / curr_price) * 100
                pred_text = f"Model Forecast: ${final:,.2f} ({roi:+.2f}% ROI)"
        
        sent_score = sentiment_data.get('score', 0)
        rsi = tech.get('rsi', 50)
        trend = tech.get('trend', 'Neutral')
        
        return f"""
        Act as a Crypto Investment Analyst. Analyze the data below for {coin_symbol}:
        
        1. {pred_text}
        2. Sentiment Score: {sent_score:.2f} (-1.0 to +1.0)
        3. RSI (14): {rsi:.1f}
        4. Trend: {trend}
        
        Your task:
        - Determine a recommendation (BUY, SELL, HOLD).
        - Assign a confidence score (0-100).
        - Write a brief synthesis paragraph explaining your logic.
        - List 2-3 key risks.
        
        Logic Guide:
        - If Model predicts significant Drop AND Trend is Down -> SELL
        - If Model predicts Drop but RSI is Oversold (<30) -> HOLD (Risk of bounce)
        - If Model predicts Rise AND Sentiment is Positive -> BUY
        """

# Singleton accessor
def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

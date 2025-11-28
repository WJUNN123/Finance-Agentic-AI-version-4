"""
Gemini LLM Integration (Text-to-JSON Regex Mode)
Most reliable method for parsing LLM outputs.
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
        temperature: float = 0.2, # Slight increase to allow better reasoning
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
        
        # 1. Build Prompt
        prompt = self._build_prompt(
            coin_symbol, market_data, sentiment_data, 
            technical_indicators, prediction_data, top_headlines, horizon_days
        )
        
        try:
            # 2. Configure for PLAIN TEXT (More reliable than Schema mode for Flash)
            config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            
            # 3. Generate
            response = self.model.generate_content(prompt, generation_config=config)
            raw_text = response.text
            
            # 4. Clean and Parse JSON Manually
            json_data = self._clean_and_parse_json(raw_text)
            
            if not json_data:
                # If parsing failed, show the RAW text so we know why
                return {
                    "recommendation": "HOLD / WAIT",
                    "score": 0.0,
                    "insight": f"**Parsing Error.**\n\nRaw AI Output:\n{raw_text[:500]}...",
                    "source": "error"
                }

            # 5. Extract Data
            synthesis = json_data.get('analysis_synthesis', 'Analysis unavailable.')
            risks = json_data.get('key_risks', [])
            score = json_data.get('confidence_score', 50)
            rec_raw = json_data.get('recommendation', 'HOLD')
            
            # Format nicely
            risk_list = "\n".join([f"- {r}" for r in risks]) if risks else "- General Volatility"
            
            formatted_insight = (
                f"**Analysis Synthesis**\n{synthesis}\n\n"
                f"**Key Risks**\n{risk_list}"
            )

            # Map to UI
            rec_map = { "BUY": "BUY", "SELL": "SELL / AVOID", "HOLD": "HOLD / WAIT" }

            return {
                "recommendation": rec_map.get(rec_raw, "HOLD / WAIT"),
                "score": float(score) / 100.0,
                "insight": formatted_insight,
                "source": "gemini",
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                "recommendation": "HOLD / WAIT",
                "score": 0.0,
                "insight": f"**API Error:** {str(e)}",
                "source": "error"
            }

    def _clean_and_parse_json(self, text: str) -> Optional[Dict]:
        """ aggressively cleans markdown to extract JSON """
        try:
            # Remove ```json and ``` lines
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            # Last ditch effort: find { and }
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    return json.loads(text[start:end])
            except:
                pass
            return None

    def _build_prompt(self, coin_symbol, market_data, sentiment_data, tech, preds, headlines, horizon):
        curr_price = market_data.get('price_usd', 0)
        
        pred_text = "No predictive models available."
        if preds and preds.get('ensemble') and len(preds['ensemble']) > 0:
            final = preds['ensemble'][-1]
            if curr_price > 0:
                roi = ((final - curr_price) / curr_price) * 100
                pred_text = f"Model Forecast: ${final:,.2f} ({roi:+.2f}% ROI)"
        
        sent_score = sentiment_data.get('score', 0)
        rsi = tech.get('rsi', 50)
        trend = tech.get('trend', 'Neutral')
        
        # We explicitly ask for JSON in the prompt text
        return f"""
        Act as a Crypto Investment Analyst. Analyze this data for {coin_symbol}:
        
        DATA:
        1. {pred_text}
        2. Sentiment Score: {sent_score:.2f} (-1.0 to +1.0)
        3. RSI (14): {rsi:.1f}
        4. Trend: {trend}
        5. News: {str(headlines[:3])}
        
        INSTRUCTIONS:
        Return a valid JSON object with exactly these keys:
        {{
            "recommendation": "BUY" | "SELL" | "HOLD",
            "confidence_score": integer (0-100),
            "analysis_synthesis": "Your concise analysis paragraph here.",
            "key_risks": ["Risk 1", "Risk 2"]
        }}
        
        Do not add any markdown. Return ONLY the JSON string.
        """

def generate_insights(api_key: str, **kwargs) -> Dict:
    generator = GeminiInsightGenerator(api_key=api_key)
    return generator.generate_insights(**kwargs)

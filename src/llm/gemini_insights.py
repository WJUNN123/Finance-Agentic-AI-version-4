"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

from typing import Dict, List, Optional
import logging
import random

logger = logging.getLogger(__name__)

# ============================================================================
# RULE-BASED INSIGHT GENERATOR (STREAMLIT CLOUD COMPATIBLE)
# ============================================================================

class RuleBasedInsightGenerator:
    """
    Generates investment insights using advanced rule-based logic
    Designed to work within Streamlit Cloud free tier limits (1GB RAM)
    """
    
    def __init__(self):
        logger.info("✅ Rule-based insight generator initialized (Streamlit Cloud compatible)")
    
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
        """Generate comprehensive investment insights using rules"""
        
        logger.info(f"🤖 Generating insights for {coin_symbol} using rule-based analysis...")
        
        # Extract data
        current_price = market_data.get('price_usd', 0)
        price_change_24h = market_data.get('pct_change_24h', 0)
        price_change_7d = market_data.get('pct_change_7d', 0)
        volume_24h = market_data.get('volume_24h', 0)
        market_cap = market_data.get('market_cap', 0)
        
        # Predictions
        ensemble_preds = prediction_data.get('ensemble', [])
        lstm_preds = prediction_data.get('lstm', [])
        xgb_preds = prediction_data.get('xgboost', [])
        
        # Calculate expected ROI
        expected_roi = 0
        if ensemble_preds and current_price > 0:
            expected_roi = ((ensemble_preds[-1] - current_price) / current_price) * 100
        
        # Model agreement
        model_agreement = 0.5
        if lstm_preds and xgb_preds and ensemble_preds:
            lstm_final = lstm_preds[-1]
            xgb_final = xgb_preds[-1]
            ensemble_final = ensemble_preds[-1]
            
            if ensemble_final > 0:
                lstm_diff = abs((lstm_final - ensemble_final) / ensemble_final)
                xgb_diff = abs((xgb_final - ensemble_final) / ensemble_final)
                avg_diff = (lstm_diff + xgb_diff) / 2
                model_agreement = max(0.3, min(1.0, 1.0 - avg_diff * 2))
        
        # Technical indicators
        rsi = technical_indicators.get('rsi', 50)
        trend = technical_indicators.get('trend', 'sideways')
        volatility = technical_indicators.get('volatility', 0.05)
        momentum = technical_indicators.get('momentum', 0)
        macd_histogram = technical_indicators.get('macd_histogram', 0)
        stoch_k = technical_indicators.get('stochastic_k', 50)
        bb_position = technical_indicators.get('bb_position', 0.5)
        
        # Sentiment
        sentiment_score = sentiment_data.get('score', 0.0)
        sentiment_breakdown = sentiment_data.get('breakdown', {})
        pos_pct = sentiment_breakdown.get('positive', 0)
        neg_pct = sentiment_breakdown.get('negative', 0)
        neu_pct = sentiment_breakdown.get('neutral', 0)
        
        # Decision logic
        recommendation, confidence, analysis, risks, reasoning, key_factors = self._analyze(
            coin_symbol=coin_symbol,
            expected_roi=expected_roi,
            model_agreement=model_agreement,
            rsi=rsi,
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            macd_histogram=macd_histogram,
            stoch_k=stoch_k,
            bb_position=bb_position,
            sentiment_score=sentiment_score,
            pos_pct=pos_pct,
            neg_pct=neg_pct,
            price_change_24h=price_change_24h,
            price_change_7d=price_change_7d,
            current_price=current_price,
            horizon_days=horizon_days
        )
        
        logger.info(f"✅ Generated: {recommendation} (confidence: {confidence:.2f})")
        
        return {
            "recommendation": recommendation,
            "score": confidence,
            "insight": analysis,
            "risks": risks,
            "reasoning": reasoning,
            "key_factors": key_factors,
            "source": "rule_based",
            "model": "advanced_rules"
        }
    
    def _analyze(
        self,
        coin_symbol: str,
        expected_roi: float,
        model_agreement: float,
        rsi: float,
        trend: str,
        volatility: float,
        momentum: float,
        macd_histogram: float,
        stoch_k: float,
        bb_position: float,
        sentiment_score: float,
        pos_pct: float,
        neg_pct: float,
        price_change_24h: float,
        price_change_7d: float,
        current_price: float,
        horizon_days: int
    ) -> tuple:
        """Advanced rule-based analysis"""
        
        # Initialize scores
        bullish_score = 0
        bearish_score = 0
        
        # === FORECAST ANALYSIS ===
        if expected_roi > 10:
            bullish_score += 3
        elif expected_roi > 5:
            bullish_score += 2
        elif expected_roi > 0:
            bullish_score += 1
        elif expected_roi < -10:
            bearish_score += 3
        elif expected_roi < -5:
            bearish_score += 2
        elif expected_roi < 0:
            bearish_score += 1
        
        # === MODEL AGREEMENT ===
        if model_agreement > 0.9:
            bullish_score += 2 if expected_roi > 0 else 0
            bearish_score += 2 if expected_roi < 0 else 0
        elif model_agreement < 0.5:
            bullish_score -= 1
            bearish_score -= 1
        
        # === RSI ANALYSIS ===
        if rsi < 30:
            bullish_score += 2  # Oversold
        elif rsi < 40:
            bullish_score += 1
        elif rsi > 70:
            bearish_score += 2  # Overbought
        elif rsi > 60:
            bearish_score += 1
        
        # === TREND ANALYSIS ===
        if trend == "uptrend":
            bullish_score += 2
        elif trend == "downtrend":
            bearish_score += 2
        
        # === MOMENTUM ===
        if momentum > 5:
            bullish_score += 1
        elif momentum < -5:
            bearish_score += 1
        
        # === MACD ===
        if macd_histogram > 0:
            bullish_score += 1
        elif macd_histogram < 0:
            bearish_score += 1
        
        # === STOCHASTIC ===
        if stoch_k < 20:
            bullish_score += 1  # Oversold
        elif stoch_k > 80:
            bearish_score += 1  # Overbought
        
        # === BOLLINGER BANDS ===
        if bb_position < 0.2:
            bullish_score += 1  # Near lower band
        elif bb_position > 0.8:
            bearish_score += 1  # Near upper band
        
        # === SENTIMENT ===
        if sentiment_score > 0.3:
            bullish_score += 1
        elif sentiment_score < -0.3:
            bearish_score += 1
        
        if pos_pct > 50:
            bullish_score += 1
        elif neg_pct > 50:
            bearish_score += 1
        
        # === RECENT PRICE ACTION ===
        if price_change_24h > 5:
            bullish_score += 1
        elif price_change_24h < -5:
            bearish_score += 1
        
        # === DECISION ===
        net_score = bullish_score - bearish_score
        
        if net_score >= 4:
            recommendation = "BUY"
            confidence = min(0.85, 0.6 + (net_score * 0.05))
        elif net_score <= -4:
            recommendation = "SELL"
            confidence = min(0.85, 0.6 + (abs(net_score) * 0.05))
        else:
            recommendation = "HOLD"
            confidence = 0.5 + (abs(net_score) * 0.03)
        
        # === GENERATE ANALYSIS ===
        analysis = self._generate_analysis(
            coin_symbol=coin_symbol,
            recommendation=recommendation,
            expected_roi=expected_roi,
            model_agreement=model_agreement,
            rsi=rsi,
            trend=trend,
            sentiment_score=sentiment_score,
            price_change_24h=price_change_24h,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            horizon_days=horizon_days
        )
        
        # === GENERATE RISKS ===
        risks = self._generate_risks(
            volatility=volatility,
            rsi=rsi,
            model_agreement=model_agreement,
            sentiment_score=sentiment_score,
            price_change_24h=price_change_24h
        )
        
        # === REASONING ===
        reasoning = self._generate_reasoning(
            recommendation=recommendation,
            expected_roi=expected_roi,
            rsi=rsi,
            trend=trend,
            bullish_score=bullish_score,
            bearish_score=bearish_score
        )
        
        # === KEY FACTORS ===
        key_factors = self._generate_key_factors(
            expected_roi=expected_roi,
            model_agreement=model_agreement,
            rsi=rsi,
            trend=trend,
            sentiment_score=sentiment_score
        )
        
        return recommendation, confidence, analysis, risks, reasoning, key_factors
    
    def _generate_analysis(
        self, coin_symbol, recommendation, expected_roi, model_agreement,
        rsi, trend, sentiment_score, price_change_24h, bullish_score,
        bearish_score, horizon_days
    ) -> str:
        """Generate natural language analysis"""
        
        if recommendation == "BUY":
            templates = [
                f"{coin_symbol} demonstrates strong bullish momentum with a {expected_roi:+.1f}% forecast over {horizon_days} days. The {model_agreement:.0%} model agreement and {trend} price action support upside potential. Current RSI of {rsi:.0f} indicates room for growth without being overbought.",
                
                f"{coin_symbol} shows positive signals with {expected_roi:+.1f}% expected return. Technical indicators align bullishly with RSI at {rsi:.0f} and {trend} momentum. The strong model consensus ({model_agreement:.0%}) adds confidence to the upward forecast.",
                
                f"Analysis suggests {coin_symbol} has upside potential of {expected_roi:+.1f}% over {horizon_days} days. The {trend} trend combined with RSI {rsi:.0f} provides favorable entry conditions. High model agreement ({model_agreement:.0%}) reinforces the bullish outlook."
            ]
        
        elif recommendation == "SELL":
            templates = [
                f"{coin_symbol} shows bearish indicators with {expected_roi:+.1f}% downside forecast over {horizon_days} days. RSI at {rsi:.0f} suggests overbought conditions, while the {trend} trend confirms selling pressure. Consider reducing exposure to limit downside risk.",
                
                f"Technical analysis indicates weakness in {coin_symbol} with {expected_roi:+.1f}% expected decline. The {trend} trend and RSI {rsi:.0f} signal potential further downside. Model agreement of {model_agreement:.0%} supports the bearish outlook.",
                
                f"{coin_symbol} faces downward pressure with {expected_roi:+.1f}% forecast decline. Current RSI {rsi:.0f} and {trend} momentum suggest caution. Consider protective measures or exit positions to preserve capital."
            ]
        
        else:  # HOLD
            templates = [
                f"{coin_symbol} presents mixed signals with {expected_roi:+.1f}% forecast over {horizon_days} days. While RSI at {rsi:.0f} is neutral, the {trend} trend and recent {price_change_24h:+.1f}% move suggest waiting for clearer directional confirmation before taking new positions.",
                
                f"Analysis shows {coin_symbol} in consolidation with {expected_roi:+.1f}% expected move. Technical indicators provide mixed signals - RSI {rsi:.0f} is neutral while trend shows {trend}. Prudent to wait for stronger confirmation before committing capital.",
                
                f"{coin_symbol} exhibits balanced technical setup with {expected_roi:+.1f}% forecast. The neutral RSI ({rsi:.0f}) and {trend} trend suggest a wait-and-see approach. Consider holding positions or waiting for clearer market direction."
            ]
        
        return random.choice(templates)
    
    def _generate_risks(
        self, volatility, rsi, model_agreement, sentiment_score, price_change_24h
    ) -> List[str]:
        """Generate relevant risk factors"""
        
        risks = []
        
        if volatility > 0.10:
            risks.append("High market volatility - expect significant price swings")
        elif volatility > 0.05:
            risks.append("Elevated volatility may lead to unexpected price movements")
        
        if rsi > 75:
            risks.append("Overbought conditions (RSI > 75) suggest potential pullback risk")
        elif rsi < 25:
            risks.append("Oversold conditions may indicate capitulation or further decline")
        
        if model_agreement < 0.6:
            risks.append(f"Low model agreement ({model_agreement:.0%}) indicates forecast uncertainty")
        
        if abs(sentiment_score) < 0.1:
            risks.append("Neutral sentiment provides limited directional conviction")
        
        if abs(price_change_24h) > 10:
            risks.append(f"Large 24h move ({price_change_24h:+.1f}%) suggests elevated risk")
        
        # Always have at least 2-3 risks
        if len(risks) < 2:
            risks.append("Standard cryptocurrency market risks apply")
        if len(risks) < 3:
            risks.append("External market factors may impact price trajectory")
        
        return risks[:3]  # Return max 3 risks
    
    def _generate_reasoning(
        self, recommendation, expected_roi, rsi, trend, bullish_score, bearish_score
    ) -> str:
        """Generate decision reasoning"""
        
        if recommendation == "BUY":
            return f"Strong bullish signals (score: +{bullish_score-bearish_score}) with {expected_roi:+.1f}% forecast, {trend} trend, and RSI {rsi:.0f} support upside case"
        
        elif recommendation == "SELL":
            return f"Bearish indicators (score: {bullish_score-bearish_score}) with {expected_roi:+.1f}% forecast, {trend} trend, and RSI {rsi:.0f} suggest downside risk"
        
        else:
            return f"Mixed signals (score: {bullish_score-bearish_score}) with {expected_roi:+.1f}% forecast and RSI {rsi:.0f} warrant cautious approach"
    
    def _generate_key_factors(
        self, expected_roi, model_agreement, rsi, trend, sentiment_score
    ) -> List[str]:
        """Generate key supporting factors"""
        
        factors = []
        
        factors.append(f"{model_agreement:.0%} model agreement indicates {'high' if model_agreement > 0.8 else 'moderate'} forecast reliability")
        factors.append(f"{expected_roi:+.1f}% expected return over forecast period")
        factors.append(f"RSI at {rsi:.0f} - {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'} territory")
        
        if abs(sentiment_score) > 0.2:
            factors.append(f"{'Positive' if sentiment_score > 0 else 'Negative'} sentiment shift supports thesis")
        
        factors.append(f"Price trend: {trend}")
        
        return factors[:3]  # Return top 3 factors


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_insights(
    api_key: str,  # Not used, kept for compatibility
    coin_symbol: str,
    market_data: Dict,
    sentiment_data: Dict,
    technical_indicators: Dict,
    prediction_data: Dict,
    top_headlines: List[str],
    horizon_days: int = 7
) -> Dict:
    """
    Generate investment insights using rule-based system
    Perfect for Streamlit Cloud free tier (1GB RAM limit)
    """
    generator = RuleBasedInsightGenerator()
    return generator.generate_insights(
        coin_symbol=coin_symbol,
        market_data=market_data,
        sentiment_data=sentiment_data,
        technical_indicators=technical_indicators,
        prediction_data=prediction_data,
        top_headlines=top_headlines,
        horizon_days=horizon_days
    )
































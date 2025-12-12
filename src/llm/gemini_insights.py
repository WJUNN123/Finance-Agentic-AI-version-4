"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

from typing import Dict, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)

class RuleBasedInsightGenerator:
    """Enhanced rule-based insight generator with robust analysis"""
    
    def __init__(self):
        logger.info("✅ Enhanced rule-based generator initialized")
    
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
        """Generate comprehensive investment insights"""
        
        logger.info(f"🤖 Generating enhanced insights for {coin_symbol}...")
        
        # Extract all data
        current_price = market_data.get('price_usd', 0)
        price_change_24h = market_data.get('pct_change_24h', 0)
        price_change_7d = market_data.get('pct_change_7d', 0)
        volume_24h = market_data.get('volume_24h', 0)
        market_cap = market_data.get('market_cap', 0)
        
        # Predictions
        ensemble_preds = prediction_data.get('ensemble', [])
        lstm_preds = prediction_data.get('lstm', [])
        xgb_preds = prediction_data.get('xgboost', [])
        
        # Calculate metrics
        expected_roi = 0
        predicted_price = current_price
        if ensemble_preds and current_price > 0:
            predicted_price = ensemble_preds[-1]
            expected_roi = ((predicted_price - current_price) / current_price) * 100
        
        # Model agreement
        model_agreement = self._calculate_model_agreement(lstm_preds, xgb_preds, ensemble_preds)
        
        # Technical indicators
        rsi = technical_indicators.get('rsi', 50)
        trend = technical_indicators.get('trend', 'sideways')
        volatility = technical_indicators.get('volatility', 0.05)
        momentum = technical_indicators.get('momentum', 0)
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        macd_histogram = technical_indicators.get('macd_histogram', 0)
        stoch_k = technical_indicators.get('stochastic_k', 50)
        stoch_d = technical_indicators.get('stochastic_d', 50)
        bb_position = technical_indicators.get('bb_position', 0.5)
        
        # Sentiment
        sentiment_score = sentiment_data.get('score', 0.0)
        sentiment_breakdown = sentiment_data.get('breakdown', {})
        sentiment_confidence = sentiment_data.get('confidence', 0.5)
        pos_pct = sentiment_breakdown.get('positive', 0)
        neg_pct = sentiment_breakdown.get('negative', 0)
        neu_pct = sentiment_breakdown.get('neutral', 0)
        
        # Liquidity
        liquidity_ratio = (volume_24h / market_cap * 100) if market_cap > 0 else 0
        
        # Perform comprehensive analysis
        analysis_result = self._comprehensive_analysis(
            coin_symbol=coin_symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            expected_roi=expected_roi,
            model_agreement=model_agreement,
            rsi=rsi,
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            bb_position=bb_position,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            pos_pct=pos_pct,
            neg_pct=neg_pct,
            neu_pct=neu_pct,
            price_change_24h=price_change_24h,
            price_change_7d=price_change_7d,
            liquidity_ratio=liquidity_ratio,
            horizon_days=horizon_days,
            headlines=top_headlines
        )
        
        logger.info(f"✅ Generated: {analysis_result['recommendation']} (confidence: {analysis_result['score']:.2f})")
        
        return analysis_result
    
    def _calculate_model_agreement(
        self, 
        lstm_preds: List, 
        xgb_preds: List, 
        ensemble_preds: List
    ) -> float:
        """Calculate model agreement score"""
        if not (lstm_preds and xgb_preds and ensemble_preds):
            return 0.5
        
        lstm_final = lstm_preds[-1]
        xgb_final = xgb_preds[-1]
        ensemble_final = ensemble_preds[-1]
        
        if ensemble_final == 0:
            return 0.5
        
        lstm_diff = abs((lstm_final - ensemble_final) / ensemble_final)
        xgb_diff = abs((xgb_final - ensemble_final) / ensemble_final)
        avg_diff = (lstm_diff + xgb_diff) / 2
        
        return max(0.3, min(1.0, 1.0 - avg_diff * 2))
    
    def _comprehensive_analysis(
        self,
        coin_symbol: str,
        current_price: float,
        predicted_price: float,
        expected_roi: float,
        model_agreement: float,
        rsi: float,
        trend: str,
        volatility: float,
        momentum: float,
        macd: float,
        macd_signal: float,
        macd_histogram: float,
        stoch_k: float,
        stoch_d: float,
        bb_position: float,
        sentiment_score: float,
        sentiment_confidence: float,
        pos_pct: float,
        neg_pct: float,
        neu_pct: float,
        price_change_24h: float,
        price_change_7d: float,
        liquidity_ratio: float,
        horizon_days: int,
        headlines: List[str]
    ) -> Dict:
        """Perform comprehensive multi-factor analysis"""
        
        # === SCORING SYSTEM ===
        bullish_score = 0
        bearish_score = 0
        factors = []
        
        # 1. FORECAST ANALYSIS (Weight: 30%)
        if expected_roi > 15:
            bullish_score += 3
            factors.append(f"Strong {expected_roi:+.1f}% forecast")
        elif expected_roi > 10:
            bullish_score += 2.5
            factors.append(f"Positive {expected_roi:+.1f}% forecast")
        elif expected_roi > 5:
            bullish_score += 2
            factors.append(f"Moderate {expected_roi:+.1f}% forecast")
        elif expected_roi > 0:
            bullish_score += 1
        elif expected_roi < -15:
            bearish_score += 3
            factors.append(f"Sharp {expected_roi:.1f}% decline forecast")
        elif expected_roi < -10:
            bearish_score += 2.5
        elif expected_roi < -5:
            bearish_score += 2
        elif expected_roi < 0:
            bearish_score += 1
        
        # 2. MODEL CONFIDENCE (Weight: 20%)
        if model_agreement > 0.9:
            confidence_boost = 2
            bullish_score += confidence_boost if expected_roi > 0 else 0
            bearish_score += confidence_boost if expected_roi < 0 else 0
            factors.append(f"{model_agreement:.0%} model consensus")
        elif model_agreement > 0.8:
            confidence_boost = 1.5
            bullish_score += confidence_boost if expected_roi > 0 else 0
            bearish_score += confidence_boost if expected_roi < 0 else 0
        elif model_agreement < 0.5:
            bullish_score -= 1
            bearish_score -= 1
        
        # 3. RSI ANALYSIS (Weight: 15%)
        if rsi < 25:
            bullish_score += 2.5
            factors.append(f"Deeply oversold (RSI {rsi:.0f})")
        elif rsi < 30:
            bullish_score += 2
            factors.append(f"Oversold (RSI {rsi:.0f})")
        elif rsi < 40:
            bullish_score += 1
        elif rsi > 75:
            bearish_score += 2.5
            factors.append(f"Extremely overbought (RSI {rsi:.0f})")
        elif rsi > 70:
            bearish_score += 2
            factors.append(f"Overbought (RSI {rsi:.0f})")
        elif rsi > 60:
            bearish_score += 1
        
        # 4. TREND ANALYSIS (Weight: 15%)
        if trend == "strong_uptrend":
            bullish_score += 2.5
            factors.append("Strong uptrend momentum")
        elif trend == "uptrend":
            bullish_score += 2
            factors.append("Uptrend established")
        elif trend == "strong_downtrend":
            bearish_score += 2.5
            factors.append("Strong downtrend pressure")
        elif trend == "downtrend":
            bearish_score += 2
            factors.append("Downtrend pattern")
        
        # 5. MACD SIGNALS (Weight: 10%)
        if macd > macd_signal and macd_histogram > 0:
            bullish_score += 1.5
            factors.append("Bullish MACD crossover")
        elif macd < macd_signal and macd_histogram < 0:
            bearish_score += 1.5
            factors.append("Bearish MACD crossover")
        
        # 6. STOCHASTIC OSCILLATOR (Weight: 10%)
        if stoch_k < 20 and stoch_k > stoch_d:
            bullish_score += 1.5
            factors.append("Stochastic reversal signal")
        elif stoch_k > 80 and stoch_k < stoch_d:
            bearish_score += 1.5
            factors.append("Stochastic topping signal")
        
        # 7. MOMENTUM (Weight: 10%)
        if momentum > 10:
            bullish_score += 1.5
            factors.append("Strong positive momentum")
        elif momentum > 5:
            bullish_score += 1
        elif momentum < -10:
            bearish_score += 1.5
            factors.append("Strong negative momentum")
        elif momentum < -5:
            bearish_score += 1
        
        # 8. BOLLINGER BANDS (Weight: 5%)
        if bb_position < 0.15:
            bullish_score += 1
            factors.append("Price near lower Bollinger band")
        elif bb_position > 0.85:
            bearish_score += 1
            factors.append("Price near upper Bollinger band")
        
        # 9. SENTIMENT (Weight: 10%)
        if sentiment_confidence > 0.7:
            if sentiment_score > 0.4:
                bullish_score += 1.5
                factors.append(f"Strong positive sentiment ({pos_pct:.0f}%)")
            elif sentiment_score < -0.4:
                bearish_score += 1.5
                factors.append(f"Strong negative sentiment ({neg_pct:.0f}%)")
        
        # 10. RECENT PRICE ACTION (Weight: 5%)
        if price_change_24h > 10:
            bullish_score += 1
        elif price_change_24h > 5:
            bullish_score += 0.5
        elif price_change_24h < -10:
            bearish_score += 1
        elif price_change_24h < -5:
            bearish_score += 0.5
        
        # === DECISION LOGIC ===
        net_score = bullish_score - bearish_score
        
        if net_score >= 5:
            recommendation = "BUY"
            confidence = min(0.90, 0.65 + (net_score * 0.04))
        elif net_score >= 3:
            recommendation = "BUY"
            confidence = min(0.80, 0.60 + (net_score * 0.04))
        elif net_score <= -5:
            recommendation = "SELL"
            confidence = min(0.90, 0.65 + (abs(net_score) * 0.04))
        elif net_score <= -3:
            recommendation = "SELL"
            confidence = min(0.80, 0.60 + (abs(net_score) * 0.04))
        else:
            recommendation = "HOLD"
            confidence = 0.50 + (abs(net_score) * 0.03)
        
        # === GENERATE DETAILED ANALYSIS ===
        analysis = self._generate_detailed_analysis(
            coin_symbol, recommendation, expected_roi, model_agreement,
            rsi, trend, sentiment_score, price_change_24h, bullish_score,
            bearish_score, horizon_days, factors[:3], volatility,
            macd_histogram, stoch_k, bb_position
        )
        
        # === GENERATE RISKS ===
        risks = self._generate_comprehensive_risks(
            volatility, rsi, model_agreement, sentiment_score,
            price_change_24h, expected_roi, liquidity_ratio
        )
        
        # === REASONING ===
        reasoning = self._generate_detailed_reasoning(
            recommendation, expected_roi, rsi, trend, bullish_score,
            bearish_score, model_agreement, sentiment_score
        )
        
        # === KEY FACTORS ===
        key_factors = factors[:4] if len(factors) >= 4 else factors
        
        return {
            "recommendation": recommendation,
            "score": confidence,
            "insight": analysis,
            "risks": risks,
            "reasoning": reasoning,
            "key_factors": key_factors,
            "source": "advanced_rule_based",
            "model": "multi_factor_v2"
        }
    
    def _generate_detailed_analysis(
        self, coin_symbol, recommendation, expected_roi, model_agreement,
        rsi, trend, sentiment_score, price_change_24h, bullish_score,
        bearish_score, horizon_days, key_factors, volatility,
        macd_histogram, stoch_k, bb_position
    ) -> str:
        """Generate detailed natural language analysis"""
        
        # Market condition
        if volatility > 0.10:
            vol_desc = "highly volatile"
        elif volatility > 0.05:
            vol_desc = "moderately volatile"
        else:
            vol_desc = "stable"
        
        # Technical setup
        tech_strength = "strong" if abs(bullish_score - bearish_score) > 5 else "moderate"
        
        if recommendation == "BUY":
            templates = [
                f"{coin_symbol} presents a compelling {tech_strength} bullish opportunity with {expected_roi:+.1f}% upside potential over {horizon_days} days. Technical analysis reveals {key_factors[0] if key_factors else 'positive signals'}, supported by {trend} momentum and RSI at {rsi:.0f} indicating room for appreciation. The {model_agreement:.0%} model consensus reinforces confidence in the forecast. Current market conditions are {vol_desc}, with recent {price_change_24h:+.1f}% price action confirming directional momentum.",
                
                f"Analysis identifies {coin_symbol} as an attractive entry point with {expected_roi:+.1f}% expected return. Key bullish factors include {key_factors[0] if key_factors else 'favorable technicals'} and {trend} price structure. The RSI reading of {rsi:.0f} suggests healthy positioning without overbought concerns. With {model_agreement:.0%} model alignment and {vol_desc} market environment, risk-reward favors long positions at current levels.",
                
                f"{coin_symbol} demonstrates {tech_strength} bullish setup targeting {expected_roi:+.1f}% gains over {horizon_days}-day horizon. Technical confluence shows {key_factors[0] if key_factors else 'positive momentum'}, {trend} trajectory, and RSI {rsi:.0f} supporting upside thesis. The {model_agreement:.0%} model agreement and {price_change_24h:+.1f}% recent performance validate the bullish outlook in this {vol_desc} market phase."
            ]
        
        elif recommendation == "SELL":
            templates = [
                f"{coin_symbol} exhibits concerning {tech_strength} bearish signals with {expected_roi:+.1f}% downside risk over {horizon_days} days. Technical deterioration evident through {key_factors[0] if key_factors else 'negative signals'}, {trend} momentum, and RSI at {rsi:.0f} indicating vulnerability. The {model_agreement:.0%} model consensus supports defensive positioning. In this {vol_desc} environment with {price_change_24h:+.1f}% recent decline, capital preservation takes priority.",
                
                f"Analysis warns of {tech_strength} bearish pressure on {coin_symbol} with {expected_roi:+.1f}% projected decline. Critical factors include {key_factors[0] if key_factors else 'weakening technicals'} and {trend} breakdown. RSI {rsi:.0f} signals potential further weakness ahead. Given {model_agreement:.0%} model alignment and {vol_desc} market conditions, reducing exposure is prudent to limit downside participation.",
                
                f"{coin_symbol} faces {tech_strength} headwinds suggesting {expected_roi:+.1f}% downside over {horizon_days}-day period. Technical analysis shows {key_factors[0] if key_factors else 'deteriorating momentum'}, {trend} pattern, and elevated RSI at {rsi:.0f}. With {model_agreement:.0%} model consensus and {price_change_24h:+.1f}% recent weakness in {vol_desc} markets, defensive strategies are warranted."
            ]
        
        else:  # HOLD
            templates = [
                f"{coin_symbol} presents mixed technical signals warranting a cautious stance. The {expected_roi:+.1f}% forecast over {horizon_days} days suggests limited directional conviction, with RSI at {rsi:.0f} and {trend} momentum providing conflicting guidance. While {model_agreement:.0%} model agreement offers moderate confidence, the {vol_desc} market environment and {price_change_24h:+.1f}% recent action favor patience over commitment. Wait for clearer technical resolution before establishing new positions.",
                
                f"Analysis on {coin_symbol} indicates balanced forces with {expected_roi:+.1f}% projected move offering limited risk-reward clarity. Technical indicators show {key_factors[0] if key_factors else 'neutral signals'} and {trend} structure, while RSI {rsi:.0f} sits in neutral territory. The {model_agreement:.0%} model consensus and {vol_desc} conditions suggest maintaining current positions rather than initiating new trades until stronger directional cues emerge.",
                
                f"{coin_symbol} consolidates with {expected_roi:+.1f}% forecast suggesting sideways action over {horizon_days} days. Mixed technicals - RSI {rsi:.0f}, {trend} pattern, and {key_factors[0] if key_factors else 'conflicting signals'} - recommend patience. In this {vol_desc} environment following {price_change_24h:+.1f}% recent move, await confirmation from {model_agreement:.0%} aligned models before tactical decisions."
            ]
        
        return random.choice(templates)
    
    def _generate_comprehensive_risks(
        self, volatility, rsi, model_agreement, sentiment_score,
        price_change_24h, expected_roi, liquidity_ratio
    ) -> List[str]:
        """Generate comprehensive risk assessment"""
        
        risks = []
        
        # Volatility risks
        if volatility > 0.15:
            risks.append(f"Extreme volatility ({volatility:.1%}) significantly increases position risk and potential slippage")
        elif volatility > 0.10:
            risks.append(f"High volatility ({volatility:.1%}) may cause sharp intraday price swings")
        elif volatility > 0.05:
            risks.append(f"Elevated volatility ({volatility:.1%}) suggests increased market uncertainty")
        
        # Technical risks
        if rsi > 80:
            risks.append(f"Severely overbought (RSI {rsi:.0f}) indicates high reversal risk")
        elif rsi > 75:
            risks.append(f"Overbought conditions (RSI {rsi:.0f}) suggest potential pullback")
        elif rsi < 20:
            risks.append(f"Deeply oversold (RSI {rsi:.0f}) may indicate capitulation or further decline")
        elif rsi < 25:
            risks.append(f"Oversold (RSI {rsi:.0f}) could lead to continued weakness")
        
        # Model uncertainty
        if model_agreement < 0.6:
            risks.append(f"Low model agreement ({model_agreement:.0%}) indicates elevated forecast uncertainty")
        elif model_agreement < 0.7:
            risks.append(f"Moderate model divergence ({model_agreement:.0%}) suggests reduced confidence")
        
        # Sentiment risks
        if abs(sentiment_score) < 0.1 and len(risks) < 3:
            risks.append("Neutral sentiment provides limited directional conviction")
        
        # Liquidity risks
        if liquidity_ratio < 2:
            risks.append(f"Low liquidity ({liquidity_ratio:.1f}%) may impact execution quality")
        
        # Price action risks
        if abs(price_change_24h) > 15:
            risks.append(f"Extreme 24h volatility ({price_change_24h:+.1f}%) increases short-term risk")
        
        # Forecast magnitude risks
        if abs(expected_roi) > 20:
            risks.append(f"Large forecast move ({expected_roi:+.1f}%) carries elevated execution risk")
        
        # Always ensure 3 risks
        while len(risks) < 3:
            default_risks = [
                "Cryptocurrency markets remain subject to regulatory developments",
                "External macroeconomic factors may override technical signals",
                "Market microstructure changes could impact price action",
                "Correlation with broader crypto market may affect independent movement"
            ]
            for risk in default_risks:
                if risk not in risks and len(risks) < 3:
                    risks.append(risk)
        
        return risks[:3]
    
    def _generate_detailed_reasoning(
        self, recommendation, expected_roi, rsi, trend, bullish_score,
        bearish_score, model_agreement, sentiment_score
    ) -> str:
        """Generate detailed reasoning explanation"""
        
        net_score = bullish_score - bearish_score
        
        if recommendation == "BUY":
            strength = "Strong" if net_score >= 5 else "Moderate"
            return (f"{strength} bullish case (score: +{net_score:.1f}) driven by {expected_roi:+.1f}% "
                   f"forecast, {trend} momentum, RSI {rsi:.0f} positioning, and {model_agreement:.0%} "
                   f"model consensus. Sentiment {'supports' if sentiment_score > 0 else 'neutral on'} the thesis.")
        
        elif recommendation == "SELL":
            strength = "Strong" if abs(net_score) >= 5 else "Moderate"
            return (f"{strength} bearish case (score: {net_score:.1f}) indicated by {expected_roi:+.1f}% "
                   f"downside forecast, {trend} pressure, RSI {rsi:.0f} vulnerability, and {model_agreement:.0%} "
                   f"model alignment. Sentiment {'confirms' if sentiment_score < 0 else 'neutral on'} risk.")
        
        else:
            return (f"Balanced signals (score: {net_score:+.1f}) with {expected_roi:+.1f}% forecast, "
                   f"RSI {rsi:.0f} neutral positioning, and {trend} structure suggest awaiting clearer "
                   f"directional confirmation. {model_agreement:.0%} model consensus offers moderate guidance.")


def generate_insights(
    api_key: str,
    coin_symbol: str,
    market_data: Dict,
    sentiment_data: Dict,
    technical_indicators: Dict,
    prediction_data: Dict,
    top_headlines: List[str],
    horizon_days: int = 7
) -> Dict:
    """Generate enhanced investment insights"""
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

















































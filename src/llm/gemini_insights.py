"""
Gemini LLM Integration
Generates AI-powered investment insights using Google's Gemini 2.0 Flash
"""

from typing import Dict, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)

class RuleBasedInsightGenerator:
    """Enhanced rule-based insight generator with robust analysis and safety checks"""
    
    def __init__(self):
        logger.info("✅ Enhanced rule-based generator initialized (SAFE VERSION)")
    
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
        """Generate comprehensive investment insights with safety checks"""
        
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
        
        # Get model agreement (passed from app.py)
        model_agreement = prediction_data.get('model_agreement', 0.5)
        
        # Calculate metrics
        expected_roi = 0
        predicted_price = current_price
        if ensemble_preds and current_price > 0:
            predicted_price = ensemble_preds[-1]
            expected_roi = ((predicted_price - current_price) / current_price) * 100
        
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
        """Perform comprehensive multi-factor analysis with SAFETY CHECKS"""
        
        bullish_score = 0
        bearish_score = 0
        factors = []
        
        # 1. FORECAST ANALYSIS
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
        
        # 2. MODEL CONFIDENCE
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
        
        # 3. RSI ANALYSIS
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
        
        # 4. TREND ANALYSIS
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
        
        # 5. MACD SIGNALS
        if macd > macd_signal and macd_histogram > 0:
            bullish_score += 1.5
            factors.append("Bullish MACD crossover")
        elif macd < macd_signal and macd_histogram < 0:
            bearish_score += 1.5
            factors.append("Bearish MACD crossover")
        
        # 6. STOCHASTIC OSCILLATOR
        if stoch_k < 20 and stoch_k > stoch_d:
            bullish_score += 1.5
            factors.append("Stochastic reversal signal")
        elif stoch_k > 80 and stoch_k < stoch_d:
            bearish_score += 1.5
            factors.append("Stochastic topping signal")
        
        # 7. MOMENTUM
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
        
        # 8. BOLLINGER BANDS
        if bb_position < 0.15:
            bullish_score += 1
            factors.append("Price near lower Bollinger band")
        elif bb_position > 0.85:
            bearish_score += 1
            factors.append("Price near upper Bollinger band")
        
        # 9. SENTIMENT
        if sentiment_confidence > 0.7:
            if sentiment_score > 0.4:
                bullish_score += 1.5
                factors.append(f"Strong positive sentiment ({pos_pct:.0f}%)")
            elif sentiment_score < -0.4:
                bearish_score += 1.5
                factors.append(f"Strong negative sentiment ({neg_pct:.0f}%)")
        
        # 10. RECENT PRICE ACTION
        if price_change_24h > 10:
            bullish_score += 1
        elif price_change_24h > 5:
            bullish_score += 0.5
        elif price_change_24h < -10:
            bearish_score += 1
        elif price_change_24h < -5:
            bearish_score += 0.5
        
        # === INITIAL DECISION ===
        net_score = bullish_score - bearish_score
        
        if net_score >= 5:
            recommendation = "BUY"
            confidence = min(0.90, 0.65 + (net_score * 0.04))
        elif net_score >= 3:
            recommendation = "BUY"
            confidence = min(0.80, 0.60 + (net_score * 0.04))
        elif net_score <= -3:
            recommendation = "SELL"
            confidence = min(0.85, 0.65 + (abs(net_score) * 0.04))
        elif net_score <= -2:
            recommendation = "SELL"
            confidence = min(0.75, 0.60 + (abs(net_score) * 0.04))
        else:
            recommendation = "HOLD"
            confidence = 0.50 + (abs(net_score) * 0.03)
        
        # === SAFETY CHECKS ===
        safety_warnings = []
        
        # Safety Check 1: Low Model Agreement
        if model_agreement < 0.60:
            logger.warning(f"⚠️ Low model agreement ({model_agreement:.0%}) - forcing HOLD")
            recommendation = "HOLD"
            confidence = min(confidence, 0.55)
            safety_warnings.append(f"⚠️ Low model consensus ({model_agreement:.0%})")
            factors.insert(0, f"Low model agreement ({model_agreement:.0%}) reduces conviction")
        
        # Safety Check 2: Minimal Forecast
        if abs(expected_roi) < 2.0:
            logger.info(f"📊 Minimal forecast ({expected_roi:+.1f}%) - forcing HOLD")
            recommendation = "HOLD"
            confidence = min(confidence, 0.60)
            safety_warnings.append(f"Minimal {expected_roi:+.1f}% movement")
        
        # Safety Check 3: Poor Risk/Reward
        if volatility > 0:
            risk_adj_return = expected_roi / (volatility * 100)
            if abs(risk_adj_return) < 0.8 and recommendation != "HOLD":
                logger.info(f"⚠️ Poor risk/reward ({risk_adj_return:.2f}) - forcing HOLD")
                recommendation = "HOLD"
                confidence = min(confidence, 0.65)
                safety_warnings.append(f"Risk/reward ({risk_adj_return:.2f}) unfavorable")
        
        # Safety Check 4: Conflicting Signals
        if trend in ["downtrend", "strong_downtrend"] and expected_roi > 3:
            logger.warning("⚠️ Conflicting: downtrend but bullish forecast")
            confidence = confidence * 0.85
            safety_warnings.append("Conflicting trend signals")
        
        # Safety Check 5: High Volatility + Low Agreement
        if volatility > 0.08 and model_agreement < 0.7:
            logger.warning("⚠️ High volatility with low model agreement")
            confidence = confidence * 0.90
        
        # === GENERATE OUTPUTS ===
        analysis = self._generate_detailed_analysis(
            coin_symbol, recommendation, expected_roi, model_agreement,
            rsi, trend, sentiment_score, price_change_24h, bullish_score,
            bearish_score, horizon_days, factors[:3], volatility,
            macd_histogram, stoch_k, bb_position, safety_warnings
        )
        
        risks = self._generate_comprehensive_risks(
            volatility, rsi, model_agreement, sentiment_score,
            price_change_24h, expected_roi, liquidity_ratio, safety_warnings
        )
        
        reasoning = self._generate_detailed_reasoning(
            recommendation, expected_roi, rsi, trend, bullish_score,
            bearish_score, model_agreement, sentiment_score, price_change_24h
        )
        
        key_factors = factors[:4] if len(factors) >= 4 else factors
        
        return {
            "recommendation": recommendation,
            "score": confidence,
            "insight": analysis,
            "risks": risks,
            "reasoning": reasoning,
            "key_factors": key_factors,
            "source": "advanced_rule_based",
            "model": "multi_factor_v2_safe"
        }
    
    def _generate_detailed_analysis(
        self, coin_symbol, recommendation, expected_roi, model_agreement,
        rsi, trend, sentiment_score, price_change_24h, bullish_score,
        bearish_score, horizon_days, key_factors, volatility,
        macd_histogram, stoch_k, bb_position, safety_warnings
    ) -> str:
        """Generate detailed natural language analysis"""
        
        if volatility > 0.10:
            vol_desc = "highly volatile"
        elif volatility > 0.05:
            vol_desc = "moderately volatile"
        else:
            vol_desc = "stable"
        
        tech_strength = "strong" if abs(bullish_score - bearish_score) > 5 else "moderate"
        
        if safety_warnings:
            warning_text = " CRITICAL: " + "; ".join(safety_warnings) + ". "
        else:
            warning_text = ""
        
        if recommendation == "BUY":
            if price_change_24h < -2:
                daily_context = f"Despite today's {price_change_24h:.1f}% pullback, "
            elif price_change_24h > 2:
                daily_context = f"Building on {price_change_24h:+.1f}% positive momentum, "
            else:
                daily_context = ""
            
            templates = [
                f"{coin_symbol} presents {tech_strength} bullish opportunity with {expected_roi:+.1f}% upside over {horizon_days} days. {warning_text}{daily_context}Technical analysis shows {key_factors[0] if key_factors else 'positive signals'}, supported by {trend} momentum and RSI {rsi:.0f} indicating room for appreciation. Model consensus of {model_agreement:.0%} reinforces forecast confidence in this {vol_desc} environment.",
                
                f"Analysis identifies {coin_symbol} as attractive entry with {expected_roi:+.1f}% expected return. {warning_text}{daily_context}Key bullish factors include {key_factors[0] if key_factors else 'favorable technicals'} and {trend} price structure. RSI {rsi:.0f} suggests healthy positioning. With {model_agreement:.0%} model alignment and {vol_desc} conditions, risk-reward favors long positions.",
            ]
        
        elif recommendation == "SELL":
            if price_change_24h > 2:
                daily_context = f"Despite today's {price_change_24h:+.1f}% bounce, "
            elif price_change_24h < -2:
                daily_context = f"Accelerating from {price_change_24h:.1f}% decline, "
            else:
                daily_context = ""
            
            templates = [
                f"{coin_symbol} exhibits {tech_strength} bearish pressure with {expected_roi:+.1f}% downside risk over {horizon_days} days. {warning_text}{daily_context}Technical deterioration: {key_factors[0] if key_factors else 'negative signals'}, {trend} momentum, RSI {rsi:.0f} vulnerability. Model consensus {model_agreement:.0%} supports defensive positioning in this {vol_desc} environment.",
                
                f"Analysis warns of {tech_strength} bearish setup for {coin_symbol} with {expected_roi:+.1f}% projected decline. {warning_text}{daily_context}Critical factors: {key_factors[0] if key_factors else 'weakening technicals'}, {trend} breakdown, RSI {rsi:.0f} signaling further weakness. {model_agreement:.0%} model alignment suggests reducing exposure.",
            ]
        
        else:  # HOLD
            if model_agreement < 0.6:
                hold_reason = f"low model agreement ({model_agreement:.0%}) creates high uncertainty"
            elif abs(expected_roi) < 2:
                hold_reason = f"minimal {expected_roi:+.1f}% forecast offers limited opportunity"
            elif volatility > 0.08:
                hold_reason = f"elevated volatility ({volatility:.1%}) increases risk"
            else:
                hold_reason = "mixed technical signals warrant caution"
            
            templates = [
                f"{coin_symbol} presents balanced forces with {expected_roi:+.1f}% forecast warranting cautious stance. {warning_text}The {hold_reason}. RSI {rsi:.0f} and {trend} structure provide conflicting guidance. Wait for clearer directional confirmation with {model_agreement:.0%} model consensus before committing capital in this {vol_desc} environment.",
                
                f"Analysis on {coin_symbol} indicates {hold_reason} with {expected_roi:+.1f}% projected move. {warning_text}Technical indicators show {key_factors[0] if key_factors else 'neutral signals'} and {trend} structure. RSI {rsi:.0f} sits in neutral territory. The {model_agreement:.0%} model consensus suggests maintaining current positions rather than new trades.",
            ]
        
        return random.choice(templates)
    
    def _generate_detailed_reasoning(
        self, recommendation, expected_roi, rsi, trend, bullish_score,
        bearish_score, model_agreement, sentiment_score, price_change_24h
    ) -> str:
        """Generate detailed reasoning explanation"""
        
        net_score = bullish_score - bearish_score
        
        if recommendation == "BUY":
            strength = "Strong" if net_score >= 5 else "Moderate"
            
            if price_change_24h < -2:
                daily_note = f"Despite {price_change_24h:.1f}% pullback, "
            elif price_change_24h > 2:
                daily_note = f"Positive {price_change_24h:+.1f}% momentum and "
            else:
                daily_note = ""
            
            return (f"{strength} bullish case (score: +{net_score:.1f}). {daily_note}"
                   f"{expected_roi:+.1f}% forecast supported by {trend} structure, "
                   f"RSI {rsi:.0f} positioning, and {model_agreement:.0%} model consensus.")
        
        elif recommendation == "SELL":
            strength = "Strong" if abs(net_score) >= 5 else "Moderate"
            
            if price_change_24h > 2:
                daily_note = f"Despite {price_change_24h:+.1f}% bounce, "
            elif price_change_24h < -2:
                daily_note = f"Accelerating {price_change_24h:.1f}% decline confirms "
            else:
                daily_note = ""
            
            return (f"{strength} bearish case (score: {net_score:.1f}). {daily_note}"
                   f"{expected_roi:+.1f}% downside forecast indicated by {trend} pressure, "
                   f"RSI {rsi:.0f} vulnerability, and {model_agreement:.0%} model alignment.")
        
        else:
            if model_agreement < 0.6:
                hold_reason = f"low model agreement ({model_agreement:.0%}) creates uncertainty"
            elif abs(expected_roi) < 2:
                hold_reason = f"minimal {expected_roi:+.1f}% forecast lacks conviction"
            else:
                hold_reason = f"mixed signals (score: {net_score:+.1f})"
            
            return (f"Neutral stance warranted due to {hold_reason}. "
                   f"RSI {rsi:.0f} neutral, {trend} structure, {price_change_24h:+.1f}% recent move. "
                   f"Await clearer directional confirmation.")
    
    def _generate_comprehensive_risks(
        self, volatility, rsi, model_agreement, sentiment_score,
        price_change_24h, expected_roi, liquidity_ratio, safety_warnings
    ) -> List[str]:
        """Generate comprehensive risk assessment"""
        
        risks = []
        
        if safety_warnings:
            for warning in safety_warnings[:2]:
                if warning not in risks:
                    risks.append(warning)
        
        if volatility > 0.15:
            risks.append(f"Extreme volatility ({volatility:.1%}) significantly increases position risk")
        elif volatility > 0.10:
            risks.append(f"High volatility ({volatility:.1%}) may cause sharp price swings")
        elif volatility > 0.05 and len(risks) < 3:
            risks.append(f"Elevated volatility ({volatility:.1%}) suggests increased uncertainty")
        
        if rsi > 80:
            risks.append(f"Severely overbought (RSI {rsi:.0f}) indicates high reversal risk")
        elif rsi > 75 and len(risks) < 3:
            risks.append(f"Overbought conditions (RSI {rsi:.0f}) suggest potential pullback")
        elif rsi < 20:
            risks.append(f"Deeply oversold (RSI {rsi:.0f}) may indicate capitulation risk")
        elif rsi < 25 and len(risks) < 3:
            risks.append(f"Oversold (RSI {rsi:.0f}) could lead to continued weakness")
        
        if model_agreement < 0.6 and len(risks) < 3:
            if f"Low model consensus ({model_agreement:.0%})" not in str(risks):
                risks.append(f"Low model agreement ({model_agreement:.0%}) indicates forecast uncertainty")
        elif model_agreement < 0.7 and len(risks) < 3:
            risks.append(f"Moderate model divergence ({model_agreement:.0%}) reduces confidence")
        
        if abs(sentiment_score) < 0.1 and len(risks) < 3:
            risks.append("Neutral sentiment provides limited directional conviction")
        
        if liquidity_ratio < 1.0 and len(risks) < 3:
            risks.append(f"Low liquidity ({liquidity_ratio:.1f}%) may impact execution quality")
        
        if abs(price_change_24h) > 15 and len(risks) < 3:
            risks.append(f"Extreme 24h volatility ({price_change_24h:+.1f}%) increases short-term risk")
        
        while len(risks) < 3:
            default_risks = [
                "Cryptocurrency markets remain subject to regulatory developments",
                "External macroeconomic factors may override technical signals",
                "Market microstructure changes could impact price action"
            ]
            for risk in default_risks:
                if risk not in risks and len(risks) < 3:
                    risks.append(risk)
                    break
        
        return risks[:3]


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
    """Generate investment insights using rule-based system"""
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

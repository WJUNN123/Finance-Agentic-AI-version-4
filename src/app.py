"""
Crypto Market Analysis Agent - Main Application
Streamlit-based interactive cryptocurrency analysis tool
"""

# ============================================================================
# NEW IMPORTS - Add to top of app.py
# ============================================================================

from models.enhanced_hybrid_predictor import train_and_predict_enhanced
from sentiment.enhanced_analyzer import get_analyzer as get_enhanced_analyzer
from llm.enhanced_gemini_insights import generate_insights_enhanced
from risk.enhanced_risk_assessment import EnhancedRiskAssessor

# Keep existing imports...
import streamlit as st
import pandas as pd
import numpy as np
import logging
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import altair as alt
from datetime import datetime

# ============================================================================
# UPDATED: analyze_cryptocurrency() function
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def analyze_cryptocurrency_enhanced(
    coin_id: str,
    horizon_days: int = 7,
    enable_backtesting: bool = False
) -> Dict:
    """
    Enhanced analysis function with all improvements
    
    Args:
        coin_id: CoinGecko coin ID
        horizon_days: Forecast horizon
        enable_backtesting: Run backtesting (disabled by default for free tier)
        
    Returns:
        Dictionary with enhanced analysis results
    """
    logger.info(f"Starting enhanced analysis for {coin_id}")
    
    try:
        # Get coin info
        coin_info = next((c for c in COINS if c['id'] == coin_id), None)
        if not coin_info:
            return {'error': f'Unknown coin: {coin_id}'}
        
        coin_symbol = coin_info['symbol']
        
        # ====================================================================
        # 1. FETCH MARKET DATA
        # ====================================================================
        logger.info("Fetching market data...")
        cg_fetcher = get_cg_fetcher()
        
        market_df = cg_fetcher.get_market_data([coin_id])
        if market_df.empty:
            return {'error': f'No market data for {coin_id}'}
        
        market_row = market_df.iloc[0]
        market_data = {
            'coin': coin_id,
            'symbol': coin_symbol.upper(),
            'price_usd': float(market_row.get('current_price', 0)),
            'pct_change_24h': float(market_row.get('price_change_percentage_24h', 0)),
            'pct_change_7d': float(market_row.get('price_change_percentage_7d_in_currency', 0)),
            'market_cap': float(market_row.get('market_cap', 0)),
            'volume_24h': float(market_row.get('total_volume', 0))
        }
        
        # Get historical data
        historical_df = cg_fetcher.get_historical_data(coin_id, days=180)
        if historical_df.empty or 'price' not in historical_df.columns:
            return {'error': 'Insufficient historical data'}
        
        price_series = historical_df['price']
        
        # ====================================================================
        # 2. CALCULATE TECHNICAL INDICATORS
        # ====================================================================
        logger.info("Calculating technical indicators...")
        technical_indicators = get_all_indicators(
            price_series,
            pct_24h=market_data['pct_change_24h'],
            pct_7d=market_data['pct_change_7d']
        )
        
        # ====================================================================
        # 3. FETCH AND ANALYZE NEWS
        # ====================================================================
        logger.info("Fetching and analyzing news...")
        news_fetcher = get_news_fetcher()
        
        articles_symbol = news_fetcher.fetch_articles(coin_symbol, max_total=25)
        articles_name = news_fetcher.fetch_articles(coin_id, max_total=25)
        
        all_articles = {a['title']: a for a in (articles_symbol + articles_name)}
        articles = list(all_articles.values())[:50]
        headlines = [a['title'] for a in articles if a.get('title')]
        
        # ====================================================================
        # 4. ENHANCED SENTIMENT ANALYSIS
        # ====================================================================
        logger.info("Analyzing sentiment with trend detection...")
        sentiment_analyzer = get_enhanced_analyzer()  # Enhanced version
        
        if headlines:
            sources = [a.get('source', 'unknown') for a in articles]
            sentiment_results = sentiment_analyzer.analyze_texts(headlines, sources)
            sentiment_score, sentiment_df = sentiment_analyzer.calculate_aggregate_sentiment(
                sentiment_results,
                use_weighting=True  # NEW: Source weighting
            )
            sentiment_breakdown = sentiment_analyzer.get_sentiment_breakdown(sentiment_results)
            
            # NEW: Get sentiment insights
            sentiment_insights = sentiment_analyzer.get_sentiment_insights(
                sentiment_results,
                sentiment_breakdown
            )
        else:
            sentiment_score = 0.0
            sentiment_df = pd.DataFrame()
            sentiment_breakdown = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
            sentiment_insights = {
                'consensus': 'no_data',
                'confidence': 0.0,
                'trend_direction': 'unknown'
            }
        
        # ====================================================================
        # 5. ENHANCED PRICE PREDICTION with Confidence Intervals
        # ====================================================================
        logger.info("Training enhanced models with confidence intervals...")
        
        try:
            predictions = train_and_predict_enhanced(
                price_series,
                horizon=horizon_days,
                window_size=CONFIG['models']['lstm']['window_size'],
                enable_backtest=enable_backtesting  # Disable for free tier
            )
            
            lstm_preds = predictions['lstm']
            xgb_preds = predictions['xgboost']
            ensemble_preds = predictions['ensemble']
            predictions_with_ci = predictions.get('predictions_with_ci', [])
            backtest_results = predictions.get('backtest')
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            lstm_preds = []
            xgb_preds = []
            ensemble_preds = []
            predictions_with_ci = []
            backtest_results = None
        
        # Build forecast table with confidence intervals
        forecast_table = []
        last_date = historical_df.index[-1]
        
        for i in range(horizon_days):
            forecast_date = last_date + pd.Timedelta(days=i+1)
            ci_data = predictions_with_ci[i] if i < len(predictions_with_ci) else {}
            
            forecast_table.append({
                'day': i + 1,
                'date': forecast_date,
                'lstm': lstm_preds[i] if i < len(lstm_preds) else None,
                'xgboost': xgb_preds[i] if i < len(xgb_preds) else None,
                'ensemble': ensemble_preds[i] if i < len(ensemble_preds) else None,
                'lower_ci': ci_data.get('lower_ci'),
                'upper_ci': ci_data.get('upper_ci'),
                'margin': ci_data.get('margin')
            })
        
        # ====================================================================
        # 6. ENHANCED RISK ASSESSMENT
        # ====================================================================
        logger.info("Assessing comprehensive risk profile...")
        risk_assessor = EnhancedRiskAssessor()
        
        risk_assessment = risk_assessor.assess_full_risk_profile(
            market_data=market_data,
            technical_indicators=technical_indicators,
            sentiment_data={
                'score': sentiment_score,
                'breakdown': sentiment_breakdown,
                'is_extreme': sentiment_insights.get('is_extreme', False),
                'trend_direction': sentiment_insights.get('trend_direction')
            },
            model_predictions={
                'lstm': lstm_preds,
                'xgboost': xgb_preds,
                'ensemble': ensemble_preds
            },
            prices=price_series
        )
        
        # ====================================================================
        # 7. ENHANCED GEMINI INSIGHTS with Multi-Stage Reasoning
        # ====================================================================
        logger.info("Generating AI insights with scenario analysis...")
        
        if API_KEYS.get('gemini'):
            try:
                insights = generate_insights_enhanced(
                    api_key=API_KEYS['gemini'],
                    coin_symbol=coin_symbol,
                    market_data=market_data,
                    sentiment_data={
                        'score': sentiment_score,
                        'breakdown': sentiment_breakdown,
                        **sentiment_insights
                    },
                    technical_indicators=technical_indicators,
                    prediction_data={
                        'lstm': lstm_preds,
                        'xgboost': xgb_preds,
                        'ensemble': ensemble_preds
                    },
                    risk_assessment=risk_assessment,
                    top_headlines=headlines[:5],
                    horizon_days=horizon_days
                )
                
                # Apply risk-adjusted confidence
                adjusted_confidence = risk_assessor.get_risk_adjusted_confidence(
                    insights.get('confidence_score', 0.5)
                )
                insights['risk_adjusted_confidence'] = adjusted_confidence
                
            except Exception as e:
                logger.error(f"Gemini insights error: {e}")
                insights = {
                    'recommendation': 'HOLD',
                    'confidence_score': 0.5,
                    'risk_adjusted_confidence': 0.4,
                    'reasoning': f'Error: {str(e)}',
                    'source': 'error'
                }
        else:
            insights = {
                'recommendation': 'HOLD',
                'confidence_score': 0.5,
                'risk_adjusted_confidence': 0.4,
                'reasoning': 'Gemini API key not configured',
                'source': 'no_api_key'
            }
        
        # ====================================================================
        # 8. COMPILE RESULTS
        # ====================================================================
        logger.info("Analysis complete!")
        
        return {
            'market': market_data,
            'technical': technical_indicators,
            'history': historical_df,
            'articles': articles,
            'headlines': headlines,
            'sentiment_score': sentiment_score,
            'sentiment_breakdown': sentiment_breakdown,
            'sentiment_insights': sentiment_insights,
            'sentiment_details': sentiment_df,
            'forecast_table': forecast_table,
            'predictions': {
                'lstm': lstm_preds,
                'xgboost': xgb_preds,
                'ensemble': ensemble_preds
            },
            'insights': insights,
            'risk_assessment': risk_assessment,
            'backtest_results': backtest_results,
            'coin_info': coin_info
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return {'error': str(e)}


# ============================================================================
# NEW: Enhanced Dashboard Rendering
# ============================================================================

def render_risk_assessment(risk_assessment: Dict):
    """NEW: Render risk assessment section"""
    st.subheader("⚠️ Risk Assessment & Market Regime")
    
    # Overall risk gauge
    overall_risk = risk_assessment.get('overall_score', 0.5)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Overall Risk Score", f"{overall_risk:.1%}")
        
        # Risk color
        if overall_risk < 0.35:
            st.success("🟢 Low Risk")
        elif overall_risk < 0.65:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
    
    with col2:
        st.progress(overall_risk, text=f"Risk Level: {overall_risk:.0%}")
    
    # Market regime
    regime = risk_assessment.get('regime')
    regime_desc = risk_assessment.get('regime_description', 'Unknown')
    st.info(regime_desc)
    
    # Top risks
    st.markdown("**Top Risk Factors:**")
    top_risks = risk_assessment.get('top_risks', [])
    for risk in top_risks[:3]:
        icon = risk.get_icon()
        severity = risk.get_severity_label()
        st.write(f"{icon} **{severity}** - {risk.description}")
        st.caption(f"  → {risk.mitigation}")


def render_confidence_intervals(forecast_table: List[Dict]):
    """NEW: Render forecast with confidence intervals"""
    st.subheader("📊 Forecast with Confidence Intervals")
    
    if not forecast_table:
        st.info("No forecast data available")
        return
    
    # Build display dataframe
    forecast_data = []
    for row in forecast_table:
        ensemble = row.get('ensemble')
        lower = row.get('lower_ci')
        upper = row.get('upper_ci')
        
        if ensemble and lower and upper:
            ci_width = upper - lower
            forecast_data.append({
                'Day': row['day'],
                'Date': row['date'].strftime('%Y-%m-%d'),
                'Forecast': f"${ensemble:,.0f}",
                'Lower CI': f"${lower:,.0f}",
                'Upper CI': f"${upper:,.0f}",
                'Margin': f"±${ci_width/2:,.0f}"
            })
    
    if forecast_data:
        st.dataframe(forecast_data, use_container_width=True)
        
        # Visualization
        plot_df = pd.DataFrame(forecast_data)
        st.caption("Confidence intervals widen for longer horizons (normal behavior)")
    else:
        st.info("Insufficient data for confidence intervals")


def render_scenarios(scenarios: List[Dict]):
    """NEW: Render bull/bear/base scenarios"""
    st.subheader("🎯 Scenario Analysis")
    
    if not scenarios:
        return
    
    cols = st.columns(len(scenarios))
    
    for col, scenario in zip(cols, scenarios):
        with col:
            scenario_type = scenario.get('type', 'unknown').upper()
            data = scenario.get('data', {})
            
            # Set color
            if scenario_type == 'BULL':
                icon = "📈"
                color = "green"
            elif scenario_type == 'BEAR':
                icon = "📉"
                color = "red"
            else:
                icon = "➡️"
                color = "blue"
            
            st.markdown(f"### {icon} {scenario_type}")
            st.write(data.get('thesis', 'N/A'))
            
            prob = data.get('probability', 0)
            st.progress(prob, text=f"Probability: {prob:.0%}")
            
            # Key levels
            if scenario_type == 'BULL':
                target = data.get('target_price')
                if target:
                    st.metric("Target", f"${target:,.0f}")
            elif scenario_type == 'BEAR':
                sl = data.get('stop_loss')
                if sl:
                    st.metric("Stop Loss", f"${sl:,.0f}")
            else:
                timeline = data.get('timeline_days')
                if timeline:
                    st.caption(f"Timeline: {timeline} days")


def render_sentiment_trend(sentiment_insights: Dict):
    """NEW: Render sentiment analysis with trend"""
    st.subheader("📊 News Sentiment Analysis")
    
    consensus = sentiment_insights.get('consensus', 'unknown')
    confidence = sentiment_insights.get('confidence', 0)
    trend = sentiment_insights.get('trend_direction', 'unknown')
    is_extreme = sentiment_insights.get('is_extreme', False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consensus_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(consensus, "❓")
        st.metric("Consensus", f"{consensus_emoji} {consensus.upper()}")
        st.caption(f"Confidence: {confidence:.0%}")
    
    with col2:
        trend_emoji = {"improving": "📈", "deteriorating": "📉", "stable": "➡️"}.get(trend, "❓")
        st.metric("Trend", f"{trend_emoji} {trend.upper()}")
    
    with col3:
        if is_extreme:
            st.warning("⚠️ EXTREME SENTIMENT")
            st.caption("High mean reversion risk")
        else:
            st.success("✓ Normal levels")
    
    # Warning if provided
    warning = sentiment_insights.get('warning')
    if warning:
        st.info(warning)


# ============================================================================
# UPDATED: Main render function - Call new functions
# ============================================================================

def render_enhanced_dashboard(result: Dict, horizon_days: int):
    """Render enhanced summary dashboard"""
    
    market = result['market']
    technical = result['technical']
    sentiment_insights = result.get('sentiment_insights', {})
    risk_assessment = result.get('risk_assessment', {})
    insights = result['insights']
    forecast_table = result.get('forecast_table', [])
    
    coin_name = result['coin_info']['name']
    symbol = market['symbol']
    price = market['price_usd']
    
    # Header
    st.markdown(f"### 🪙 {coin_name} ({symbol})")
    
    # Price + metrics
    cols = st.columns([1.5, 1.2, 1.2, 1.2])
    with cols[0]:
        st.markdown("**Price**")
        st.markdown(f"<span style='font-size:2rem;font-weight:800'>${price:,.2f}</span>", unsafe_allow_html=True)
    
    with cols[1]:
        st.metric("Market Cap", format_money(market.get('market_cap', 0)))
    with cols[2]:
        st.metric("24h Volume", format_money(market.get('volume_24h', 0)))
    with cols[3]:
        st.metric("RSI (14)", f"{technical.get('rsi', 50):.1f}")
    
    st.divider()
    
    # NEW: Risk Assessment
    render_risk_assessment(risk_assessment)
    st.divider()
    
    # NEW: Sentiment with insights
    render_sentiment_trend(sentiment_insights)
    st.divider()
    
    # Recommendation
    rec = insights.get('recommendation', 'HOLD').upper()
    confidence = insights.get('risk_adjusted_confidence', insights.get('confidence_score', 0.5))
    
    st.subheader(f"✅ Recommendation: {rec}")
    st.progress(confidence, text=f"Confidence: {confidence:.0%}")
    st.write(insights.get('reasoning', ''))
    
    st.divider()
    
    # NEW: Scenarios
    if insights.get('scenarios'):
        render_scenarios(insights.get('scenarios', []))
        st.divider()
    
    # NEW: Confidence intervals
    render_confidence_intervals(forecast_table)
    st.divider()
    
    # Catalysts & Action Items
    if insights.get('catalysts'):
        st.subheader("🔥 Key Catalysts")
        for catalyst in insights.get('catalysts', []):
            st.write(f"• {catalyst}")
    
    if insights.get('action_items'):
        st.subheader("📋 Action Items")
        for item in insights.get('action_items', []):
            st.write(f"• {item}")


# ============================================================================
# UPDATED: Main function - Use enhanced analyzer
# ============================================================================

def main():
    """Updated main app"""
    st.set_page_config(page_title="Crypto Analysis Agent", page_icon="🗣️", layout="wide")
    
    st.markdown("""
    <div style='display:flex;align-items:center;gap:0.6rem;'>
        <h1 style='margin:0;'>🚀 Enhanced Crypto Analysis Agent</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("AI-powered cryptocurrency analysis with **confidence intervals**, **risk assessment**, and **scenario analysis**.")
    
    # Session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Input
    user_message = st.text_input("Your Question", placeholder="e.g., 'Bitcoin forecast' or 'Should I buy ETH?'")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if analyze_button and user_message.strip():
        with st.spinner("Analyzing... This may take 30-60 seconds..."):
            parsed = parse_user_message(user_message)
            result = analyze_cryptocurrency_enhanced(
                coin_id=parsed['coin_id'],
                horizon_days=parsed['horizon_days'],
                enable_backtesting=False  # Disabled for free tier
            )
            
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.session_state.last_result = result
    
    # Display results
    if st.session_state.last_result:
        st.markdown("---")
        render_enhanced_dashboard(
            st.session_state.last_result,
            st.session_state.last_result.get('horizon_days', 7)
        )


if __name__ == "__main__":
    main()

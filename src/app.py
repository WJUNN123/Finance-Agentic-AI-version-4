"""
Crypto Market Analysis Agent - Main Application
Streamlit-based interactive cryptocurrency analysis tool
"""

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from data_fetchers.coingecko import get_fetcher as get_cg_fetcher
from data_fetchers.news import get_fetcher as get_news_fetcher
from sentiment.analyzer import get_analyzer
from models.hybrid_predictor import train_and_predict
from llm.gemini_insights import generate_insights
from utils.technical_indicators import get_all_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config() -> dict:
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return get_default_config()

def get_default_config() -> dict:
    return {
        'cryptocurrencies': [
            {'name': 'Bitcoin', 'id': 'bitcoin', 'symbol': 'BTC'},
            {'name': 'Ethereum', 'id': 'ethereum', 'symbol': 'ETH'},
            {'name': 'Solana', 'id': 'solana', 'symbol': 'SOL'},
        ],
        'models': {
            'lstm': {'window_size': 30, 'epochs': 15, 'batch_size': 16},
        }
    }

CONFIG = load_config()
COINS = CONFIG.get('cryptocurrencies', [])
COIN_NAME_TO_ID = {c['name'].lower(): c['id'] for c in COINS}
COIN_SYMBOL_TO_ID = {c['symbol'].lower(): c['id'] for c in COINS}

# ============================================================================
# NEW: RISK ENGINE
# ============================================================================
def calculate_risk_assessment(
    tech: Dict, 
    sentiment: Dict, 
    pred_risk: float, 
    market: Dict
) -> Dict:
    """
    Multi-factor Risk Assessment Engine
    """
    risks = []
    score = 0  # 0 (Safe) to 10 (Extreme Risk)
    
    # 1. Technical Risk (RSI)
    rsi = tech.get('rsi', 50)
    if rsi > 75: 
        risks.append("Extreme Overbought (RSI > 75)")
        score += 3
    elif rsi < 25:
        risks.append("Extreme Oversold (RSI < 25)")
        score += 2
        
    # 2. Volatility Risk
    vol = tech.get('volatility', 0)
    if vol > 80:
        risks.append(f"High Volatility ({vol:.0f}%)")
        score += 2
        
    # 3. Sentiment Risk (Contrarian Indicator)
    sent_score = sentiment.get('score', 0)
    if sent_score > 0.6:
        risks.append("Euphoric Sentiment (Contrarian Risk)")
        score += 2
    elif sent_score < -0.6:
        risks.append("Extreme Fear (Panic Selling)")
        score += 1
        
    # 4. Prediction Risk (Model Disagreement)
    if pred_risk > 0.05: # >5% disagreement
        risks.append("High Model Uncertainty")
        score += 2
        
    # 5. Liquidity Risk
    mcap = market.get('market_cap', 0)
    vol_24h = market.get('volume_24h', 0)
    if mcap > 0 and (vol_24h / mcap) < 0.02:
        risks.append("Low Liquidity (<2% Turnover)")
        score += 2

    # Normalize Score
    risk_level = "LOW"
    if score >= 7: risk_level = "CRITICAL"
    elif score >= 4: risk_level = "HIGH"
    elif score >= 2: risk_level = "MODERATE"
    
    return {
        "level": risk_level,
        "score": min(score, 10),
        "factors": risks
    }

# ============================================================================
# ANALYSIS LOGIC
# ============================================================================
def parse_user_message(message: str) -> Dict:
    import re
    msg_lower = message.lower()
    coin_id = "bitcoin" # Default
    
    for name, cid in COIN_NAME_TO_ID.items():
        if name in msg_lower: coin_id = cid
    for sym, cid in COIN_SYMBOL_TO_ID.items():
        if re.search(r'\b' + re.escape(sym) + r'\b', msg_lower): coin_id = cid
            
    horizon_days = 7
    m = re.search(r'(\d+)\s*(day|days|d)\b', msg_lower)
    if m: horizon_days = int(m.group(1))
    
    return {'coin_id': coin_id, 'horizon_days': horizon_days}

@st.cache_data(ttl=300)
def analyze_cryptocurrency(coin_id: str, horizon_days: int = 7) -> Dict:
    logger.info(f"Starting analysis for {coin_id}")
    
    # 1. Fetch Market Data
    cg_fetcher = get_cg_fetcher()
    market_df = cg_fetcher.get_market_data([coin_id])
    if market_df.empty: return {'error': f'No data for {coin_id}'}
    
    row = market_df.iloc[0]
    market_data = {
        'coin': coin_id,
        'symbol': row.get('symbol', '').upper(),
        'price_usd': float(row.get('current_price', 0)),
        'pct_change_24h': float(row.get('price_change_percentage_24h', 0)),
        'pct_change_7d': float(row.get('price_change_percentage_7d_in_currency', 0)),
        'market_cap': float(row.get('market_cap', 0)),
        'volume_24h': float(row.get('total_volume', 0))
    }
    
    # 2. Historical Data
    hist_df = cg_fetcher.get_historical_data(coin_id, days=180)
    if hist_df.empty: return {'error': 'Insufficient history'}
    price_series = hist_df['price']
    
    # 3. Technicals (Enhanced)
    technical_indicators = get_all_indicators(price_series)
    
    # 4. Sentiment (Enhanced)
    news_fetcher = get_news_fetcher()
    analyzer = get_analyzer()
    # Fetch articles using both ID and Symbol for better coverage
    articles = news_fetcher.fetch_articles(coin_id, max_total=20)
    sentiment_result = analyzer.analyze_news(articles)
    
    # 5. Prediction (Enhanced with CI)
    pred_result = train_and_predict(price_series, horizon_days)
    
    # 6. Risk Assessment (NEW)
    risk_profile = calculate_risk_assessment(
        tech=technical_indicators,
        sentiment=sentiment_result,
        pred_risk=pred_result['risk_metrics']['model_disagreement'],
        market=market_data
    )
    
    # 7. AI Insights
    # Prepare simplified context for LLM to save tokens
    headlines = [a['title'] for a in articles[:5]]
    insights = generate_insights(
        api_key=st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
        coin_symbol=market_data['symbol'],
        market_data=market_data,
        sentiment_data={'score': sentiment_result['score'], 'label': sentiment_result['interpretation']},
        technical_indicators=technical_indicators,
        prediction_data={'ensemble_mean': pred_result['ensemble']},
        top_headlines=headlines,
        horizon_days=horizon_days
    )
    
    return {
        'market': market_data,
        'technical': technical_indicators,
        'sentiment': sentiment_result,
        'risk_profile': risk_profile,
        'predictions': {
            'ensemble': pred_result['ensemble'],
            'lower_ci': pred_result['confidence_intervals']['lower'],
            'upper_ci': pred_result['confidence_intervals']['upper']
        },
        'insights': insights,
        'coin_info': next((c for c in COINS if c['id'] == coin_id), {'name': coin_id})
    }

# ============================================================================
# UI RENDERING
# ============================================================================
def render_summary_dashboard(result: Dict, horizon: int):
    m = result['market']
    tech = result['technical']
    sent = result['sentiment']
    risk = result['risk_profile']
    preds = result['predictions']
    
    # --- Header ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title(f"{result['coin_info']['name']} ({m['symbol']})")
        st.write(f"**${m['price_usd']:,.2f}**")
        
        # Color for price change
        color = "green" if m['pct_change_24h'] >= 0 else "red"
        st.markdown(f"<span style='color:{color}'>{m['pct_change_24h']:+.2f}% (24h)</span>", unsafe_allow_html=True)

    with col2:
        # RISK GAUGE
        risk_color = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "HIGH": "#ef4444", "CRITICAL": "#7f1d1d"}
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 8px; border: 1px solid {risk_color[risk['level']]}; text-align: center;">
            <small>RISK LEVEL</small><br>
            <strong style="color: {risk_color[risk['level']]}; font-size: 1.2em;">{risk['level']}</strong><br>
            <small>{risk['score']}/10</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # SENTIMENT GAUGE
        sent_color = "green" if sent['score'] > 0.2 else "red" if sent['score'] < -0.2 else "gray"
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 8px; border: 1px solid {sent_color}; text-align: center;">
            <small>SENTIMENT</small><br>
            <strong style="color: {sent_color}; font-size: 1.2em;">{sent['interpretation']}</strong><br>
            <small>{len(sent['details'])} articles</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- AI Recommendation ---
    if result['insights'].get('recommendation'):
        rec = result['insights']['recommendation']
        rec_color = "#22c55e" if "BUY" in rec.upper() else "#ef4444" if "SELL" in rec.upper() else "#f59e0b"
        st.markdown(f"""
        <div style="background-color: {rec_color}22; padding: 15px; border-radius: 10px; border-left: 5px solid {rec_color};">
            <h4 style="margin:0; color: {rec_color};">🤖 AI Recommendation: {rec}</h4>
            <p style="margin-top: 5px;">{result['insights'].get('insight')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

    # --- Risk Factors (if any) ---
    if risk['factors']:
        with st.expander(f"⚠️ Risk Factors Detected ({len(risk['factors'])})", expanded=True):
            for f in risk['factors']:
                st.markdown(f"- {f}")

    # --- Forecast Chart with Bands ---
    st.subheader(f"🔮 {horizon}-Day Price Forecast")
    
    dates = pd.date_range(start=datetime.now(), periods=horizon+1)[1:]
    chart_data = pd.DataFrame({
        'Date': dates,
        'Base Case': preds['ensemble'],
        'Bull Case': preds['upper_ci'],
        'Bear Case': preds['lower_ci']
    })
    
    base = alt.Chart(chart_data).encode(x=alt.X('Date', axis=alt.Axis(format='%b %d')))
    
    # Line
    line = base.mark_line(color='#3b82f6', strokeWidth=3).encode(
        y=alt.Y('Base Case', scale=alt.Scale(zero=False)),
        tooltip=['Date', 'Base Case']
    )
    
    # Confidence Band
    band = base.mark_area(opacity=0.2, color='#3b82f6').encode(
        y='Bear Case',
        y2='Bull Case',
        tooltip=['Bear Case', 'Bull Case']
    )
    
    st.altair_chart(band + line, use_container_width=True)
    
    # --- Metrics Grid ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI (14)", f"{tech['rsi']:.1f}", delta="Overbought" if tech['rsi']>70 else "Oversold" if tech['rsi']<30 else "Neutral", delta_color="off")
    c2.metric("Volatility", f"{tech['volatility']:.1f}%")
    c3.metric("Trend", tech['trend'].title())
    c4.metric("Model Uncertainty", f"{result['predictions']['upper_ci'][-1] - result['predictions']['lower_ci'][-1]:.0f}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Crypto Analysis Agent", layout="wide")
    
    if 'last_result' not in st.session_state: st.session_state.last_result = None
    
    # Input
    user_input = st.text_input("Ask about a crypto...", placeholder="e.g. Bitcoin 7 day forecast")
    
    if st.button("Analyze") and user_input:
        with st.spinner("Analyzing markets, reading news, and running models..."):
            parsed = parse_user_message(user_input)
            res = analyze_cryptocurrency(parsed['coin_id'], parsed['horizon_days'])
            
            if 'error' in res:
                st.error(res['error'])
            else:
                st.session_state.last_result = res
                st.session_state.last_horizon = parsed['horizon_days']

    # Display
    if st.session_state.last_result:
        render_summary_dashboard(st.session_state.last_result, st.session_state.get('last_horizon', 7))

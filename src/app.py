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
    handlers=[
        logging.StreamHandler()
    ]
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
        logger.warning("Config file not found, using defaults")
        return get_default_config()

def get_default_config() -> dict:
    """Default configuration if YAML not found"""
    return {
        'cryptocurrencies': [
            {'name': 'Bitcoin', 'id': 'bitcoin', 'symbol': 'BTC'},
            {'name': 'Ethereum', 'id': 'ethereum', 'symbol': 'ETH'},
            {'name': 'Binance Coin', 'id': 'binancecoin', 'symbol': 'BNB'},
            {'name': 'Ripple', 'id': 'ripple', 'symbol': 'XRP'},
            {'name': 'Solana', 'id': 'solana', 'symbol': 'SOL'},
            {'name': 'Cardano', 'id': 'cardano', 'symbol': 'ADA'},
            {'name': 'Dogecoin', 'id': 'dogecoin', 'symbol': 'DOGE'}
        ],
        'models': {
            'lstm': {'window_size': 30, 'epochs': 20, 'batch_size': 16},
            'ensemble': {'lstm_weight': 0.7, 'xgboost_weight': 0.3}
        }
    }

CONFIG = load_config()

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_keys() -> dict:
    """Get API keys from Streamlit secrets or environment variables"""
    keys = {}
    
    # Try Streamlit secrets first
    try:
        keys['gemini'] = st.secrets.get('gemini', {}).get('api_key')
        keys['hf_token'] = st.secrets.get('huggingface', {}).get('token')
    except Exception:
        pass
    
    # Fallback to environment variables
    if not keys.get('gemini'):
        keys['gemini'] = os.getenv('GEMINI_API_KEY')
    if not keys.get('hf_token'):
        keys['hf_token'] = os.getenv('HF_TOKEN')
    
    return keys

API_KEYS = get_api_keys()

# ============================================================================
# COIN MAPPING
# ============================================================================

COINS = CONFIG.get('cryptocurrencies', [])
COIN_NAME_TO_ID = {c['name'].lower(): c['id'] for c in COINS}
COIN_SYMBOL_TO_ID = {c['symbol'].lower(): c['id'] for c in COINS}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_money(value: float) -> str:
    """Format number as money"""
    if pd.isna(value):
        return "—"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000_000:
        return f"${value/1_000_000_000_000:.2f}T"
    elif abs_val >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    else:
        return f"${value:,.2f}"

def get_rsi_zone(rsi: float) -> str:
    """Get RSI zone description"""
    if pd.isna(rsi):
        return "—"
    if rsi >= 70:
        return "Overbought"
    elif rsi <= 30:
        return "Oversold"
    else:
        return "Neutral"

def sentiment_bar(pos: float, neu: float, neg: float, width: int = 20) -> str:
    """Create visual sentiment bar"""
    pos_blocks = int(round(width * (pos / 100.0)))
    neu_blocks = int(round(width * (neu / 100.0)))
    neg_blocks = max(0, width - pos_blocks - neu_blocks)
    return "🟩" * pos_blocks + "⬜" * neu_blocks + "🟥" * neg_blocks

def get_recommendation_style(rating: str) -> Tuple[str, str, str]:
    """Get styling for recommendation"""
    rating_lower = (rating or "").lower()
    if "buy" in rating_lower:
        return ("BUY", "🟢", "#16a34a")
    elif "sell" in rating_lower or "avoid" in rating_lower:
        return ("SELL / AVOID", "🔴", "#ef4444")
    else:
        return ("HOLD / WAIT", "🟡", "#f59e0b")

def parse_user_message(message: str) -> Dict:
    """Parse user message to extract intent"""
    import re
    msg_lower = message.lower()
    
    # Extract coin
    coin_id = None
    for name, cid in COIN_NAME_TO_ID.items():
        if name in msg_lower:
            coin_id = cid
            break
    if not coin_id:
        for sym, cid in COIN_SYMBOL_TO_ID.items():
            if re.search(r'\b' + re.escape(sym) + r'\b', msg_lower):
                coin_id = cid
                break
    if not coin_id:
        coin_id = "bitcoin"  # Default
    
    # Extract horizon
    horizon_days = 7  # Default
    m = re.search(r'(\d+)\s*(day|days|d)\b', msg_lower)
    if m:
        horizon_days = int(m.group(1))
    
    return {
        'coin_id': coin_id,
        'horizon_days': horizon_days
    }

# ============================================================================
# CORE ANALYSIS FUNCTION
# ============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def analyze_cryptocurrency(
    coin_id: str,
    horizon_days: int = 7
) -> Dict:
    """
    Main analysis function that orchestrates all components
    
    Args:
        coin_id: CoinGecko coin ID
        horizon_days: Forecast horizon in days
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info(f"Starting analysis for {coin_id}, horizon={horizon_days}")
    
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
        market_data['rsi_14'] = technical_indicators['rsi']
        
        # ====================================================================
        # 3. FETCH AND ANALYZE NEWS
        # ====================================================================
        logger.info("Fetching news articles...")
        news_fetcher = get_news_fetcher()
        
        # Fetch articles for both symbol and name
        articles_symbol = news_fetcher.fetch_articles(coin_symbol, max_total=25)
        articles_name = news_fetcher.fetch_articles(coin_id, max_total=25)
        
        # Merge and deduplicate
        all_articles = {a['title']: a for a in (articles_symbol + articles_name)}
        articles = list(all_articles.values())[:50]
        
        # Extract headlines
        headlines = [a['title'] for a in articles if a.get('title')]
        
        # ====================================================================
        # 4. SENTIMENT ANALYSIS
        # ====================================================================
        logger.info("Analyzing sentiment...")
        sentiment_analyzer = get_analyzer()
        
        if headlines:
            sentiment_results = sentiment_analyzer.analyze_texts(headlines)
            sentiment_score, sentiment_df = sentiment_analyzer.calculate_aggregate_sentiment(
                sentiment_results
            )
            sentiment_breakdown = sentiment_analyzer.get_sentiment_breakdown(sentiment_results)
        else:
            sentiment_score = 0.0
            sentiment_df = pd.DataFrame()
            sentiment_breakdown = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        # ====================================================================
        # 5. PRICE PREDICTION
        # ====================================================================
        logger.info("Training models and generating forecast...")
        
        try:
            predictions = train_and_predict(
                price_series,
                horizon=horizon_days,
                window_size=CONFIG['models']['lstm']['window_size']
            )
            
            lstm_preds = predictions['lstm']
            xgb_preds = predictions['xgboost']
            ensemble_preds = predictions['ensemble']
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            lstm_preds = []
            xgb_preds = []
            ensemble_preds = []
        
        # Build forecast table
        forecast_table = []
        last_date = historical_df.index[-1]
        
        for i in range(horizon_days):
            forecast_date = last_date + pd.Timedelta(days=i+1)
            forecast_table.append({
                'day': i + 1,
                'date': forecast_date,
                'lstm': lstm_preds[i] if i < len(lstm_preds) else None,
                'xgboost': xgb_preds[i] if i < len(xgb_preds) else None,
                'ensemble': ensemble_preds[i] if i < len(ensemble_preds) else None
            })
        
        # ====================================================================
        # 6. GENERATE AI INSIGHTS (GEMINI)
        # ====================================================================
        logger.info("Generating AI insights...")
        
        if API_KEYS.get('gemini'):
            try:
                # 1. Package Prediction Data
                prediction_data = {
                    'lstm': lstm_preds,
                    'xgboost': xgb_preds,
                    'ensemble': ensemble_preds # Pass full list for trend analysis
                }
                
                # 2. Package Sentiment Data
                sentiment_data = {
                    'score': sentiment_score,
                    'breakdown': sentiment_breakdown # Pass full breakdown (pos/neg %)
                }

                # 3. Call Generator
                insights = generate_insights(
                    api_key=API_KEYS['gemini'],
                    coin_symbol=coin_symbol,
                    market_data=market_data,
                    sentiment_data=sentiment_data,     # NEW: Passing dict
                    technical_indicators=technical_indicators,
                    prediction_data=prediction_data,   # NEW: Passing dict
                    top_headlines=headlines[:5],
                    horizon_days=horizon_days
                )
            except Exception as e:
                logger.error(f"Gemini insights error: {e}")
                insights = {
                    'recommendation': 'HOLD / WAIT',
                    'score': 0.0,
                    'insight': f'Error generating insights: {str(e)}',
                    'source': 'error'
                }
        else:
            # Keep existing fallback logic
            insights = {
                'recommendation': 'HOLD / WAIT',
                'score': 0.0,
                'insight': 'Gemini API key not configured. Please add your API key to enable AI-powered insights.',
                'source': 'no_api_key'
            }
        
        # ====================================================================
        # 7. COMPILE RESULTS
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
            'sentiment_details': sentiment_df,
            'forecast_table': forecast_table,
            'predictions': {
                'lstm': lstm_preds,
                'xgboost': xgb_preds,
                'ensemble': ensemble_preds
            },
            'insights': insights,
            'coin_info': coin_info
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return {'error': str(e)}

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_summary_dashboard(result: Dict, horizon_days: int):
    """Render the main summary dashboard with all insights - FIXED VERSION"""
    
    market = result['market']
    technical = result['technical']
    sentiment_breakdown = result['sentiment_breakdown']
    insights = result['insights']
    
    # Extract data
    coin_name = result['coin_info']['name']
    symbol = market['symbol']
    price = market['price_usd']
    pct_24h = market['pct_change_24h']
    pct_7d = market['pct_change_7d']
    market_cap = market['market_cap']
    volume = market['volume_24h']
    rsi = technical['rsi']
    
    # Recommendation styling - FIXED: Use consistent recommendation from insights
    rec_text = insights.get('recommendation', 'HOLD / WAIT').upper().strip()
    if "BUY" in rec_text:
        rec_label = "BUY"
        rec_emoji = "🟢"
        rec_color = "#16a34a"
    elif "SELL" in rec_text:
        rec_label = "SELL / AVOID"
        rec_emoji = "🔴"
        rec_color = "#ef4444"
    else:
        rec_label = "HOLD / WAIT"
        rec_emoji = "🟡"
        rec_color = "#f59e0b"
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.subheader("✅ AI-Powered Insights & Risk Assessment")
    
    main_col, risk_col = st.columns([2.5, 1])
    
    with main_col:
        # Recommendation - CONSISTENT with top
        st.markdown(
            f"<span style='display:inline-block;padding:8px 16px;border-radius:12px;"
            f"background:{rec_color}22;color:{rec_color};font-weight:800;font-size:1.1rem'>"
            f"{rec_emoji} {rec_label}</span>",
            unsafe_allow_html=True
        )
        
        # Confidence score - FIXED: Display as percentage
        rec_score = insights.get('score', 0.5)
        if not pd.isna(rec_score):
            if isinstance(rec_score, float):
                score_100 = max(0, min(100, int(round(rec_score * 100))))
            else:
                score_100 = int(rec_score)
            st.progress(score_100 / 100.0, text=f"Confidence: {score_100}/100")
        
        # Source indicator
        source = insights.get('source', 'unknown')
        if source == 'gemini':
            st.caption("🤖 Powered by Google Gemini 2.0 Flash")
        elif source == 'fallback':
            st.caption("⚙️ Rule-based analysis")
        elif source == 'no_api_key':
            st.warning("⚠️ Gemini API key not configured. Add your API key for AI-powered insights.")
        
        st.write("")
        
        # Sentiment visualization
        st.markdown("**📊 News Sentiment Analysis**")
        pos = sentiment_breakdown['positive']
        neu = sentiment_breakdown['neutral']
        neg = sentiment_breakdown['negative']
        
        st.markdown(sentiment_bar(pos, neu, neg))
        st.caption(f"Positive {pos:.1f}% · Neutral {neu:.1f}% · Negative {neg:.1f}%")
        
        # Insights text - FIXED: Clear, readable content
        st.write("")
        st.markdown("**📋 Analysis Summary**")
        insight_text = insights.get('insight', '')
        if insight_text and len(insight_text) > 20:
            st.info(insight_text)
        else:
            st.info(f"**{rec_label}** - Based on current market conditions and sentiment analysis.")
    
    with risk_col:
        st.markdown("**⚠️ Risk Factors**")
        
        # Display risks from Gemini if available
        risks = insights.get('risks', [])
        if risks and isinstance(risks, list):
            for risk in risks[:3]:  # Show top 3 risks
                st.write(f"• {risk}")
        else:
            # Fallback risk calculation
            if not pd.isna(volume) and not pd.isna(market_cap) and market_cap > 0:
                liq_pct = (volume / market_cap) * 100
                if liq_pct < 5:
                    st.write("🔴 Low liquidity risk")
                elif liq_pct < 10:
                    st.write("🟡 Medium liquidity")
                else:
                    st.write("🟢 Good liquidity")
            
            volatility = technical.get('volatility', 0)
            if not pd.isna(volatility):
                if volatility > 0.10:
                    st.write("🔴 High volatility")
                elif volatility > 0.05:
                    st.write("🟡 Medium volatility")
                else:
                    st.write("🟢 Low volatility")
            
            if not pd.isna(rsi):
                if rsi >= 70:
                    st.write("🟡 Overbought (RSI)")
                elif rsi <= 30:
                    st.write("🟡 Oversold (RSI)")
        
        st.write("")
        st.markdown("**📈 Technical Signals**")
        
        # Momentum
        momentum = technical.get('momentum', 0)
        if not pd.isna(momentum):
            if momentum > 5:
                st.write("🟢 Strong upward momentum")
            elif momentum > 0:
                st.write("🟡 Slight upward momentum")
            elif momentum > -5:
                st.write("🟡 Slight downward momentum")
            else:
                st.write("🔴 Strong downward momentum")
        
        # Trend
        trend = technical.get('trend', 'sideways')
        if trend == 'uptrend':
            st.write("🟢 Uptrend detected")
        elif trend == 'downtrend':
            st.write("🔴 Downtrend detected")
        else:
            st.write("🟡 Sideways movement")
    
    st.divider()
    
    # ========================================================================
    # FORECAST SECTION
    # ========================================================================
    st.subheader(f"🎯 {horizon_days}-Day Price Forecast")
    
    forecast_table = result['forecast_table']
    history_df = result.get('history')
    
    if forecast_table:
        # Build forecast DataFrame
        forecast_rows = []
        for row in forecast_table:
            date = row['date']
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            ensemble = row.get('ensemble')
            forecast_rows.append({
                'Date': date_str,
                'Forecast ($)': ensemble if ensemble is not None else None
            })
        
        df_forecast = pd.DataFrame(forecast_rows).set_index('Date')
        
        chart_col, table_col = st.columns([1.3, 1])
        
        with table_col:
            st.dataframe(
                df_forecast.style.format({'Forecast ($)': '${:,.2f}'}),
                use_container_width=True
            )
        
        with chart_col:
            # Create combined chart
            combined_df = pd.DataFrame()
            
            # Add history
            if history_df is not None and not history_df.empty and 'price' in history_df.columns:
                hist_series = history_df['price'].tail(90)
                combined_df['History'] = hist_series
            
            # Add forecast
            if not df_forecast.empty:
                forecast_series = df_forecast['Forecast ($)'].astype(float)
                forecast_series.index = pd.to_datetime(forecast_series.index)
                combined_df = pd.concat([combined_df, forecast_series.rename('Forecast')])
            
            if not combined_df.empty:
                # Reset index for Altair
                plot_df = combined_df.reset_index()
                plot_df.columns = ['Date', 'History', 'Forecast']
                plot_df = plot_df.melt('Date', var_name='Series', value_name='Price')
                plot_df = plot_df.dropna(subset=['Price'])
                
                # Create chart
                chart = alt.Chart(plot_df).mark_line(size=2).encode(
                    x=alt.X('Date:T', title='Date'),
                    y=alt.Y('Price:Q', title='Price (USD)', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'Series:N',
                        scale=alt.Scale(domain=['History', 'Forecast'], range=['#4e79a7', '#ff4d4f'])
                    ),
                    tooltip=['Date:T', 'Series:N', alt.Tooltip('Price:Q', format=',.2f')]
                ).properties(height=350)
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Insufficient data for chart")
    else:
        st.info("No forecast data available")

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Crypto Analysis Agent",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 2.4rem; }
.stButton > button { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:0.6rem;'>
        <div style='width:36px;height:36px;display:inline-flex;align-items:center;
                    justify-content:center;background:linear-gradient(135deg,#7c3aed33,#06b6d433);
                    border:1px solid #24324a;border-radius:12px;font-size:1.1rem;'>💬</div>
        <h1 style='margin:0;'>Crypto Analysis Agent</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color:#96a7bf;margin:-0.15rem 0 1.1rem 0;'>
    AI-powered cryptocurrency analysis with live data, sentiment analysis, and price forecasting.
    <strong>Educational purposes only</strong> — not financial advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'last_horizon' not in st.session_state:
        st.session_state.last_horizon = 7
    
    # Quick actions
    st.markdown("##### 🚀 Quick Start")
    cols = st.columns(len(COINS))
    for i, coin in enumerate(COINS):
        with cols[i]:
            if st.button(coin['symbol'], use_container_width=True, key=f"quick_{coin['id']}"):
                st.session_state.last_query = f"{coin['symbol']} 7-day forecast"
    
    # Input section
    st.markdown("---")
    user_message = st.text_input(
        "**Your Question**",
        placeholder="e.g., 'Should I buy ETH?' or 'BTC 7-day forecast'",
        key="user_input"
    )
    
    analyze_button = st.button("🔍 Analyze", type="primary", help="Click to start analysis")
    
    # Process query
    if analyze_button and user_message.strip():
        with st.spinner("🔄 Analyzing... This may take 30-60 seconds..."):
            parsed = parse_user_message(user_message)
            coin_id = parsed['coin_id']
            horizon = parsed['horizon_days']
            
            result = analyze_cryptocurrency(coin_id, horizon)
            
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.session_state.last_result = result
                st.session_state.last_horizon = horizon
    
    # Display results
    if st.session_state.last_result:
        st.markdown("---")
        render_summary_dashboard(
            st.session_state.last_result,
            st.session_state.last_horizon
        )
    else:
        st.info("👆 Enter a query above to get started! Try 'Bitcoin forecast' or 'Should I buy ETH?'")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

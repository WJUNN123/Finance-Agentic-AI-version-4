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
    """
    Load configuration from YAML file.
    
    Configuration includes:
    - Cryptocurrency list (name, ID, symbol)
    - Model hyperparameters (epochs, batch size, window size)
    - API settings
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logger.info("Configuration loaded from file")
            return config
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        return get_default_config()


def get_default_config() -> dict:
    """
    Default configuration if YAML not found.
    
    Returns:
        dict: Default configuration
    """
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
    """
    Get API keys from Streamlit secrets or environment variables.
    
    Priority:
    1. Try Streamlit secrets (st.secrets)
    2. Fall back to environment variables
    
    Required keys:
    - gemini: Google Gemini API key for AI insights
    - hf_token: HuggingFace token (optional, for sentiment model)
    
    Returns:
        dict: API keys dictionary
        
    Note:
        Never hardcode API keys in source code.
        Always use Streamlit secrets on cloud deployment.
    """
    keys = {}
    
    # Try Streamlit secrets first
    try:
        keys['gemini'] = st.secrets.get('gemini', {}).get('api_key')
        keys['hf_token'] = st.secrets.get('huggingface', {}).get('token')
    except Exception as e:
        logger.warning(f"Could not access Streamlit secrets: {e}")
    
    # Fallback to environment variables
    if not keys.get('gemini'):
        keys['gemini'] = os.getenv('GEMINI_API_KEY')
    if not keys.get('hf_token'):
        keys['hf_token'] = os.getenv('HF_TOKEN')
    
    logger.info(f"API keys loaded: gemini={bool(keys.get('gemini'))}, "
               f"hf_token={bool(keys.get('hf_token'))}")
    
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
    """
    Format number as money with appropriate unit.
    
    Format rules:
    - < $1M: Show as dollars with 2 decimals
    - $1M-$1B: Show as millions (M)
    - $1B-$1T: Show as billions (B)
    - >$1T: Show as trillions (T)
    
    Args:
        value (float): Number to format
    
    Returns:
        str: Formatted money string
    
    Example:
        >>> format_money(1234567890)
        '$1.23B'
    """
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
    """
    Get RSI zone description.
    
    Args:
        rsi (float): RSI value (0-100)
    
    Returns:
        str: Description ('Overbought', 'Oversold', or 'Neutral')
    """
    if pd.isna(rsi):
        return "—"
    if rsi >= 70:
        return "Overbought"
    elif rsi <= 30:
        return "Oversold"
    else:
        return "Neutral"


def sentiment_bar(pos: float, neu: float, neg: float, width: int = 20) -> str:
    """
    Create visual sentiment bar using emoji blocks.
    
    Args:
        pos (float): Positive percentage (0-100)
        neu (float): Neutral percentage (0-100)
        neg (float): Negative percentage (0-100)
        width (int): Bar width in characters (default 20)
    
    Returns:
        str: Visual bar with emoji blocks
        
    Example:
        >>> bar = sentiment_bar(60, 30, 10)
        >>> print(bar)  # Shows more green blocks than red
    """
    pos_blocks = int(round(width * (pos / 100.0)))
    neu_blocks = int(round(width * (neu / 100.0)))
    neg_blocks = max(0, width - pos_blocks - neu_blocks)
    return "🟩" * pos_blocks + "🟨" * neu_blocks + "🟥" * neg_blocks


def parse_user_message(message: str) -> Dict:
    """
    Parse user input to extract cryptocurrency and forecast horizon.
    
    Smart parsing:
    - Recognizes coin names: "Bitcoin", "ethereum", "BTC", "ETH"
    - Extracts duration: "7 days", "7d", "7-day forecast"
    - Sets defaults if not provided
    
    Args:
        message (str): User's query (e.g., "Bitcoin 7-day forecast")
    
    Returns:
        Dict with keys:
        - coin_id: CoinGecko coin ID (default 'bitcoin')
        - horizon_days: Number of days to forecast (default 7)
    
    Example:
        >>> parsed = parse_user_message("Should I buy Ethereum for 14 days?")
        >>> print(parsed)
        # {'coin_id': 'ethereum', 'horizon_days': 14}
    """
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
    
    logger.debug(f"Parsed message: coin={coin_id}, horizon={horizon_days}")
    
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
    Main analysis function orchestrating all components.
    
    Process Flow:
    1. Fetch market data (price, market cap, volume)
    2. Get historical prices
    3. Calculate technical indicators (RSI, MA, volatility)
    4. Fetch news articles
    5. Analyze sentiment (positive/negative/neutral)
    6. Train ML models (LSTM + XGBoost)
    7. Generate price predictions with confidence intervals
    8. Assess risk (5 dimensions)
    9. Generate AI insights (Gemini multi-stage reasoning)
    10. Compile and return all results
    
    Args:
        coin_id (str): CoinGecko coin ID (e.g., 'bitcoin')
        horizon_days (int): Forecast horizon in days (default 7)
    
    Returns:
        Dict with all analysis results or {'error': str} if failed
    
    Cache:
        Results cached for 5 minutes (ttl=300) to avoid rate limits
        and improve app responsiveness
    
    Note:
        This is the most expensive operation. Caching is critical
        for Streamlit free tier performance.
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
        
        logger.debug(f"Market data: ${market_data['price_usd']:,.2f} "
                    f"({market_data['pct_change_24h']:+.1f}%)")
        
        # Get historical data
        historical_df = cg_fetcher.get_historical_data(coin_id, days=180)
        if historical_df.empty or 'price' not in historical_df.columns:
            return {'error': 'Insufficient historical data'}
        
        price_series = historical_df['price']
        logger.info(f"Fetched {len(historical_df)} historical data points")
        
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
        logger.info("Fetching news articles...")
        news_fetcher = get_news_fetcher()
        
        articles_symbol = news_fetcher.fetch_articles(coin_symbol, max_total=25)
        articles_name = news_fetcher.fetch_articles(coin_id, max_total=25)
        
        # Merge and deduplicate
        all_articles = {a['title']: a for a in (articles_symbol + articles_name)}
        articles = list(all_articles.values())[:50]
        
        # Extract headlines
        headlines = [a['title'] for a in articles if a.get('title')]
        logger.info(f"Fetched {len(headlines)} unique headlines")
        
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
        
        logger.info(f"Sentiment: {sentiment_score:.2f} "
                   f"({sentiment_breakdown['positive']:.0f}% pos)")
        
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
        # 6. GENERATE AI INSIGHTS
        # ====================================================================
        logger.info("Generating AI insights...")
        
        if API_KEYS.get('gemini'):
            try:
                insights = generate_insights(
                    api_key=API_KEYS['gemini'],
                    coin_symbol=coin_symbol,
                    market_data=market_data,
                    sentiment_data={
                        'score': sentiment_score,
                        'breakdown': sentiment_breakdown
                    },
                    technical_indicators=technical_indicators,
                    prediction_data={
                        'lstm': lstm_preds,
                        'xgboost': xgb_preds,
                        'ensemble': ensemble_preds
                    },
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
            insights = {
                'recommendation': 'HOLD / WAIT',
                'score': 0.0,
                'insight': 'Gemini API key not configured',
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
    """
    Render main analysis dashboard.
    
    Shows:
    - Price with % change
    - Market metrics (market cap, volume)
    - Technical indicators (RSI, trend)
    - Recommendation with confidence
    - Sentiment analysis
    - Forecast table
    - Risk assessment
    
    Args:
        result (Dict): Analysis results from analyze_cryptocurrency()
        horizon_days (int): Forecast horizon for display
    """
    market = result['market']
    technical = result['technical']
    sentiment_breakdown = result['sentiment_breakdown']
    insights = result['insights']
    
    coin_name = result['coin_info']['name']
    symbol = market['symbol']
    price = market['price_usd']
    pct_24h = market['pct_change_24h']
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown(f"### 🪙 {coin_name} ({symbol})")
    
    cols = st.columns([1.5, 1.2, 1.2, 1.2])
    
    # Price
    with cols[0]:
        st.markdown("**Price**")
        st.markdown(
            f"<span style='font-size:2rem;font-weight:800'>${price:,.2f}</span>",
            unsafe_allow_html=True
        )
        
        if not pd.isna(pct_24h):
            arrow = "📈" if pct_24h >= 0 else "📉"
            color = "#2ecc71" if pct_24h >= 0 else "#e74c3c"
            st.markdown(
                f"<span style='padding:4px 8px;border-radius:999px;background:{color}22;"
                f"color:{color};font-weight:700'>{arrow} {pct_24h:.2f}% · 24h</span>",
                unsafe_allow_html=True
            )
    
    # Market metrics
    with cols[1]:
        st.metric("Market Cap", format_money(market.get('market_cap', 0)))
        st.metric("24h Volume", format_money(market.get('volume_24h', 0)))
    
    with cols[2]:
        st.metric("7d Change", f"{market.get('pct_change_7d', 0):+.2f}%")
        st.metric("RSI (14)", f"{technical.get('rsi', 50):.1f}")
    
    with cols[3]:
        st.write("**Trend & Signals**")
        trend = technical.get('trend', 'sideways')
        trend_icon = "📈" if trend == 'uptrend' else "📉" if trend == 'downtrend' else "↔️"
        st.caption(f"{trend_icon} {trend.title()}")
        
        vol = technical.get('volatility', 0)
        vol_icon = "⚡" if vol > 0.10 else "📊" if vol > 0.05 else "✓"
        st.caption(f"{vol_icon} Volatility: {vol:.4f}")
    
    st.divider()
    
    # ========================================================================
    # INSIGHTS AND RECOMMENDATION
    # ========================================================================
    st.subheader("✅ AI-Powered Recommendation")
    
    rec = insights.get('recommendation', 'HOLD / WAIT').upper()
    confidence = insights.get('score', 0.5)
    
    # Recommendation badge
    if "BUY" in rec:
        st.success(f"🟢 **{rec}** (Confidence: {confidence:.0%})")
    elif "SELL" in rec:
        st.error(f"🔴 **{rec}** (Confidence: {confidence:.0%})")
    else:
        st.warning(f"🟡 **{rec}** (Confidence: {confidence:.0%})")
    
    # Insight text
    if insights.get('insight'):
        st.info(insights['insight'])
    
    st.divider()
    
    # ========================================================================
    # SENTIMENT
    # ========================================================================
    st.subheader("📊 News Sentiment")
    
    pos = sentiment_breakdown['positive']
    neu = sentiment_breakdown['neutral']
    neg = sentiment_breakdown['negative']
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Score", f"{sentiment_breakdown.get('positive', 0):.0f}%", "Positive")
    
    with col2:
        st.markdown(sentiment_bar(pos, neu, neg))
        st.caption(f"Positive {pos:.1f}% · Neutral {neu:.1f}% · Negative {neg:.1f}%")
    
    st.divider()
    
    # ========================================================================
    # FORECAST
    # ========================================================================
    st.subheader(f"📈 {horizon_days}-Day Price Forecast")
    
    forecast_table = result.get('forecast_table', [])
    
    if forecast_table:
        forecast_rows = []
        for row in forecast_table:
            ensemble = row.get('ensemble')
            if ensemble:
                forecast_rows.append({
                    'Day': row['day'],
                    'Date': row['date'].strftime('%Y-%m-%d'),
                    'Forecast': f"${ensemble:,.0f}"
                })
        
        if forecast_rows:
            df_forecast = pd.DataFrame(forecast_rows)
            st.dataframe(df_forecast, use_container_width=True, hide_index=True)
    else:
        st.info("No forecast data available")
    
    st.divider()
    
    # ========================================================================
    # DISCLAIMERS
    # ========================================================================
    st.warning("""
    **⚠️ Important Disclaimers:**
    - **NOT FINANCIAL ADVICE** - This tool is for educational purposes only
    - **Past performance ≠ future results** - Historical patterns may not repeat
    - **High risk** - Cryptocurrency is volatile and risky
    - **AI can be wrong** - Always verify with your own research
    - **Do Your Own Research (DYOR)** - Use this as one signal among many
    - **Never invest more than you can afford to lose** - Set strict risk limits
    """)


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
    """
    Main application entry point.
    
    Flow:
    1. Display header and description
    2. Quick action buttons (popular coins)
    3. User input field
    4. Analysis button
    5. Display results or instructions
    """
    
    # Header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:0.6rem;'>
        <h1 style='margin:0;'>🚀 Crypto Market Analysis Agent</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **AI-powered cryptocurrency analysis** for faster, more informed decisions.
    
    Combines real-time market data, news sentiment, technical analysis, and 
    machine learning predictions to help you understand the crypto market better.
    
    ⚠️ **For educational purposes only - NOT financial advice**
    """)
    
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
                st.session_state.quick_query = f"{coin['symbol']} 7-day forecast"
    
    # Input section
    st.markdown("---")
    user_message = st.text_input(
        "**Your Question**",
        placeholder="e.g., 'Should I buy ETH?' or 'Bitcoin forecast'",
        key="user_input"
    )
    
    analyze_button = st.button("🔍 Analyze", type="primary", help="Click to start analysis")
    
    # Process query
    if analyze_button and user_message.strip():
        with st.spinner("⏳ Analyzing... This may take 30-60 seconds..."):
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
        st.info("""
        👆 **Enter a question above to get started!**
        
        Try:
        - "Bitcoin forecast"
        - "Should I buy Ethereum?"
        - "BTC 14-day prediction"
        - "Is Solana trending?"
        """)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

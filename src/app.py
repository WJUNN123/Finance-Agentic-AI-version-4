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
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
import uuid
import altair as alt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
# NOTE: Adjust these imports based on your folder structure
# If files are in same directory, use:
from data_fetchers.coingecko import get_fetcher as get_cg_fetcher
from data_fetchers.news import get_fetcher as get_news_fetcher
from sentiment.analyzer import get_analyzer
from models.hybrid_predictor import train_and_predict
from llm.gemini_insights import generate_insights
from utils.technical_indicators import get_all_indicators

# If files are in nested folders, use:
# from data_fetchers.coingecko import get_fetcher as get_cg_fetcher
# from data_fetchers.news import get_fetcher as get_news_fetcher
# from sentiment.analyzer import get_analyzer
# from models.hybrid_predictor import train_and_predict
# from llm.gemini_insights import generate_insights
# from utils.technical_indicators import get_all_indicators

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
# STAGE 3: INPUT VALIDATION FUNCTIONS (NEW)
# ============================================================================

def validate_coin_id(coin_id: str) -> Tuple[bool, str]:
    """
    Validate coin ID
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not coin_id or not isinstance(coin_id, str):
        return False, "Coin ID must be a non-empty string"
    
    valid_coins = [c['id'] for c in COINS]
    if coin_id not in valid_coins:
        available = ', '.join([c['symbol'] for c in COINS])
        return False, f"❌ Unsupported coin: {coin_id}. Available: {available}"
    
    return True, "Valid"

def validate_horizon(horizon_days: int) -> Tuple[bool, str]:
    """
    Validate forecast horizon
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(horizon_days, int):
        return False, "Horizon must be an integer"
    
    if horizon_days < 1:
        return False, "Horizon must be at least 1 day"
    
    if horizon_days > 30:
        return False, "Horizon cannot exceed 30 days (model reliability)"
    
    return True, "Valid"

# ============================================================================
# CORE ANALYSIS FUNCTION (ENHANCED)
# ============================================================================

@st.cache_data(ttl=600, show_spinner=False)  # STAGE 1: Increased from 300 to 600
def analyze_cryptocurrency(
    coin_id: str,
    horizon_days: int = 7
) -> Dict:
    """
    Main analysis function with validation and better error handling
    
    Args:
        coin_id: CoinGecko coin ID
        horizon_days: Forecast horizon in days
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info(f"Starting analysis for {coin_id}, horizon={horizon_days}")
    
    # STAGE 3: Input validation
    valid_coin, coin_msg = validate_coin_id(coin_id)
    if not valid_coin:
        return {
            'error': coin_msg,
            'error_type': 'validation',
            'suggestion': 'Please choose from the available cryptocurrencies above.'
        }
    
    valid_horizon, horizon_msg = validate_horizon(horizon_days)
    if not valid_horizon:
        return {
            'error': horizon_msg,
            'error_type': 'validation',
            'suggestion': 'Please use a forecast period between 1-30 days.'
        }
    
    logger.info(f"✅ Validation passed for {coin_id}, horizon={horizon_days}")
    
    try:
        # Get coin info
        coin_info = next((c for c in COINS if c['id'] == coin_id), None)
        if not coin_info:
            return {
                'error': f'Unknown coin: {coin_id}',
                'error_type': 'validation',
                'suggestion': 'Please choose a valid cryptocurrency.'
            }
        
        coin_symbol = coin_info['symbol']
        
        # ====================================================================
        # 1. FETCH MARKET DATA (WITH ENHANCED ERROR HANDLING)
        # ====================================================================
        logger.info("Fetching market data...")
        cg_fetcher = get_cg_fetcher()
        
        try:
            market_df = cg_fetcher.get_market_data([coin_id])
            if market_df.empty:
                return {
                    'error': f'❌ No market data available for {coin_id}',
                    'suggestion': 'This coin might not be supported yet. Try: Bitcoin, Ethereum, or Solana.',
                    'error_type': 'data_not_found'
                }
        except requests.exceptions.Timeout:
            return {
                'error': '⏱️ Request timed out',
                'suggestion': 'The API is taking too long. Please try again in a moment.',
                'error_type': 'timeout'
            }
        except requests.exceptions.ConnectionError:
            return {
                'error': '🌐 Connection error',
                'suggestion': 'Unable to connect to data provider. Check your internet connection.',
                'error_type': 'connection'
            }
        except Exception as e:
            logger.error(f"Market data error: {e}", exc_info=True)
            return {
                'error': f'❌ Error fetching market data: {str(e)}',
                'suggestion': 'Please try again. If the problem persists, try a different cryptocurrency.',
                'error_type': 'api_error'
            }
        
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
            return {
                'error': 'Insufficient historical data',
                'suggestion': 'This cryptocurrency might not have enough price history.',
                'error_type': 'data_not_found'
            }
        
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
        # 4. SENTIMENT ANALYSIS (ENHANCED WITH CONFIDENCE)
        # ====================================================================
        logger.info("Analyzing sentiment...")
        sentiment_analyzer = get_analyzer()
        
        sentiment_confidence = 0.5  # Default
        if headlines:
            sentiment_results = sentiment_analyzer.analyze_texts(headlines)
            sentiment_score, sentiment_df = sentiment_analyzer.calculate_aggregate_sentiment(
                sentiment_results,
                use_recency_bias=True  # STAGE 2: Recency weighting
            )
            sentiment_breakdown = sentiment_analyzer.get_sentiment_breakdown(sentiment_results)
            sentiment_confidence = sentiment_analyzer.get_sentiment_confidence(sentiment_results)  # STAGE 2: Confidence
        else:
            sentiment_score = 0.0
            sentiment_df = pd.DataFrame()
            sentiment_breakdown = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        
        # ====================================================================
        # 5. PRICE PREDICTION (WITH MODEL CACHING)
        # ====================================================================
        logger.info("Training models and generating forecast...")
        
        try:
            predictions = train_and_predict(
                price_series,
                horizon=horizon_days,
                coin_id=coin_id,  # STAGE 1: For caching
                use_cache=True,   # STAGE 1: Enable caching
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
        # 6. GENERATE AI INSIGHTS (GEMINI) WITH CONFIDENCE
        # ====================================================================
        logger.info("Generating AI insights...")
        
        if API_KEYS.get('gemini'):
            try:
                # Package data with confidence scores (STAGE 2)
                prediction_data = {
                    'lstm': lstm_preds,
                    'xgboost': xgb_preds,
                    'ensemble': ensemble_preds
                }
                
                sentiment_data = {
                    'score': sentiment_score,
                    'breakdown': sentiment_breakdown,
                    'confidence': sentiment_confidence  # STAGE 2: Pass confidence
                }
                
                insights = generate_insights(
                    api_key=API_KEYS['gemini'],
                    coin_symbol=coin_symbol,
                    market_data=market_data,
                    sentiment_data=sentiment_data,
                    technical_indicators=technical_indicators,
                    prediction_data=prediction_data,
                    top_headlines=headlines[:5],
                    horizon_days=horizon_days
                )
            except Exception as e:
                logger.error(f"Gemini insights error: {e}")
                insights = {
                    'recommendation': 'HOLD / WAIT',
                    'score': 0.0,
                    'insight': f'Error generating insights: {str(e)}',
                    'source': 'error',
                    'risks': []
                }
        else:
            insights = {
                'recommendation': 'HOLD / WAIT',
                'score': 0.0,
                'insight': 'Gemini API key not configured. Please add your API key to enable AI-powered insights.',
                'source': 'no_api_key',
                'risks': []
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
            'sentiment_confidence': sentiment_confidence,  # STAGE 2
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
        return {
            'error': f'❌ Unexpected error: {str(e)}',
            'suggestion': 'Please try again. If the problem persists, the service might be temporarily unavailable.',
            'error_type': 'unknown'
        }


# ============================================================================
# UI RENDERING FUNCTIONS (ENHANCED)
# ============================================================================

def build_analysis_summary(
    insight_text: str,
    recommendation: str,
    market_data: Dict,
    technical: Dict,
    sentiment_breakdown: Dict,
    forecast_table: List,
    horizon_days: int,
    risks: List
) -> str:
    """Build comprehensive analysis summary combining AI insight with key data points"""
    lines = []
    
    # AI Insight
    if insight_text and len(insight_text) > 20:
        lines.append("**AI Analysis:**")
        lines.append(insight_text)
        lines.append("")
    
    # Price Forecast
    if forecast_table:
        current_price = market_data.get('price_usd', 0)
        final_forecast = forecast_table[-1].get('ensemble') if forecast_table else None
        
        if final_forecast:
            price_change = final_forecast - current_price
            price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
            
            direction = "📈 UP" if price_change_pct > 0 else "📉 DOWN"
            lines.append(f"**{horizon_days}-Day Forecast:**")
            lines.append(f"{direction} {abs(price_change_pct):.2f}% to ${final_forecast:,.0f}")
            lines.append("")
    
    # Technical signals
    technical_signals = []
    rsi = technical.get('rsi', 50)
    if not pd.isna(rsi):
        if rsi >= 70:
            technical_signals.append(f"RSI {rsi:.0f} (Overbought)")
        elif rsi <= 30:
            technical_signals.append(f"RSI {rsi:.0f} (Oversold)")
        else:
            technical_signals.append(f"RSI {rsi:.0f} (Neutral)")
    
    trend = technical.get('trend', 'sideways')
    if trend == 'uptrend':
        technical_signals.append("📈 Uptrend")
    elif trend == 'downtrend':
        technical_signals.append("📉 Downtrend")
    else:
        technical_signals.append("〰️ Sideways")
    
    if technical_signals:
        lines.append("**Technical Signals:**")
        lines.append(" | ".join(technical_signals))
        lines.append("")
    
    # Sentiment
    pos = sentiment_breakdown.get('positive', 0)
    neg = sentiment_breakdown.get('negative', 0)
    neu = sentiment_breakdown.get('neutral', 0)
    
    sentiment_signal = ""
    if pos > neg and pos > 40:
        sentiment_signal = "🟢 Positive"
    elif neg > pos and neg > 40:
        sentiment_signal = "🔴 Negative"
    else:
        sentiment_signal = "⚪ Neutral"
    
    lines.append("**Sentiment:**")
    lines.append(f"{sentiment_signal} ({pos:.0f}% pos, {neu:.0f}% neu, {neg:.0f}% neg)")
    lines.append("")
    
    # Key risks
    if risks and isinstance(risks, list) and len(risks) > 0:
        lines.append("**⚠️ Key Risks:**")
        for i, risk in enumerate(risks[:3], 1):
            lines.append(f"{i}. {risk}")
        lines.append("")
    
    # Recommendation rationale
    price_change_24h = market_data.get('pct_change_24h', 0)
    recommendation_reason = ""
    if recommendation == "BUY":
        recommendation_reason = (
            f"Strong upward signals with {price_change_24h:+.1f}% daily change "
            f"and favorable technicals suggest entry opportunity."
        )
    elif recommendation == "SELL / AVOID":
        recommendation_reason = (
            f"Downward pressure with {price_change_24h:+.1f}% daily change "
            f"and weak technicals suggest caution or exit."
        )
    else:
        recommendation_reason = (
            f"Mixed signals with {price_change_24h:+.1f}% daily change warrant "
            f"a consolidation period before new positions."
        )
    
    lines.append("**Why " + recommendation + "?**")
    lines.append(recommendation_reason)
    
    return "\n\n".join(lines)


# ============================================================================
# STAGE 3: ENHANCED DASHBOARD SECTIONS (NEW)
# ============================================================================

def render_enhanced_dashboard_sections(result: Dict, market: Dict, technical: Dict):
    """Add enhanced market context and prediction quality sections"""
    
    coin_id = result['market']['coin']
    curr_price = market['price_usd']
    volume = market['volume_24h']
    market_cap = market['market_cap']
    predictions = result.get('predictions', {})
    
    # Market Context Section
    st.divider()
    st.subheader("📊 Market Context")
    
    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
    
    with comp_col1:
        if coin_id != 'bitcoin':
            st.metric("Market Leader", "Bitcoin", help="Track Bitcoin for market direction")
        else:
            st.metric("Market Position", "#1", help="Bitcoin is the market leader")
    
    with comp_col2:
        # Liquidity score
        if not pd.isna(volume) and not pd.isna(market_cap) and market_cap > 0:
            volume_quality = (volume / market_cap) * 100
            if volume_quality > 15:
                quality_label = "🟢 Excellent"
                quality_desc = "Very high trading activity"
            elif volume_quality > 10:
                quality_label = "🟢 High"
                quality_desc = "Strong trading activity"
            elif volume_quality > 5:
                quality_label = "🟡 Medium"
                quality_desc = "Moderate trading activity"
            else:
                quality_label = "🔴 Low"
                quality_desc = "Limited trading activity"
            
            st.metric("Liquidity Score", quality_label, f"{volume_quality:.1f}%", help=quality_desc)
    
    with comp_col3:
        # Volatility classification
        volatility = technical.get('volatility', 0)
        if not pd.isna(volatility):
            if volatility > 0.15:
                vol_label = "🔴 Very High"
                vol_desc = "Extreme price swings expected"
            elif volatility > 0.10:
                vol_label = "🟠 High"
                vol_desc = "Significant price movements"
            elif volatility > 0.05:
                vol_label = "🟡 Medium"
                vol_desc = "Moderate price fluctuations"
            else:
                vol_label = "🟢 Low"
                vol_desc = "Stable price action"
            
            st.metric("Volatility", vol_label, f"{volatility:.2%}", help=vol_desc)
    
    with comp_col4:
        # Price position relative to bands
        bb_upper = technical.get('bb_upper', curr_price * 1.05)
        bb_lower = technical.get('bb_lower', curr_price * 0.95)
        
        if curr_price > bb_upper:
            position = "🔴 Extended"
            position_desc = "Price above upper band"
        elif curr_price < bb_lower:
            position = "🟢 Compressed"
            position_desc = "Price below lower band"
        else:
            position = "🟡 Normal"
            position_desc = "Within bands"
        
        st.metric("Price Position", position, help=position_desc)
    
    # Prediction Quality Section
    if predictions and predictions.get('ensemble'):
        st.divider()
        st.subheader("🎯 Prediction Quality Metrics")
        
        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
        
        lstm_preds = predictions.get('lstm', [])
        xgb_preds = predictions.get('xgboost', [])
        ensemble_preds = predictions.get('ensemble', [])
        
        if len(lstm_preds) > 0 and len(xgb_preds) > 0 and len(ensemble_preds) > 0:
            lstm_final = lstm_preds[-1]
            xgb_final = xgb_preds[-1]
            ensemble_final = ensemble_preds[-1]
            
            with pred_col1:
                # Model agreement
                lstm_diff = abs((lstm_final - ensemble_final) / ensemble_final * 100)
                xgb_diff = abs((xgb_final - ensemble_final) / ensemble_final * 100)
                avg_diff = (lstm_diff + xgb_diff) / 2
                confidence = max(50, 100 - min(avg_diff * 10, 50))
                
                if confidence > 85:
                    agreement_label = "🟢 Strong"
                elif confidence > 70:
                    agreement_label = "🟡 Moderate"
                else:
                    agreement_label = "🔴 Weak"
                
                st.metric("Model Agreement", agreement_label, f"{confidence:.0f}%", 
                         help="How much LSTM and XGBoost agree")
            
            with pred_col2:
                pred_range = abs((lstm_final - xgb_final) / curr_price * 100)
                st.metric("Forecast Range", f"±{pred_range:.1f}%", 
                         help="Spread between model predictions")
            
            with pred_col3:
                pred_return = ((ensemble_final - curr_price) / curr_price * 100)
                volatility = technical.get('volatility', 0.05)
                risk_adj = pred_return / (volatility * 100 + 1) if volatility > 0 else 0
                st.metric("Risk-Adj Return", f"{risk_adj:.2f}", 
                         help="Expected return ÷ volatility")
            
            with pred_col4:
                direction = "📈 Bullish" if pred_return > 2 else ("📉 Bearish" if pred_return < -2 else "↔️ Neutral")
                st.metric("Forecast Signal", direction, f"{pred_return:+.1f}%", 
                         help="Expected price movement")


# ============================================================================
# STAGE 3: ENHANCED CHART (NEW)
# ============================================================================

def create_enhanced_chart(combined_df, market_data, technical, coin_symbol, horizon_days, prediction_data=None):
    """Create interactive chart with DYNAMIC support/resistance lines based on predictions"""
    if combined_df.empty:
        st.info("Insufficient data for chart")
        return
    
    # Reset index for Altair
    plot_df = combined_df.reset_index()
    if 'Date' not in plot_df.columns:
        # First column is likely the date index
        date_col = plot_df.columns[0]
        plot_df = plot_df.rename(columns={date_col: 'Date'})
    
    # Get current price
    current_price = float(market_data.get('price_usd', 0))
    
    # === DYNAMIC SUPPORT/RESISTANCE CALCULATION ===
    # Calculate based on PREDICTED price (not current price)
    if prediction_data and 'ensemble' in prediction_data:
        ensemble_preds = prediction_data.get('ensemble', [])
        if ensemble_preds and len(ensemble_preds) > 0:
            # Use predicted price for support/resistance
            predicted_price = float(ensemble_preds[-1])
            # Support = 5% below predicted price
            support_level = predicted_price * 0.95
            # Resistance = 5% above predicted price
            resistance_level = predicted_price * 1.05
            
            logger.info(f"📊 Dynamic levels - Predicted: ${predicted_price:,.2f}, "
                       f"Support: ${support_level:,.2f}, Resistance: ${resistance_level:,.2f}")
        else:
            # Fallback to current price if no predictions
            predicted_price = current_price
            support_level = current_price * 0.95
            resistance_level = current_price * 1.05
    else:
        # Fallback to current price if no prediction data
        predicted_price = current_price
        support_level = current_price * 0.95
        resistance_level = current_price * 1.05
    
    # Melt for Altair
    value_cols = [col for col in plot_df.columns if col != 'Date']
    if not value_cols:
        st.warning("No data columns found for charting")
        return
    
    plot_df_melted = plot_df.melt('Date', value_vars=value_cols, var_name='Series', value_name='Price')
    plot_df_melted = plot_df_melted.dropna(subset=['Price'])
    
    if plot_df_melted.empty:
        st.warning("No valid data for charting")
        return
    
    # Create base chart
    base = alt.Chart(plot_df_melted).mark_line(size=2.5).encode(
        x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%b %d')),
        y=alt.Y('Price:Q', title='Price (USD)', scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Series:N',
            scale=alt.Scale(domain=['History', 'Forecast'], range=['#4e79a7', '#ff4d4f']),
            legend=alt.Legend(title="Data Type", orient='top')
        ),
        strokeWidth=alt.condition(alt.datum.Series == 'Forecast', alt.value(3), alt.value(2)),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Series:N', title='Type'),
            alt.Tooltip('Price:Q', title='Price', format='$,.2f')
        ]
    )
    
    # Support line (green dashed) - DYNAMIC based on prediction
    support_df = pd.DataFrame({
        'y': [support_level], 
        'label': [f'Support: ${support_level:,.2f}']
    })
    support_line = alt.Chart(support_df).mark_rule(
        strokeDash=[8, 4], 
        color='#10b981',  # Tailwind green-500
        size=2.5, 
        opacity=0.8
    ).encode(
        y='y:Q', 
        tooltip=alt.Tooltip('label:N', title='Support Level')
    )
    
    # Resistance line (red dashed) - DYNAMIC based on prediction
    resistance_df = pd.DataFrame({
        'y': [resistance_level], 
        'label': [f'Resistance: ${resistance_level:,.2f}']
    })
    resistance_line = alt.Chart(resistance_df).mark_rule(
        strokeDash=[8, 4], 
        color='#ef4444',  # Tailwind red-500
        size=2.5, 
        opacity=0.8
    ).encode(
        y='y:Q', 
        tooltip=alt.Tooltip('label:N', title='Resistance Level')
    )
    
    # Combine all layers
    chart = (base + support_line + resistance_line).properties(
        height=400,
        title=f"{coin_symbol} Price: History & {horizon_days}-Day Forecast"
    ).configure_title(
        fontSize=16, 
        font='Arial', 
        anchor='start',
        color='#1f2937'
    ).interactive()
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # Enhanced caption with actual values
    roi_pct = ((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0
    
    st.caption(
        f"🟢 **Support: ${support_level:,.2f}** (95% of target) | "
        f"🔴 **Resistance: ${resistance_level:,.2f}** (105% of target) | "
        f"🎯 **Target: ${predicted_price:,.2f}** ({roi_pct:+.1f}%) | "
        f"📊 Blue = History | 📈 Red = Forecast"
    )


# ============================================================================
# MAIN DASHBOARD RENDERING
# ============================================================================

def render_summary_dashboard(result: Dict, horizon_days: int):
    """Render the main summary dashboard with all insights"""
    
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
    
    # Recommendation styling
    rec_text = insights.get('recommendation', 'HOLD / WAIT').upper().strip()
    if "BUY" in rec_text:
        rec_label, rec_emoji, rec_color = "BUY", "🟢", "#16a34a"
    elif "SELL" in rec_text:
        rec_label, rec_emoji, rec_color = "SELL / AVOID", "🔴", "#ef4444"
    else:
        rec_label, rec_emoji, rec_color = "HOLD / WAIT", "🟡", "#f59e0b"
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    st.markdown(f"### 📊 {coin_name} ({symbol})")
    
    cols = st.columns([1.5, 1.2, 1.2, 1.2])
    
    with cols[0]:
        st.markdown("**Price**")
        st.markdown(f"<span style='font-size:2rem;font-weight:800'>${price:,.2f}</span>", unsafe_allow_html=True)
        
        if not pd.isna(pct_24h):
            arrow = "🔺" if pct_24h >= 0 else "🔻"
            color = "#2ecc71" if pct_24h >= 0 else "#e74c3c"
            st.markdown(
                f"<span style='padding:4px 8px;border-radius:999px;background:{color}22;"
                f"color:{color};font-weight:700'>{arrow} {pct_24h:.2f}% · 24h</span>",
                unsafe_allow_html=True
            )
        
        st.markdown(
            f"<span style='display:inline-block;margin-top:8px;padding:6px 12px;"
            f"border-radius:12px;background:{rec_color}22;color:{rec_color};"
            f"font-weight:800;font-size:1.0rem'>{rec_emoji} {rec_label}</span>",
            unsafe_allow_html=True
        )
    
    with cols[1]:
        st.metric("Market Cap", format_money(market_cap))
        st.metric("24h Volume", format_money(volume))
    
    with cols[2]:
        pct_7d_display = f"{pct_7d:+.2f}%" if not pd.isna(pct_7d) else "—"
        st.metric("7d Change", pct_7d_display)
        
        rsi_display = f"{rsi:.1f} ({get_rsi_zone(rsi)})" if not pd.isna(rsi) else "—"
        st.metric("RSI (14)", rsi_display)
    
    with cols[3]:
        st.write("**Quick Info**")
        
        if not pd.isna(volume) and not pd.isna(market_cap) and market_cap > 0:
            liq_pct = (volume / market_cap) * 100
            liq_icon = "🟢" if liq_pct > 10 else "🟡" if liq_pct > 5 else "🔴"
            st.caption(f"{liq_icon} Liquidity: {liq_pct:.1f}%")
        
        st.caption(f"📊 {get_rsi_zone(rsi)}")
        
        trend_icon = "📈" if technical.get('trend') == 'uptrend' else "📉" if technical.get('trend') == 'downtrend' else "〰️"
        st.caption(f"{trend_icon} Trend: {technical.get('trend', 'N/A').title()}")
    
    # STAGE 3: Enhanced dashboard sections
    render_enhanced_dashboard_sections(result, market, technical)
    
    st.divider()
    
    # ========================================================================
    # INSIGHTS AND RISK SECTION
    # ========================================================================
    st.subheader("✅ AI-Powered Insights & Risk Assessment")
    
    main_col, risk_col = st.columns([2.5, 1])
    
    with main_col:
        st.markdown(
            f"<span style='display:inline-block;padding:8px 16px;border-radius:12px;"
            f"background:{rec_color}22;color:{rec_color};font-weight:800;font-size:1.1rem'>"
            f"{rec_emoji} {rec_label}</span>",
            unsafe_allow_html=True
        )
        
        rec_score = insights.get('score', 0.5)
        if not pd.isna(rec_score):
            score_100 = max(0, min(100, int(round(rec_score * 100)) if isinstance(rec_score, float) else int(rec_score)))
            st.progress(score_100 / 100.0, text=f"Confidence: {score_100}/100")
        
        source = insights.get('source', 'unknown')
        if source == 'gemini':
            st.caption("🤖 Powered by Google Gemini 2.0 Flash")
        elif source == 'no_api_key':
            st.warning("⚠️ Gemini API key not configured.")
        
        st.write("")
        st.markdown("**📊 News Sentiment Analysis**")
        pos = sentiment_breakdown['positive']
        neu = sentiment_breakdown['neutral']
        neg = sentiment_breakdown['negative']
        
        st.markdown(sentiment_bar(pos, neu, neg))
        st.caption(f"Positive {pos:.1f}% · Neutral {neu:.1f}% · Negative {neg:.1f}%")
        
        # STAGE 2: Show sentiment confidence
        sentiment_conf = result.get('sentiment_confidence', 0.5)
        if sentiment_conf > 0:
            st.caption(f"📊 Sentiment Confidence: {sentiment_conf:.0%}")
        
        st.write("")
        st.markdown("**📋 Analysis Summary**")
        
        analysis_summary = build_analysis_summary(
            insight_text=insights.get('insight', ''),
            recommendation=rec_label,
            market_data=market,
            technical=technical,
            sentiment_breakdown=sentiment_breakdown,
            forecast_table=result.get('forecast_table', []),
            horizon_days=horizon_days,
            risks=insights.get('risks', [])
        )
        
        st.info(analysis_summary)
    
    with risk_col:
        st.markdown("**⚠️ Risk Factors**")
        
        risks = insights.get('risks', [])
        if risks and isinstance(risks, list):
            for risk in risks[:3]:
                st.write(f"• {risk}")
        else:
            # Fallback risks
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
        
        st.write("")
        st.markdown("**📈 Technical Signals**")
        
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
        
        trend = technical.get('trend', 'sideways')
        if trend == 'uptrend':
            st.write("🟢 Uptrend detected")
        elif trend == 'downtrend':
            st.write("🔴 Downtrend detected")
        else:
            st.write("🟡 Sideways movement")
    
    st.divider()
    
    # ========================================================================
    # FORECAST SECTION WITH ENHANCED CHART
    # ========================================================================
    st.subheader(f"🎯 {horizon_days}-Day Price Forecast")

    forecast_table = result['forecast_table']
    history_df = result.get('history')
    
    if forecast_table:
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
            
            if history_df is not None and not history_df.empty and 'price' in history_df.columns:
                hist_series = history_df['price'].tail(90)
                combined_df['History'] = hist_series
            
            if not df_forecast.empty:
                forecast_series = df_forecast['Forecast ($)'].astype(float)
                forecast_series.index = pd.to_datetime(forecast_series.index)
                combined_df = pd.concat([combined_df, forecast_series.rename('Forecast')])
            
            # STAGE 3: Enhanced chart with support/resistance
            if not combined_df.empty:
                # Extract ensemble predictions
                ensemble_preds = [row.get('ensemble') for row in forecast_table if row.get('ensemble') is not None]
                
                if ensemble_preds:
                    # Use min/max of forecast as support/resistance
                    min_forecast = min(ensemble_preds)
                    max_forecast = max(ensemble_preds)
                    
                    # Add 2% buffer for better visualization
                    support_level = min_forecast * 0.98
                    resistance_level = max_forecast * 1.02
                    
                    # Override technical indicators with forecast-based levels
                    technical['support'] = support_level
                    technical['resistance'] = resistance_level
                
                # Package prediction data
                prediction_data = {
                    'ensemble': ensemble_preds
                }
                
                create_enhanced_chart(
                    combined_df=combined_df,
                    market_data=market,
                    technical=technical,
                    coin_symbol=symbol,
                    horizon_days=horizon_days,
                    prediction_data=prediction_data
                )
        else:
            st.info("Insufficient data for chart")
    else:
        st.info("No forecast data available")

# ============================================================================
# STAGE 3: ENHANCED ERROR DISPLAY (NEW)
# ============================================================================

def display_enhanced_error(result):
    """Display error with helpful suggestions"""
    error_msg = result.get('error', 'Unknown error')
    error_type = result.get('error_type', 'unknown')
    suggestion = result.get('suggestion', 'Please try again.')
    
    icon_map = {
        'validation': '⚠️',
        'timeout': '⏱️',
        'connection': '🌐',
        'data_not_found': '❌',
        'api_error': '🔌',
        'unknown': '❓'
    }
    icon = icon_map.get(error_type, '❌')
    
    st.error(f"{icon} {error_msg}")
    
    if suggestion:
        st.info(f"💡 **Suggestion:** {suggestion}")
    
    if error_type in ['timeout', 'connection', 'api_error']:
        if st.button("🔄 Retry", type="secondary"):
            st.rerun()


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
                display_enhanced_error(result)  # STAGE 3: Enhanced error display
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
















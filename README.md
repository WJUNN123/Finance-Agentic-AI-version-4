# 🚀 Finance Agentic AI - Crypto Market Analysis Agent

An intelligent **agentic AI-powered** cryptocurrency analysis system that combines live market data, sentiment analysis, hybrid ML models, and LLM reasoning with conflict resolution and self-reflection capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 🎯 Key Features

### Agentic AI Capabilities
- **Conflict Resolution**: Automatically detects and resolves conflicting signals (RSI vs forecast, sentiment vs technical, model disagreement)
- **Self-Reflection**: LLM critiques its own recommendations before finalizing, adjusting confidence when issues are found
- **Signal Priority Framework**: Structured decision-making when indicators disagree

### Data & Analysis
- **Live Market Data**: Real-time prices from CoinGecko API with retry logic and rate limiting
- **News Sentiment Analysis**: Twitter-RoBERTa model with recency-weighted scoring and confidence metrics
- **18+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ATR, OBV, and more
- **Risk Assessment**: Comprehensive analysis including liquidity, volatility, and market conditions

### Machine Learning
- **Hybrid LSTM + XGBoost Ensemble**: Balanced predictions with configurable weights
- **20+ Engineered Features**: MA distances, momentum, volatility, RSI multi-period, MACD histogram
- **Conservative Adjustments**: Trend dampening and mean reversion for realistic forecasts
- **Model Caching**: Automatic model persistence for faster subsequent analyses

### LLM Integration
- **Gemini 2.0 Flash**: AI-powered insight generation with structured JSON output
- **Multi-Model Fallback**: Automatic fallback through multiple Gemini model variants
- **Enhanced Fallback Logic**: Rule-based insights when API is unavailable

## 📊 Supported Cryptocurrencies

| Coin | Symbol | CoinGecko ID |
|------|--------|--------------|
| Bitcoin | BTC | bitcoin |
| Ethereum | ETH | ethereum |
| Binance Coin | BNB | binancecoin |
| Ripple | XRP | ripple |
| Solana | SOL | solana |
| Cardano | ADA | cardano |
| Dogecoin | DOGE | dogecoin |

## 🏗️ Architecture

```
crypto-analysis-agent/
├── app.py                      # Main Streamlit application
├── data_fetchers/
│   ├── coingecko.py           # CoinGecko API client with retry logic
│   └── news.py                # RSS feed fetcher for crypto news
├── sentiment/
│   └── analyzer.py            # Twitter-RoBERTa sentiment analysis
├── models/
│   └── hybrid_predictor.py    # LSTM + XGBoost ensemble model
├── llm/
│   └── gemini_insights.py     # Gemini LLM with conflict resolution
├── utils/
│   └── technical_indicators.py # 18+ technical indicators
├── config/
│   └── config.yaml            # Configuration settings
└── models_cache/              # Cached trained models
```

## 🔧 Technical Indicators

The system calculates the following indicators from price data:

| Category | Indicators |
|----------|------------|
| Momentum | RSI (7, 14, 21), MACD, Stochastic Oscillator (%K, %D) |
| Trend | Moving Averages (7, 14, 30, 50), Trend Identification |
| Volatility | Bollinger Bands, ATR, EWMA Volatility |
| Volume | OBV Trend (approximated from price) |
| Support/Resistance | Rolling window extrema |

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key (for AI insights)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crypto-analysis-agent.git
cd crypto-analysis-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.12.0
xgboost>=1.7.0
scikit-learn>=1.2.0
transformers>=4.30.0
torch>=2.0.0
feedparser>=6.0.0
requests>=2.28.0
plotly>=5.14.0
altair>=5.0.0
pyyaml>=6.0
google-genai>=0.3.0
```

4. **Configure API keys**

Create `.streamlit/secrets.toml`:
```toml
[gemini]
api_key = "your-gemini-api-key"

[huggingface]
token = "your-hf-token"  # Optional
```

Or set environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export HF_TOKEN="your-hf-token"  # Optional
```

5. **Run the application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

## 📖 Usage

### Natural Language Queries
```
"Should I buy Bitcoin?"
"ETH 7-day forecast"
"What's the sentiment for Solana?"
"BTC risk analysis with 14-day horizon"
```

### Response Components

1. **Recommendation Card**: BUY / HOLD / SELL with confidence score
2. **Market Overview**: Price, 24h/7d changes, market cap, volume
3. **Technical Analysis**: RSI zones, MACD, support/resistance levels
4. **Sentiment Analysis**: News sentiment breakdown with visual bar
5. **Price Forecast Chart**: LSTM, XGBoost, and ensemble predictions
6. **AI Insights**: Gemini-powered analysis with conflict resolution
7. **Risk Assessment**: Key risks and action plan

## 🧠 Agentic AI Features

### Conflict Detection
The system automatically detects conflicts between:
- **RSI vs Forecast**: When oversold/overbought signals contradict price predictions
- **Sentiment vs Technical**: When news sentiment conflicts with chart patterns
- **Model Disagreement**: When LSTM and XGBoost predict different directions

### Self-Reflection
Before finalizing recommendations, the LLM:
1. Reviews potential issues with its analysis
2. Adjusts confidence if concerns are found
3. Performs a final sanity check

### Signal Priority (when conflicts exist)
1. Model Agreement > 80% → Trust ML forecast
2. RSI extreme (< 30 or > 70) → Trust RSI signal
3. Strong confirmed trend → Trust trend direction
4. High sentiment confidence (> 70%) → Consider sentiment
5. Still unclear → Recommend HOLD

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
cryptocurrencies:
  - name: Bitcoin
    id: bitcoin
    symbol: BTC
  # ... more coins

models:
  lstm:
    window_size: 30
    epochs: 20
    batch_size: 16
  ensemble:
    lstm_weight: 0.5
    xgboost_weight: 0.5
```

## 🚢 Deployment

### Streamlit Cloud (Free Tier)

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in dashboard settings
5. Deploy!

**Free Tier Optimizations:**
- Model caching reduces repeated training
- 10-minute result caching (`@st.cache_data(ttl=600)`)
- Rate limiting for API calls
- Multi-model fallback for Gemini quota management

### Docker

```bash
docker build -t crypto-agent .
docker run -p 8501:8501 -e GEMINI_API_KEY="your-key" crypto-agent
```

## 🔌 API Reference

### Core Functions

```python
# Main analysis function
analyze_cryptocurrency(coin_id: str, horizon_days: int = 7) -> Dict

# Hybrid prediction
train_and_predict(price_series, horizon=7, coin_id='bitcoin', use_cache=True) -> Dict

# Gemini insights with conflict resolution
generate_insights(api_key, coin_symbol, market_data, sentiment_data, 
                  technical_indicators, prediction_data, top_headlines, horizon_days) -> Dict

# Technical indicators
get_all_indicators(prices: pd.Series, pct_24h=None, pct_7d=None) -> Dict

# Sentiment analysis
analyzer.analyze_texts(headlines: List[str]) -> List[Dict]
analyzer.calculate_aggregate_sentiment(results, use_recency_bias=True) -> Tuple[float, pd.DataFrame]
```

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** It does not constitute financial advice. Cryptocurrency trading involves substantial risk of loss. Always:
- Do your own research (DYOR)
- Never invest more than you can afford to lose
- Consult with a qualified financial advisor

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [CoinGecko](https://www.coingecko.com) - Cryptocurrency market data
- [Hugging Face](https://huggingface.co) - Twitter-RoBERTa sentiment model
- [Google Gemini](https://deepmind.google/technologies/gemini/) - AI insights
- [Streamlit](https://streamlit.io) - Web application framework

---

Made with ❤️ by WJ

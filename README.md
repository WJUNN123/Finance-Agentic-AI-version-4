# 🚀 Crypto Market Analysis Agent

An intelligent AI-powered cryptocurrency analysis system that helps traders make faster, data-driven investment decisions. This agent combines live market data, sentiment analysis, and machine learning to provide comprehensive crypto insights.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 🎯 Features

- **Live Data Integration**: Real-time crypto prices from CoinGecko + RSS news feeds
- **Advanced Sentiment Analysis**: Twitter-RoBERTa model for market sentiment detection
- **AI-Powered Insights**: Gemini 2.0 Flash for intelligent investment recommendations
- **Hybrid Price Prediction**: XGBoost + LSTM ensemble for accurate 7-day forecasts
- **Risk Assessment**: Comprehensive analysis including RSI, volatility, and liquidity metrics
- **Interactive Dashboard**: Beautiful Streamlit UI with real-time charts and visualizations

## 📊 Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Ripple (XRP)
- Solana (SOL)
- Cardano (ADA)
- Dogecoin (DOGE)

## 🏗️ Architecture

```
crypto-analysis-agent/
├── src/
│   ├── data_fetchers/     # Live data collection
│   ├── models/            # ML models (LSTM, XGBoost)
│   ├── sentiment/         # Twitter-RoBERTa sentiment
│   ├── llm/              # Gemini integration
│   ├── utils/            # Helper functions
│   └── app.py            # Main Streamlit app
├── config/               # Configuration files
├── tests/               # Unit tests
├── docs/                # Documentation
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key
- (Optional) Hugging Face token for sentiment model

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crypto-analysis-agent.git
cd crypto-analysis-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**

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
streamlit run src/app.py
```

Visit `http://localhost:8501` in your browser!

## 📖 Usage

### Basic Query
```
"Should I buy Bitcoin?"
"ETH 7-day forecast"
"What's the sentiment for Solana?"
```

### Advanced Analysis
```
"BTC risk analysis with 14-day horizon"
"Compare ETH and SOL for short-term trading"
"ADA technical indicators and news sentiment"
```

### Response Components

1. **Analysis Summary**: Quick overview with buy/hold/sell recommendation
2. **Market Data**: Current price, 24h/7d changes, RSI, market cap, volume
3. **Sentiment Analysis**: News sentiment breakdown (positive/neutral/negative)
4. **Risk Management**: Liquidity risk, regulatory risk, volatility metrics
5. **7-Day Forecast**: Price predictions with confidence intervals
6. **Technical Analysis**: RSI zones, momentum indicators, support/resistance
7. **AI Insights**: Gemini-powered strategic recommendations

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Data sources
data_sources:
  coingecko_api: "https://api.coingecko.com/api/v3"
  rss_feeds:
    - "https://www.coindesk.com/arc/outboundfeeds/rss/"
    - "https://cointelegraph.com/rss"

# Model settings
models:
  lstm:
    window_size: 30
    epochs: 20
    batch_size: 16
  xgboost:
    n_estimators: 100
    max_depth: 5

# Sentiment analysis
sentiment:
  model: "cardiffnlp/twitter-roberta-base-sentiment-latest"
  batch_size: 32

# LLM settings
llm:
  model: "gemini-2.0-flash"
  temperature: 0.3
  max_tokens: 500
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=src tests/
```

## 📦 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in dashboard settings
5. Deploy!

### Docker

```bash
docker build -t crypto-agent .
docker run -p 8501:8501 -e GEMINI_API_KEY="your-key" crypto-agent
```

### Heroku

```bash
heroku create your-app-name
git push heroku main
heroku config:set GEMINI_API_KEY="your-key"
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [API Keys Setup](docs/API_KEYS.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

**This tool is for educational purposes only.** It does not provide financial advice. Cryptocurrency trading involves substantial risk of loss. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CoinGecko](https://www.coingecko.com) for crypto market data
- [Hugging Face](https://huggingface.co) for sentiment analysis models
- [Google Gemini](https://deepmind.google/technologies/gemini/) for AI insights
- [Streamlit](https://streamlit.io) for the amazing web framework

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/crypto-analysis-agent](https://github.com/yourusername/crypto-analysis-agent)

---

Made with ❤️ for the crypto community
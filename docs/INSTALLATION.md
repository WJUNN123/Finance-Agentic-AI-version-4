# 📦 Installation and Deployment Guide

Complete guide for setting up and deploying the Crypto Market Analysis Agent.

## 📋 Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [API Keys Configuration](#api-keys-configuration)
3. [Testing the Application](#testing-the-application)
4. [Deployment Options](#deployment-options)
5. [Troubleshooting](#troubleshooting)

---

## 🖥️ Local Development Setup

### Prerequisites

- **Python 3.9 or higher** (check with `python --version`)
- **pip** package manager
- **Git** for version control
- **4GB+ RAM** recommended
- **Internet connection** for API calls

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/crypto-analysis-agent.git
cd crypto-analysis-agent
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all necessary packages. It may take 5-10 minutes depending on your internet speed.

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import tensorflow; print('All packages installed successfully!')"
```

---

## 🔑 API Keys Configuration

### Required API Keys

1. **Google Gemini API Key** (Required)
   - Visit: https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy the key

2. **Hugging Face Token** (Optional, for sentiment models)
   - Visit: https://huggingface.co/settings/tokens
   - Create a new token with "read" access
   - Copy the token

### Configuration Methods

#### Method 1: Streamlit Secrets (Recommended for Streamlit Cloud)

Create `.streamlit/secrets.toml`:

```bash
mkdir -p .streamlit
```

Create/edit `.streamlit/secrets.toml`:

```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY_HERE"

[huggingface]
token = "YOUR_HF_TOKEN_HERE"  # Optional
```

#### Method 2: Environment Variables (Recommended for Docker/Heroku)

**On macOS/Linux:**
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"  # Optional
```

**On Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
set HF_TOKEN=YOUR_HF_TOKEN_HERE
```

**On Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
$env:HF_TOKEN="YOUR_HF_TOKEN_HERE"
```

#### Method 3: .env File

Create `.env` file in project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
HF_TOKEN=YOUR_HF_TOKEN_HERE
```

---

## 🧪 Testing the Application

### Run Locally

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

### Test Queries

Try these sample queries:
- "Should I buy Bitcoin?"
- "ETH 7-day forecast"
- "What's the sentiment for Solana?"
- "BTC risk analysis"

### Run Unit Tests

```bash
# Install test dependencies if not already installed
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/
```

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

**Best for**: Quick deployment, free hosting, automatic updates

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub repository
   - Select the main branch
   - Set main file path: `src/app.py`

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, click your app
   - Go to Settings → Secrets
   - Paste your secrets in TOML format:
   ```toml
   [gemini]
   api_key = "your-key-here"
   ```

4. **Deploy!**
   - Click "Deploy"
   - Your app will be live in minutes at `https://your-app-name.streamlit.app`

### Option 2: Docker

**Best for**: Containerized deployment, cloud platforms

1. **Create Dockerfile** (already included in repo)

2. **Build Docker image**
   ```bash
   docker build -t crypto-agent .
   ```

3. **Run container**
   ```bash
   docker run -p 8501:8501 \
     -e GEMINI_API_KEY="your-key" \
     -e HF_TOKEN="your-token" \
     crypto-agent
   ```

4. **Access app**
   - Open http://localhost:8501

### Option 3: Heroku

**Best for**: Production deployment with custom domain

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Windows (download from heroku.com)
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create app**
   ```bash
   heroku create your-crypto-agent
   ```

4. **Add buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   ```

5. **Set environment variables**
   ```bash
   heroku config:set GEMINI_API_KEY="your-key"
   heroku config:set HF_TOKEN="your-token"
   ```

6. **Deploy**
   ```bash
   git push heroku main
   ```

7. **Open app**
   ```bash
   heroku open
   ```

### Option 4: AWS EC2

**Best for**: Full control, scalability

1. **Launch EC2 instance**
   - Ubuntu 22.04 LTS
   - t2.medium or larger (4GB+ RAM)
   - Open port 8501 in security group

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```

4. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/crypto-analysis-agent.git
   cd crypto-analysis-agent
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Set environment variables**
   ```bash
   export GEMINI_API_KEY="your-key"
   export HF_TOKEN="your-token"
   ```

6. **Run with systemd** (for auto-restart)
   
   Create `/etc/systemd/system/crypto-agent.service`:
   ```ini
   [Unit]
   Description=Crypto Analysis Agent
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/crypto-analysis-agent
   Environment="GEMINI_API_KEY=your-key"
   Environment="HF_TOKEN=your-token"
   ExecStart=/home/ubuntu/crypto-analysis-agent/venv/bin/streamlit run src/app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Start service:
   ```bash
   sudo systemctl enable crypto-agent
   sudo systemctl start crypto-agent
   ```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Module not found" errors

**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Tensorflow GPU not working

**Solution**: Install CUDA-compatible TensorFlow
```bash
pip install tensorflow-gpu==2.14.0
```

#### 3. Sentiment model download fails

**Solution**: Set Hugging Face cache directory
```bash
export TRANSFORMERS_CACHE="./models/cache"
```

#### 4. "Rate limit exceeded" from CoinGecko

**Solution**: The app has built-in rate limiting. Wait 1 minute between requests.

#### 5. Streamlit shows "Connection error"

**Solution**: Check if port 8501 is available
```bash
# macOS/Linux
lsof -i :8501

# Windows
netstat -ano | findstr :8501
```

#### 6. Gemini API errors

**Solution**: 
- Verify API key is correct
- Check your Google Cloud quota
- Ensure billing is enabled (Gemini 2.0 Flash is free but requires billing setup)

### Performance Optimization

#### Reduce Memory Usage

In `config/config.yaml`, adjust:
```yaml
models:
  lstm:
    batch_size: 8  # Reduce from 16
  sentiment:
    batch_size: 16  # Reduce from 32
```

#### Speed Up Model Loading

Enable caching:
```yaml
cache:
  enabled: true
  ttl:
    models: 7200  # Cache for 2 hours
```

### Getting Help

1. **Check logs**: Look at `logs/app.log` for error details
2. **GitHub Issues**: Open an issue at https://github.com/yourusername/crypto-analysis-agent/issues
3. **Discussions**: Ask questions in GitHub Discussions

---

## 📈 Monitoring and Maintenance

### Check Application Health

```bash
# Check if running
ps aux | grep streamlit

# Check logs
tail -f logs/app.log

# Monitor resource usage
htop  # or top on macOS
```

### Update Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Restart app
# (varies by deployment method)
```

### Backup Data

The app stores minimal data, but you may want to backup:
- Configuration: `config/config.yaml`
- Secrets: `.streamlit/secrets.toml` (encrypted!)
- Logs: `logs/` directory

---

## 🎉 Next Steps

Once deployed successfully:

1. ✅ Test all cryptocurrencies (BTC, ETH, BNB, XRP, SOL, ADA, DOGE)
2. ✅ Try different query types (forecast, sentiment, risk analysis)
3. ✅ Monitor performance and logs
4. ✅ Set up monitoring (optional: Datadog, New Relic)
5. ✅ Configure alerts for errors
6. ✅ Share with users!

---

## 📞 Support

Need help? Contact:
- Email: your-email@example.com
- Twitter: @yourhandle
- GitHub: https://github.com/yourusername

**Remember**: This tool is for educational purposes. Always do your own research before making investment decisions!

---

Made with ❤️ for the crypto community
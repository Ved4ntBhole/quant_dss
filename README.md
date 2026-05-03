# QuantDSS India — Quant-Driven Investment Decision Support System

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine-learning-powered investment decision support system for the Indian stock market, built with Python, scikit-learn, yfinance, and Streamlit.

---

## 🚀 Key Features

- 🎯 AI risk profiling via 5-question questionnaire  
- 🔍 Real-time asset analysis using 7 quantitative metrics  
- 📊 ML-powered risk classification (RandomForest, ~85–87% accuracy)  
- 💡 Personalized ETF recommendations using live NSE data  
- 🤖 AI-generated insights via Gemini API  
- 📈 Full portfolio visualizations and reports  
- 📥 Download reports as JSON, CSV, or PDF  

---

## 📁 Project Structure

```
quant_dss/
├── config/
│   ├── investment_policy.json
│   ├── etf_universe.json
│   ├── questionnaire_scoring.json
│   └── ticker_universe.json
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model_trainer.py
│   ├── recommender.py
│   ├── gemini_client.py
│   └── api_client.py
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── logs/
├── train.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/quant_dss.git
cd quant_dss
```

### 2. Create Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys.

**Optional APIs:**
- Gemini API → https://ai.google.dev  
- Alpha Vantage → https://www.alphavantage.co  

---

## 📊 Training the Model

```bash
python train.py
```

### Expected Output

```
============================================================
QuantDSS India — Model Training Pipeline
Benchmark: ^NSEI
============================================================

[Step 1] Loading price data...
  Source: yfinance (40 NSE tickers, period=2y)

[Step 2] Computing features...
  Feature matrix: 40 tickers × 7 features

[Step 3] Generating risk labels...
  Auto-labels → Low / Medium / High

[Step 4] Training RandomForest classifier...
  5-fold CV Accuracy: ~0.84
  Test Accuracy: ~0.86

Model saved → models/risk_classifier.pkl
============================================================
```

---

## ▶️ Run the App

```bash
streamlit run app/main.py
```

Open: http://localhost:8501

---

## 🎯 Usage Guide

### Page 1: Risk Questionnaire
- Answer 5 questions  
- Score (0–15)  
- Profile: Low / Medium / High  

### Page 2: Live Asset Analysis
- Enter NSE/BSE tickers  
- Computes 7 quantitative metrics  
- ML-based risk classification  

**Example:**
```
RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
```

### Page 3: Recommendations
- ETF suggestions based on risk profile  
- AI-generated summaries  
- Portfolio allocation insights  

### Page 4: Report
- JSON / CSV / PDF export  

---

## 🧠 ML Model Details

| Component | Details |
|----------|--------|
| Algorithm | RandomForestClassifier |
| Features | Volatility, Return, Sharpe, Beta, Drawdown, VaR, Downside Dev |
| Classes | Low, Medium, High |
| Accuracy | ~82–88% |

### Feature Formulas

```
Annualized Volatility = σ_daily × √252
Beta = Cov(stock, NIFTY) / Var(NIFTY)
Sharpe Ratio = (μ - rf) / σ
Max Drawdown = max((peak - trough) / peak)
VaR 95% = 5th percentile returns
Downside Deviation = std(returns < 0) × √252
```

---

## 🔑 API Integration

### Gemini API
- AI summaries
- Free tier available
- Model: gemini-2.5-flash

### Alpha Vantage
- Live price enrichment
- Limited NSE coverage

### yfinance
- Primary data source
- No API key required

---

## ⚠️ Known Limitations

- Alpha Vantage has limited Indian coverage  
- API rate limits apply  
- Data gaps possible  
- Model is not financial advice  

---

## 🚀 Deployment

### Local

```bash
streamlit run app/main.py --logger.level=debug
```

### Streamlit Cloud

1. Push to GitHub  
2. Go to https://share.streamlit.io  
3. Select repo + `app/main.py`  
4. Add secrets:
```
GEMINI_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

---

## 📊 Example Output

**Risk Profile:**
```
Medium Risk (8/15)
```

**Asset:**
```
RELIANCE.NS
Risk: Medium | Confidence: 87%
Return: 18% | Volatility: 22%
```

---

## 🤝 Contributing

1. Fork repo  
2. Create branch  
3. Commit changes  
4. Open PR  

---

## 📄 License

MIT License

---

## ⚖️ Disclaimer

Educational purposes only. Not financial advice.

---

**Happy investing! 📈**
# 📈 Stock Market Exploratory Data Analysis

End-to-end EDA of synthetic stock price data for 8 major tech companies (AAPL, GOOGL, MSFT, AMZN, META, NVDA, TSLA, NFLX) from 2019 to mid-2024 — including a simulated COVID market crash.

## 📊 What It Does

- Simulates realistic daily stock prices using Geometric Brownian Motion (GBM)
- Calculates annualised return, volatility, Sharpe Ratio, and max drawdown
- Analyses cross-asset correlations
- Visualises moving averages and rolling volatility

## 📈 Outputs

| File | Description |
|------|-------------|
| `01_normalized_prices.png` | All stocks normalised to 100 at start |
| `02_correlation_heatmap.png` | Lower-triangle return correlation matrix |
| `03_rolling_volatility.png` | 30-day rolling annualised volatility |
| `04_risk_return.png` | Risk vs Return scatter plot |
| `05_NVDA_moving_averages.png` | 50/200-day MA for NVDA |
| `stats_summary.csv` | Full performance metrics table |

## 🛠️ Tech Stack

- **Python 3.8+**
- `pandas`, `numpy` — data simulation & wrangling
- `scipy` — statistical metrics
- `matplotlib` + `seaborn` — visualizations

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/analysis.py
```

## 📁 Project Structure

```
02_stock_market_eda/
├── src/
│   └── analysis.py
├── outputs/
├── requirements.txt
└── README.md
```

## 🔍 Key Metrics Computed

- **Annualised Return** — average yearly gain
- **Sharpe Ratio** — risk-adjusted return
- **Max Drawdown** — largest peak-to-trough decline
- **30-day Rolling Volatility** — time-varying risk

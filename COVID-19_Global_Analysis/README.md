# 🦠 COVID-19 Global Data Analysis

A comprehensive exploratory data analysis of COVID-19 trends across 8 major countries, covering daily cases, deaths, recovery rates, and mortality statistics from Jan 2020 to Dec 2023.

## 📊 What It Does

- Generates realistic simulated COVID-19 time-series data
- Computes 7-day rolling averages for smoother trend visualization
- Calculates country-level mortality rates
- Produces publication-ready charts

## 📈 Outputs

| File | Description |
|------|-------------|
| `01_daily_cases_rolling.png` | 7-day rolling average for each country |
| `02_deaths_heatmap.png` | Annual deaths heatmap |
| `03_mortality_rate.png` | Horizontal bar chart of mortality rates |
| `04_cumulative_cases.png` | Cumulative case growth over time |
| `summary.txt` | Key global statistics |

## 🛠️ Tech Stack

- **Python 3.8+**
- `pandas` — data wrangling
- `numpy` — numerical simulation
- `matplotlib` + `seaborn` — visualizations

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/analysis.py
```

## 📁 Project Structure

```
01_covid_analysis/
├── src/
│   └── analysis.py       # Main analysis script
├── outputs/              # Generated charts & summary
├── requirements.txt
└── README.md
```

## 🔍 Key Insights

- Compares pandemic waves across 8 countries
- Identifies peak case periods using rolling averages
- Ranks countries by mortality rate
- Tracks total case progression over 3 years

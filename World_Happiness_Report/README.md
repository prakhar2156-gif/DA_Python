# 😊 World Happiness Report Analysis

Comprehensive analysis of simulated World Happiness Report data (2015–2023) across 60+ countries and 9 regions, including machine learning clustering.

## 📊 What It Does

- Tracks happiness trends by region over 9 years
- Identifies top/bottom countries by happiness score
- Correlates happiness with GDP, social support, health, freedom, and generosity
- Applies **K-Means clustering** + **PCA** to group countries by socioeconomic profile

## 📈 Outputs

| File | Description |
|------|-------------|
| `01_top_bottom_countries.png` | Top 10 happiest vs bottom 10 |
| `02_regional_trends.png` | Regional happiness over time (2015–2023) |
| `03_factor_correlations.png` | Correlation heatmap |
| `04_gdp_vs_happiness.png` | Scatter: GDP per capita vs score |
| `05_kmeans_clusters.png` | PCA-projected country clusters |
| `country_clusters.csv` | Cluster assignments per country |

## 🛠️ Tech Stack

- `pandas`, `numpy` — data engineering
- `matplotlib`, `seaborn` — visualizations
- `scikit-learn` — KMeans clustering + PCA

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/analysis.py
```

## 📁 Project Structure

```
04_world_happiness/
├── src/
│   └── analysis.py
├── outputs/
├── requirements.txt
└── README.md
```

## 🔍 Factors Analysed

- GDP per capita
- Social support
- Healthy life expectancy
- Freedom to make life choices
- Generosity
- Perceptions of corruption

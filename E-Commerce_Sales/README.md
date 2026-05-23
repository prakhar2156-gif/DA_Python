# 🛒 E-Commerce Sales Analysis

A full data analysis pipeline on 50,000 synthetic e-commerce orders across categories, regions, and payment methods — including RFM customer segmentation.

## 📊 What It Does

- Generates realistic transaction data with seasonal patterns (Q4 spike, weekend boost)
- Computes monthly revenue with trend line
- Segments customers using **RFM (Recency, Frequency, Monetary)** analysis
- Breaks down revenue by category, region, and payment method

## 📈 Outputs

| File | Description |
|------|-------------|
| `01_monthly_revenue.png` | Bar + moving average revenue chart |
| `02_category_breakdown.png` | Revenue by category (bar + pie) |
| `03_rfm_segments.png` | Customer segmentation count & revenue |
| `04_region_category_heatmap.png` | Revenue heatmap by region × category |
| `05_payment_methods.png` | Revenue split by payment mode |
| `rfm_segments.csv` | Full RFM scoring table |

## 🛠️ Tech Stack

- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualizations

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python src/analysis.py
```

## 📁 Project Structure

```
03_ecommerce_sales/
├── src/
│   └── analysis.py
├── outputs/
├── requirements.txt
└── README.md
```

## 🔍 Customer Segments

| Segment | Description |
|---------|-------------|
| **Champions** | Recent, frequent, high-value buyers |
| **Loyal** | Regular customers with good spend |
| **Potential** | Promising but not yet consistent |
| **At-Risk** | Haven't purchased recently |

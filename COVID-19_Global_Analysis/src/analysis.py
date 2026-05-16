"""
COVID-19 Global Data Analysis
==============================
Analyzes global COVID-19 trends including cases, deaths, and recovery rates.
Generates visualizations and summary statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#0d1117",
                     "axes.facecolor": "#161b22", "text.color": "white",
                     "axes.labelcolor": "white", "xtick.color": "white",
                     "ytick.color": "white", "axes.titlecolor": "white"})

COUNTRIES = ["USA", "India", "Brazil", "UK", "France", "Germany", "Russia", "Italy"]
COLORS    = ["#ff4f4f", "#ff9f43", "#48dbfb", "#1dd1a1",
             "#feca57", "#ff6b9d", "#c8d6e5", "#a29bfe"]


# ── Data Generation ────────────────────────────────────────────────────────────
np.random.seed(42)

def generate_covid_data() -> pd.DataFrame:
    """Simulate realistic COVID-19 time-series data for multiple countries."""
    dates = pd.date_range(start="2020-01-22", end="2023-12-31", freq="D")
    records = []

    peak_cases = {"USA": 900_000, "India": 400_000, "Brazil": 300_000,
                  "UK": 250_000, "France": 500_000, "Germany": 350_000,
                  "Russia": 200_000, "Italy": 220_000}

    for country in COUNTRIES:
        peak = peak_cases[country]
        n    = len(dates)
        t    = np.linspace(0, 4 * np.pi, n)
        base = (np.sin(t) * 0.5 + 0.5) * peak
        noise        = np.random.normal(0, peak * 0.05, n)
        daily_cases  = np.clip(base + noise, 0, None).astype(int)
        daily_deaths = (daily_cases * np.random.uniform(0.01, 0.03, n)).astype(int)
        daily_recov  = (daily_cases * np.random.uniform(0.85, 0.95, n)).astype(int)

        for i, d in enumerate(dates):
            records.append({
                "date": d, "country": country,
                "daily_cases":    daily_cases[i],
                "daily_deaths":   daily_deaths[i],
                "daily_recovered": daily_recov[i],
                "total_cases":    int(daily_cases[:i+1].sum()),
                "total_deaths":   int(daily_deaths[:i+1].sum()),
            })

    return pd.DataFrame(records)


# ── Analysis Functions ─────────────────────────────────────────────────────────
def compute_mortality_rate(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.groupby("country").last().reset_index()
    latest["mortality_rate"] = (latest["total_deaths"] / latest["total_cases"] * 100).round(2)
    return latest[["country", "total_cases", "total_deaths", "mortality_rate"]]


def weekly_rolling(df: pd.DataFrame, country: str) -> pd.DataFrame:
    sub = df[df["country"] == country].copy()
    sub["7day_avg"] = sub["daily_cases"].rolling(7, min_periods=1).mean()
    return sub


# ── Visualisations ─────────────────────────────────────────────────────────────
def plot_daily_cases_rolling(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("COVID-19  ·  7-Day Rolling Average Daily Cases by Country",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    for ax, country, color in zip(axes.flat, COUNTRIES, COLORS):
        sub = weekly_rolling(df, country)
        ax.fill_between(sub["date"], sub["7day_avg"], alpha=0.3, color=color)
        ax.plot(sub["date"], sub["7day_avg"], color=color, linewidth=1.5)
        ax.set_title(country, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))))
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_daily_cases_rolling.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_mortality_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="country", columns=df["date"].dt.year,
        values="daily_deaths", aggfunc="sum"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Total Deaths"})
    ax.set_title("COVID-19  ·  Annual Deaths by Country", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Country")
    path = os.path.join(OUTPUT_DIR, "02_deaths_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_mortality_rate_bar(df: pd.DataFrame):
    stats = compute_mortality_rate(df).sort_values("mortality_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(stats["country"], stats["mortality_rate"],
                   color=sns.color_palette("coolwarm", len(stats)))
    ax.bar_label(bars, fmt="%.2f%%", padding=4, color="white")
    ax.set_title("COVID-19  ·  Overall Mortality Rate by Country",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Mortality Rate (%)")
    ax.set_xlim(0, stats["mortality_rate"].max() * 1.25)
    path = os.path.join(OUTPUT_DIR, "03_mortality_rate.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_total_cases_line(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    for country, color in zip(COUNTRIES, COLORS):
        sub = df[df["country"] == country]
        ax.plot(sub["date"], sub["total_cases"] / 1e6,
                label=country, color=color, linewidth=2)
    ax.set_title("COVID-19  ·  Cumulative Cases (Millions)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Cases (M)")
    ax.legend(loc="upper left", framealpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    path = os.path.join(OUTPUT_DIR, "04_cumulative_cases.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def generate_summary(df: pd.DataFrame):
    stats = compute_mortality_rate(df)
    summary = {
        "generated_at":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "countries_analysed": COUNTRIES,
        "date_range":       f"{df['date'].min().date()} → {df['date'].max().date()}",
        "global_cases":     int(df.groupby("country")["total_cases"].last().sum()),
        "global_deaths":    int(df.groupby("country")["total_deaths"].last().sum()),
        "avg_mortality_%":  round(stats["mortality_rate"].mean(), 2),
        "highest_mortality": stats.loc[stats["mortality_rate"].idxmax(), "country"],
        "lowest_mortality":  stats.loc[stats["mortality_rate"].idxmin(), "country"],
    }
    path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k:<25}: {v}\n")
    print(f"  Saved → {path}")
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n🦠  COVID-19 Global Analysis")
    print("=" * 45)

    print("  Generating dataset …")
    df = generate_covid_data()
    df.to_csv(os.path.join(OUTPUT_DIR, "covid_data.csv"), index=False)
    print(f"  Dataset shape: {df.shape}")

    print("\n  Plotting …")
    plot_daily_cases_rolling(df)
    plot_mortality_heatmap(df)
    plot_mortality_rate_bar(df)
    plot_total_cases_line(df)

    print("\n  Summary Statistics")
    print("  " + "-" * 40)
    summary = generate_summary(df)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    print("\n✅  All outputs saved to /outputs/\n")


if __name__ == "__main__":
    main()

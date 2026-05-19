"""
Stock Market Exploratory Data Analysis
========================================
Simulates and analyses stock price data for top tech companies.
Covers returns, volatility, correlation, and moving averages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fafafa", "axes.facecolor": "#ffffff",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "figure.dpi": 120,
})
sns.set_palette("Set2")

TICKERS  = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "NFLX"]
START    = "2019-01-01"
END      = "2024-06-30"


# ── Data Generation ────────────────────────────────────────────────────────────
np.random.seed(99)

INIT_PRICES = {"AAPL": 157, "GOOGL": 1070, "MSFT": 104, "AMZN": 1750,
               "META": 131, "NVDA": 34,  "TSLA": 63,  "NFLX": 267}
ANNUAL_RET  = {"AAPL": 0.32, "GOOGL": 0.22, "MSFT": 0.30, "AMZN": 0.18,
               "META": 0.15, "NVDA": 0.55,  "TSLA": 0.45, "NFLX": 0.12}
VOLATILITY  = {"AAPL": 0.22, "GOOGL": 0.24, "MSFT": 0.21, "AMZN": 0.28,
               "META": 0.32, "NVDA": 0.50,  "TSLA": 0.65, "NFLX": 0.38}


def generate_prices() -> pd.DataFrame:
    dates = pd.bdate_range(START, END)
    n     = len(dates)
    price_dict = {}

    for tk in TICKERS:
        mu      = ANNUAL_RET[tk] / 252
        sigma   = VOLATILITY[tk] / np.sqrt(252)
        shocks  = np.random.normal(mu, sigma, n)
        # Add a market crash event (March 2020)
        crash_mask = (dates >= "2020-02-20") & (dates <= "2020-03-23")
        shocks[crash_mask] -= np.random.uniform(0.03, 0.06, crash_mask.sum())
        cumulative   = np.exp(np.cumsum(shocks))
        price_dict[tk] = INIT_PRICES[tk] * cumulative

    df = pd.DataFrame(price_dict, index=dates)
    df.index.name = "date"
    return df


# ── Analysis ───────────────────────────────────────────────────────────────────
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_stats(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tk in TICKERS:
        r = returns[tk]
        rows.append({
            "Ticker":        tk,
            "Ann. Return %": round((1 + r.mean()) ** 252 - 1, 4) * 100,
            "Ann. Volatility %": round(r.std() * np.sqrt(252), 4) * 100,
            "Sharpe Ratio":  round((r.mean() * 252) / (r.std() * np.sqrt(252)), 2),
            "Max Drawdown %": round(((prices[tk] / prices[tk].cummax()) - 1).min() * 100, 2),
            "Skewness":      round(stats.skew(r), 3),
            "Kurtosis":      round(stats.kurtosis(r), 3),
        })
    return pd.DataFrame(rows).set_index("Ticker")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_normalized_prices(prices: pd.DataFrame):
    normed = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(14, 6))
    for tk in TICKERS:
        ax.plot(normed.index, normed[tk], label=tk, linewidth=2)
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("Normalised Stock Prices (Base = 100)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Indexed Price")
    ax.legend(ncol=4, loc="upper left")
    # Shade COVID crash
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-23"),
               alpha=0.15, color="red", label="COVID Crash")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_normalized_prices.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_correlation_heatmap(returns: pd.DataFrame):
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Return Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_rolling_volatility(returns: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    for tk in TICKERS:
        vol = returns[tk].rolling(30).std() * np.sqrt(252) * 100
        ax.plot(vol.index, vol, label=tk, linewidth=1.5, alpha=0.85)
    ax.set_title("30-Day Rolling Annualised Volatility (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Volatility (%)")
    ax.legend(ncol=4, loc="upper right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_rolling_volatility.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_risk_return(stats_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("Set2", len(TICKERS))
    for i, (tk, row) in enumerate(stats_df.iterrows()):
        ax.scatter(row["Ann. Volatility %"], row["Ann. Return %"],
                   s=180, color=colors[i], zorder=5)
        ax.annotate(tk, (row["Ann. Volatility %"], row["Ann. Return %"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10, fontweight="bold")
    ax.set_title("Risk vs Return", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_risk_return.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_ma_single(prices: pd.DataFrame, ticker: str = "AAPL"):
    sub = prices[[ticker]].copy()
    sub["MA_50"]  = sub[ticker].rolling(50).mean()
    sub["MA_200"] = sub[ticker].rolling(200).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub.index, sub[ticker],   color="#2d3436", linewidth=1.2, label="Price", alpha=0.8)
    ax.plot(sub.index, sub["MA_50"],  color="#e17055", linewidth=2,   label="50-day MA")
    ax.plot(sub.index, sub["MA_200"], color="#0984e3", linewidth=2,   label="200-day MA")
    ax.set_title(f"{ticker}  ·  Price with Moving Averages", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"05_{ticker}_moving_averages.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n📈  Stock Market EDA")
    print("=" * 45)

    print("  Generating price data …")
    prices  = generate_prices()
    returns = compute_returns(prices)
    prices.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"))
    returns.to_csv(os.path.join(OUTPUT_DIR, "returns.csv"))

    stats_df = compute_stats(returns, prices)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "stats_summary.csv"))

    print("\n  Performance Summary")
    print("  " + "-" * 55)
    print(stats_df.to_string())

    print("\n  Plotting …")
    plot_normalized_prices(prices)
    plot_correlation_heatmap(returns)
    plot_rolling_volatility(returns)
    plot_risk_return(stats_df)
    plot_ma_single(prices, "NVDA")

    print("\n✅  All outputs saved to /outputs/\n")


if __name__ == "__main__":
    main()

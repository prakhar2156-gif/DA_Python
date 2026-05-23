"""
E-Commerce Sales Analysis
===========================
Analyses synthetic e-commerce transaction data to uncover:
- Revenue trends by month / category / region
- Customer segmentation (RFM)
- Top products and cohort patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

CATEGORIES  = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books", "Beauty"]
REGIONS     = ["North", "South", "East", "West", "Central"]
PAYMENT_MET = ["Credit Card", "UPI", "Net Banking", "COD", "Wallet"]
N_ORDERS    = 50_000


# ── Data Generation ────────────────────────────────────────────────────────────
np.random.seed(7)

def generate_orders() -> pd.DataFrame:
    dates  = pd.date_range("2021-01-01", "2023-12-31")
    # Weight more orders towards weekends & Q4
    weights        = np.ones(len(dates))
    weights[dates.dayofweek >= 5] *= 1.4
    weights[(dates.month >= 10)] *= 1.6
    weights /= weights.sum()

    order_dates = np.random.choice(dates, size=N_ORDERS, replace=True, p=weights)

    cat_weights    = [0.30, 0.20, 0.18, 0.12, 0.10, 0.10]
    categories     = np.random.choice(CATEGORIES, N_ORDERS, p=cat_weights)
    base_prices    = {"Electronics": 450, "Clothing": 90, "Home & Kitchen": 150,
                      "Sports": 120, "Books": 25, "Beauty": 60}
    unit_prices = np.array([base_prices[c] * np.random.uniform(0.5, 2.5)
                            for c in categories]).round(2)
    quantities  = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5], N_ORDERS)
    discounts   = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], N_ORDERS,
                                   p=[0.55, 0.15, 0.15, 0.10, 0.05])

    df = pd.DataFrame({
        "order_id":      range(1, N_ORDERS + 1),
        "customer_id":   np.random.randint(1000, 15000, N_ORDERS),
        "order_date":    order_dates,
        "category":      categories,
        "region":        np.random.choice(REGIONS, N_ORDERS),
        "payment_method": np.random.choice(PAYMENT_MET, N_ORDERS),
        "unit_price":    unit_prices,
        "quantity":      quantities,
        "discount":      discounts,
    })
    df["revenue"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).round(2)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["month"]  = df["order_date"].dt.to_period("M")
    df["year"]   = df["order_date"].dt.year
    return df.sort_values("order_date").reset_index(drop=True)


# ── RFM Segmentation ───────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    rfm = df.groupby("customer_id").agg(
        Recency   = ("order_date", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("order_id",   "count"),
        Monetary  = ("revenue",    "sum"),
    ).reset_index()

    for col, labels in [("R", [4, 3, 2, 1]), ("F", [1, 2, 3, 4]), ("M", [1, 2, 3, 4])]:
        src = "Recency" if col == "R" else ("Frequency" if col == "F" else "Monetary")
        rfm[f"{col}_score"] = pd.qcut(rfm[src], 4, labels=labels, duplicates="drop")

    rfm["RFM_Score"] = (rfm["R_score"].astype(int) +
                        rfm["F_score"].astype(int) +
                        rfm["M_score"].astype(int))

    def segment(score):
        if score >= 10: return "Champions"
        elif score >= 8: return "Loyal"
        elif score >= 6: return "Potential"
        else: return "At-Risk"

    rfm["Segment"] = rfm["RFM_Score"].apply(segment)
    return rfm


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_monthly_revenue(df: pd.DataFrame):
    monthly = df.groupby("month")["revenue"].sum().reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()
    monthly["MA3"] = monthly["revenue"].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly["month_dt"], monthly["revenue"] / 1e6,
           color="#74b9ff", alpha=0.8, label="Monthly Revenue")
    ax.plot(monthly["month_dt"], monthly["MA3"] / 1e6,
            color="#d63031", linewidth=2.5, label="3-Month MA")
    ax.set_title("Monthly Revenue (₹ Millions)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Revenue (M)"); ax.set_xlabel("Month")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_monthly_revenue.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_category_breakdown(df: pd.DataFrame):
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar
    bars = axes[0].bar(cat_rev.index, cat_rev.values / 1e6,
                       color=sns.color_palette("pastel"))
    axes[0].set_title("Revenue by Category", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Revenue (₹M)"); axes[0].set_xlabel("")
    axes[0].bar_label(bars, fmt="₹%.1fM", padding=3, fontsize=9)
    axes[0].tick_params(axis="x", rotation=20)

    # Pie
    axes[1].pie(cat_rev.values, labels=cat_rev.index,
                autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("Set3"))
    axes[1].set_title("Share of Revenue", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_category_breakdown.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_rfm_segments(rfm: pd.DataFrame):
    seg_counts = rfm["Segment"].value_counts()
    seg_rev    = rfm.groupby("Segment")["Monetary"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"Champions": "#00b894", "Loyal": "#0984e3",
               "Potential": "#fdcb6e", "At-Risk": "#d63031"}

    axes[0].bar(seg_counts.index, seg_counts.values,
                color=[palette[s] for s in seg_counts.index])
    axes[0].set_title("Customer Count by RFM Segment", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Customers")

    axes[1].bar(seg_rev.index, seg_rev.values / 1e6,
                color=[palette[s] for s in seg_rev.index])
    axes[1].set_title("Revenue by RFM Segment", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Revenue (₹M)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:.1f}M"))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_rfm_segments.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_region_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="region", columns="category", values="revenue",
        aggfunc="sum"
    ) / 1e6

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Revenue (₹M)"})
    ax.set_title("Revenue by Region & Category (₹M)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_region_category_heatmap.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_payment_methods(df: pd.DataFrame):
    pm = df.groupby("payment_method")["revenue"].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(pm.index, pm.values / 1e6,
                   color=sns.color_palette("husl", len(pm)))
    ax.bar_label(bars, fmt="₹%.1fM", padding=4, fontsize=9)
    ax.set_title("Revenue by Payment Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("Revenue (₹M)"); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_payment_methods.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n🛒  E-Commerce Sales Analysis")
    print("=" * 45)

    print("  Generating orders …")
    df = generate_orders()
    df.to_csv(os.path.join(OUTPUT_DIR, "orders.csv"), index=False)
    print(f"  Orders: {len(df):,}  |  Revenue: ₹{df['revenue'].sum()/1e6:.1f}M")

    snapshot = pd.Timestamp("2024-01-01")
    rfm      = compute_rfm(df, snapshot)
    rfm.to_csv(os.path.join(OUTPUT_DIR, "rfm_segments.csv"), index=False)

    print("\n  RFM Segment Summary")
    print("  " + "-" * 40)
    print(rfm.groupby("Segment")[["Frequency", "Monetary"]].mean().round(2).to_string())

    print("\n  Plotting …")
    plot_monthly_revenue(df)
    plot_category_breakdown(df)
    plot_rfm_segments(rfm)
    plot_region_heatmap(df)
    plot_payment_methods(df)

    print("\n✅  All outputs saved to /outputs/\n")


if __name__ == "__main__":
    main()

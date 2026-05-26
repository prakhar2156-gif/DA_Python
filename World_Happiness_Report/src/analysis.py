"""
World Happiness Report Analysis
==================================
Analyses World Happiness Report data (2015-2023) to explore:
- Happiness scores by country and region
- Correlation between happiness and socioeconomic factors
- Trends over time
- Country clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

YEARS   = list(range(2015, 2024))
REGIONS = ["Western Europe", "North America", "Australia & NZ",
           "Latin America", "Eastern Europe", "Middle East",
           "Southeast Asia", "South Asia", "Sub-Saharan Africa"]

REGION_COUNTRIES = {
    "Western Europe":   ["Finland", "Denmark", "Iceland", "Switzerland", "Netherlands",
                         "Norway", "Sweden", "Austria", "Germany", "France", "UK", "Belgium"],
    "North America":    ["Canada", "USA", "Mexico"],
    "Australia & NZ":   ["Australia", "New Zealand"],
    "Latin America":    ["Brazil", "Argentina", "Chile", "Colombia", "Peru", "Uruguay"],
    "Eastern Europe":   ["Czech Republic", "Poland", "Slovakia", "Romania", "Hungary", "Russia"],
    "Middle East":      ["Israel", "UAE", "Saudi Arabia", "Kuwait", "Jordan", "Lebanon"],
    "Southeast Asia":   ["Singapore", "Thailand", "Malaysia", "Philippines", "Vietnam", "Indonesia"],
    "South Asia":       ["India", "Pakistan", "Bangladesh", "Sri Lanka", "Nepal"],
    "Sub-Saharan Africa": ["South Africa", "Ghana", "Nigeria", "Kenya", "Ethiopia", "Tanzania"],
}

REGION_BASE_HAPPINESS = {
    "Western Europe": 7.2, "North America": 7.1, "Australia & NZ": 7.3,
    "Latin America": 6.0, "Eastern Europe": 5.6, "Middle East": 5.5,
    "Southeast Asia": 5.8, "South Asia": 4.4, "Sub-Saharan Africa": 4.0,
}

FACTORS = ["gdp_per_capita", "social_support", "health_life_expectancy",
           "freedom", "generosity", "corruption_perception"]


# ── Data Generation ────────────────────────────────────────────────────────────
np.random.seed(21)

def generate_dataset() -> pd.DataFrame:
    rows = []
    for region, countries in REGION_COUNTRIES.items():
        base = REGION_BASE_HAPPINESS[region]
        for country in countries:
            country_offset = np.random.uniform(-0.8, 0.8)
            for year in YEARS:
                trend = np.random.uniform(-0.05, 0.08)
                happiness = np.clip(
                    base + country_offset + trend * (year - 2015) + np.random.normal(0, 0.15),
                    1.5, 9.5
                )
                rows.append({
                    "year":    year,
                    "country": country,
                    "region":  region,
                    "happiness_score": round(happiness, 3),
                    "gdp_per_capita":          round(np.clip(happiness * 0.12 + np.random.normal(0, 0.06), 0, 2.0), 3),
                    "social_support":          round(np.clip(happiness * 0.10 + np.random.normal(0, 0.05), 0, 1.5), 3),
                    "health_life_expectancy":  round(np.clip(happiness * 0.08 + np.random.normal(0, 0.04), 0, 1.2), 3),
                    "freedom":                 round(np.clip(np.random.uniform(0.2, 0.7), 0, 1.0), 3),
                    "generosity":              round(np.clip(np.random.uniform(0.0, 0.5), 0, 0.7), 3),
                    "corruption_perception":   round(np.clip(np.random.uniform(0.0, 0.4), 0, 0.6), 3),
                })
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_top_bottom(df: pd.DataFrame, year: int = 2023):
    latest = df[df["year"] == year].sort_values("happiness_score", ascending=False)
    top10  = latest.head(10)
    bot10  = latest.tail(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_top = sns.color_palette("Greens_r", 10)
    colors_bot = sns.color_palette("Reds",     10)

    for ax, data, colors, title in zip(
        axes, [top10, bot10], [colors_top, colors_bot],
        ["🏆 Top 10 Happiest Countries", "😔 Bottom 10 Countries"]
    ):
        bars = ax.barh(data["country"][::-1], data["happiness_score"][::-1], color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Happiness Score")
        ax.set_xlim(0, 10)

    plt.suptitle(f"World Happiness Report {year}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_top_bottom_countries.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_regional_trends(df: pd.DataFrame):
    trend = df.groupby(["year", "region"])["happiness_score"].mean().reset_index()
    palette = sns.color_palette("tab10", len(REGIONS))

    fig, ax = plt.subplots(figsize=(13, 6))
    for region, color in zip(REGIONS, palette):
        sub = trend[trend["region"] == region]
        ax.plot(sub["year"], sub["happiness_score"], marker="o",
                label=region, color=color, linewidth=2, markersize=5)

    ax.set_title("Average Happiness Score by Region Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Happiness Score")
    ax.set_xticks(YEARS)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_regional_trends.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_factor_correlations(df: pd.DataFrame):
    latest = df[df["year"] == 2023]
    corr_df = latest[["happiness_score"] + FACTORS].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Happiness vs Factor Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_factor_correlations.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_gdp_vs_happiness(df: pd.DataFrame, year: int = 2023):
    sub = df[df["year"] == year].copy()
    palette = dict(zip(REGIONS, sns.color_palette("tab10", len(REGIONS))))

    fig, ax = plt.subplots(figsize=(11, 7))
    for region in REGIONS:
        rd = sub[sub["region"] == region]
        ax.scatter(rd["gdp_per_capita"], rd["happiness_score"],
                   label=region, color=palette[region], s=80, alpha=0.85, edgecolors="white")
        for _, row in rd.iterrows():
            ax.annotate(row["country"],
                        (row["gdp_per_capita"], row["happiness_score"]),
                        fontsize=6.5, alpha=0.7,
                        textcoords="offset points", xytext=(4, 2))

    ax.set_title(f"GDP per Capita vs Happiness Score ({year})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("GDP per Capita (normalized)")
    ax.set_ylabel("Happiness Score")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_gdp_vs_happiness.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")


def plot_kmeans_clusters(df: pd.DataFrame, year: int = 2023, k: int = 4):
    sub    = df[df["year"] == year].copy()
    X      = sub[["happiness_score", "gdp_per_capita", "social_support",
                  "health_life_expectancy"]].fillna(0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_sc)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sc)

    sub["cluster"]  = labels
    sub["PC1"]      = X_pca[:, 0]
    sub["PC2"]      = X_pca[:, 1]

    colors = sns.color_palette("Set1", k)
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(k):
        cd = sub[sub["cluster"] == c]
        ax.scatter(cd["PC1"], cd["PC2"], color=colors[c],
                   label=f"Cluster {c+1}", s=80, alpha=0.8, edgecolors="white")
        for _, row in cd.iterrows():
            ax.annotate(row["country"], (row["PC1"], row["PC2"]),
                        fontsize=6, alpha=0.65,
                        textcoords="offset points", xytext=(3, 2))

    ax.set_title(f"K-Means Country Clusters (k={k}) — PCA Projection",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_kmeans_clusters.png")
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved → {path}")
    return sub[["country", "region", "happiness_score", "cluster"]]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n😊  World Happiness Analysis")
    print("=" * 45)

    print("  Generating dataset …")
    df = generate_dataset()
    df.to_csv(os.path.join(OUTPUT_DIR, "happiness_data.csv"), index=False)
    print(f"  Records: {len(df):,}  |  Countries: {df['country'].nunique()}")

    print("\n  Plotting …")
    plot_top_bottom(df)
    plot_regional_trends(df)
    plot_factor_correlations(df)
    plot_gdp_vs_happiness(df)
    clusters = plot_kmeans_clusters(df)
    clusters.to_csv(os.path.join(OUTPUT_DIR, "country_clusters.csv"), index=False)

    latest = df[df["year"] == 2023]
    print("\n  2023 Global Summary")
    print("  " + "-" * 35)
    print(f"  Global Avg Score : {latest['happiness_score'].mean():.2f}")
    print(f"  Happiest Country : {latest.nlargest(1, 'happiness_score')['country'].values[0]}")
    print(f"  Lowest Country   : {latest.nsmallest(1, 'happiness_score')['country'].values[0]}")

    print("\n✅  All outputs saved to /outputs/\n")


if __name__ == "__main__":
    main()

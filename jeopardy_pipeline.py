"""
How to Win Jeopardy! — Elevated Strategy Analysis Pipeline
Project Elevate — Ian P. Cox

Analyses:
  1.  Data overview & round distribution
  2.  Top categories by frequency
  3.  Value distribution by round
  4.  Category difficulty (answer-in-question ratio)
  5.  Daily Double location strategy (by round and value)
  6.  High-value vs. low-value clue word clouds (TF-IDF)
  7.  NLP: top TF-IDF terms by category cluster
  8.  Temporal trends: category popularity over seasons
  9.  Answer length distribution (shorter answers = easier?)
  10. Chi-squared: term enrichment in high-value clues
  11. Logistic regression: predicting high-value clues
  12. Feature importance for the prediction model
  + Interactive Plotly dashboard
"""

import warnings
from pathlib import Path
import re
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              ConfusionMatrixDisplay, roc_curve)
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings("ignore")

DATA_PATH = Path("/home/ubuntu/jeopardy.csv")
OUT_DIR   = Path("/home/ubuntu/jeopardy_outputs")
OUT_DIR.mkdir(exist_ok=True)

JEOPARDY_BLUE  = "#060CE9"
JEOPARDY_GOLD  = "#FFD700"
JEOPARDY_BLACK = "#1A1A2E"
JEOPARDY_RED   = "#E74C3C"
JEOPARDY_TEAL  = "#1ABC9C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.facecolor": "#F8F9FA",
    "figure.facecolor": "white",
})


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Clean value column
    df["Value"] = (df["Value"].astype(str)
                   .str.replace(r"[\$,]", "", regex=True)
                   .str.strip())
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"]).copy()

    # Parse air date
    df["Air Date"] = pd.to_datetime(df["Air Date"], errors="coerce")
    df["Year"] = df["Air Date"].dt.year

    # Normalize text
    def norm(s):
        if pd.isna(s): return ""
        return re.sub(r"[^a-z0-9\s]", "", str(s).lower())

    df["clean_q"] = df["Question"].apply(norm)
    df["clean_a"] = df["Answer"].apply(norm)
    df["Category"] = df["Category"].str.strip().str.upper()

    # High-value flag (above median value)
    median_val = df["Value"].median()
    df["high_value"] = (df["Value"] >= median_val).astype(int)

    # Answer-in-question ratio
    def aiq(row):
        q = set(row["clean_q"].split())
        a = [w for w in row["clean_a"].split() if w and w != "the"]
        if not a: return 0.0
        return sum(1 for w in a if w in q) / len(a)
    df["aiq"] = df.apply(aiq, axis=1)

    # Answer word count
    df["answer_words"] = df["clean_a"].str.split().str.len()
    df["question_words"] = df["clean_q"].str.split().str.len()

    print(f"Dataset loaded: {len(df):,} clues")
    print(f"  Rounds: {df['Round'].unique().tolist()}")
    print(f"  Value range: ${df['Value'].min():,.0f} – ${df['Value'].max():,.0f}")
    print(f"  Date range: {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Unique categories: {df['Category'].nunique():,}")
    return df


# ── Chart 1: Dataset Overview ─────────────────────────────────────────────────

def plot_overview(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Round distribution
    round_counts = df["Round"].value_counts()
    axes[0].bar(round_counts.index, round_counts.values,
                color=[JEOPARDY_BLUE, JEOPARDY_GOLD, JEOPARDY_RED][:len(round_counts)])
    axes[0].set_title("Clues by Round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Number of Clues")
    for i, v in enumerate(round_counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)

    # Value distribution
    axes[1].hist(df["Value"], bins=30, color=JEOPARDY_GOLD, edgecolor="black", alpha=0.8)
    axes[1].axvline(df["Value"].median(), color=JEOPARDY_RED, lw=2, linestyle="--",
                    label=f"Median: ${df['Value'].median():,.0f}")
    axes[1].set_title("Clue Value Distribution")
    axes[1].set_xlabel("Value ($)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Clues per year
    year_counts = df.groupby("Year").size().reset_index(name="count")
    year_counts = year_counts.dropna()
    axes[2].plot(year_counts["Year"], year_counts["count"],
                 color=JEOPARDY_BLUE, lw=2, marker="o", markersize=4)
    axes[2].set_title("Clues per Season Year")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Number of Clues")
    axes[2].fill_between(year_counts["Year"], year_counts["count"],
                         alpha=0.15, color=JEOPARDY_BLUE)

    plt.suptitle("Jeopardy! Dataset Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_overview.png")
    plt.close()
    print("Saved: 01_overview.png")


# ── Chart 2: Top Categories ───────────────────────────────────────────────────

def plot_top_categories(df):
    top_cats = df["Category"].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [JEOPARDY_BLUE if i < 5 else JEOPARDY_GOLD for i in range(len(top_cats))]
    bars = ax.barh(top_cats.index[::-1], top_cats.values[::-1], color=colors[::-1])
    ax.set_title("Top 20 Most Frequent Jeopardy! Categories", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Clues")
    for bar, val in zip(bars, top_cats.values[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{val}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_top_categories.png")
    plt.close()
    print("Saved: 02_top_categories.png")


# ── Chart 3: Value by Round ───────────────────────────────────────────────────

def plot_value_by_round(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    rounds = ["Jeopardy!", "Double Jeopardy!", "Final Jeopardy!"]
    data_by_round = [df[df["Round"] == r]["Value"].dropna() for r in rounds if r in df["Round"].values]
    labels_used = [r for r in rounds if r in df["Round"].values]
    bp = axes[0].boxplot(data_by_round, labels=[r.replace("!", "!") for r in labels_used],
                         patch_artist=True,
                         boxprops=dict(facecolor=JEOPARDY_GOLD, alpha=0.7))
    axes[0].set_title("Value Distribution by Round")
    axes[0].set_ylabel("Clue Value ($)")
    axes[0].set_xlabel("Round")

    # Mean value heatmap by category (top 15)
    top15 = df["Category"].value_counts().head(15).index
    cat_val = df[df["Category"].isin(top15)].groupby("Category")["Value"].mean().sort_values(ascending=False)
    axes[1].barh(cat_val.index[::-1], cat_val.values[::-1],
                 color=JEOPARDY_BLUE, alpha=0.8)
    axes[1].set_title("Mean Clue Value — Top 15 Categories")
    axes[1].set_xlabel("Mean Value ($)")
    for i, v in enumerate(cat_val.values[::-1]):
        axes[1].text(v + 5, i, f"${v:,.0f}", va="center", fontsize=8)

    plt.suptitle("Jeopardy! Value Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_value_by_round.png")
    plt.close()
    print("Saved: 03_value_by_round.png")


# ── Chart 4: Answer-in-Question (Difficulty Proxy) ───────────────────────────

def plot_aiq(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of AIQ ratio
    axes[0].hist(df["aiq"], bins=30, color=JEOPARDY_TEAL, edgecolor="black", alpha=0.8)
    axes[0].axvline(df["aiq"].mean(), color=JEOPARDY_RED, lw=2, linestyle="--",
                    label=f"Mean: {df['aiq'].mean():.3f}")
    axes[0].set_title("Answer-in-Question Ratio Distribution\n(Higher = answer is hinted in the clue)")
    axes[0].set_xlabel("AIQ Ratio")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # AIQ by value bucket
    df["value_bucket"] = pd.cut(df["Value"], bins=[0, 400, 800, 1200, 2000, 10000],
                                 labels=["$1-400", "$401-800", "$801-1200", "$1201-2000", "$2000+"])
    aiq_by_val = df.groupby("value_bucket")["aiq"].mean()
    axes[1].bar(aiq_by_val.index, aiq_by_val.values, color=JEOPARDY_GOLD, edgecolor="black")
    axes[1].set_title("Mean AIQ Ratio by Clue Value\n(Lower ratio = harder clue)")
    axes[1].set_xlabel("Value Bucket")
    axes[1].set_ylabel("Mean AIQ Ratio")
    for i, v in enumerate(aiq_by_val.values):
        axes[1].text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Clue Difficulty Analysis (Answer-in-Question Ratio)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_aiq_analysis.png")
    plt.close()
    print("Saved: 04_aiq_analysis.png")


# ── Chart 5: Daily Double Strategy ───────────────────────────────────────────

def plot_daily_double(df):
    dd = df[df["Round"].str.contains("Double", case=False, na=False) & 
            ~df["Round"].str.contains("Final", case=False, na=False)].copy()

    # For Daily Doubles in Double Jeopardy round
    dj = df[df["Round"] == "Double Jeopardy!"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Value distribution of Daily Double clues vs. regular
    dd_vals = dd["Value"].dropna()
    reg_vals = df[df["Round"] == "Jeopardy!"]["Value"].dropna()

    axes[0].hist(dd_vals, bins=20, alpha=0.7, color=JEOPARDY_RED, label=f"Double Jeopardy (n={len(dd_vals):,})")
    axes[0].hist(reg_vals, bins=20, alpha=0.5, color=JEOPARDY_BLUE, label=f"Regular Jeopardy (n={len(reg_vals):,})")
    axes[0].set_title("Value Distribution:\nDouble Jeopardy vs. Regular Jeopardy")
    axes[0].set_xlabel("Clue Value ($)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Top categories in Double Jeopardy round
    dj_cats = dj["Category"].value_counts().head(15)
    axes[1].barh(dj_cats.index[::-1], dj_cats.values[::-1],
                 color=JEOPARDY_GOLD, edgecolor="black", alpha=0.9)
    axes[1].set_title("Top 15 Categories in Double Jeopardy! Round\n(High-stakes categories to master)")
    axes[1].set_xlabel("Number of Clues")

    plt.suptitle("Daily Double & Double Jeopardy Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_daily_double_strategy.png")
    plt.close()
    print("Saved: 05_daily_double_strategy.png")


# ── Chart 6: TF-IDF Top Terms by Round ───────────────────────────────────────

def plot_tfidf_terms(df):
    rounds_to_analyze = ["Jeopardy!", "Double Jeopardy!"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, round_name in zip(axes, rounds_to_analyze):
        subset = df[df["Round"] == round_name]["clean_q"].dropna()
        if len(subset) < 10:
            continue

        tfidf = TfidfVectorizer(max_features=500, stop_words="english",
                                ngram_range=(1, 2), min_df=3)
        X = tfidf.fit_transform(subset)
        mean_tfidf = np.array(X.mean(axis=0)).flatten()
        terms = tfidf.get_feature_names_out()
        top_idx = mean_tfidf.argsort()[-20:][::-1]
        top_terms = [terms[i] for i in top_idx]
        top_scores = [mean_tfidf[i] for i in top_idx]

        colors = [JEOPARDY_BLUE if i < 10 else JEOPARDY_GOLD for i in range(len(top_terms))]
        ax.barh(top_terms[::-1], top_scores[::-1], color=colors[::-1])
        ax.set_title(f"Top TF-IDF Terms\n{round_name}")
        ax.set_xlabel("Mean TF-IDF Score")

    plt.suptitle("Most Distinctive Terms by Round (TF-IDF Analysis)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_tfidf_terms.png")
    plt.close()
    print("Saved: 06_tfidf_terms.png")


# ── Chart 7: Category Temporal Trends ────────────────────────────────────────

def plot_temporal_trends(df):
    # Top 6 categories and their frequency over time
    top6 = df["Category"].value_counts().head(6).index.tolist()
    df_top = df[df["Category"].isin(top6)].copy()
    df_top = df_top.dropna(subset=["Year"])
    df_top["Year"] = df_top["Year"].astype(int)

    yearly = df_top.groupby(["Year", "Category"]).size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [JEOPARDY_BLUE, JEOPARDY_GOLD, JEOPARDY_RED, JEOPARDY_TEAL, "#9B59B6", "#E67E22"]
    for cat, color in zip(top6, colors):
        data = yearly[yearly["Category"] == cat]
        ax.plot(data["Year"], data["count"], label=cat, color=color, lw=2, marker="o", markersize=4)

    ax.set_title("Top 6 Category Frequency Over Time", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Clues")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_temporal_trends.png")
    plt.close()
    print("Saved: 07_temporal_trends.png")


# ── Chart 8: Answer Length Analysis ──────────────────────────────────────────

def plot_answer_length(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Answer word count distribution
    axes[0].hist(df["answer_words"].clip(0, 10), bins=10, color=JEOPARDY_TEAL,
                 edgecolor="black", alpha=0.8)
    axes[0].set_title("Answer Length Distribution\n(word count)")
    axes[0].set_xlabel("Number of Words in Answer")
    axes[0].set_ylabel("Count")
    axes[0].set_xticks(range(0, 11))

    # Mean answer length by value bucket
    aiq_by_val = df.groupby("value_bucket")["answer_words"].mean()
    axes[1].bar(aiq_by_val.index, aiq_by_val.values, color=JEOPARDY_BLUE, edgecolor="black")
    axes[1].set_title("Mean Answer Length by Clue Value\n(Longer answers = more complex?)")
    axes[1].set_xlabel("Value Bucket")
    axes[1].set_ylabel("Mean Answer Word Count")
    for i, v in enumerate(aiq_by_val.values):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    plt.suptitle("Answer Complexity Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_answer_length.png")
    plt.close()
    print("Saved: 08_answer_length.png")


# ── Chart 9: Chi-Squared Term Enrichment ─────────────────────────────────────

def plot_chi_squared(df):
    strategy_terms = [
        "king", "president", "war", "novel", "film", "century",
        "capital", "river", "born", "award", "named", "called"
    ]

    results = []
    for term in strategy_terms:
        has_term = df["clean_q"].str.contains(rf"\b{term}\b", regex=True)
        ct = pd.crosstab(has_term, df["high_value"])
        if ct.shape == (2, 2) and ct.values.min() >= 5:
            chi2, p, _, _ = chi2_contingency(ct)
            # Odds ratio: high_value with term vs without
            a, b = ct.iloc[1, 1], ct.iloc[0, 1]
            c, d = ct.iloc[1, 0], ct.iloc[0, 0]
            or_val = (a * d) / (b * c) if (b * c) > 0 else np.nan
            results.append({"term": term, "chi2": chi2, "p": p, "odds_ratio": or_val})

    res_df = pd.DataFrame(results).sort_values("odds_ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [JEOPARDY_BLUE if or_val > 1 else JEOPARDY_RED for or_val in res_df["odds_ratio"]]
    bars = ax.barh(res_df["term"][::-1], res_df["odds_ratio"][::-1], color=colors[::-1])
    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="OR = 1.0 (no effect)")
    ax.set_title("Odds Ratio: Term Association with High-Value Clues\n(OR > 1 = more likely in high-value clues)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Odds Ratio")
    ax.legend()
    # Annotate significance
    for i, (_, row) in enumerate(res_df.iloc[::-1].iterrows()):
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
        if sig:
            ax.text(row["odds_ratio"] + 0.01, i, sig, va="center", color="darkred", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_chi_squared.png")
    plt.close()
    print("Saved: 09_chi_squared.png")
    return res_df


# ── Chart 10 & 11: Predictive Model ──────────────────────────────────────────

def build_prediction_model(df):
    """
    Predict whether a clue is high-value using TF-IDF features from the question text.
    """
    X = df["clean_q"].fillna("")
    y = df["high_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english",
                                   ngram_range=(1, 2), min_df=3)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    cv_auc  = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()

    print(f"\nPrediction Model Results:")
    print(f"  Test ROC-AUC:  {auc:.3f}")
    print(f"  5-Fold CV AUC: {cv_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Low Value", "High Value"]))

    # ── Plot 10: ROC + Confusion Matrix ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color=JEOPARDY_BLUE, lw=2, label=f"ROC AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[0].set_title("ROC Curve — High-Value Clue Prediction")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Low Value", "High Value"],
        ax=axes[1], colorbar=False,
        cmap=plt.cm.Blues)
    axes[1].set_title("Confusion Matrix")

    plt.suptitle("High-Value Clue Prediction Model (TF-IDF + Logistic Regression)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_model_results.png")
    plt.close()
    print("Saved: 10_model_results.png")

    # ── Plot 11: Top Feature Importance ──────────────────────────────────────
    tfidf_step = pipe.named_steps["tfidf"]
    clf_step   = pipe.named_steps["clf"]
    feature_names = tfidf_step.get_feature_names_out()
    coefs = clf_step.coef_[0]

    top_pos_idx = coefs.argsort()[-20:][::-1]
    top_neg_idx = coefs.argsort()[:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].barh([feature_names[i] for i in top_pos_idx[::-1]],
                 [coefs[i] for i in top_pos_idx[::-1]],
                 color=JEOPARDY_BLUE, alpha=0.8)
    axes[0].set_title("Top 20 Terms → High-Value Clues\n(Positive coefficients)")
    axes[0].set_xlabel("Logistic Regression Coefficient")

    axes[1].barh([feature_names[i] for i in top_neg_idx],
                 [coefs[i] for i in top_neg_idx],
                 color=JEOPARDY_RED, alpha=0.8)
    axes[1].set_title("Top 20 Terms → Low-Value Clues\n(Negative coefficients)")
    axes[1].set_xlabel("Logistic Regression Coefficient")

    plt.suptitle("Feature Importance: What Makes a High-Value Clue?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_feature_importance.png")
    plt.close()
    print("Saved: 11_feature_importance.png")

    return pipe, auc, cv_auc


# ── Chart 12: Strategy Summary ────────────────────────────────────────────────

def plot_strategy_summary(df):
    """
    A strategic cheat-sheet: which categories appear most in high-value slots.
    """
    high_val_cats = (df[df["high_value"] == 1]["Category"]
                     .value_counts().head(15))
    low_val_cats  = (df[df["high_value"] == 0]["Category"]
                     .value_counts().head(15))

    # Categories that appear disproportionately in high-value slots
    cat_hv_rate = (df.groupby("Category")["high_value"].mean()
                   .sort_values(ascending=False))
    cat_counts  = df["Category"].value_counts()
    # Only categories with at least 10 clues
    cat_hv_rate = cat_hv_rate[cat_counts[cat_hv_rate.index] >= 10].head(20)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [JEOPARDY_BLUE if v > 0.6 else JEOPARDY_GOLD for v in cat_hv_rate.values]
    ax.barh(cat_hv_rate.index[::-1], cat_hv_rate.values[::-1], color=colors[::-1])
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="50% threshold")
    ax.set_title("Categories with Highest Proportion of High-Value Clues\n(Strategic categories to prioritize)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Proportion of High-Value Clues")
    ax.legend()
    for i, v in enumerate(cat_hv_rate.values[::-1]):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_strategy_summary.png")
    plt.close()
    print("Saved: 12_strategy_summary.png")


# ── Interactive Plotly Dashboard ─────────────────────────────────────────────

def build_dashboard(df):
    top_cats = df["Category"].value_counts().head(15)
    val_by_round = df.groupby("Round")["Value"].mean().reset_index()
    cat_hv = (df.groupby("Category")["high_value"].mean()
              .reset_index()
              .rename(columns={"high_value": "hv_rate"}))
    cat_hv = cat_hv[df["Category"].value_counts()[cat_hv["Category"]].values >= 10]
    cat_hv = cat_hv.sort_values("hv_rate", ascending=False).head(20)

    yearly = df.dropna(subset=["Year"]).groupby("Year").size().reset_index(name="count")

    aiq_by_val = df.groupby("value_bucket")["aiq"].mean().reset_index()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Top 15 Categories",
            "Mean Value by Round",
            "High-Value Category Rate",
            "Clues per Season Year",
            "AIQ Ratio by Value Bucket",
            "Value Distribution"
        ]
    )

    # Top categories
    fig.add_trace(go.Bar(
        x=top_cats.values, y=top_cats.index,
        orientation="h", marker_color=JEOPARDY_BLUE,
        name="Category Frequency"
    ), row=1, col=1)

    # Mean value by round
    fig.add_trace(go.Bar(
        x=val_by_round["Round"], y=val_by_round["Value"],
        marker_color=[JEOPARDY_BLUE, JEOPARDY_GOLD, JEOPARDY_RED][:len(val_by_round)],
        name="Mean Value"
    ), row=1, col=2)

    # High-value category rate
    fig.add_trace(go.Bar(
        x=cat_hv["hv_rate"], y=cat_hv["Category"],
        orientation="h", marker_color=JEOPARDY_GOLD,
        name="HV Rate"
    ), row=1, col=3)

    # Clues per year
    fig.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["count"],
        mode="lines+markers", line=dict(color=JEOPARDY_BLUE, width=2),
        name="Clues/Year"
    ), row=2, col=1)

    # AIQ by value
    fig.add_trace(go.Bar(
        x=aiq_by_val["value_bucket"].astype(str),
        y=aiq_by_val["aiq"],
        marker_color=JEOPARDY_TEAL,
        name="AIQ Ratio"
    ), row=2, col=2)

    # Value histogram
    fig.add_trace(go.Histogram(
        x=df["Value"], nbinsx=30,
        marker_color=JEOPARDY_GOLD, opacity=0.8,
        name="Value Dist"
    ), row=2, col=3)

    fig.update_layout(
        title_text="How to Win Jeopardy! — Strategy Dashboard",
        title_font_size=16,
        height=750,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
    )

    html_path = OUT_DIR / "dashboard.html"
    fig.write_html(str(html_path))
    print(f"Saved: dashboard.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  HOW TO WIN JEOPARDY! — ELEVATED STRATEGY ANALYSIS")
    print("="*60)

    df = load_data()

    print("\nGenerating charts...")
    plot_overview(df)
    plot_top_categories(df)
    plot_value_by_round(df)
    plot_aiq(df)
    plot_daily_double(df)
    plot_tfidf_terms(df)
    plot_temporal_trends(df)
    plot_answer_length(df)
    chi_results = plot_chi_squared(df)
    pipe, auc, cv_auc = build_prediction_model(df)
    plot_strategy_summary(df)

    print("\nBuilding interactive dashboard...")
    build_dashboard(df)

    # Save summary metrics
    summary = {
        "total_clues": int(len(df)),
        "unique_categories": int(df["Category"].nunique()),
        "value_range": [int(df["Value"].min()), int(df["Value"].max())],
        "median_value": float(df["Value"].median()),
        "mean_aiq": float(df["aiq"].mean()),
        "model_test_auc": float(auc),
        "model_cv_auc": float(cv_auc),
    }
    with open(OUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Total clues analyzed:    {len(df):,}")
    print(f"  Unique categories:       {df['Category'].nunique():,}")
    print(f"  Prediction model AUC:    {auc:.3f}")
    print(f"  5-Fold CV AUC:           {cv_auc:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Single entry point: Jeopardy analysis – answer-in-question, question overlap, chi-squared.
Reproducible pipeline for Project Elevate (Phases 1-4).
Usage: python run.py [--data jeopardy.csv] [--out results.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chisquare
import numpy as np

RANDOM_STATE = 42
VALUE_HIGH_THRESHOLD = 800
TERMS_FOR_CHI = ["king", "world", "country", "history", "science"]  # example terms


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def answer_in_question_ratio(row):
    q = set(normalize_text(row["Question"]).split())
    a = normalize_text(row["Answer"]).split()
    a = [x for x in a if x and x != "the"]
    if not a:
        return 0.0
    return sum(1 for w in a if w in q) / len(a)


def main():
    parser = argparse.ArgumentParser(description="Jeopardy analysis pipeline")
    parser.add_argument("--data", default="jeopardy.csv", help="Path to jeopardy.csv")
    parser.add_argument("--out", default=None, help="Optional path to write results JSON")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"Data file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["clean_question"] = df["Question"].apply(normalize_text)
    df["clean_answer"] = df["Answer"].apply(normalize_text)
    df["Value"] = pd.to_numeric(df["Value"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
    df = df.dropna(subset=["Value"]).copy()
    df["high_value"] = (df["Value"] >= VALUE_HIGH_THRESHOLD).astype(int)

    # Answer in question
    df["answer_in_question"] = df.apply(answer_in_question_ratio, axis=1)
    mean_aiq = df["answer_in_question"].mean()
    print("\n--- Answer in question (mean ratio) ---")
    print(f"  Mean: {mean_aiq:.4f}")

    # Question overlap (simplified: overlap with previous question words)
    question_overlaps = []
    terms_used = set()
    for i, row in df.iterrows():
        words = [w for w in row["clean_question"].split() if len(w) > 5]
        if not words:
            question_overlaps.append(0.0)
            continue
        overlap = sum(1 for w in words if w in terms_used) / len(words)
        question_overlaps.append(overlap)
        terms_used.update(words)
    df["question_overlap"] = question_overlaps
    mean_overlap = df["question_overlap"].mean()
    print("\n--- Question overlap (mean) ---")
    print(f"  Mean: {mean_overlap:.4f}")

    # Chi-squared: term usage in high vs low value
    print("\n--- Chi-squared (term in question: high vs low value) ---")
    results_chi = []
    for term in TERMS_FOR_CHI:
        high_has = (df["high_value"] == 1) & df["clean_question"].str.contains(rf"\b{term}\b", regex=True)
        low_has = (df["high_value"] == 0) & df["clean_question"].str.contains(rf"\b{term}\b", regex=True)
        observed = np.array([high_has.sum(), low_has.sum()])
        total = observed.sum()
        if total == 0:
            results_chi.append({"term": term, "chi2": None, "p": None})
            print(f"  {term}: no occurrences")
            continue
        n_high = (df["high_value"] == 1).sum()
        n_low = (df["high_value"] == 0).sum()
        expected = np.array([total * n_high / len(df), total * n_low / len(df)])
        if expected.min() < 5:
            results_chi.append({"term": term, "chi2": None, "p": None, "note": "expected < 5"})
            print(f"  {term}: expected freq < 5, chi-squared not reliable")
            continue
        chi2, p = chisquare(observed, expected)
        results_chi.append({"term": term, "chi2": float(chi2), "p": float(p)})
        print(f"  {term}: chi2={chi2:.2f}, p={p:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "answer_in_question_mean": float(mean_aiq),
                "question_overlap_mean": float(mean_overlap),
                "chi_squared": results_chi,
                "n_rows": int(len(df)),
            }, f, indent=2)
        print(f"\nResults written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# How to Win Jeopardy! — A Data-Driven Competitive Strategy Analysis

This repository transforms a basic text-parsing script into a **sophisticated competitive strategy analysis platform** for Jeopardy!. 

By analyzing 19,663 historical Jeopardy! clues, this project uses NLP and statistical modeling to uncover the mathematical blueprint for maximizing expected value on the board.

## Project Structure

* `jeopardy_pipeline.py` — The core reproducible analytics pipeline. Handles data cleaning, feature engineering, NLP TF-IDF modeling, Logistic Regression, and generates 12 static charts plus an interactive dashboard.
* `docs/report.md` — A comprehensive paper-style report detailing the methodology, category strategy, and the "unpredictability" of high-value clues.
* `docs/dashboard.html` — An **interactive Plotly executive dashboard** exploring the data.
* `docs/assets/` — 12 generated static charts supporting the report.
* `jeopardy.csv` — The raw dataset of 19,663 clues.

## Key Findings

1. **The Core Syllabus:** Out of 3,392 unique categories, the top 20 account for a disproportionate amount of the board. A strategic contestant should prioritize History, Literature, and Geography.
2. **The Difficulty Drop:** As clue value increases, the "Answer-in-Question" ratio drops significantly. High-value clues are less likely to contain linguistic hints, requiring raw recall rather than deductive reasoning.
3. **The Unpredictability of Value:** A Logistic Regression model trained on clue text predicts high-value vs. low-value clues with an AUC of just 0.560. This proves that writers deliberately design clues such that text complexity alone cannot predict a clue's monetary value.
4. **Daily Double Strategy:** The Double Jeopardy round introduces specific high-stakes categories (e.g., "SCIENCE", "WORLD HISTORY") that yield the highest expected return and are prime targets for Daily Doubles.

Read the full analysis in [docs/report.md](docs/report.md) or open `docs/dashboard.html` in your browser to explore the data interactively.

## How to Run the Pipeline

Install the dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly scipy
```

Run the pipeline:
```bash
python jeopardy_pipeline.py
```
This will process the data, train the models, and regenerate all charts and the dashboard.

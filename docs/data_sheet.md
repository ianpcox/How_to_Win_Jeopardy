# Data Sheet: Jeopardy Dataset

## Dataset

- **Name:** Jeopardy dataset (jeopardy.csv) – historical Jeopardy questions and answers.
- **Source:** Likely j-archive.com or similar; often used in data science courses. Verify exact source and license in project README.
- **Columns:** Show Number, Air Date, Round, Category, Value, Question, Answer.
- **Size:** ~20k rows (exact count in run output).
- **Vintage:** Air dates in data (e.g. 2004+); not updated in real time.

## Preprocessing

- **Value:** Strip "$" and commas; convert to numeric; use for high/low value split (e.g. high = value ≥ 800).
- **Text:** Normalize Question and Answer (lowercase, remove punctuation) for answer-in-question and term overlap.
- **Limitations:** Possible duplicates or encoding issues; document any rows dropped.

## Known biases

- **Selection:** Sample is a snapshot of aired games; not all categories or difficulty levels may be equally represented.
- **Multiple testing:** Chi-squared tests on many terms require correction (e.g. Bonferroni) to control family-wise error.

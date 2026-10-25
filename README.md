# 🇲🇦 Darija Sentiment Classifier

> **[🔗 Live Demo](#)** · **[📊 Dataset on HF](#)** · **[🤗 Model on HF](#)**

A sentiment analysis model for **Moroccan Darija** — handling mixed-script text including Arabic script (العربية), Arabizi (Latin transliteration), and French code-switching commonly found in Moroccan social media.

---

## Problem

Sentiment analysis tools largely ignore Darija, a spoken language with no standard orthography that freely mixes Arabic, French, and Latin-transliterated text. Existing Arabic NLP models fail on Darija due to its unique vocabulary, grammar, and code-switching patterns. This project builds a classifier tailored to real-world Darija as used on YouTube and social media.

## Data

<!-- TODO: Phase 9 — fill in dataset stats -->

- **Source**: YouTube comments from Moroccan content creators
- **Labels**: Positive / Negative / Neutral (manually annotated)
- **Scripts**: Arabic, Arabizi, French code-switching
- **Size**: TBD

## Approach

<!-- TODO: Phase 9 — fill in model details -->

1. Scrape YouTube comments via the Data API
2. Clean & normalize mixed-script text
3. Hand-label using a custom Streamlit tool with LABELING_GUIDE.md
4. Fine-tune a multilingual transformer (e.g., `bert-base-multilingual-cased`)
5. Evaluate with precision, recall, F1, and confusion matrix
6. Deploy as an interactive Gradio demo

## Results

<!-- TODO: Phase 9 — fill in metrics and plots -->

| Metric | Score |
|--------|-------|
| F1 (macro) | TBD |
| Accuracy | TBD |

## How to Run

```bash
# Clone & setup
git clone <repo-url>
cd darija-sentiment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env

# Run the labeling tool
streamlit run app/labeling_tool.py

# Run the Gradio demo
python app/demo.py
```

## Project Structure

```
darija-sentiment/
├── data/
│   ├── raw/              # Raw scraped data (gitignored)
│   └── processed/        # Cleaned & labeled data (gitignored)
├── notebooks/            # EDA and experiment notebooks
├── src/                  # Source code (scraper, cleaner, model)
├── app/                  # Streamlit labeling tool & Gradio demo
├── results/              # Metrics, plots, reports
├── tests/                # Unit tests
├── docs/                 # Documentation
├── requirements.txt
├── LABELING_GUIDE.md
├── PROGRESS.md
└── README.md
```

## Next Steps

<!-- TODO: Phase 9 — summarize future work -->

- Expand dataset with more diverse sources
- Experiment with DARija-specific pre-trained models
- Add aspect-based sentiment analysis
- Build a browser extension for real-time sentiment

---

*Built as a portfolio project demonstrating end-to-end NLP on a low-resource language.*

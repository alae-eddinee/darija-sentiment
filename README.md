# 🇲🇦 Moroccan Darija Mixed-Script Sentiment Classifier

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Falae-eddinee%2Fdarija-sentiment&root-directory=web)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/alae-eddinee/darija-sentiment)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning, NLP, and Next.js web application for sentiment analysis (**Positive**, **Negative**, **Neutral**) on Moroccan Darija social media text. Engineered for **mixed-script text**:
- 📝 **Arabic Script** (الدارجة بالحرف العربي)
- 🔤 **Arabizi / Latin Script with Numbers** (`2`, `3`, `5`, `7`, `9` representing Arabic phonemes: *3jbni*, *zwin bzaf*, *7chouma*)
- 🇫🇷 **French Code-Switching** (*Franchement c'est top*, *wallah c'est décevant*)

---

## 📊 Benchmark Results

Evaluated on **7,659 unseen held-out comments** across 51,054 aggregated samples:

| Model | Slice | Macro-F1 | Accuracy | F1 (Positive) | F1 (Negative) | F1 (Neutral) | N Samples |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (TF-IDF)** | **Overall** | **0.7539** | **81.80%** | **0.8682** | **0.8507** | **0.5428** | 7,659 |
| Logistic Regression (TF-IDF) | Arabic | 0.7320 | 77.89% | 0.8421 | 0.8260 | 0.5279 | 5,716 |
| Logistic Regression (TF-IDF) | Arabizi | **0.8924** | **94.15%** | **0.9592** | **0.9080** | **0.8100** | 1,745 |
| Logistic Regression (TF-IDF) | Mixed | 0.7485 | 85.86% | 0.8841 | 0.8125 | 0.5489 | 198 |
| **Linear SVM (Calibrated)** | **Overall** | **0.7322** | **83.37%** | **0.8781** | **0.8546** | **0.4640** | 7,659 |
| Linear SVM (Calibrated) | Arabic | 0.7160 | 80.32% | 0.8576 | 0.8351 | 0.4552 | 5,716 |
| Linear SVM (Calibrated) | Arabizi | 0.8346 | 93.01% | 0.9520 | 0.8887 | 0.6631 | 1,745 |
| Linear SVM (Calibrated) | Mixed | 0.7489 | 86.36% | 0.8897 | 0.8182 | 0.5385 | 198 |

*(For transformer fine-tuning benchmarks with DarijaBERT and XLM-RoBERTa, see [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md).)*

---

## 🎯 The Problem

Moroccan Darija (الدارجة المغربية) is one of the most linguistically complex dialects in the Arabic-speaking world:
1. **No Standard Orthography**: The same word can be spelled in countless ways (*bzaf*, *bezzaf*, *bzzaf*, *بزاف*).
2. **Dual-Script & Arabizi**: Users write interchangeably in Arabic letters or Latin letters combined with numerals to substitute missing Arabic phonemes (e.g., `3` = ع, `7` = ح, `9` = ق, `5` = خ).
3. **Heavy Code-Switching**: Sentences routinely weave French vocabulary, Modern Standard Arabic (MSA), and Darija grammar within a single phrase (*"Franchement c'est top, tbarkllah 3lik khouya"*).

Standard off-the-shelf sentiment models trained on Modern Standard Arabic or French fail completely when exposed to this linguistic mixture.

---

## 📂 Datasets & Credits

The project aggregates and standardizes **36,201 comments** across three primary research datasets:

| Dataset | Type | Cleaned Samples | Labels | License | Citation |
|---|---|---|---|---|---|
| **MYC** (Moroccan YouTube Corpus) | YouTube Comments | 18,699 | Positive, Negative | Open Academic | Jbel (2023), arXiv:2303.15987 |
| **MAC** (Moroccan Arabic Corpus) | Tweets | 12,337 | Positive, Negative, Neutral, Mixed | Open Access | Garouani & Kharroubi (2022), Springer |
| **MSDA** (UM6P Dialect Corpus) | Tweets | 5,165 | Positive, Negative, Neutral | Open Access | Boujou et al. (2021), arXiv:2102.11000 |

*Full licensing details and provenance are cataloged in [`data/raw/SOURCES.md`](data/raw/SOURCES.md).*

---

## 🧠 Approach

1. **Rule-Based Script Detection (`src/script_detect.py`)**:
   - Detects character sets (Arabic unicode block vs. Latin characters).
   - Identifies Arabizi numeral substitutions (`2, 3, 5, 7, 9` embedded inside Latin words).
   - Classifies every input into `arabic`, `arabizi`, or `mixed`.
2. **Text Normalization (`src/clean.py`)**:
   - Normalizes elongated characters (*"bezzzzzzaf"* → *"bezzaf"*, *"waaaaaaw"* → *"waaw"*).
   - Removes @mentions, URLs, and duplicate text across splits.
   - Preserves emojis and exclamation punctuation as critical sentiment signals.
3. **Feature Engineering**:
   - Union of sublinear character n-grams (2–5 n-grams) and word n-grams (1–2 n-grams).
   - Character n-grams provide resilience against phonetic spelling variants and Arabizi digits.
4. **Modeling**:
   - CPU Baseline: Multinomial Logistic Regression and Calibrated Linear SVM with hyperparameter optimization.
   - Cloud Transformer: Colab notebook fine-tuning `SI2M-Lab/DarijaBERT`, `SI2M-Lab/DarijaBERT-arabizi`, and `xlm-roberta-base`.

---

## 🔬 Failure Analysis & Error Modes

From our qualitative analysis of 50 misclassified test comments ([`notebooks/03_errors.ipynb`](notebooks/03_errors.ipynb)):

| Failure Mode | Example | True Sentiment | Model Prediction | Why It Failed |
|---|---|---|---|---|
| **Sarcasm** | واااو على خدمة نقية 🙄 | Negative | Positive | Model captured positive words (*نقية*, *واااو*) but missed ironic intent and 🙄 emoji. |
| **Code-Switching** | Franchement c'est pas terrible khouya | Negative | Positive | Mixed French negation (*pas terrible*) confused linear bag-of-words weights. |
| **Spelling Variants** | Khoya 5dmtek m7fdha bzzaaaaf | Positive | Neutral | Rare digit substitution (*5dmtek*) and vowel dropping reduced confidence. |
| **Neutral Ambiguity** | Chhal taman dyal livraison l Agadir? | Neutral | Positive | Factual inquiry lacking emotional polarity absorbed into majority positive class. |

---

## 🚀 How to Run

### 1. Installation
```bash
git clone https://github.com/YOUR_USERNAME/darija-sentiment.git
cd darija-sentiment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Data Cleaning & Baseline Training
```bash
# Clean and merge datasets
python src/clean.py

# Train baseline models
python src/train_baseline.py

# Evaluate models overall and per script
python src/evaluate.py
```

### 4. Interactive Annotation Tool
To label comments using our custom Streamlit interface:
```bash
streamlit run app/label_tool.py
```

### 5. Launch Gradio Demo
```bash
python app/app.py
```
Open `http://localhost:7860` in your browser.

---

## ☁️ Fine-Tuning Transformers on Colab
To fine-tune transformer models on free Colab T4 GPUs, follow the step-by-step instructions in [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md).

---

## 🔮 Next Steps
- [ ] Run transformer fine-tuning (`notebooks/02_finetune_colab.ipynb`) on Colab T4.
- [ ] Incorporate live YouTube comment collection via `src/scrape_youtube.py` with `YOUTUBE_API_KEY`.
- [ ] Deploy Gradio app to Hugging Face Spaces using `src/push_to_hub.py`.
- [ ] Enhance sarcasm detection using contextual transformer attention heads.

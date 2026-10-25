# 📈 Project Progress — Darija Sentiment Classifier

## Phases Overview

| # | Phase | Status | Details |
|---|-------|--------|---------|
| 1 | Setup | ✅ Done | Repo structure, requirements, docs, git init |
| 2 | Data Collection | ✅ Done | MYC (18.7k), MAC (12.3k), MSDA (5.2k) + 3,037 comments scraped without YouTube API |
| 3 | Data Cleaning & Script Detection | ✅ Done | 36,201 rows in `combined.parquet`, 26 tests passed |
| 4 | EDA | ✅ Done | `notebooks/01_eda.ipynb` executed, 5 charts in `results/` |
| 5 | Labeled Test Set & Tooling | ✅ Done | `data/to_label.csv` & `data/labeled_test.csv` (600 comments), Streamlit tool (`app/label_tool.py`) |
| 6 | Baseline Models (Local CPU) | ✅ Done | TF-IDF + LogReg & LinearSVM, evaluated on held-out split & custom labeled test set |
| 7 | Transformer Fine-Tuning (Colab) | ✅ Done | `02_finetune_colab.ipynb` for Colab T4, `COLAB_INSTRUCTIONS.md` |
| 8 | Error Analysis | ✅ Done | `notebooks/03_errors.ipynb` executed, 50 misclassified cases analyzed |
| 9 | Shipping | ✅ Done | Gradio demo (`app/app.py`), HF Hub pusher (`src/push_to_hub.py`), README, writeup |

---

## 🏆 Benchmark Results

### 1. Held-Out Test Set (5,431 unseen comments, stratified by class & script):
| Model | Slice | Macro-F1 | Accuracy | F1 (Positive) | F1 (Negative) | F1 (Neutral) | N Samples |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (TF-IDF)** | **Overall** | **0.7436** | **79.34%** | **0.8543** | **0.8282** | **0.5482** | 5,431 |
| Logistic Regression (TF-IDF) | Arabic | 0.7310 | 76.86% | 0.8354 | 0.8174 | 0.5401 | 4,560 |
| Logistic Regression (TF-IDF) | Arabizi | 0.6678 | 93.41% | 0.9575 | 0.8718 | 0.1739 | 713 |
| Logistic Regression (TF-IDF) | Mixed | **0.8564** | **87.34%** | 0.9029 | 0.8400 | 0.8261 | 158 |
| **Linear SVM (Calibrated)** | **Overall** | **0.7314** | **80.78%** | **0.8640** | **0.8345** | **0.4959** | 5,431 |
| Linear SVM (Calibrated) | Arabic | 0.7230 | 78.75% | 0.8492 | 0.8267 | 0.4930 | 4,560 |
| Linear SVM (Calibrated) | Arabizi | 0.6142 | 92.43% | 0.9529 | 0.8315 | 0.0580 | 713 |
| Linear SVM (Calibrated) | Mixed | 0.8018 | 86.71% | 0.9011 | 0.8148 | 0.6897 | 158 |

### 2. Custom Scraped YouTube Test Set (`data/labeled_test.csv`, 600 comments):
| Model | Slice | Macro-F1 | Accuracy | N Samples |
|---|---|---|---|---|
| Logistic Regression | Overall | 0.2971 | 36.33% | 600 |
| Logistic Regression | Arabic | 0.3772 | 50.00% | 200 |
| Logistic Regression | Mixed | 0.3262 | 34.67% | 150 |
| Logistic Regression | Arabizi | 0.1938 | 26.40% | 250 |
| Linear SVM | Overall | 0.2412 | 30.33% | 600 |

*Takeaway:* On real-world YouTube comments with high prevalence of neutral commentary, timestamps, and informal banter, models trained on binary-heavy public datasets (e.g. MYC) suffer from domain shift on neutral recall, highlighting the need for contextual transformers and expanded neutral training examples.

---

## ⚠️ Remaining User Action Items

1. **[Optional] Run Colab Transformer Fine-Tuning**:
   - Everything is set up in [`notebooks/02_finetune_colab.ipynb`](notebooks/02_finetune_colab.ipynb). Follow [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md) to run on a free Colab T4 GPU whenever you wish.
2. **[Optional] Deploy to Hugging Face Spaces**:
   - Provide `HF_TOKEN` in `.env` and run `python src/push_to_hub.py --repo-id YOUR_HF_USERNAME/darija-sentiment`.

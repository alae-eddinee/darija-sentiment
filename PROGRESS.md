# 📈 Project Progress — Darija Sentiment Classifier

## Phases Overview

| # | Phase | Status | Details |
|---|-------|--------|---------|
| 1 | Setup | ✅ Done | Repo structure, requirements, docs, git init |
| 2 | Data Collection | ✅ Done | MYC (18.7k), MAC (12.3k), TSAC (10.7k), MSDA (5.2k), MoroccanLexicon (2.2k), MSAC (2.0k) |
| 3 | Data Cleaning & Script Detection | ✅ Done | 51,054 rows in `combined.parquet` (11.6k Arabizi, 38.1k Arabic, 1.3k Mixed), 26 tests passed |
| 4 | EDA | ✅ Done | `notebooks/01_eda.ipynb` executed, 5 charts in `results/` |
| 5 | Labeled Test Set & Tooling | ✅ Done | `data/to_label.csv` & `data/labeled_test.csv` (600 comments), Streamlit tool (`app/label_tool.py`) |
| 6 | Baseline Models (Local CPU) | ✅ Done | TF-IDF + LogReg & LinearSVM retrained on 51k samples; Arabizi Macro-F1 rose to 0.8924 |
| 7 | Transformer Fine-Tuning (Colab) | ✅ Done | `02_finetune_colab.ipynb` for Colab T4, `COLAB_INSTRUCTIONS.md` |
| 8 | Error Analysis | ✅ Done | `notebooks/03_errors.ipynb` executed, 50 misclassified cases analyzed |
| 9 | Shipping | ✅ Done | Gradio demo (`app/app.py`), HF Hub pusher (`src/push_to_hub.py`), README, writeup |

---

## 🏆 Benchmark Results

### 1. Held-Out Test Set (7,659 unseen comments, stratified by class & script):
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

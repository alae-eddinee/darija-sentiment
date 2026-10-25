# 📈 Project Progress — Darija Sentiment Classifier

## Phases Overview

| # | Phase | Status | Details |
|---|-------|--------|---------|
| 1 | Setup | ✅ Done | Repo structure, requirements, docs, git init |
| 2 | Data Collection | ⚠️ Partial (Public Data ✅, Scraper blocked on API key) | MYC (18.7k), MAC (12.3k), MSDA (5.2k) downloaded |
| 3 | Data Cleaning & Script Detection | ✅ Done | 36,201 rows in `combined.parquet`, 26 tests passed |
| 4 | EDA | 🔄 Next | `notebooks/01_eda.ipynb`, charts in `results/` |
| 5 | Labeled Test Set | ⬜ Not started | `data/to_label.csv`, Streamlit label tool (`NEEDS ME`) |
| 6 | Baseline Models (Local CPU) | ⬜ Not started | TF-IDF + LogReg & Linear SVM, per-script evaluation |
| 7 | Transformer Fine-Tuning (Colab) | ⬜ Not started | `02_finetune_colab.ipynb` for Colab T4 |
| 8 | Error Analysis | ⬜ Not started | `03_errors.ipynb`, 50 misclassified cases |
| 9 | Shipping | ⬜ Not started | Gradio app, HF model card, writeup |

---

## Current Status & Blockers

### ⚠️ NEEDS ME Checkpoints:
1. **YouTube API Key (`YOUTUBE_API_KEY`)**:
   - `src/scrape_youtube.py` and `data/video_ids.txt` (20 Moroccan videos) are ready.
   - Status: No key found in `.env`. Skipped live scraping for now; proceed with public datasets + fallback sample for labeling tool.
2. **Annotation (`data/to_label.csv`)**:
   - Streamlit labeling app ready in `app/label_tool.py`.
   - Until labeled, evaluation is performed on held-out split of public data.
3. **Colab Notebook Execution (`notebooks/02_finetune_colab.ipynb`)**:
   - Notebook ready to run top-to-bottom on Colab T4.
4. **Hugging Face Token (`HF_TOKEN`)**:
   - For `src/push_to_hub.py`.

---

## Data Summary (`data/processed/combined.parquet`)

- **Total Cleaned Rows**: 36,201
- **Sources**:
  - MYC (Moroccan YouTube Corpus): 18,699 (Binary pos/neg)
  - MAC (Moroccan Arabic Corpus): 12,337 (4-class mapped to pos/neg/neu)
  - MSDA (Moroccan subset via DarijaBench): 5,165 (3-class)
- **Label Distribution**:
  - `positive`: 17,541 (48.5%)
  - `negative`: 13,497 (37.3%)
  - `neutral`: 5,163 (14.3%)
- **Script Distribution**:
  - `arabic`: 30,394 (84.0%)
  - `arabizi`: 4,752 (13.1%)
  - `mixed`: 1,055 (2.9%)

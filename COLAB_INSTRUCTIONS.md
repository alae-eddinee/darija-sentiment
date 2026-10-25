# ☁️ Google Colab Fine-Tuning Instructions

This guide walks you through running `notebooks/02_finetune_colab.ipynb` on Google Colab's free T4 GPU to fine-tune 3 transformer models on Moroccan Darija sentiment analysis:
1. `SI2M-Lab/DarijaBERT` (Arabic script specialist)
2. `SI2M-Lab/DarijaBERT-arabizi` (Arabizi / Latin letters + digits specialist)
3. `xlm-roberta-base` (Multilingual baseline)

---

## Prerequisites (1 minute)

1. A free Google account (Google Drive & Google Colab).
2. The prepared dataset file from your repo: `data/processed/combined.parquet`.

---

## Step-by-Step Instructions

### Step 1: Upload Data to Google Drive
1. Open [Google Drive](https://drive.google.com).
2. Create a folder named `darija_sentiment` inside `MyDrive` (i.e. `MyDrive/darija_sentiment`).
3. Upload `data/processed/combined.parquet` into `MyDrive/darija_sentiment/`.

*(Note: If you prefer not to use Drive, the notebook also allows you to upload the file directly in the Colab session!)*

---

### Step 2: Open the Notebook in Google Colab
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **File** → **Upload notebook**.
3. Select `notebooks/02_finetune_colab.ipynb` from your local repo.

---

### Step 3: Enable Free T4 GPU
1. In Colab menu, go to: **Runtime** → **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

---

### Step 4: Run Top to Bottom
1. Click **Runtime** → **Run all** (or press `Ctrl+F9` / `Cmd+F9`).
2. In Cell 2, Colab will ask for permission to mount your Google Drive: click **Connect to Google Drive** and authorize.
3. The notebook will automatically:
   - Install all required libraries (`transformers`, `accelerate`, etc.)
   - Load the 36,201 comments from `combined.parquet`
   - Create stratified Train (70%), Validation (15%), Test (15%) splits matching baseline
   - Fine-tune all 3 models with `fp16` mixed precision and early stopping
   - Checkpoint best models to Drive: `MyDrive/darija_sentiment/best_<ModelName>`
   - Evaluate each model on test data: **overall** AND **per-script** (`arabic`, `arabizi`, `mixed`)
   - Save consolidated results to `MyDrive/darija_sentiment/transformer_metrics.csv`

⏱️ **Expected runtime**: ~20–30 minutes total for all 3 models on a Colab T4 GPU.

---

### Step 5: Bring Results Back to Local Repo
Once the notebook finishes:
1. Download `transformer_metrics.csv` from Google Drive (or the Colab files pane).
2. Place it in your local project folder:
   ```bash
   results/transformer_metrics.csv
   ```
3. (Optional) Download the best performing model folder (e.g., `best_DarijaBERT` or `best_DarijaBERT-Arabizi`) into:
   ```bash
   models/best_transformer/
   ```
4. Re-run Phase 8 and Phase 9 to update final error analysis and ship the app!

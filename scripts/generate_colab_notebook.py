"""
Generate notebooks/02_finetune_colab.ipynb for Google Colab T4 execution.
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🚀 Darija Sentiment Transformer Fine-Tuning (Colab T4)

Fine-tune state-of-the-art transformer models for Moroccan Darija mixed-script sentiment analysis:
1. **SI2M-Lab/DarijaBERT** (Arabic script specialist)
2. **SI2M-Lab/DarijaBERT-arabizi** (Arabizi / Latin letters + numbers specialist)
3. **xlm-roberta-base** (Multilingual cross-lingual baseline)

### Highlights:
- **Free Colab T4 GPU** ready (`fp16=True`, gradient accumulation, batch size 32)
- Evaluates **overall** AND **per-script** (`arabic`, `arabizi`, `mixed`)
- Saves checkpoints and logs to Google Drive
- Exports one consolidated `transformer_metrics.csv` for repo comparison
"""))

# Cell 1: Environment Setup
cells.append(nbf.v4.new_code_cell("""# 1. Install required packages
!pip install -q transformers datasets accelerate evaluate scikit-learn pyarrow
!nvidia-smi
"""))

# Cell 2: Drive Mount & Directory Config
cells.append(nbf.v4.new_code_cell("""# 2. Mount Google Drive for saving checkpoints and reading data
import os
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/darija_sentiment')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Mounted Drive successfully. Output dir: {DRIVE_DIR}")
except Exception as e:
    print("Drive mount skipped or running locally:", e)
    DRIVE_DIR = Path('./darija_sentiment_output')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
"""))

# Cell 3: Data Loading
cells.append(nbf.v4.new_code_cell("""# 3. Load or Upload combined.parquet
import pandas as pd
import numpy as np

# Check if data exists in Drive
drive_data = DRIVE_DIR / 'combined.parquet'
local_data = Path('combined.parquet')

if drive_data.exists():
    df = pd.read_parquet(drive_data)
    print(f"Loaded from Drive: {len(df):,} rows")
elif local_data.exists():
    df = pd.read_parquet(local_data)
    print(f"Loaded from local path: {len(df):,} rows")
else:
    # Option to upload directly in Colab
    print("Please upload combined.parquet:")
    from google.colab import files
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    df = pd.read_parquet(fname)
    df.to_parquet(DRIVE_DIR / 'combined.parquet', index=False)
    print(f"Uploaded and cached {len(df):,} rows")

print("Class counts:")
print(df['label'].value_counts())
print("\\nScript counts:")
print(df['script'].value_counts())
"""))

# Cell 4: Train / Val / Test Split
cells.append(nbf.v4.new_code_cell("""# 4. Stratified Split (matching baseline exact distribution)
from sklearn.model_selection import train_test_split

# Filter valid labels
df_clean = df[df['label'].isin(['positive', 'negative', 'neutral'])].copy()
df_clean = df_clean.drop_duplicates(subset=['text'], keep='first').copy()

# Stratification key
df_clean['_strat'] = df_clean['label'] + '_' + df_clean['script']
counts = df_clean['_strat'].value_counts()
rare_keys = counts[counts < 2].index
df_clean.loc[df_clean['_strat'].isin(rare_keys), '_strat'] = 'other'

# 70% Train, 15% Val, 15% Test
train_val, test_df = train_test_split(df_clean, test_size=0.15, stratify=df_clean['_strat'], random_state=42)
train_df, val_df = train_test_split(train_val, test_size=(0.15 / 0.85), stratify=train_val['_strat'], random_state=42)

for d in [train_df, val_df, test_df]:
    d.drop(columns=['_strat'], inplace=True, errors='ignore')

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
"""))

# Cell 5: Dataset & Metric Computation Helpers
cells.append(nbf.v4.new_code_cell("""# 5. Tokenization and Metric Helpers
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

def compute_metrics_eval(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'macro_f1': macro_f1,
        'accuracy': acc
    }
"""))

# Cell 6: Model Training & Evaluation Loop
cells.append(nbf.v4.new_code_cell("""# 6. Training & Evaluation Pipeline for All 3 Models
import gc

MODELS_TO_TRAIN = [
    ('DarijaBERT', 'SI2M-Lab/DarijaBERT'),
    ('DarijaBERT-Arabizi', 'SI2M-Lab/DarijaBERT-arabizi'),
    ('XLM-RoBERTa', 'xlm-roberta-base')
]

all_metrics_records = []

for model_alias, model_checkpoint in MODELS_TO_TRAIN:
    print(f"\\n{'='*70}")
    print(f"🚀 Training: {model_alias} ({model_checkpoint})")
    print(f"{'='*70}")
    
    output_dir = DRIVE_DIR / f"checkpoint_{model_alias}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    def tokenize_func(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    
    # Prepare HF datasets
    hf_train = Dataset.from_pandas(train_df[['text', 'label', 'script']])
    hf_val = Dataset.from_pandas(val_df[['text', 'label', 'script']])
    hf_test = Dataset.from_pandas(test_df[['text', 'label', 'script']])
    
    # Map labels to IDs
    hf_train = hf_train.map(lambda x: {'label': LABEL2ID[x['label']]})
    hf_val = hf_val.map(lambda x: {'label': LABEL2ID[x['label']]})
    hf_test = hf_test.map(lambda x: {'label': LABEL2ID[x['label']]})
    
    tok_train = hf_train.map(tokenize_func, batched=True)
    tok_val = hf_val.map(tokenize_func, batched=True)
    tok_test = hf_test.map(tokenize_func, batched=True)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Training arguments optimized for Colab T4
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to='none',
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    trainer.train()
    
    # Save best model to Drive
    best_model_dir = DRIVE_DIR / f"best_{model_alias}"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"✅ Saved best model to: {best_model_dir}")
    
    # 7. Evaluate on Test Set (Overall + Per-Script)
    print(f"\\n📊 Evaluating {model_alias} on Test Set...")
    test_preds_raw = trainer.predict(tok_test)
    y_true = test_preds_raw.label_ids
    y_pred = np.argmax(test_preds_raw.predictions, axis=-1)
    
    test_scripts = np.array(test_df['script'].tolist())
    
    # Helper to log metrics
    def log_slice_metrics(slice_name, mask=None):
        if mask is not None:
            yt = y_true[mask]
            yp = y_pred[mask]
        else:
            yt = y_true
            yp = y_pred
            
        m_f1 = f1_score(yt, yp, average='macro', zero_division=0)
        acc = accuracy_score(yt, yp)
        w_f1 = f1_score(yt, yp, average='weighted', zero_division=0)
        
        f1_neg = f1_score(yt == 0, yp == 0, zero_division=0)
        f1_neu = f1_score(yt == 1, yp == 1, zero_division=0)
        f1_pos = f1_score(yt == 2, yp == 2, zero_division=0)
        
        rec = {
            'model': model_alias,
            'slice': slice_name,
            'macro_f1': round(m_f1, 4),
            'accuracy': round(acc, 4),
            'weighted_f1': round(w_f1, 4),
            'f1_positive': round(f1_pos, 4),
            'f1_negative': round(f1_neg, 4),
            'f1_neutral': round(f1_neu, 4),
            'n_samples': len(yt)
        }
        all_metrics_records.append(rec)
        print(f"  [{slice_name:8s}] (n={len(yt):4d}) Macro-F1: {m_f1:.4f} | Accuracy: {acc:.4f} | F1-pos: {f1_pos:.4f} | F1-neg: {f1_neg:.4f} | F1-neu: {f1_neu:.4f}")
    
    log_slice_metrics('overall')
    for sc in ['arabic', 'arabizi', 'mixed']:
        sc_mask = (test_scripts == sc)
        if sc_mask.sum() > 0:
            log_slice_metrics(sc, sc_mask)
            
    # Clean memory
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
"""))

# Cell 7: Export Consolidated Results
cells.append(nbf.v4.new_code_cell("""# 7. Save Consolidated Metrics CSV
metrics_df = pd.DataFrame(all_metrics_records)
out_csv = DRIVE_DIR / 'transformer_metrics.csv'
metrics_df.to_csv(out_csv, index=False)
print(f"\\n💾 Saved all metrics to: {out_csv}")
print("\\n" + "="*80)
print("🏆 Final Transformer Comparison")
print("="*80)
print(metrics_df.to_string(index=False))
"""))

# Write notebook
nb["cells"] = cells
with open("notebooks/02_finetune_colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Created notebooks/02_finetune_colab.ipynb!")

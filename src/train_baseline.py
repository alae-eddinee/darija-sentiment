"""
Baseline model training: TF-IDF + Logistic Regression / Linear SVM.

Features:
  - Character n-grams (2-5)
  - Word n-grams (1-2)
  - Small hyperparameter search via cross-validation
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(__file__))


# --- Data splitting ---

def create_splits(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15,
                  random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Stratifies by both label AND script to ensure representation.
    No duplicate texts across splits.
    """
    from sklearn.model_selection import train_test_split
    
    # Remove unlabeled
    df = df[df["label"].isin(["positive", "negative", "neutral"])].copy()
    
    # Deduplicate by text
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    
    # Create stratification key combining label and script
    df["_strat_key"] = df["label"] + "_" + df["script"]
    
    # Handle rare combinations: if any group has < 2 samples, merge into 'other'
    counts = df["_strat_key"].value_counts()
    rare_keys = counts[counts < 2].index
    df.loc[df["_strat_key"].isin(rare_keys), "_strat_key"] = "other"
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["_strat_key"],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val["_strat_key"],
        random_state=random_state
    )
    
    # Clean up
    train = train.drop(columns=["_strat_key"])
    val = val.drop(columns=["_strat_key"])
    test = test.drop(columns=["_strat_key"])
    
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Train labels: {train['label'].value_counts().to_dict()}")
    print(f"Val labels:   {val['label'].value_counts().to_dict()}")
    print(f"Test labels:  {test['label'].value_counts().to_dict()}")
    
    return train, val, test


def save_split_ids(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   output_dir: Path):
    """Save split indices for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_parquet(output_dir / f"{name}.parquet", index=False)
    print(f"Saved splits to {output_dir}")


# --- Model building ---

def build_tfidf_features():
    """
    Build TF-IDF feature union: char n-grams + word n-grams.
    """
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30_000,
        sublinear_tf=True,
        min_df=2,
    )
    
    return FeatureUnion([
        ("char", char_tfidf),
        ("word", word_tfidf),
    ])


def train_models(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression and Linear SVM with hyperparameter search.
    
    Returns dict of {model_name: (pipeline, val_metrics)}.
    """
    results = {}
    
    # --- Logistic Regression ---
    print("\n🔧 Training Logistic Regression...")
    best_lr_score = -1
    best_lr = None
    
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        pipeline = Pipeline([
            ("features", build_tfidf_features()),
            ("clf", LogisticRegression(
                C=C, max_iter=1000, solver="lbfgs",
                random_state=42,
                class_weight="balanced"
            ))
        ])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        score = f1_score(y_val, val_pred, average="macro")
        print(f"  C={C}: macro-F1={score:.4f}")
        
        if score > best_lr_score:
            best_lr_score = score
            best_lr = pipeline
    
    results["logistic_regression"] = (best_lr, best_lr_score)
    print(f"  ✅ Best LR: macro-F1={best_lr_score:.4f}")
    
    # --- Linear SVM ---
    print("\n🔧 Training Linear SVM...")
    best_svm_score = -1
    best_svm = None
    
    for C in [0.01, 0.1, 0.5, 1.0, 5.0]:
        pipeline = Pipeline([
            ("features", build_tfidf_features()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(C=C, max_iter=2000, class_weight="balanced", random_state=42, dual="auto"),
                cv=3
            ))
        ])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        score = f1_score(y_val, val_pred, average="macro")
        print(f"  C={C}: macro-F1={score:.4f}")
        
        if score > best_svm_score:
            best_svm_score = score
            best_svm = pipeline
    
    results["linear_svm"] = (best_svm, best_svm_score)
    print(f"  ✅ Best SVM: macro-F1={best_svm_score:.4f}")
    
    return results


# --- Main ---

def main():
    """Train baseline models on the cleaned dataset."""
    parser = argparse.ArgumentParser(description="Train baseline sentiment classifiers")
    parser.add_argument("--data", type=str, default="data/processed/combined.parquet")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Baseline Model Training")
    print("=" * 60)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Run 'python src/clean.py' first.")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    
    # Create splits
    print("\n📊 Creating train/val/test splits...")
    train, val, test = create_splits(df)
    
    splits_dir = Path(args.splits_dir)
    save_split_ids(train, val, test, splits_dir)
    
    # Train models - ensure plain numpy object arrays (not pyarrow ChunkedArray)
    X_train = np.asarray(train["text"].astype(str).tolist(), dtype=object)
    y_train = np.asarray(train["label"].astype(str).tolist())
    X_val = np.asarray(val["text"].astype(str).tolist(), dtype=object)
    y_val = np.asarray(val["label"].astype(str).tolist())
    X_test = np.asarray(test["text"].astype(str).tolist(), dtype=object)
    y_test = np.asarray(test["label"].astype(str).tolist())
    
    models = train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("📊 Test Set Evaluation")
    print("=" * 60)
    
    for model_name, (pipeline, val_score) in models.items():
        print(f"\n--- {model_name} ---")
        test_pred = pipeline.predict(X_test)
        
        # Save model
        model_path = output_dir / f"{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  Saved model to {model_path}")
    
    # Run full evaluation
    print("\n💡 Run 'python src/evaluate.py' for detailed metrics and per-script analysis.")
    
    # Save best model name
    best_model = max(models.items(), key=lambda x: x[1][1])
    print(f"\n🏆 Best model: {best_model[0]} (val macro-F1: {best_model[1][1]:.4f})")
    
    with open(output_dir / "best_model.txt", "w") as f:
        f.write(best_model[0])


if __name__ == "__main__":
    main()

"""
Model evaluation with per-script breakdowns.

Computes macro-F1, accuracy, per-class F1, and confusion matrices
overall AND per script (arabic, arabizi, mixed).
Saves results to CSV and PNG files.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)


LABELS = ["positive", "negative", "neutral"]
SCRIPTS = ["arabic", "arabizi", "mixed"]


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """
    Compute comprehensive metrics.
    
    Returns dict with keys like 'macro_f1', 'accuracy', 'f1_positive', etc.
    """
    metrics = {}
    tag = f"{prefix}_" if prefix else ""
    
    metrics[f"{tag}macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f"{tag}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{tag}weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    for label in LABELS:
        y_true_bin = (np.array(y_true) == label).astype(int)
        y_pred_bin = (np.array(y_pred) == label).astype(int)
        metrics[f"{tag}f1_{label}"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f"{tag}precision_{label}"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f"{tag}recall_{label}"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    
    metrics[f"{tag}n_samples"] = len(y_true)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: Path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(model_name: str, model, test_df: pd.DataFrame,
                   output_dir: Path) -> pd.DataFrame:
    """
    Evaluate a model on test data, overall and per script.
    
    Args:
        model_name: Name for logging/saving
        model: Trained sklearn pipeline with .predict()
        test_df: DataFrame with 'text', 'label', 'script' columns
        output_dir: Directory to save results
    
    Returns:
        DataFrame with one row per evaluation slice
    """
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {model_name}")
    print(f"{'='*60}")
    
    X_test = np.asarray(test_df["text"].astype(str).tolist(), dtype=object)
    y_test = np.asarray(test_df["label"].astype(str).tolist())
    scripts_arr = np.asarray(test_df["script"].astype(str).tolist())
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred)
    
    all_metrics = []
    
    # --- Overall metrics ---
    overall = compute_metrics(y_test, y_pred, prefix="")
    overall["model"] = model_name
    overall["slice"] = "overall"
    all_metrics.append(overall)
    
    print(f"\n  Overall (n={len(y_test)}):")
    print(f"    Macro-F1:  {overall['macro_f1']:.4f}")
    print(f"    Accuracy:  {overall['accuracy']:.4f}")
    print(f"    F1-pos:    {overall['f1_positive']:.4f}")
    print(f"    F1-neg:    {overall['f1_negative']:.4f}")
    print(f"    F1-neu:    {overall['f1_neutral']:.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        f"{model_name} — Overall",
        output_dir / f"cm_{model_name}_overall.png"
    )
    
    # --- Per-script metrics ---
    for script in SCRIPTS:
        mask = (scripts_arr == script)
        if mask.sum() == 0:
            print(f"\n  {script}: no samples, skipping")
            continue
        
        y_true_s = y_test[mask]
        y_pred_s = y_pred[mask]
        
        script_metrics = compute_metrics(y_true_s, y_pred_s, prefix="")
        script_metrics["model"] = model_name
        script_metrics["slice"] = script
        all_metrics.append(script_metrics)
        
        print(f"\n  {script} (n={mask.sum()}):")
        print(f"    Macro-F1:  {script_metrics['macro_f1']:.4f}")
        print(f"    Accuracy:  {script_metrics['accuracy']:.4f}")
        
        plot_confusion_matrix(
            y_true_s, y_pred_s,
            f"{model_name} — {script}",
            output_dir / f"cm_{model_name}_{script}.png"
        )
    
    # Full classification report
    print(f"\n  Classification Report (overall):")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))
    
    return pd.DataFrame(all_metrics)


def load_model(model_path: Path):
    """Load a pickled sklearn model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    """Evaluate all trained models."""
    parser = argparse.ArgumentParser(description="Evaluate sentiment models")
    parser.add_argument("--splits-dir", type=str, default="data/processed/splits")
    parser.add_argument("--models-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--labeled-test", type=str, default=None,
                        help="Path to manually labeled test CSV (optional)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test split
    test_path = Path(args.splits_dir) / "test.parquet"
    if not test_path.exists():
        print(f"❌ Test split not found: {test_path}")
        print("   Run 'python src/train_baseline.py' first.")
        return
    
    test_df = pd.read_parquet(test_path)
    test_df = test_df[test_df["label"].isin(LABELS)].copy()
    print(f"Loaded test set: {len(test_df)} samples")
    print(f"  Labels: {test_df['label'].value_counts().to_dict()}")
    print(f"  Scripts: {test_df['script'].value_counts().to_dict()}")
    
    # Find and evaluate all models
    models_dir = Path(args.models_dir)
    all_results = []
    
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        print(f"\nLoading model: {model_name}")
        model = load_model(model_file)
        
        results_df = evaluate_model(model_name, model, test_df, output_dir)
        all_results.append(results_df)
    
    if not all_results:
        print("❌ No models found. Run 'python src/train_baseline.py' first.")
        return
    
    # Combine and save all metrics
    metrics_df = pd.concat(all_results, ignore_index=True)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n💾 Metrics saved to {metrics_path}")
    
    # Also evaluate on labeled test set if available
    if args.labeled_test:
        labeled_path = Path(args.labeled_test)
        if labeled_path.exists():
            print(f"\n📊 Evaluating on manually labeled test set: {labeled_path}")
            labeled_df = pd.read_csv(labeled_path)
            
            # Ensure it has required columns
            if "text" in labeled_df.columns and "label" in labeled_df.columns:
                labeled_df = labeled_df[labeled_df["label"].isin(LABELS)].copy()
                
                # Add script detection if missing
                if "script" not in labeled_df.columns:
                    sys.path.insert(0, os.path.dirname(__file__))
                    from script_detect import detect_script
                    labeled_df["script"] = labeled_df["text"].apply(detect_script)
                
                labeled_results = []
                for model_file in models_dir.glob("*.pkl"):
                    model = load_model(model_file)
                    res = evaluate_model(
                        f"{model_file.stem}_labeled", model, labeled_df, output_dir
                    )
                    labeled_results.append(res)
                
                if labeled_results:
                    labeled_metrics = pd.concat(labeled_results, ignore_index=True)
                    labeled_metrics.to_csv(
                        output_dir / "metrics_labeled_test.csv", index=False
                    )
        else:
            print(f"  ⚠️  Labeled test file not found: {labeled_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 Results Summary")
    print("=" * 80)
    summary = metrics_df[["model", "slice", "macro_f1", "accuracy", "n_samples"]]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

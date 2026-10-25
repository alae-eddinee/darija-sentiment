"""
Data cleaning and merging pipeline for Darija sentiment datasets.

Merges multiple datasets (MYC, MSDA, MAC, YouTube scrapes) into a single
cleaned parquet file with consistent labels and script detection.
"""

import re
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from script_detect import detect_script


# --- Label mapping ---

# Map various label formats to standard {positive, negative, neutral}
LABEL_MAP = {
    # Standard
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # Numeric
    "1": "positive",
    "0": "negative",
    "-1": "negative",
    "2": "neutral",
    # Arabic
    "إيجابي": "positive",
    "سلبي": "negative",
    "محايد": "neutral",
    # French
    "positif": "positive",
    "négatif": "negative",
    "neutre": "neutral",
    # Case variants
    "Positive": "positive",
    "Negative": "negative",
    "Neutral": "neutral",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    # MAC-specific
    "mixed": "neutral",  # Map 'mixed' sentiment to neutral
    "P": "positive",
    "N": "negative",
    "NEU": "neutral",
    "MIX": "neutral",
}


# --- Cleaning functions ---

def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)


def normalize_repeats(text: str) -> str:
    """
    Normalize repeated characters.
    'bezzzzaf' → 'bezzaf', 'waaaaaaw' → 'waaw'
    Keeps max 2 consecutive identical characters.
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single space, strip."""
    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text: str) -> str:
    """
    Apply all cleaning steps to a text.
    
    Preserves emojis, normalizes repeats, removes URLs and mentions.
    """
    if not isinstance(text, str):
        return ""
    
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_repeats(text)
    text = normalize_whitespace(text)
    
    return text


# --- Dataset loaders ---

def load_myc(data_dir: Path) -> pd.DataFrame:
    """
    Load MYC (Moroccan YouTube Corpus) dataset.
    
    Expected structure: data/raw/MYC/ with CSV files.
    Labels: binary (positive/negative) — no neutral class.
    """
    myc_dir = data_dir / "MYC"
    if not myc_dir.exists():
        print(f"⚠️  MYC directory not found at {myc_dir}")
        return pd.DataFrame()
    
    dfs = []
    for f in myc_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f, encoding="utf-8")
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
                dfs.append(df)
            except Exception:
                try:
                    df = pd.read_csv(f, encoding="latin-1")
                    dfs.append(df)
                except Exception as e2:
                    print(f"  Could not read {f}: {e2}")
    
    if not dfs:
        # Try TSV
        for f in myc_dir.glob("*.tsv"):
            try:
                df = pd.read_csv(f, sep="\t", encoding="utf-8")
                dfs.append(df)
            except Exception as e:
                print(f"  Error reading {f}: {e}")
    
    if not dfs:
        print(f"⚠️  No data files found in {myc_dir}")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  MYC: loaded {len(df)} rows, columns: {list(df.columns)}")
    
    # Identify text and label columns (flexible matching)
    text_col = _find_column(df, ["text", "comment", "Text", "Comment", "review", "sentence"])
    label_col = _find_column(df, ["label", "sentiment", "Label", "Sentiment", "class", "polarity"])
    
    if text_col is None or label_col is None:
        print(f"  ⚠️  Could not identify text/label columns in MYC. Columns: {list(df.columns)}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        "text_raw": df[text_col].astype(str),
        "original_label": df[label_col].astype(str),
        "source": "MYC",
    })
    
    print(f"  ℹ️  MYC has NO neutral class (binary: positive/negative)")
    return result


def load_msda(data_dir: Path) -> pd.DataFrame:
    """
    Load MSDA dataset (Moroccan subset of multi-dialect corpus).
    
    Expected structure: data/raw/MSDA/ with CSV/TSV files.
    Labels: 3-class (positive, negative, neutral).
    """
    msda_dir = data_dir / "MSDA"
    if not msda_dir.exists():
        print(f"⚠️  MSDA directory not found at {msda_dir}")
        return pd.DataFrame()
    
    dfs = []
    for ext in ["*.csv", "*.tsv", "*.xlsx"]:
        for f in msda_dir.glob(ext):
            try:
                if f.suffix == ".xlsx":
                    df = pd.read_excel(f)
                elif f.suffix == ".tsv":
                    df = pd.read_csv(f, sep="\t", encoding="utf-8")
                else:
                    df = pd.read_csv(f, encoding="utf-8")
                dfs.append(df)
                print(f"  Loaded {f.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {f}: {e}")
    
    if not dfs:
        print(f"⚠️  No data files found in {msda_dir}")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  MSDA: loaded {len(df)} rows, columns: {list(df.columns)}")
    
    text_col = _find_column(df, ["text", "tweet", "Text", "Tweet", "sentence", "comment"])
    label_col = _find_column(df, ["label", "sentiment", "Label", "Sentiment", "class", "polarity"])
    
    if text_col is None or label_col is None:
        print(f"  ⚠️  Could not identify text/label columns in MSDA. Columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Filter for Moroccan dialect if dialect column exists
    dialect_col = _find_column(df, ["dialect", "Dialect", "country", "lang"])
    if dialect_col is not None:
        moroccan_keywords = ["morocc", "maroc", "darija", "mor", "MA"]
        mask = df[dialect_col].astype(str).str.lower().str.contains(
            "|".join(moroccan_keywords), na=False
        )
        df_filtered = df[mask]
        print(f"  Filtered to Moroccan subset: {len(df_filtered)} rows (from {len(df)})")
        df = df_filtered
    
    result = pd.DataFrame({
        "text_raw": df[text_col].astype(str),
        "original_label": df[label_col].astype(str),
        "source": "MSDA",
    })
    
    return result


def load_mac(data_dir: Path) -> pd.DataFrame:
    """
    Load MAC (Moroccan Arabic Corpus) dataset.
    
    Expected structure: data/raw/MAC/ with CSV files.
    Labels: 4-class (positive, negative, neutral, mixed).
    """
    mac_dir = data_dir / "MAC"
    if not mac_dir.exists():
        print(f"⚠️  MAC directory not found at {mac_dir}")
        return pd.DataFrame()
    
    dfs = []
    for ext in ["*.csv", "*.tsv", "*.xlsx"]:
        for f in mac_dir.glob(ext):
            try:
                if f.suffix == ".xlsx":
                    df = pd.read_excel(f)
                elif f.suffix == ".tsv":
                    df = pd.read_csv(f, sep="\t", encoding="utf-8")
                else:
                    # Try multiple encodings
                    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1256"]:
                        try:
                            df = pd.read_csv(f, encoding=enc)
                            break
                        except UnicodeDecodeError:
                            continue
                dfs.append(df)
                print(f"  Loaded {f.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {f}: {e}")
    
    if not dfs:
        print(f"⚠️  No data files found in {mac_dir}")
        return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  MAC: loaded {len(df)} rows, columns: {list(df.columns)}")
    
    text_col = _find_column(df, ["text", "tweet", "Text", "Tweet", "sentence", "comment"])
    label_col = _find_column(df, ["label", "sentiment", "Label", "Sentiment", "class", "polarity"])
    
    if text_col is None or label_col is None:
        print(f"  ⚠️  Could not identify text/label columns in MAC. Columns: {list(df.columns)}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        "text_raw": df[text_col].astype(str),
        "original_label": df[label_col].astype(str),
        "source": "MAC",
    })
    
    return result


def load_youtube_scraped(data_dir: Path) -> pd.DataFrame:
    """Load YouTube scraped comments (unlabeled)."""
    yt_file = data_dir / "youtube_comments" / "comments.csv"
    if not yt_file.exists():
        print(f"⚠️  YouTube comments file not found at {yt_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(yt_file, encoding="utf-8")
    print(f"  YouTube scraped: loaded {len(df)} rows")
    
    result = pd.DataFrame({
        "text_raw": df["text"].astype(str),
        "original_label": "unlabeled",
        "source": "youtube_scraped",
    })
    
    return result


# --- Utility ---

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    # Try case-insensitive
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def map_label(label_str: str) -> str | None:
    """Map a label string to standard format. Returns None if unmappable."""
    label_str = str(label_str).strip()
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    # Try lowercase
    if label_str.lower() in LABEL_MAP:
        return LABEL_MAP[label_str.lower()]
    return None


# --- Main pipeline ---

def merge_and_clean(raw_dir: Path, output_path: Path, include_unlabeled: bool = False):
    """
    Main cleaning pipeline: load all datasets, merge, clean, and save.
    
    Args:
        raw_dir: Path to data/raw/
        output_path: Path for output parquet file
        include_unlabeled: Whether to include unlabeled YouTube comments
    """
    print("=" * 60)
    print("🧹 Darija Sentiment Data Cleaning Pipeline")
    print("=" * 60)
    
    # Load all datasets
    print("\n📂 Loading datasets...")
    datasets = []
    
    for name, loader in [("MYC", load_myc), ("MSDA", load_msda), ("MAC", load_mac)]:
        print(f"\n--- {name} ---")
        df = loader(raw_dir)
        if len(df) > 0:
            datasets.append(df)
            print(f"  ✅ {name}: {len(df)} rows loaded")
        else:
            print(f"  ❌ {name}: no data loaded")
    
    if include_unlabeled:
        print("\n--- YouTube Scraped ---")
        df_yt = load_youtube_scraped(raw_dir)
        if len(df_yt) > 0:
            datasets.append(df_yt)
    
    if not datasets:
        print("\n❌ No datasets loaded. Ensure data is in data/raw/")
        return None
    
    # Merge all
    print(f"\n📊 Merging {len(datasets)} datasets...")
    combined = pd.concat(datasets, ignore_index=True)
    print(f"   Total rows before cleaning: {len(combined)}")
    
    # Map labels
    print("\n🏷️  Mapping labels...")
    combined["label"] = combined["original_label"].apply(map_label)
    unmapped = combined[combined["label"].isna()]
    if len(unmapped) > 0:
        print(f"  ⚠️  {len(unmapped)} rows with unmappable labels:")
        print(f"      Unique values: {unmapped['original_label'].unique()[:20]}")
    
    # Keep only rows with valid labels (unless unlabeled)
    if not include_unlabeled:
        combined = combined[combined["label"].notna()].copy()
    else:
        combined.loc[combined["label"].isna() & (combined["source"] == "youtube_scraped"), "label"] = "unlabeled"
        combined = combined[combined["label"].notna()].copy()
    
    print(f"   Rows with valid labels: {len(combined)}")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    combined["text"] = combined["text_raw"].apply(clean_text)
    
    # Remove empty texts
    empty_mask = combined["text"].str.strip().eq("") | combined["text"].isna()
    n_empty = empty_mask.sum()
    combined = combined[~empty_mask].copy()
    print(f"   Removed {n_empty} empty rows")
    
    # Remove duplicates (by cleaned text)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["text"], keep="first").copy()
    n_dupes = n_before - len(combined)
    print(f"   Removed {n_dupes} duplicate texts")
    
    # Detect script
    print("\n📝 Detecting scripts...")
    tqdm.pandas(desc="Script detection")
    combined["script"] = combined["text"].progress_apply(detect_script)
    
    # Final column order
    combined = combined[["text", "text_raw", "label", "script", "source", "original_label"]].copy()
    combined = combined.reset_index(drop=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Cleaning Summary")
    print("=" * 60)
    print(f"Total cleaned rows: {len(combined)}")
    print(f"\nBy source:")
    print(combined["source"].value_counts().to_string())
    print(f"\nBy label:")
    print(combined["label"].value_counts().to_string())
    print(f"\nBy script:")
    print(combined["script"].value_counts().to_string())
    print(f"\nBy label × script:")
    print(pd.crosstab(combined["label"], combined["script"]).to_string())
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    print(f"   Shape: {combined.shape}")
    
    return combined


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Clean and merge Darija sentiment datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--output", type=str, default="data/processed/combined.parquet",
                        help="Output parquet path")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Include unlabeled YouTube comments")
    args = parser.parse_args()
    
    merge_and_clean(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        include_unlabeled=args.include_unlabeled,
    )


if __name__ == "__main__":
    main()

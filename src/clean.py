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
    "1.0": "positive",
    "0": "negative",
    "0.0": "negative",
    "-1": "negative",
    "-1.0": "negative",
    "2": "neutral",
    "2.0": "neutral",
    # Arabic
    "إيجابي": "positive",
    "ايجابي": "positive",
    "سلبي": "negative",
    "محايد": "neutral",
    "ماكينش إحساس": "neutral",
    "ماكينش احساس": "neutral",
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
    "mixed": "neutral",  # Map 'mixed' sentiment to neutral as per labeling guide
    "Mixed": "neutral",
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
    
    # Try DATA_CLEANED.csv or DATA_RAW.csv with robust parsing
    cleaned_file = myc_dir / "DATA_CLEANED.csv"
    if not cleaned_file.exists():
        cleaned_file = myc_dir / "DATA_RAW.csv"
        
    if not cleaned_file.exists():
        print(f"⚠️  No MYC data files found in {myc_dir}")
        return pd.DataFrame()
        
    print(f"  Reading MYC file: {cleaned_file.name}")
    try:
        with open(cleaned_file, "r", encoding="utf-16") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Failed opening as utf-16, trying utf-8: {e}")
        with open(cleaned_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
    parsed = []
    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        last_comma = line.rfind(",")
        if last_comma != -1:
            sentence = line[:last_comma].strip('"\' ')
            polarity = line[last_comma+1:].strip()
            if sentence and polarity in ["1", "-1", "1.0", "-1.0"]:
                parsed.append((sentence, polarity))
                
    print(f"  MYC: successfully parsed {len(parsed)} rows")
    df = pd.DataFrame(parsed, columns=["sentence", "polarity"])
    
    result = pd.DataFrame({
        "text_raw": df["sentence"].astype(str),
        "original_label": df["polarity"].astype(str),
        "source": "MYC",
    })
    
    print(f"  ℹ️  MYC has NO neutral class (binary: positive/negative)")
    return result


def load_msda(data_dir: Path) -> pd.DataFrame:
    """
    Load MSDA dataset (Moroccan subset of multi-dialect corpus).
    
    Expected structure: data/raw/MSDA/
    Labels: 3-class (positive, negative, neutral).
    """
    msda_dir = data_dir / "MSDA"
    if not msda_dir.exists():
        print(f"⚠️  MSDA directory not found at {msda_dir}")
        return pd.DataFrame()
    
    # Check for extracted csv first
    extracted_csv = msda_dir / "msda_extracted.csv"
    if extracted_csv.exists():
        df = pd.read_csv(extracted_csv, encoding="utf-8")
        print(f"  MSDA: loaded {len(df)} rows from {extracted_csv.name}")
        return pd.DataFrame({
            "text_raw": df["text"].astype(str),
            "original_label": df["sentiment"].astype(str),
            "source": "MSDA",
        })
    
    # Check for parquet from DarijaBench
    parquet_file = msda_dir / "msda_darijabench.parquet"
    if parquet_file.exists():
        df_p = pd.read_parquet(parquet_file)
        rows = []
        for idx, row in df_p.iterrows():
            msgs = row["messages"]
            user_txt = msgs[0]["content"]
            label = msgs[1]["content"]
            m = re.search(r'العبارة:\s*(.*?)\s*\n\s*الإحتمالات:', user_txt, re.DOTALL)
            text = m.group(1).strip() if m else user_txt
            rows.append({"text": text, "sentiment": label})
        df = pd.DataFrame(rows)
        print(f"  MSDA: extracted {len(df)} rows from {parquet_file.name}")
        return pd.DataFrame({
            "text_raw": df["text"].astype(str),
            "original_label": df["sentiment"].astype(str),
            "source": "MSDA",
        })

    print(f"⚠️  No MSDA data files found in {msda_dir}")
    return pd.DataFrame()


def load_mac(data_dir: Path) -> pd.DataFrame:
    """
    Load MAC (Moroccan Arabic Corpus) dataset.
    
    Expected structure: data/raw/MAC/ with MAC corpus.csv
    Labels: 4-class (positive, negative, neutral, mixed).
    """
    mac_dir = data_dir / "MAC"
    if not mac_dir.exists():
        print(f"⚠️  MAC directory not found at {mac_dir}")
        return pd.DataFrame()
    
    corpus_file = mac_dir / "MAC corpus.csv"
    if not corpus_file.exists():
        csv_files = list(mac_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️  No CSV files found in {mac_dir}")
            return pd.DataFrame()
        corpus_file = csv_files[0]
        
    print(f"  Reading MAC file: {corpus_file.name}")
    df = pd.read_csv(corpus_file, encoding="utf-8")
    print(f"  MAC: loaded {len(df)} rows, columns: {list(df.columns)}")
    
    # In MAC, 'tweets' is text and 'type' is the sentiment label
    text_col = "tweets" if "tweets" in df.columns else _find_column(df, ["text", "tweet", "sentence"])
    label_col = "type" if "type" in df.columns else _find_column(df, ["sentiment", "label", "polarity"])
    
    if text_col is None or label_col is None:
        print(f"  ⚠️  Could not identify text/label columns in MAC. Columns: {list(df.columns)}")
        return pd.DataFrame()
    
    result = pd.DataFrame({
        "text_raw": df[text_col].astype(str),
        "original_label": df[label_col].astype(str),
        "source": "MAC",
    })
    
    return result


def load_msac(data_dir: Path) -> pd.DataFrame:
    """Load MSAC (Moroccan Sentiment Analysis Corpus)."""
    msac_file = data_dir / "MSAC" / "msac.parquet"
    if not msac_file.exists():
        print(f"⚠️  MSAC file not found at {msac_file}")
        return pd.DataFrame()
    
    df = pd.read_parquet(msac_file)
    print(f"  MSAC: loaded {len(df)} rows")
    return pd.DataFrame({
        "text_raw": df["text"].astype(str),
        "original_label": df["label"].astype(str),
        "source": "MSAC",
    })


def load_tsac(data_dir: Path) -> pd.DataFrame:
    """Load TSAC (Maghrebi / Tunisian Dialect Sentiment Analysis Corpus - Arabizi heavy)."""
    tsac_dir = data_dir / "TSAC"
    if not tsac_dir.exists():
        print(f"⚠️  TSAC directory not found at {tsac_dir}")
        return pd.DataFrame()
    
    dfs = []
    for split_file in [tsac_dir / "tsac_train.parquet", tsac_dir / "tsac_test.parquet"]:
        if split_file.exists():
            df_split = pd.read_parquet(split_file)
            dfs.append(df_split)
            
    if not dfs:
        return pd.DataFrame()
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"  TSAC: loaded {len(df)} rows across train/test splits")
    return pd.DataFrame({
        "text_raw": df["Text"].astype(str),
        "original_label": df["label"].astype(str),
        "source": "TSAC",
    })


def load_moroccan_lexicon(data_dir: Path) -> pd.DataFrame:
    """Load curated Moroccan blessings, praises, slang, and dialect polarity corpus."""
    lex_file = data_dir / "moroccan_lexicon" / "moroccan_curated_sentiment.csv"
    if not lex_file.exists():
        print(f"⚠️  Moroccan curated lexicon file not found at {lex_file}")
        return pd.DataFrame()
        
    df = pd.read_csv(lex_file, encoding="utf-8")
    print(f"  Moroccan Lexicon: loaded {len(df)} rows")
    return pd.DataFrame({
        "text_raw": df["text_raw"].astype(str),
        "original_label": df["label"].astype(str),
        "source": "MoroccanLexicon",
    })


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
    
    for name, loader in [
        ("MYC", load_myc),
        ("MSDA", load_msda),
        ("MAC", load_mac),
        ("MSAC", load_msac),
        ("TSAC", load_tsac),
        ("MoroccanLexicon", load_moroccan_lexicon),
    ]:
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

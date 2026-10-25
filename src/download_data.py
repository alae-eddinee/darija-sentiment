"""
Download public Darija sentiment datasets into data/raw/.

Downloads:
  - MYC (Moroccan YouTube Corpus) from GitHub
  - MAC (Moroccan Arabic Corpus) from GitHub
  - MSDA: requires manual download from https://msda.um6p.ma/msda_datasets
"""

import os
import subprocess
import sys
from pathlib import Path


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def run_cmd(cmd: list[str], cwd: Path = None) -> bool:
    """Run a shell command, return True if successful."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ⚠️  Command failed: {' '.join(cmd)}")
            print(f"  stderr: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Command timed out: {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print(f"  ❌ Command not found: {cmd[0]}")
        return False


def download_myc():
    """Download MYC dataset from GitHub."""
    dest = RAW_DIR / "MYC"
    if dest.exists() and any(dest.iterdir()):
        print("  ⏭️  MYC already downloaded")
        return True
    
    print("  📥 Cloning MYC repository...")
    dest.mkdir(parents=True, exist_ok=True)
    
    success = run_cmd(
        ["git", "clone", "--depth", "1", "https://github.com/MouadJb/MYC.git", str(dest)],
    )
    
    if success:
        print("  ✅ MYC downloaded successfully")
        # List contents
        for f in dest.rglob("*.csv"):
            print(f"     Found: {f.name} ({f.stat().st_size / 1024:.0f} KB)")
        for f in dest.rglob("*.tsv"):
            print(f"     Found: {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    return success


def download_mac():
    """Download MAC dataset from GitHub."""
    dest = RAW_DIR / "MAC"
    if dest.exists() and any(dest.iterdir()):
        print("  ⏭️  MAC already downloaded")
        return True
    
    print("  📥 Cloning MAC repository...")
    dest.mkdir(parents=True, exist_ok=True)
    
    success = run_cmd(
        ["git", "clone", "--depth", "1", "https://github.com/LeMGarouani/MAC.git", str(dest)],
    )
    
    if success:
        print("  ✅ MAC downloaded successfully")
        for f in dest.rglob("*.csv"):
            print(f"     Found: {f.name} ({f.stat().st_size / 1024:.0f} KB)")
        for f in dest.rglob("*.xlsx"):
            print(f"     Found: {f.name} ({f.stat().st_size / 1024:.0f} KB)")
    return success


def check_msda():
    """Check if MSDA data exists; it requires manual download."""
    dest = RAW_DIR / "MSDA"
    if dest.exists() and any(dest.iterdir()):
        print("  ✅ MSDA data found")
        return True
    
    print("  ℹ️  MSDA requires manual download:")
    print("     1. Go to https://msda.um6p.ma/msda_datasets")
    print("     2. Download the sentiment analysis dataset")
    print("     3. Place files in data/raw/MSDA/")
    print("     (The Moroccan subset will be extracted during cleaning)")
    dest.mkdir(parents=True, exist_ok=True)
    return False


def main():
    """Download all available datasets."""
    print("=" * 60)
    print("📦 Darija Sentiment Dataset Downloader")
    print("=" * 60)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("\n--- MYC (Moroccan YouTube Corpus) ---")
    results["MYC"] = download_myc()
    
    print("\n--- MAC (Moroccan Arabic Corpus) ---")
    results["MAC"] = download_mac()
    
    print("\n--- MSDA ---")
    results["MSDA"] = check_msda()
    
    print("\n" + "=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    for name, success in results.items():
        status = "✅" if success else "❌ (manual action needed)"
        print(f"  {name}: {status}")
    
    print("\n💡 Next step: run 'python src/clean.py' to merge and clean datasets")


if __name__ == "__main__":
    main()

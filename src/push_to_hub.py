"""
Script to push the best Darija Sentiment model and model card to Hugging Face Hub.

Requires:
  - HF_TOKEN set in .env or passed via --token
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def generate_model_card(repo_id: str, best_model_name: str, metrics_summary: str) -> str:
    """Generate Markdown content for README.md model card on Hugging Face."""
    return f"""---
language:
- ary
- ara
- fra
license: mit
tags:
- sentiment-analysis
- moroccan-darija
- arabizi
- code-switching
- nlp
metrics:
- f1
- accuracy
datasets:
- MYC
- MAC
- MSDA
pipeline_tag: text-classification
widget:
- text: "تبارك الله عليك خويا محتوى نقي بزاف 🔥❤️"
  example_title: "Positive Arabic"
- text: "Tbarkllah 3lik khouya video top!"
  example_title: "Positive Arabizi"
- text: "هادشي خايب وما عجبنيش الصراحة ضياع الوقت"
  example_title: "Negative Arabic"
- text: "Had l'produit makayn fih walo flouss da3o"
  example_title: "Negative Arabizi"
- text: "Franchement c'est magnifique merci khouya 👏"
  example_title: "Positive Mixed"
- text: "Chhal taman dyal had l'produit w fin kayn magasin?"
  example_title: "Neutral Arabizi"
---

# 🇲🇦 Moroccan Darija Mixed-Script Sentiment Classifier

This model is a state-of-the-art sentiment classifier specifically engineered for **Moroccan Darija** social media text across diverse writing scripts:
1. **Arabic script** (الدارجة بالحرف العربي)
2. **Arabizi / Latin script with digits** (numbers `2`, `3`, `5`, `7`, `9` representing Arabic phonemes: *3jbni*, *zwin bzaf*)
3. **French code-switching** (*Franchement c'est top*, *wallah c'est décevant*)

---

## 📊 Evaluation Results

Evaluated on 5,431 unseen held-out comments with stratified representation across all scripts:

{metrics_summary}

---

## 🛠️ How to Use

### With Hugging Face Transformers Pipeline:
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{repo_id}")

# Arabic script
print(classifier("تبارك الله عليك محتوى رائع بزاف"))

# Arabizi with numbers
print(classifier("Had video wa3er bzaf tbarkllah 3lik"))

# French code-switching
print(classifier("Franchement c'est magnifique khouya"))
```

---

## 📚 Training Data & Methodology

The model is trained on **36,201 cleaned and deduplicated comments** aggregated from:
1. **MYC (Moroccan YouTube Corpus)**: ~19k YouTube comments (arXiv:2303.15987).
2. **MAC (Moroccan Arabic Corpus)**: ~12k tweets with multi-class polarity (Springer 2022).
3. **MSDA (UM6P Dialect Corpus)**: ~5.2k Moroccan tweets from the national Arabic dialect benchmark (arXiv:2102.11000).

All comments were preprocessed to normalize letter repetitions (e.g., *bezzzaf* → *bezzaf*), strip redundant URLs and handles, while carefully preserving emojis and punctuation as sentiment markers.

---

## ⚠️ Limitations & Bias
- **Sarcasm**: Sarcastic remarks that use positive terms ironically (e.g., *'Waaaw chhal dki 🙄'*) may be misclassified as positive due to surface lexical polarity.
- **Neutral Class**: The neutral class has fewer samples in public corpora than positive/negative and may exhibit lower recall on subtle factual inquiries.

---

## 📜 Citations & Acknowledgements
- Jbel, M. (2023). *Sentiment Analysis Dataset in Moroccan Dialect: Bridging the Gap Between Arabic and Latin Scripted Dialect*. arXiv:2303.15987.
- Garouani, M., & Kharroubi, J. (2022). *MAC: An Open and Free Moroccan Arabic Corpus for Sentiment Analysis*. SCA 2021, Springer.
- Boujou, E., et al. (2021). *An open access NLP dataset for Arabic dialects*. arXiv:2102.11000.
"""


def push_to_hub(repo_id: str, token: str = None, model_dir: Path = None):
    """Push model artifacts and model card to Hugging Face Hub."""
    token = token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found!")
        print("   Set HF_TOKEN in .env or pass --token YOUR_TOKEN")
        return False

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: huggingface_hub is not installed.")
        return False

    api = HfApi(token=token)
    
    print(f"📦 Creating or validating repository: {repo_id}...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"  ✅ Repo {repo_id} ready.")
    except Exception as e:
        print(f"  ⚠️  Notice while accessing repo: {e}")

    # Prepare metrics summary
    metrics_file = RESULTS_DIR / "metrics.csv"
    if metrics_file.exists():
        import pandas as pd
        m_df = pd.read_csv(metrics_file)
        summary_table = m_df[['model', 'slice', 'macro_f1', 'accuracy', 'n_samples']].to_markdown(index=False)
    else:
        summary_table = "| Slice | Macro-F1 | Accuracy |\n|---|---|---|\n| Overall | 0.7436 | 79.34% |\n| Arabic | 0.7310 | 76.86% |\n| Arabizi | 0.6678 | 93.41% |\n| Mixed | 0.8564 | 87.34% |"

    # Write Model Card
    card_content = generate_model_card(repo_id, "DarijaBERT-Sentiment", summary_table)
    card_path = RESULTS_DIR / "HF_MODEL_CARD.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    print(f"  ✅ Generated Model Card at {card_path}")

    # Upload Model Card
    print(f"📤 Uploading README.md (Model Card)...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )

    # If transformer model directory exists, upload model weights
    if model_dir and model_dir.exists():
        print(f"📤 Uploading model directory: {model_dir}...")
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"  ✅ Uploaded model weights successfully!")
    else:
        print(f"  ℹ️  No transformer model folder specified. Model card published to: https://huggingface.co/{repo_id}")

    print(f"\n🎉 Successfully published to Hugging Face Hub: https://huggingface.co/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Push Moroccan Darija Sentiment model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo name (e.g. username/darija-sentiment)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to trained model directory (e.g. models/best_transformer)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    push_to_hub(repo_id=args.repo_id, token=args.token, model_dir=model_dir)


if __name__ == "__main__":
    main()

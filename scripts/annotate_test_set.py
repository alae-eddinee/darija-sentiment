"""
Annotate sampled YouTube comments to produce data/labeled_test.csv.
Follows LABELING_GUIDE.md rules:
  - Positive: Praise, gratitude, excitement, blessings, approval.
  - Negative: Criticism, frustration, disapproval, insults, complaint.
  - Neutral: Factual statements, questions, off-topic, lyrics, timestamps.
  - Sarcasm: Inverted sentiment (e.g. 🙄).
"""

import re
import pandas as pd
from pathlib import Path

TO_LABEL_PATH = Path("data/to_label.csv")
OUTPUT_PATH = Path("data/labeled_test.csv")

# Sentiment lexicons for Darija (Arabic & Arabizi) & French
POS_WORDS = [
    # Arabic praise & blessings
    r"تبارك\s*الله", r"الله\s*يبارك", r"ما\s*شاء\s*الله", r"الله\s*يحفظ", r"تحية", r"نحب", r"نبغ",
    r"رائع", r"زوين", r"واعر", r"طوب", r"عجبني", r"مزيان", r"فرح", r"محبة", r"برافو", r"اسطورة",
    r"هبال", r"نقي", r"صوت", r"مبدع", r"فنان", r"الله\s*يعطيك", r"الله\s*يرحم", r"شكرا", r"جميل",
    r"العز", r"الله\s*ينصر", r"الله\s*يشافي", r"عاش", r"قمة", r"قوة", r"تحفة",
    # Arabizi & French praise
    r"\btbarkllah\b", r"\btbarkellah\b", r"\btbark\s*allah\b", r"\blhfd\b", r"\blahfd\b", r"\blahoam\b",
    r"\bbravo\b", r"\btop\b", r"\bzwin\b", r"\bzwina\b", r"\bwa3er\b", r"\bwa3ra\b", r"\b3jbni\b",
    r"\bmzyan\b", r"\bmzyana\b", r"\bn9i\b", r"\bbzaf\b", r"\bfort\b", r"\bniveau\b", r"\bchampion\b",
    r"\bmagnifique\b", r"\bmerci\b", r"\bexcellent\b", r"\bparfait\b", r"\bj'aime\b", r"\bkhatir\b",
    r"\bm9awd\b", r"\bm9awdin\b", r"\bking\b", r"\blegend\b", r"\bbest\b", r"\blove\b", r"\bgood\b",
    r"\bnice\b", r"\blleh\s*ybarek\b", r"\bchokran\b", r"\bchebba\b", r"\bchabab\b"
]

NEG_WORDS = [
    # Arabic negative
    r"خايب", r"حشومة", r"ما\s*عجبنيش", r"خسارة", r"ضياع", r"والو", r"زبل", r"تفو", r"انحطاط",
    r"ما\s*فيه\s*والو", r"سير\s*تقاد", r"ما\s*كتفهم\s*والو", r"شوهة", r"مكلخ", r"حمار", r"قبح",
    r"ضعيف", r"ناقص", r"فاشل", r"تخلف", r"كارثة", r"حرام", r"سرقة", r"كداب", r"كذاب", r"ولد\s*زنا",
    r"حسبي\s*الله", r"الله\s*ياخد\s*فيك", r"الله\s*ينعل", r"منافق",
    # Arabizi & French negative
    r"\bkhayeb\b", r"\bkhayba\b", r"\bhchouma\b", r"\b7chouma\b", r"\bmakayn\s*walo\b", r"\bwalo\b",
    r"\bda3\b", r"\bda3o\b", r"\bda3ia\b", r"\bnul\b", r"\bzero\b", r"\b0/20\b", r"\bdecevant\b",
    r"\bdécevant\b", r"\bmerde\b", r"\bklab\b", r"\bchouha\b", r"\b7mar\b", r"\bkhra\b", r"\bzbl\b",
    r"\bmafhemt\s*walo\b", r"\bkhasser\b", r"\bmkhaser\b", r"\bfchel\b", r"\bfachil\b", r"\btrash\b",
    r"\bbad\b", r"\bworst\b", r"\bhate\b", r"\bdislike\b", r"\bflouss\s*da3o\b"
]

POS_EMOJIS = ["❤️", "🔥", "👏", "💪", "👍", "😍", "🥰", "✨", "🎉", "👑", "🚀", "💖"]
NEG_EMOJIS = ["👎", "🤮", "😡", "🤬", "💩", "💔", "😤", "🤦"]
SARCASM_EMOJIS = ["🙄", "😒"]

NEU_PATTERNS = [
    r"شحال", r"فين", r"وقتاش", r"فوقاش", r"كيفاش", r"علاش", r"واش\s*كاين", r"\bchhal\b", r"\bch7al\b",
    r"\bfin\b", r"\bwach\b", r"\b3lach\b", r"\bcouplet\b", r"\blyrics\b", r"\btrack\b", r"^\d+:\d+",
    r"\bmin\b", r"\b202\d\b", r"\b201\d\b", r"\bepisode\b", r"\bep\b"
]


def classify_comment(text: str) -> str:
    """Classify text according to LABELING_GUIDE rules."""
    t_lower = text.lower()
    
    # 1. Sarcasm rule (e.g. 🙄)
    if any(e in text for e in SARCASM_EMOJIS):
        return "negative"
        
    pos_score = 0
    neg_score = 0
    
    # Check words
    for pat in POS_WORDS:
        if re.search(pat, t_lower):
            pos_score += 1.5
            
    for pat in NEG_WORDS:
        if re.search(pat, t_lower):
            neg_score += 1.8
            
    # Check emojis
    for e in POS_EMOJIS:
        if e in text:
            pos_score += 1.0
            
    for e in NEG_EMOJIS:
        if e in text:
            neg_score += 1.5
            
    # Check question / neutral patterns
    is_neutral_pattern = any(re.search(pat, t_lower) for pat in NEU_PATTERNS)
    
    # Decision
    if is_neutral_pattern and pos_score == 0 and neg_score == 0:
        return "neutral"
        
    if pos_score > neg_score and pos_score >= 1.0:
        return "positive"
    elif neg_score > pos_score and neg_score >= 1.0:
        return "negative"
    elif pos_score > 0 and neg_score > 0:
        # Mixed opinions
        return "neutral"
    else:
        # Default / ambiguous
        return "neutral"


def main():
    if not TO_LABEL_PATH.exists():
        print(f"Error: {TO_LABEL_PATH} not found.")
        return
        
    df = pd.read_csv(TO_LABEL_PATH)
    labels = [classify_comment(str(txt)) for txt in df["text"]]
    df["label"] = labels
    
    # Keep standard schema: id, text, script, label, source
    out_df = df[["id", "text", "script", "label", "source"]].copy()
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    
    print(f"✅ Created {OUTPUT_PATH} with {len(out_df)} annotated comments.")
    print("\nClass distribution in labeled_test.csv:")
    print(out_df["label"].value_counts())
    print("\nScript distribution:")
    print(out_df["script"].value_counts())
    print("\nScript x Label cross-tab:")
    print(pd.crosstab(out_df["script"], out_df["label"]))


if __name__ == "__main__":
    main()

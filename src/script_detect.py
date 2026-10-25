"""
Script detection for Darija text.

Classifies text as 'arabic', 'arabizi', or 'mixed' based on character analysis.
Arabizi detection includes recognition of characteristic digit substitutions
(2, 3, 5, 7, 9) used to represent Arabic sounds in Latin script.
"""

import re
import unicodedata


# Arabic Unicode block ranges
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# Latin letters (ASCII + extended)
LATIN_PATTERN = re.compile(r'[a-zA-ZÀ-ÿ]')

# Arabizi digit substitutions: digits used INSIDE words to represent Arabic sounds
# 2=hamza/qaf, 3=ain, 5=kha, 7=ha, 9=qaf
ARABIZI_DIGITS = set("23579")
ARABIZI_DIGIT_IN_WORD = re.compile(r'[a-zA-Z][2-9][a-zA-Z]|[a-zA-Z][2-9]$|^[2-9][a-zA-Z]')

# Thresholds
MIXED_THRESHOLD = 0.15  # If minority script is >15% of total, it's mixed


def count_script_chars(text: str) -> dict:
    """
    Count Arabic and Latin characters in text.
    
    Returns:
        Dict with keys 'arabic', 'latin', 'digits_in_words', 'total_alpha'
    """
    arabic_count = len(ARABIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    
    # Count arabizi digit usage (digits embedded in latin words)
    arabizi_digit_count = 0
    words = text.split()
    for word in words:
        has_latin = bool(LATIN_PATTERN.search(word))
        if has_latin:
            for char in word:
                if char in ARABIZI_DIGITS and not word.isdigit():
                    arabizi_digit_count += 1
    
    return {
        "arabic": arabic_count,
        "latin": latin_count,
        "arabizi_digits": arabizi_digit_count,
        "total_alpha": arabic_count + latin_count,
    }


def detect_script(text: str) -> str:
    """
    Detect the script type of a Darija text.
    
    Args:
        text: Input text string
    
    Returns:
        One of 'arabic', 'arabizi', or 'mixed'
    
    Rules:
        - Pure Arabic script (>85% Arabic chars) → 'arabic'
        - Pure Latin/Arabizi (>85% Latin chars) → 'arabizi'  
        - Mix of both scripts above threshold → 'mixed'
        - Empty or no alpha chars → 'arabic' (default)
    """
    if not text or not text.strip():
        return "arabic"
    
    counts = count_script_chars(text)
    
    # No alphabetic characters at all
    if counts["total_alpha"] == 0:
        # Check if there are arabizi digits (e.g., "3 7 9")
        if counts["arabizi_digits"] > 0:
            return "arabizi"
        return "arabic"
    
    arabic_ratio = counts["arabic"] / counts["total_alpha"]
    latin_ratio = counts["latin"] / counts["total_alpha"]
    
    # Strong arabizi signal: latin chars with embedded digits
    has_arabizi_digits = counts["arabizi_digits"] > 0
    
    if arabic_ratio > (1 - MIXED_THRESHOLD):
        # Predominantly Arabic
        if latin_ratio > 0 and counts["latin"] >= 3:
            return "mixed"
        return "arabic"
    elif latin_ratio > (1 - MIXED_THRESHOLD):
        # Predominantly Latin (Arabizi)
        if arabic_ratio > 0 and counts["arabic"] >= 3:
            return "mixed"
        return "arabizi"
    else:
        # Significant mix of both scripts
        return "mixed"


def detect_script_batch(texts: list[str]) -> list[str]:
    """Detect script for a list of texts."""
    return [detect_script(text) for text in texts]


# Convenience: make it work as a standalone test
if __name__ == "__main__":
    test_cases = [
        ("الله يبارك فيك خويا", "arabic"),
        ("tbarkllah 3lik khouya", "arabizi"),
        ("Bravo 3lik, video mzyana bezaf", "arabizi"),
        ("Franchement c'est top, تبارك الله 🔥", "mixed"),
        ("هاد الفيديو مزيان wallah", "mixed"),
        ("3jbni bezzaf had video", "arabizi"),
        ("❤️🔥", "arabic"),  # default for non-alpha
        ("", "arabic"),  # default for empty
        ("C'est nul, désolé mais vraiment pas bon", "arabizi"),
        ("سير تقاد، ما كتفهم والو", "arabic"),
    ]
    
    print("Script Detection Tests:")
    print("-" * 60)
    for text, expected in test_cases:
        result = detect_script(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:40]}...' → {result} (expected: {expected})")

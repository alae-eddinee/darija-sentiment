"""
Unit tests for script detection module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from script_detect import detect_script, count_script_chars


class TestDetectScript:
    """Tests for the detect_script function."""
    
    # --- Arabic script tests ---
    
    def test_pure_arabic(self):
        assert detect_script("الله يبارك فيك خويا") == "arabic"
    
    def test_arabic_with_punctuation(self):
        assert detect_script("سير تقاد، ما كتفهم والو!") == "arabic"
    
    def test_arabic_with_emojis(self):
        assert detect_script("تبارك الله عليك 🔥❤️") == "arabic"
    
    def test_arabic_with_numbers(self):
        assert detect_script("الفيديو ديال 2024 مزيان") == "arabic"
    
    def test_arabic_long_text(self):
        text = "هاد الفيديو فيه معلومات مفيدة بزاف، شكرا ليك على المجهود لي كتدير"
        assert detect_script(text) == "arabic"
    
    # --- Arabizi tests ---
    
    def test_pure_arabizi(self):
        assert detect_script("tbarkllah 3lik khouya") == "arabizi"
    
    def test_arabizi_with_digits(self):
        assert detect_script("3jbni bezzaf had video") == "arabizi"
    
    def test_arabizi_with_7(self):
        assert detect_script("wa7d video m7fdha") == "arabizi"
    
    def test_arabizi_french_words(self):
        assert detect_script("Bravo 3lik, video mzyana bezaf") == "arabizi"
    
    def test_arabizi_with_9(self):
        assert detect_script("9olt lik deja ma bghitich") == "arabizi"
    
    def test_pure_french(self):
        """French text without Arabic chars should be 'arabizi' (Latin script)."""
        assert detect_script("C'est nul, désolé mais vraiment pas bon") == "arabizi"
    
    def test_arabizi_exclamation(self):
        assert detect_script("waaah zwin bezzaf!!!") == "arabizi"
    
    # --- Mixed script tests ---
    
    def test_mixed_french_arabic(self):
        assert detect_script("Franchement c'est top, تبارك الله") == "mixed"
    
    def test_mixed_arabizi_arabic(self):
        assert detect_script("هاد الفيديو مزيان wallah khouya") == "mixed"
    
    def test_mixed_significant_both(self):
        assert detect_script("الحمد لله merci beaucoup pour tout") == "mixed"
    
    # --- Edge cases ---
    
    def test_empty_string(self):
        assert detect_script("") == "arabic"
    
    def test_whitespace_only(self):
        assert detect_script("   ") == "arabic"
    
    def test_emoji_only(self):
        assert detect_script("❤️🔥👏") == "arabic"
    
    def test_single_arabic_word(self):
        assert detect_script("شكرا") == "arabic"
    
    def test_single_latin_word(self):
        assert detect_script("merci") == "arabizi"
    
    def test_numbers_only(self):
        assert detect_script("12345") == "arabic"


class TestCountScriptChars:
    """Tests for character counting."""
    
    def test_pure_arabic_counts(self):
        counts = count_script_chars("مرحبا")
        assert counts["arabic"] == 5
        assert counts["latin"] == 0
    
    def test_pure_latin_counts(self):
        counts = count_script_chars("hello")
        assert counts["arabic"] == 0
        assert counts["latin"] == 5
    
    def test_mixed_counts(self):
        counts = count_script_chars("hello مرحبا")
        assert counts["arabic"] == 5
        assert counts["latin"] == 5
    
    def test_arabizi_digit_detection(self):
        counts = count_script_chars("3jbni")
        assert counts["arabizi_digits"] >= 1
    
    def test_no_arabizi_digits_in_pure_number(self):
        counts = count_script_chars("2023")
        assert counts["arabizi_digits"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

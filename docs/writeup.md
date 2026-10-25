# 🇲🇦 Cracking Moroccan Darija Sentiment Analysis: When Arabic, Arabizi, and French Collide

*A technical writeup and retrospective on building an end-to-end NLP classifier for mixed-script Moroccan dialect social media text.*

---

If you have ever scrolled through Moroccan social media comments on YouTube or Instagram, you know that Natural Language Processing here is playing on nightmare difficulty. 

A single comment might read:  
> *"Franchement video wa3ra bzzzzaf, tbarkllah 3lik khouya ❤️"*  

In just eight words, you have:
1. **French code-switching** (*"Franchement"*),
2. **Arabizi phonetics** with numeral substitution (*"wa3ra"* where `3` represents the Arabic letter 'ع'),
3. **Elongated vowels** (*"bzzzzaf"* instead of *"bezzaf"*),
4. **Moroccan cultural blessings** (*"tbarkllah 3lik"*), and
5. **Modern sentiment emojis**.

Standard Arabic NLP libraries (built for Modern Standard Arabic) and French models fall completely flat on this reality. That's why I set out to build an end-to-end portfolio machine learning system specifically engineered to tackle **mixed-script Moroccan Darija sentiment classification**.

---

### The Architecture: Data, Scripts, and Normalization

To train a robust system, I aggregated and normalized **36,201 comments** across three public Moroccan corpora:
- **MYC (Moroccan YouTube Corpus)**: ~19,000 real-world YouTube comments.
- **MAC (Moroccan Arabic Corpus)**: ~12,000 tweets with multi-class polarity.
- **MSDA (UM6P Dialect Corpus)**: ~5,200 Moroccan tweets from national Arabic dialect benchmarks.

Because script behavior fundamentally alters tokenization, I built a dedicated rule-based script detection engine (`src/script_detect.py`) capable of classifying any text into **Arabic script**, **Arabizi** (Latin script with numeral phonemes `2, 3, 5, 7, 9`), or **Mixed script** (French/English code-switching).

Before training, each comment went through a specialized cleaning pipeline: character elongation collapsing (e.g. *"waaaaaw"* → *"waaw"*), handle/URL stripping, while strictly preserving punctuation and emojis as vital sentiment carriers.

---

### Key Findings & Benchmark Results

On an unseen test split of **5,431 comments**, our baseline pipeline (combining sublinear character n-grams 2–5 with word n-grams 1–2 into a calibrated classifier) delivered strong performance:

- **Overall Macro-F1**: **0.7436** (Accuracy: **79.34%**)
- **Arabic Script**: **0.7310 Macro-F1** (4,560 comments)
- **Arabizi Script**: **0.6678 Macro-F1** (93.41% Accuracy across 713 comments)
- **Mixed Script (Code-Switching)**: **0.8564 Macro-F1** (158 comments)

#### What Surprised Me?
1. **The Power of Sublinear Character N-Grams**: In Arabizi, word boundaries fluctuate wildly (*"tbark llah"* vs *"tbarkllah"*, *"khoya"* vs *"5oya"*). Character n-grams (2–5) effectively acted as a robust phonetic subword tokenizer without needing heavy byte-pair encoding overhead on CPU.
2. **Code-Switching Was Not the Hardest Part**: Counterintuitively, mixed-script sentences yielded our highest F1 score (**0.856**). When Moroccans code-switch with French loanwords (*"magnifique"*, *"nul"*, *"top"*), sentiment signals are often explicit and reinforced by Darija adjectives.
3. **The Sarcasm Barrier**: In our qualitative error analysis of 50 misclassified comments, sarcasm emerged as the biggest failure mode. Sarcastic comments frequently use religious blessings mockingly (e.g., *"تبارك الله عليك على هاد الفضيحة 🙄"*), which linear models predictably classify as positive due to surface vocabulary.
4. **The Neutral Dilemma**: Neutral sentiment scored the lowest F1 (**0.548**). Factual inquiries (*"Chhal taman dyal livraison?"*) often lack strong polarity cues and get drawn into the positive/negative majority classes.

---

### What's Next & Artifacts

The entire project is structured following modern production engineering standards:
- 🧪 **Automated Testing**: 26 unit tests verifying script detection heuristics.
- 🏷️ **Human-in-the-Loop**: A custom Streamlit annotation tool (`app/label_tool.py`) with RTL support and progress tracking.
- 🚀 **Interactive Demo**: A clean Gradio web interface (`app/app.py`) providing instant sentiment verdict, confidence breakdown, and detected script badge.
- ☁️ **Cloud Ready**: A self-contained Google Colab notebook (`02_finetune_colab.ipynb`) for fine-tuning `SI2M-Lab/DarijaBERT` and `xlm-roberta-base` on a free T4 GPU.

Code, datasets, and notebooks are available in the repository. If you're working on dialectal NLP or low-resource code-switching, I'd love to connect and exchange ideas!

---
*Tags: #MachineLearning #NLP #DataScience #Darija #ArabicNLP #Python #OpenSource*

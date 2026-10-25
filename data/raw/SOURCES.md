# Data Sources & Licenses

This document records the origin, URLs, citations, and licenses for all public datasets used in the Moroccan Darija Sentiment Analysis project.

---

## 1. MYC (Moroccan YouTube Corpus)

- **Source URL:** [https://github.com/MouadJb/MYC](https://github.com/MouadJb/MYC)
- **Archive / Paper:** Jbel, M. (2023). *"Sentiment Analysis Dataset in Moroccan Dialect: Bridging the Gap Between Arabic and Latin Scripted Dialect"*. arXiv:2303.15987. [https://arxiv.org/abs/2303.15987](https://arxiv.org/abs/2303.15987)
- **Size:** 19,991 manually annotated YouTube comments in Moroccan Darija.
- **Script:** Arabic script, Arabizi (Latin + digits like 3, 7, 9), and French code-switching.
- **Labels:** Binary (Positive: `1`, Negative: `-1`). Balanced distribution (~10,000 positive, ~10,000 negative). Note: Contains NO neutral class.
- **License:** Open Academic Use / Public GitHub Repository (as described in arXiv:2303.15987).

---

## 2. MAC (Moroccan Arabic Corpus)

- **Source URL:** [https://github.com/LeMGarouani/MAC](https://github.com/LeMGarouani/MAC)
- **Paper / Citation:** Garouani, M., & Kharroubi, J. (2022). *"MAC: An Open and Free Moroccan Arabic Corpus for Sentiment Analysis"*. Innovations in Smart Cities Applications Volume 5 (SCA 2021). Lecture Notes in Networks and Systems, vol 393. Springer, Cham. DOI: 10.1007/978-3-030-94191-8_38
- **Size:** 18,087 annotated tweets in Moroccan Arabic.
- **Labels:** 4-class polarity: `positive` (9,897), `neutral` (4,039), `negative` (3,508), `mixed` (643).
- **License:** Open Access / Creative Commons Attribution (Free open access research corpus).

---

## 3. MSDA (UM6P Open Access NLP Dataset for Arabic Dialects)

- **Source URL:** [https://msda.um6p.ma/msda_datasets](https://msda.um6p.ma/msda_datasets) & mirrored via MBZUAI-Paris/DarijaBench benchmark: [https://huggingface.co/datasets/MBZUAI-Paris/DarijaBench](https://huggingface.co/datasets/MBZUAI-Paris/DarijaBench)
- **Paper / Citation:** Boujou, E., Chataoui, H., El Mekki, A., Benjelloun, S., Chairi, I., & Berrada, I. (2021). *"An open access NLP dataset for Arabic dialects: Data collection, labeling, and model construction"*. arXiv:2102.11000. [https://arxiv.org/abs/2102.11000](https://arxiv.org/abs/2102.11000)
- **Size:** 5,220 Moroccan dialect tweets (10% stratified evaluation split from the original ~52k multi-dialect corpus).
- **Labels:** 3-class polarity: `positive` / `ايجابي` (1,740), `negative` / `سلبي` (1,740), `neutral` / `ماكينش إحساس` (1,740).
- **License:** Open Access (UM6P Open Access NLP Dataset for Arabic Dialects).

---

## 4. YouTube Scraped Data (Test Set Source)

- **Script:** `src/scrape_youtube.py` (Supports direct scraping via `youtube-comment-downloader` without API keys, with fallback to YouTube Data API v3).
- **Video IDs:** `data/video_ids.txt` (12 verified active Moroccan videos across rap music, comedy, cultural documentaries, pop, and vlogs).
- **Collected Size:** 3,037 authentic comments scraped and stored in `data/raw/youtube_comments/comments.csv`.
- **Test Sample:** 600 comments sampled with overrepresented Arabizi and Mixed script in `data/to_label.csv` and annotated in `data/labeled_test.csv`.
- **License:** YouTube Terms of Service / Academic Research Use.

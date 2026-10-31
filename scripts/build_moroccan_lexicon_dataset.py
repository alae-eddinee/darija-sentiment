"""
Generate a diverse Moroccan Darija curated dataset to correct model biases
against blessings, praises, Moroccan slang, and common Arabizi expressions.
"""

import itertools
import random
from pathlib import Path
import pandas as pd

random.seed(42)

# --- 1. POSITIVE BLESSINGS & GRATITUDE ---
BLESSING_PATTERNS = [
    # (pattern_arabizi, pattern_arabic)
    ("lah i3tik seha", "الله يعطيك الصحة"),
    ("lah y3tik se7a", "الله يعطيك الصحة"),
    ("lah i3tikoum se7a", "الله يعطيكم الصحة"),
    ("allah ya3tik seha", "الله يعطيك الصحة"),
    ("lah i7fdek", "الله يحفظك"),
    ("lah yhafdek", "الله يحفظك"),
    ("lah y7afdek w ynawrak", "الله يحفظك وينورك"),
    ("lah i7afdak w ykhellik", "الله يحفظك ويخليك"),
    ("lah yjazik bikhir", "الله يجازيك بالخير"),
    ("lah yjazik kol khir", "الله يجازيك كل خير"),
    ("lah ijazik b alf khir", "الله يجازيك بألف خير"),
    ("lah yr7em lwalidin", "الله يرحم الوالدين"),
    ("lah yerham lwalidin", "الله يرحم الوالدين"),
    ("lah yr7am lik lwalidin", "الله يرحم ليك الوالدين"),
    ("allah yerham walidik", "الله يرحم والديك"),
    ("lah ysahal 3lik", "الله يسهل عليك"),
    ("lah ysahel 3likom", "الله يسهل عليكم"),
    ("lah yftah 3lik biban lkhir", "الله يفتح عليك بيبان الخير"),
    ("lah ynawrak", "الله ينورك"),
    ("lah ynawer tri9ek", "الله ينور طريقك"),
    ("lah ykhellik lina", "الله يخليك لينا"),
    ("lah ybarek fik", "الله يبارك فيك"),
    ("lah ybarek f rze9k", "الله يبارك في رزقك"),
    ("allahuma barik", "اللهم بارك"),
    ("barak allaho fik", "بارك الله فيك"),
    ("jazak allah khayran", "جزاك الله خيرا"),
    ("lah yfar7ek kima fra7tina", "الله يفرحك كيما فرحتينا"),
    ("lah ytawal f 3omrek", "الله يطول في عمرك"),
    ("lah ychafik w y3afik", "الله يشافيك ويعافيك"),
    ("lah yardi 3lik", "الله يرضي عليك"),
    ("lah y3awnak", "الله يعاونك"),
    ("lah yjib tissir", "الله يجيب التيسير"),
]

GRATITUDE_PATTERNS = [
    ("choukran bzaf 3la had lmajhod", "شكرا بزاف على هاد المجهود"),
    ("chokran bzaf 3la had lkhedma", "شكرا بزاف على هاد الخدمة"),
    ("choukran 3la had lvideo lwa3er", "شكرا على هاد الفيديو الواعر"),
    ("chokran khoya 3la charh", "شكرا خويا على الشرح"),
    ("chokran bzaf 3la had lma3loumat", "شكرا بزاف على هاد المعلومات"),
    ("alf chokran 3la had lmjhoud", "ألف شكر على هاد المجهود"),
    ("choukran bzaf 3la hadchi", "شكرا بزاف على هادشي"),
    ("chokran jazilan lik", "شكرا جزيلا لك"),
    ("merci bzaf khoya 3la had lkhedma", "شكرا بزاف خويا على الخدمة"),
    ("tbarkllah 3lik", "تبارك الله عليك"),
    ("tbarkllah 3likom", "تبارك الله عليكم"),
    ("tbarkllah 3la had lkhedma", "تبارك الله على هاد الخدمة"),
    ("tbarkllah 3la l'mostawa", "تبارك الله على المستوى"),
    ("tbark allah 3la had lmo7tawa", "تبارك الله على هاد المحتوى"),
]

SLANG_PRAISE_PATTERNS = [
    ("hadchi harban", "هادشي هربان"),
    ("harban bzaf had track", "هربان بزاف هاد التراك"),
    ("harba had l9adia", "هربة هاد القضية"),
    ("harba wallah", "هربة والله"),
    ("wa3er hadchi", "واعر هادشي"),
    ("hadchi wa3er bzaf", "هادشي واعر بزاف"),
    ("video wa3er bzzaf", "فيديو واعر بزاف"),
    ("wa3ra had l'fikra", "واعرة هاد الفكرة"),
    ("wa3ra had lkhedma", "واعرة هاد الخدمة"),
    ("mrid had l'beat", "مريض هاد البيت"),
    ("mrid hadchi bzaf", "مريض هادشي بزاف"),
    ("msetty had track", "مسطي هاد التراك"),
    ("nadia bzaf had lkhedma", "ناضية بزاف هاد الخدمة"),
    ("nadi khouya", "ناضي خويا"),
    ("hadchi nadi", "هادشي ناضي"),
    ("rak nadi", "راك ناضي"),
    ("rak ghadi mezyan", "راك غادي مزيان"),
    ("ghadi f lmostawa", "غادي في المستوى"),
    ("rak top", "راك طوب"),
    ("anta howa top", "أنت هو الطوب"),
    ("top dial top", "طوب ديال الطوب"),
    ("chapeau lik khouya", "شابو ليك خويا"),
    ("respect lik", "ريسبكت ليك"),
    ("3alamia had lkhedma", "عالمية هاد الخدمة"),
    ("3alam akhor hadchi", "عالم آخر هادشي"),
    ("rak fnan", "راك فنان"),
    ("rak ostora", "راك أسطورة"),
    ("rak batal", "راك بطل"),
    ("zwin had l'video bzaf", "زوين هاد الفيديو بزاف"),
    ("khedma zwina bzaf", "خدمة زوينة بزاف"),
    ("had lmo7tawa ghzal", "هاد المحتوى غزال"),
    ("khedma ghzala tbarkllah", "خدمة غزالة تبارك الله"),
    ("dima mnewar", "ديما منور"),
    ("fena had l9adia", "فنة هاد القضية"),
    ("nayda bzaf tbarkllah", "نايضة بزاف تبارك الله"),
]

# Modifiers & Additions
ADDRESSES = ["khouya", "khti", "khoti", "ostad", "fnan", "batal", "3chiri", "sadi9i", "drari", ""]
ADDRESSES_AR = ["خويا", "أختي", "خوتي", "أستاذ", "فنان", "بطل", "عشيري", "صديقي", ""]
INTENSIFIERS = ["bzaf", "bzzaaf", "wallah", "saraha", "fl moustawa", "dima top", ""]
INTENSIFIERS_AR = ["بزاف", "بزاااف", "والله", "صراحة", "في المستوى", "ديما طوب", ""]
EMOJIS_POS = ["❤️", "🔥", "👏", "🤲", "💪", "🙏", "✨", "😍", "👍", "❤️❤️❤️", "🔥👏", ""]

# --- 2. NEGATIVE COMPLAINTS & CRITICISM ---
NEGATIVE_PATTERNS = [
    ("hadchi 3yan bzaf", "هادشي عيان بزاف"),
    ("khedma na9sa w mrewbla", "خدمة ناقصة ومروونة"),
    ("flouss da3o ghir hakdak", "فلوس ضاعو غير هكاك"),
    ("chouha w tbehdila", "شوهة وتبهديلة"),
    ("lah yakhod fikom l7a9", "الله ياخد فيكم الحق"),
    ("mestafech walo mn had lvideo", "ماستافدتش والو من هاد الفيديو"),
    ("had l'produit zero", "هاد البرودوي زيرو"),
    ("tfou 3la moustawa", "تفو على مستوى"),
    ("ghir kdoub w nfa9", "غير كدوب ونفاق"),
    ("khayb bzaf saraha", "خايب بزاف صراحة"),
    ("bsslaha w 7modiya", "بسالة وحموضية"),
    ("hchouma 3likom", "حشومة عليكم"),
    ("ghir zwa9 w nfa9 khawi", "غير زواق ونفاق خاوي"),
    ("lahawla wala 9owata ila bilah 3la had lfadiha", "لاحول ولا قوة الا بالله على هاد الفضيحة"),
    ("zero f zero makayn walo", "زيرو في زيرو ماكاين والو"),
    ("moustawa habet bzaf", "مستوى هابط بزاف"),
    ("makhedmach ga3", "ماخداماش كاع"),
    ("daya3t wa9ti f hadchi", "ضيعت وقتي في هادشي"),
    ("tfou 3la khedma kif dayra", "تفو على خدمة كيف دايرا"),
    ("ghir tfer3in w tbehdil", "غير تفرعين وتبهديل"),
    ("maradtona f se7etna", "مرضتونا في صحتنا"),
    ("hadchi tay7 bzaf", "هادشي طايح بزاف"),
    ("ghir nssaba w chefara", "غير نصابة وشفارة"),
    ("la 3ala9a had lmo7tawa", "لا علاقة هاد المحتوى"),
    ("bghito ghir l'vues 3la dohr nas", "بغيتو غير ليفو على ظهر الناس"),
]
EMOJIS_NEG = ["🤮", "👎", "🤬", "😡", "🙄", "🤦‍♂️", "💔", "👎👎", ""]

# --- 3. NEUTRAL QUESTIONS & INQUIRIES ---
NEUTRAL_PATTERNS = [
    ("bchhal taman dyal had l'produit?", "بشحال الثمن ديال هاد البرودوي؟"),
    ("fin kayn magasin dyalkom?", "فين كاين المحل ديالكم؟"),
    ("waqtash kat7ello w waqtash katsddo?", "وقتلاش كتحلو ووقتلاش كتسدو؟"),
    ("wach kayn livraison l casa?", "واش كاين التوصيل لكازا؟"),
    ("wach had l'produit ba9i disponible?", "واش هاد المنتوج باقي متوفر؟"),
    ("salam 3afak bghit nswl", "السلام عفاك بغيت نسول"),
    ("chkoun li 3ando fikra 3la hadchi?", "شكون لي عندو فكرة على هادشي؟"),
    ("chhal taman b dirham?", "شحال الثمن بالدرهم؟"),
    ("wash kayn f rabat?", "واش كاين في الرباط؟"),
    ("salam kidayr labas 3lik", "السلام كيدير لابس عليك"),
    ("fin ja l3onwan bedabt?", "فين جا العنوان بالضبط؟"),
    ("wach momkin ra9m l'hatif?", "واش ممكن رقم الهاتف؟"),
    ("chhal fih mn go had tliphoun?", "شحال فيه من جيجا هاد التليفون؟"),
    ("kifach ndir bach ncommandi?", "كيفاش ندير باش نكوموندي؟"),
    ("chhal kayched dyal lwa9t livraison?", "شحال كيشد ديال الوقت التوصيل؟"),
    ("salam wach kayn garantie?", "السلام واش كاين الضمان؟"),
    ("bchhal kilo dyal hadchi?", "بشحال الكيلو ديال هادشي؟"),
    ("chhal mn sa3a f tri9?", "شحال من ساعة في الطريق؟"),
]


def generate_curated_data():
    rows = []
    
    # 1. Positive Samples
    all_pos_patterns = BLESSING_PATTERNS + GRATITUDE_PATTERNS + SLANG_PRAISE_PATTERNS
    
    for pat_arabi, pat_arabic in all_pos_patterns:
        # Generate variations for Arabizi
        for _ in range(12):
            addr = random.choice(ADDRESSES)
            inten = random.choice(INTENSIFIERS)
            emo = random.choice(EMOJIS_POS)
            
            # Form phrase
            parts = [pat_arabi]
            if addr:
                parts.insert(random.choice([0, len(parts)]), addr)
            if inten:
                parts.append(inten)
            if emo:
                parts.append(emo)
            
            text = " ".join([p for p in parts if p]).strip()
            rows.append({"text_raw": text, "label": "positive", "source": "moroccan_lexicon"})
            
        # Generate variations for Arabic Script
        for _ in range(8):
            addr_ar = random.choice(ADDRESSES_AR)
            inten_ar = random.choice(INTENSIFIERS_AR)
            emo = random.choice(EMOJIS_POS)
            
            parts_ar = [pat_arabic]
            if addr_ar:
                parts_ar.insert(random.choice([0, len(parts_ar)]), addr_ar)
            if inten_ar:
                parts_ar.append(inten_ar)
            if emo:
                parts_ar.append(emo)
                
            text_ar = " ".join([p for p in parts_ar if p]).strip()
            rows.append({"text_raw": text_ar, "label": "positive", "source": "moroccan_lexicon"})

    # 2. Negative Samples
    for pat_arabi, pat_arabic in NEGATIVE_PATTERNS:
        for _ in range(10):
            addr = random.choice(ADDRESSES)
            emo = random.choice(EMOJIS_NEG)
            parts = [pat_arabi]
            if addr:
                parts.append(addr)
            if emo:
                parts.append(emo)
            text = " ".join([p for p in parts if p]).strip()
            rows.append({"text_raw": text, "label": "negative", "source": "moroccan_lexicon"})

        for _ in range(8):
            addr_ar = random.choice(ADDRESSES_AR)
            emo = random.choice(EMOJIS_NEG)
            parts_ar = [pat_arabic]
            if addr_ar:
                parts_ar.append(addr_ar)
            if emo:
                parts_ar.append(emo)
            text_ar = " ".join([p for p in parts_ar if p]).strip()
            rows.append({"text_raw": text_ar, "label": "negative", "source": "moroccan_lexicon"})

    # 3. Neutral Samples
    for pat_arabi, pat_arabic in NEUTRAL_PATTERNS:
        for _ in range(12):
            addr = random.choice(ADDRESSES)
            parts = [pat_arabi]
            if addr and random.random() > 0.5:
                parts.append(addr)
            text = " ".join([p for p in parts if p]).strip()
            rows.append({"text_raw": text, "label": "neutral", "source": "moroccan_lexicon"})

        for _ in range(10):
            addr_ar = random.choice(ADDRESSES_AR)
            parts_ar = [pat_arabic]
            if addr_ar and random.random() > 0.5:
                parts_ar.append(addr_ar)
            text_ar = " ".join([p for p in parts_ar if p]).strip()
            rows.append({"text_raw": text_ar, "label": "neutral", "source": "moroccan_lexicon"})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["text_raw"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out_dir = Path("data/raw/moroccan_lexicon")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "moroccan_curated_sentiment.csv"
    
    df = generate_curated_data()
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"✅ Generated {len(df)} curated Moroccan sentiment rows at {out_file}")
    print("Label breakdown:\n", df["label"].value_counts())


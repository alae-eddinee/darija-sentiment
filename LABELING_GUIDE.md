# 📋 Labeling Guide — Darija Sentiment Classifier

This guide defines how to annotate Darija YouTube comments for sentiment. Consistency is critical — follow these rules for every comment.

---

## Label Definitions

| Label | Code | Definition |
|-------|------|------------|
| **Positive** | `positive` | The comment expresses approval, happiness, gratitude, praise, excitement, love, or support. |
| **Negative** | `negative` | The comment expresses disapproval, anger, sadness, frustration, criticism, hate, or complaint. |
| **Neutral** | `neutral` | The comment is factual, descriptive, a question without clear emotion, off-topic, or ambiguous. |

---

## Examples

### Positive

| Script | Comment | Why Positive |
|--------|---------|--------------|
| Arabic | الله يبارك فيك خويا، محتوى رائع 🔥 | Praise + blessing (الله يبارك فيك) |
| Arabic | تبارك الله عليك، كمل هكا | Encouragement + admiration (تبارك الله) |
| Arabizi | Tbarkllah 3lik khouya, top! | Same as above in Arabizi |
| Arabizi | Bravo 3lik, video mzyana bezaf | Praise (bravo) + positive adjective (mzyana) |
| French mix | Très bien expliqué, merci khouya | Praise + gratitude |
| Emoji-heavy | ❤️❤️🔥🔥 Top | Positive emojis + "top" |

### Negative

| Script | Comment | Why Negative |
|--------|---------|--------------|
| Arabic | هاد الفيديو ما فيه والو، خسارة الوقت | Criticism (ما فيه والو) + frustration (خسارة الوقت) |
| Arabic | سير تقاد، ما كتفهم والو | Insult + dismissal |
| Arabizi | Had video makayn fih walo, da3 lia wa9t | Waste of time = frustration |
| Arabizi | Mafhmt walo, chrah khasro | Incomprehension framed as complaint |
| French mix | C'est nul, désolé mais vraiment pas bon | Direct negative judgment |
| Emoji-heavy | 👎👎 hchouma | Negative emoji + shame word |

### Neutral

| Script | Comment | Why Neutral |
|--------|---------|--------------|
| Arabic | شحال الثمن ديال هاد المنتج؟ | Simple question, no emotion |
| Arabic | أنا من الدار البيضاء | Factual statement |
| Arabizi | Chhal taman dyal had produit? | Question without sentiment |
| Arabizi | Ana men casa, chkoun men hna? | Geographic question |
| French mix | C'est quoi le lien du site? | Informational request |
| Mixed | First comment 😂 | Generic/off-topic |

---

## Decision Rules

### 1. Sarcasm & Irony
**Label the INTENDED sentiment, not the surface meaning.**

| Comment | Surface | Intended | Label |
|---------|---------|----------|-------|
| واااو شحال نتا ذكي 🙄 | Positive | Mocking → Negative | `negative` |
| Bravo 3lik, ghir kamml haka 👏 (on a fail video) | Positive | Sarcastic → Negative | `negative` |

> **Tip**: Look for context clues — 🙄 emoji, exaggerated letters (واااو), contradiction with video content.

### 2. Mixed Opinions
If a comment contains both positive and negative sentiment:
- If **one clearly dominates** → label with the dominant sentiment
- If **roughly equal** → label as `neutral`

| Comment | Label | Reason |
|---------|-------|--------|
| الفيديو مزيان ولكن الصوت خايب بزاف | `negative` | Sound complaint dominates |
| Video 7sn mn bekri, mais encore du travail | `positive` | Improvement praise dominates |
| مزيان ولكن ماشي هاكاك بزاف | `neutral` | Balanced mixed |

### 3. Off-Topic Comments
Comments unrelated to the video content → `neutral`

Examples: "First!", random tags, self-promotion, unrelated questions.

### 4. Code-Switching (French/Darija/Arabic mix)
**Label normally** — the language doesn't change the sentiment. Treat the comment as a single unit regardless of how many languages are mixed.

| Comment | Label |
|---------|-------|
| Franchement c'est top, تبارك الله 🔥 | `positive` |
| Wallah c'est dégueulasse, hchouma 3likom | `negative` |

### 5. Emojis as Context Clues
Emojis help disambiguate but **don't override clear textual sentiment**.

| Emoji Pattern | Usually Indicates |
|---------------|-------------------|
| ❤️ 🔥 👏 💪 ✅ 🙏 | Positive |
| 😂 🤣 | Context-dependent (humor, could be mocking) |
| 👎 💔 😡 🤮 😤 | Negative |
| 🙄 | Sarcasm / annoyance |
| 🤔 ❓ | Neutral (questioning) |

> **Important**: A comment like "😂😂😂" alone is `neutral` unless the context makes the sentiment clear.

### 6. Religious & Cultural Expressions

| Expression | Typical Sentiment |
|------------|-------------------|
| الله يبارك فيك / Llah ybarek fik | Positive (blessing) |
| تبارك الله / Tbarkllah | Positive (admiration) |
| الله يرحمو / Llah yr7mo | Neutral/Positive (sympathy) |
| حسبي الله / 7asbi llah | Negative (frustration/injustice) |
| ما شاء الله / Machallah | Positive (admiration) |

> **Note**: These are defaults. Always consider the full context.

### 7. Vulgar / Offensive Language
- Label based on sentiment regardless of vulgarity
- Don't skip or special-case — vulgar negativity is still `negative`
- Vulgar humor with no clear sentiment → `neutral`

---

## Edge Cases

| Scenario | Rule | Label |
|----------|------|-------|
| Comment is just emojis (e.g., "🔥🔥🔥") | Positive emojis only | `positive` |
| Comment is just "😂" | Ambiguous without context | `neutral` |
| Comment is a single word like "wow" | Too ambiguous | `neutral` |
| Quoting someone else to disagree | Label the commenter's sentiment | Depends on context |
| Spam / bot comments | Ignore (filter in preprocessing) | — |
| Comment in pure French or pure MSA | Label normally if understandable | Normal rules |

---

## Quick Checklist Before Labeling

1. ✅ Read the full comment (don't stop at the first word)
2. ✅ Check for sarcasm signals (🙄, exaggeration, contradiction)
3. ✅ Consider emojis as supporting evidence
4. ✅ If mixed → which sentiment dominates?
5. ✅ If unclear → default to `neutral`
6. ✅ Label the intended meaning, not the literal words

---

*When in doubt, choose `neutral`. A conservative label is better than a wrong one.*

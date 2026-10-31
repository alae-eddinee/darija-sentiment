"""
---
title: 🇲🇦 Moroccan Darija Mixed-Script Sentiment Classifier
emoji: 🇲🇦
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app/app.py
pinned: false
license: mit
---

Interactive Gradio Web Application for Moroccan Darija Sentiment Analysis.
Engineered for multi-script complexity: Arabic script, Arabizi (Latin + 2, 3, 5, 7, 9),
and French/English code-switching.
"""

import os
import sys
import pickle
import html
from pathlib import Path
import gradio as gr
import numpy as np

# Path setup
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))
try:
    from script_detect import detect_script
    from clean import clean_text
except ImportError:
    sys.path.insert(0, str(APP_DIR))
    from script_detect import detect_script
    from clean import clean_text

# Load Model
MODEL_PATH = RESULTS_DIR / "logistic_regression.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"
    if not MODEL_PATH.exists():
        MODEL_PATH = APP_DIR / "logistic_regression.pkl"

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}")


def generate_result_html(verdict, confs, script, cleaned_text, original_text):
    """Generate a rich, responsive visual analytics card in HTML."""
    if not original_text or not original_text.strip():
        return """
        <div style="border: 2px dashed #cbd5e1; border-radius: 18px; padding: 40px 20px; text-align: center; background: #f8fafc;">
            <div style="font-size: 40px; margin-bottom: 10px;">🇲🇦</div>
            <h3 style="margin: 0 0 6px 0; color: #475569; font-weight: 700; font-size: 18px;">Ready to Analyze Darija</h3>
            <p style="margin: 0; color: #94a3b8; font-size: 14px;">Enter a comment on the left or select an example below to view sentiment breakdown.</p>
        </div>
        """

    pos_prob = float(confs.get("Positive", 0.0))
    neg_prob = float(confs.get("Negative", 0.0))
    neu_prob = float(confs.get("Neutral", 0.0))

    pos_pct = round(pos_prob * 100, 1)
    neg_pct = round(neg_prob * 100, 1)
    neu_pct = round(neu_prob * 100, 1)

    # Style profiles per sentiment
    profiles = {
        "positive": {
            "title": "Positive Sentiment",
            "title_ar": "إيجابي",
            "emoji": "🟢",
            "bg": "linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.15) 100%)",
            "border": "#10b981",
            "badge_bg": "#059669",
            "text_color": "#065f46",
            "icon_bg": "#d1fae5",
            "tag": "Encouraging / Praise / Satisfaction"
        },
        "negative": {
            "title": "Negative Sentiment",
            "title_ar": "سلبي",
            "emoji": "🔴",
            "bg": "linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.15) 100%)",
            "border": "#ef4444",
            "badge_bg": "#dc2626",
            "text_color": "#991b1b",
            "icon_bg": "#fee2e2",
            "tag": "Critique / Dissatisfaction / Complaint"
        },
        "neutral": {
            "title": "Neutral Sentiment",
            "title_ar": "محايد",
            "emoji": "⚪",
            "bg": "linear-gradient(135deg, rgba(100, 116, 139, 0.08) 0%, rgba(71, 85, 105, 0.15) 100%)",
            "border": "#64748b",
            "badge_bg": "#475569",
            "text_color": "#1e293b",
            "icon_bg": "#f1f5f9",
            "tag": "Inquiry / Factual / Question"
        }
    }

    v_key = verdict.lower()
    p = profiles.get(v_key, profiles["neutral"])
    top_score = max(pos_pct, neg_pct, neu_pct)

    script_badges = {
        "arabic": ("📝 Arabic Script", "#0284c7", "#e0f2fe", "الدارجة بالحرف العربي"),
        "arabizi": ("🔤 Arabizi (Digits 2, 3, 5, 7, 9)", "#8b5cf6", "#ede9fe", "أرابيزي بالأرقام اللاتينية"),
        "mixed": ("🇫🇷 Code-Switching (French + Darija)", "#ea580c", "#ffedd5", "مزج بين الفرنسية والدارجة")
    }
    s_label, s_color, s_bg, s_ar = script_badges.get(script, (script.capitalize(), "#64748b", "#f1f5f9", ""))

    safe_original = html.escape(original_text)
    safe_cleaned = html.escape(cleaned_text)

    html_out = f"""
    <div style="border: 2px solid {p['border']}; border-radius: 20px; padding: 24px; background: {p['bg']}; box-shadow: 0 12px 30px -8px rgba(0,0,0,0.08); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        
        <!-- Header Banner -->
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="width: 52px; height: 52px; border-radius: 14px; background: {p['icon_bg']}; display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                    {p['emoji']}
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <h2 style="margin: 0; font-size: 24px; font-weight: 800; color: {p['text_color']};">{p['title']}</h2>
                        <span style="font-size: 18px; font-weight: 700; color: {p['text_color']}; font-family: 'Noto Naskh Arabic', Tahoma, sans-serif;">({p['title_ar']})</span>
                    </div>
                    <span style="font-size: 13px; color: #64748b; font-weight: 500;">{p['tag']}</span>
                </div>
            </div>
            <div style="background: {p['badge_bg']}; color: white; padding: 8px 18px; border-radius: 9999px; font-size: 15px; font-weight: 800; letter-spacing: 0.3px; box-shadow: 0 4px 10px rgba(0,0,0,0.12);">
                {top_score}% Confidence
            </div>
        </div>

        <!-- Confidence Meter Bars -->
        <div style="margin-bottom: 22px;">
            <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #475569; margin-bottom: 12px;">
                📊 Confidence Breakdown across Classes
            </div>
            
            <!-- Positive Bar -->
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #065f46;">
                    <span>🟢 Positive (إيجابي)</span>
                    <span style="font-weight: 700;">{pos_pct}%</span>
                </div>
                <div style="height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;">
                    <div style="width: {pos_pct}%; height: 100%; background: linear-gradient(90deg, #10b981 0%, #059669 100%); border-radius: 6px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                </div>
            </div>

            <!-- Negative Bar -->
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #991b1b;">
                    <span>🔴 Negative (سلبي)</span>
                    <span style="font-weight: 700;">{neg_pct}%</span>
                </div>
                <div style="height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;">
                    <div style="width: {neg_pct}%; height: 100%; background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); border-radius: 6px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                </div>
            </div>

            <!-- Neutral Bar -->
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #334155;">
                    <span>⚪ Neutral (محايد)</span>
                    <span style="font-weight: 700;">{neu_pct}%</span>
                </div>
                <div style="height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden;">
                    <div style="width: {neu_pct}%; height: 100%; background: linear-gradient(90deg, #94a3b8 0%, #64748b 100%); border-radius: 6px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);"></div>
                </div>
            </div>
        </div>

        <!-- Linguistic & Script Pills -->
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; padding-top: 14px; border-top: 1px dashed rgba(0,0,0,0.1);">
            <div style="background: {s_bg}; color: {s_color}; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 700; display: flex; align-items: center; gap: 6px;">
                <span>{s_label}</span>
                <span style="opacity: 0.8;">• {s_ar}</span>
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; color: #475569; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;">
                🧹 Normalized: <span style="font-style: italic; color: #0f172a;">"{safe_cleaned}"</span>
            </div>
        </div>

    </div>
    """
    return html_out


def predict_sentiment(text: str):
    """Predict sentiment, probabilities, and script for Darija text."""
    if not text or not text.strip():
        return generate_result_html("", {}, "None", "", "")
    
    # 1. Script Detection
    script = detect_script(text)
    
    # 2. Text Normalization
    cleaned = clean_text(text)
    
    if model is None:
        return "<div style='color: red;'>Model not loaded.</div>"
    
    # 3. Model Inference
    probs = model.predict_proba([cleaned])[0]
    classes = model.classes_
    
    confidences = {classes[i].capitalize(): float(probs[i]) for i in range(len(classes))}
    best_label = classes[np.argmax(probs)]
    
    return generate_result_html(best_label, confidences, script, cleaned, text)


# Categorized Moroccan Darija Examples
EXAMPLES_CATEGORIES = {
    "🌟 Blessings & Gratitude (الأدعية والتشجيع)": [
        "lah i3tik seha khouya 3la had l'video zwin bzaf ❤️🤲",
        "lah i7fdek wynawrak akhi lkarim",
        "choukran bzaf 3la had lkhedma lwa3ra tbarkllah 3lik",
        "الله يجازيك بالخير خويا ويرحم ليك الوالدين على هاد الشرح",
        "تبارك الله عليك محتوى نقي وهادف الله يوفقك 🔥",
    ],
    "🔥 Moroccan Slang & Praises (المدح والدارجة الشبابية)": [
        "hadchi harban bzaf wallah 🔥👏",
        "wa3er hadchi rak ghadi f lmostawa top",
        "khedma nadia bzaf khouya rak batal",
        "3alamia had lkhedma tbarkllah 3lik",
        "هادشي ناضي بزاف واعر طوب ديما منور أسطورة",
    ],
    "⚠️ Critiques & Complaints (الانتقادات والشكايات)": [
        "hadchi 3yan bzaf w flouss da3o ghir hakdak 🤮",
        "khedma na9sa bzaf w mrewbla saraha",
        "هادشي ما عجبنيش الصراحة، خدمة ناقصة وضياع الوقت 🤦‍♂️",
        "ghir kdoub w nfa9 khawi la 3ala9a had l'produit",
        "صفر على عشرة ماكاين والو غير تفرعين وتبهديل",
    ],
    "🇫🇷 Code-Switching (المزج بين الفرنسية والدارجة)": [
        "Franchement c'est magnifique, merci beaucoup khouya 👏",
        "C'est du grand n'importe quoi hadchi saraha zéro",
        "Wallah c'est top niveau, continue comme ça mon frère 💪",
        "Produit très décevant floussi mchaw f lkhewa",
    ],
    "💬 Neutral Questions & Inquiries (الأسئلة والاستفسارات)": [
        "bchhal taman dyal had l'produit w fin kayn magasin?",
        "waqtash kat7ello w wach kayn livraison l rabat?",
        "السلام عفاك واش هاد المنتوج باقي متوفر وبشحال الثمن؟",
        "salam kidayr labas wach momkin ra9m l'hatif dyalkom?",
    ]
}

# Flat list for Gradio Examples component
FLAT_EXAMPLES = [
    ["lah i3tik seha khouya 3la had l'video zwin bzaf ❤️🤲"],
    ["hadchi harban bzaf wallah 🔥👏"],
    ["الله يجازيك بالخير خويا ويرحم ليك الوالدين على هاد الشرح"],
    ["hadchi 3yan bzaf w flouss da3o ghir hakdak 🤮"],
    ["Franchement c'est magnifique, merci beaucoup khouya 👏"],
    ["bchhal taman dyal had l'produit w fin kayn magasin?"]
]

# Custom CSS for Modern Moroccan Aesthetic
CUSTOM_CSS = """
/* Google Font & Smooth Rendering */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Naskh+Arabic:wght@500;700&display=swap');

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
    background-color: #f8fafc !important;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #064e3b 100%);
    border-radius: 24px;
    padding: 32px 28px;
    color: white;
    box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.25);
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-pill {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Action Button */
#submit-btn {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border-radius: 14px !important;
    padding: 12px 24px !important;
    box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.2s ease !important;
}

#submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 25px -5px rgba(16, 185, 129, 0.5) !important;
}

#clear-btn {
    border-radius: 14px !important;
    font-weight: 600 !important;
    border: 1px solid #cbd5e1 !important;
    background: white !important;
    color: #475569 !important;
}

/* Quick Pills */
.pill-button {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 9999px !important;
    font-size: 13px !important;
    color: #334155 !important;
    padding: 6px 14px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}

.pill-button:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px) !important;
}

/* Input area */
textarea {
    border-radius: 16px !important;
    border: 1.5px solid #cbd5e1 !important;
    font-size: 15px !important;
    transition: border-color 0.2s ease !important;
}

textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
}
"""

with gr.Blocks(title="🇲🇦 Darija Sentiment AI") as demo:
    
    # Hero Banner
    gr.HTML(
        """
        <div class="hero-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 32px;">🇲🇦</span>
                        <h1 style="margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">Moroccan Darija AI</h1>
                    </div>
                    <p style="margin: 0 0 14px 0; color: #cbd5e1; font-size: 15px; max-width: 620px; line-height: 1.5;">
                        Production-grade sentiment analysis engineered specifically for Moroccan social media.
                        Mastering <strong>Arabic Script</strong>, <strong>Arabizi</strong> (<span style="color: #6ee7b7;">2, 3, 5, 7, 9</span>), and <strong>French Code-Switching</strong>.
                    </p>
                    <div style="font-family: 'Noto Naskh Arabic', Tahoma; font-size: 15px; color: #94a3b8; direction: rtl;">
                        الذكاء الاصطناعي لتحليل المشاعر في الدارجة المغربية بكل حروفها
                    </div>
                </div>
                
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div class="hero-pill">📊 <strong>51,054</strong> Cleaned Comments</div>
                    <div class="hero-pill">⚡ <strong>94.1%</strong> Arabizi Accuracy</div>
                    <div class="hero-pill">🛡️ Fixed Blessing Bias</div>
                </div>
            </div>
        </div>
        """
    )
    
    # Main Interactive Row
    with gr.Row(equal_height=True):
        
        # Left: Input Form
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Enter Moroccan Darija Comment")
            
            input_text = gr.Textbox(
                label="Comment Text / نص التعليق",
                placeholder="Ex: Tbarkllah 3lik khouya / تبارك الله عليك / Franchement c'est magnifique...",
                lines=4,
                show_label=False
            )
            
            # Quick Chips / Fast Test Prompts
            gr.Markdown("<div style='font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase;'>⚡ Quick Try Examples:</div>")
            with gr.Row():
                chip1 = gr.Button("🤲 lah i3tik seha", elem_classes=["pill-button"])
                chip2 = gr.Button("🔥 hadchi harban", elem_classes=["pill-button"])
                chip3 = gr.Button("❤️ تبارك الله عليك", elem_classes=["pill-button"])
                chip4 = gr.Button("⚠️ hadchi 3yan", elem_classes=["pill-button"])
                chip5 = gr.Button("❓ bchhal taman?", elem_classes=["pill-button"])

            with gr.Row(elem_id="action-row"):
                clear_btn = gr.Button("🧹 Clear", elem_id="clear-btn", scale=1)
                submit_btn = gr.Button("⚡ Analyze Sentiment / تحليل المشاعر", elem_id="submit-btn", scale=3)
        
        # Right: Rich Visual Output
        with gr.Column(scale=6):
            gr.Markdown("### 🎯 Sentiment Verdict & Analytics")
            output_card = gr.HTML(value=generate_result_html("", {}, "None", "", ""))

    # Wire up Quick Chips
    chip1.click(lambda: "lah i3tik seha khouya 3la had l'video zwin bzaf ❤️🤲", outputs=[input_text]).then(
        predict_sentiment, inputs=[input_text], outputs=[output_card]
    )
    chip2.click(lambda: "hadchi harban bzaf wallah 🔥👏", outputs=[input_text]).then(
        predict_sentiment, inputs=[input_text], outputs=[output_card]
    )
    chip3.click(lambda: "تبارك الله عليك خويا محتوى واعر ونقي بزاف 🔥❤️", outputs=[input_text]).then(
        predict_sentiment, inputs=[input_text], outputs=[output_card]
    )
    chip4.click(lambda: "hadchi 3yan bzaf w flouss da3o ghir hakdak 🤮", outputs=[input_text]).then(
        predict_sentiment, inputs=[input_text], outputs=[output_card]
    )
    chip5.click(lambda: "bchhal taman dyal had l'produit w fin kayn magasin?", outputs=[input_text]).then(
        predict_sentiment, inputs=[input_text], outputs=[output_card]
    )

    # Wire up Submit & Clear
    submit_btn.click(
        fn=predict_sentiment,
        inputs=[input_text],
        outputs=[output_card]
    )
    
    input_text.submit(
        fn=predict_sentiment,
        inputs=[input_text],
        outputs=[output_card]
    )
    
    clear_btn.click(
        lambda: ("", generate_result_html("", {}, "None", "", "")),
        outputs=[input_text, output_card]
    )

    # Arabizi Decoder Accordion
    with gr.Accordion("💡 Moroccan Arabizi Phoneme Decoder (أرقام الأرابيزي المقابلة للحروف)", open=False):
        gr.HTML(
            """
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 10px;">
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                    <div style="font-size: 22px; font-weight: 800; color: #8b5cf6; margin-bottom: 4px;">3 ➔ ع (Ain)</div>
                    <div style="font-size: 13px; color: #475569;"><strong>Examples:</strong> <em>3afak</em> (please), <em>wa3er</em> (awesome), <em>3lach</em> (why)</div>
                </div>
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                    <div style="font-size: 22px; font-weight: 800; color: #059669; margin-bottom: 4px;">7 ➔ ح (Haa)</div>
                    <div style="font-size: 13px; color: #475569;"><strong>Examples:</strong> <em>7chouma</em> (shame), <em>se7a</em> (health), <em>7bibi</em> (my friend)</div>
                </div>
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                    <div style="font-size: 22px; font-weight: 800; color: #dc2626; margin-bottom: 4px;">9 ➔ ق (Qaf)</div>
                    <div style="font-size: 13px; color: #475569;"><strong>Examples:</strong> <em>9rib</em> (near), <em>9ahwa</em> (coffee), <em>9bel</em> (before)</div>
                </div>
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                    <div style="font-size: 22px; font-weight: 800; color: #d97706; margin-bottom: 4px;">5 ➔ خ (Khaa)</div>
                    <div style="font-size: 13px; color: #475569;"><strong>Examples:</strong> <em>5oya</em> (brother), <em>5ir</em> (good), <em>5edma</em> (work)</div>
                </div>
                <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                    <div style="font-size: 22px; font-weight: 800; color: #0284c7; margin-bottom: 4px;">2 ➔ ء (Hamza)</div>
                    <div style="font-size: 13px; color: #475569;"><strong>Examples:</strong> <em>ra2i</em> (opinion), <em>so2al</em> (question)</div>
                </div>
            </div>
            """
        )

    # Categorized Examples Explorer
    with gr.Accordion("📂 Explore Moroccan Darija Category Presets", open=False):
        for category_name, examples_list in EXAMPLES_CATEGORIES.items():
            gr.Markdown(f"**{category_name}**")
            with gr.Row():
                for ex in examples_list[:3]:
                    btn = gr.Button(ex, elem_classes=["pill-button"], size="sm")
                    btn.click(lambda t=ex: t, outputs=[input_text]).then(
                        predict_sentiment, inputs=[input_text], outputs=[output_card]
                    )

    # Standard Gradio Examples for one-click test
    gr.Examples(
        examples=FLAT_EXAMPLES,
        inputs=[input_text],
        fn=predict_sentiment,
        outputs=[output_card],
        cache_examples=False,
        label="Click Any Benchmark Example to Test Instantly"
    )

    # Footer
    gr.HTML(
        """
        <div style="margin-top: 30px; padding: 20px 0; border-top: 1px solid #e2e8f0; text-align: center; color: #94a3b8; font-size: 13px;">
            🇲🇦 Moroccan Darija Sentiment Analysis Pipeline • Trained on 51,054 Comments (MYC, MAC, TSAC, MSDA, MSAC, Curated Lexicon)
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=CUSTOM_CSS)

"""
---
title: Moroccan Darija Mixed-Script Sentiment Classifier
emoji: 🇲🇦
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app/app.py
pinned: false
license: mit
---

Gradio demo for Moroccan Darija Sentiment Analysis.
Handles mixed scripts: Arabic script, Arabizi (Latin + numbers 2, 3, 5, 7, 9), and French code-switching.
"""

import os
import sys
import pickle
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
    # Fallback if running directly inside app/
    sys.path.insert(0, str(APP_DIR))
    from script_detect import detect_script
    from clean import clean_text

# Load Model
MODEL_PATH = RESULTS_DIR / "logistic_regression.pkl"
if not MODEL_PATH.exists():
    # Look for fallback in APP_DIR or models/
    MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"
    if not MODEL_PATH.exists():
        MODEL_PATH = APP_DIR / "logistic_regression.pkl"

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = f.read()
        model = pickle.loads(model)
else:
    print(f"Warning: Model not found at {MODEL_PATH}")


def predict_sentiment(text: str):
    """
    Predict sentiment, confidence breakdown, and detected script for Darija text.
    """
    if not text or not text.strip():
        return "Please enter a comment.", {}, "None"
    
    # 1. Script Detection
    script = detect_script(text)
    
    # 2. Text Normalization
    cleaned = clean_text(text)
    
    if model is None:
        return "Model not loaded.", {}, script
    
    # 3. Model Inference
    # Model pipeline takes raw cleaned text array
    probs = model.predict_proba([cleaned])[0]
    classes = model.classes_
    
    # Create confidence dictionary formatted for Gradio Label
    confidences = {classes[i].capitalize(): float(probs[i]) for i in range(len(classes))}
    best_label = classes[np.argmax(probs)]
    
    # Script presentation
    script_display = {
        "arabic": "Arabic Script (العربية)",
        "arabizi": "Arabizi / Latin Script (3, 7, 9)",
        "mixed": "Mixed Script / Code-Switching (Français + Darija)"
    }.get(script, script.capitalize())
    
    # Friendly status badge
    sentiment_emoji = {"positive": "🟢 Positive (إيجابي)", "negative": "🔴 Negative (سلبي)", "neutral": "⚪ Neutral (محايد)"}
    result_text = sentiment_emoji.get(best_label, best_label.capitalize())
    
    return result_text, confidences, script_display


# 6 curated Moroccan Darija examples covering different scripts and sentiments
EXAMPLES = [
    ["تبارك الله عليك خويا محتوى واعر ونقي بزاف 🔥❤️"],  # Positive Arabic
    ["Had l'video zwin bzaf, tbarkllah 3likoum khouya top!"],  # Positive Arabizi
    ["هادشي ما عجبنيش الصراحة، خدمة ناقصة وضياع الوقت"],  # Negative Arabic
    ["Makayn walo f had l'produit, flouss da3o ghir hakdak"],  # Negative Arabizi
    ["Franchement c'est magnifique, merci beaucoup khouya 👏"],  # Positive Mixed/Code-Switching
    ["Chhal taman dyal had l'produit w fin kayn magasin?"]   # Neutral question Arabizi
]

# Gradio Interface
with gr.Blocks(title="🇲🇦 Darija Sentiment Classifier") as demo:
    gr.Markdown(
        """
        # 🇲🇦 Moroccan Darija Mixed-Script Sentiment Classifier
        Accurately predict sentiment (**Positive**, **Negative**, **Neutral**) in Moroccan social media comments.
        Specialized in handling **Arabic script**, **Arabizi** (*digits 3, 7, 9*), and **French code-switching**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Enter Moroccan Darija Comment / أدخل التعليق هنا",
                placeholder="Ex: Tbarkllah 3lik khouya / تبارك الله عليك / Franchement c'est top...",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment / تحليل الإحساس", variant="primary")
            
        with gr.Column(scale=2):
            output_verdict = gr.Textbox(label="Verdict / النتيجة", interactive=False)
            output_script = gr.Textbox(label="Detected Script / نوع الخط", interactive=False)
            output_probs = gr.Label(label="Confidence Breakdown / نسب الثقة", num_top_classes=3)
    
    submit_btn.click(
        fn=predict_sentiment,
        inputs=[input_text],
        outputs=[output_verdict, output_probs, output_script]
    )
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text],
        fn=predict_sentiment,
        outputs=[output_verdict, output_probs, output_script],
        cache_examples=False
    )
    
    gr.Markdown(
        """
        ---
        **Project Details:**
        - **Data Sources:** MYC (Moroccan YouTube Corpus), MAC (Moroccan Arabic Corpus), MSDA (UM6P).
        - **Architecture:** TF-IDF (character n-grams 2–5 + word n-grams 1–2) + Logistic Regression / Transformer Fine-Tuning.
        - **Evaluation:** Evaluated overall and per script (Arabic, Arabizi, Mixed).
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

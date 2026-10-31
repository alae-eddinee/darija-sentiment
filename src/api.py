"""
FastAPI backend service for Darija Sentiment Analysis.
Provides fast inference endpoints for the Next.js frontend.
"""

import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Path setup
import sys
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
MODEL_PATH = PROJECT_ROOT / "results" / "logistic_regression.pkl"

from script_detect import detect_script
from clean import clean_text

app = FastAPI(title="Darija Sentiment API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Loaded sentiment model from {MODEL_PATH}")
else:
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}")


class SentimentRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: SentimentRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Sentiment model is not loaded.")
    
    script = detect_script(text)
    cleaned = clean_text(text)
    
    # Model inference
    probs = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    
    scores = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
    top_idx = int(np.argmax(probs))
    verdict = classes[top_idx]
    confidence = round(float(probs[top_idx]), 4)
    
    script_labels = {
        "arabic": "Arabic Script",
        "arabizi": "Arabizi (3, 7, 9)",
        "mixed": "French Code-Switching"
    }
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "scores": {
            "positive": scores.get("positive", 0.0),
            "negative": scores.get("negative", 0.0),
            "neutral": scores.get("neutral", 0.0)
        },
        "script": script,
        "script_label": script_labels.get(script, script.capitalize()),
        "cleaned_text": cleaned
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


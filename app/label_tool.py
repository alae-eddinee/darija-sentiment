"""
Streamlit labeling tool for Moroccan Darija sentiment dataset.

Features:
  - Shows one comment at a time with large readable text
  - Displays detected script type (Arabic, Arabizi, Mixed)
  - 3 sentiment buttons: Positive (Green), Negative (Red), Neutral (Gray) + Skip
  - Saves annotations to data/labeled_test.csv
  - Automatically resumes where you left off
  - Shows progress bar and statistics (total annotated, per-class breakdown)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TO_LABEL_FILE = DATA_DIR / "to_label.csv"
LABELED_FILE = DATA_DIR / "labeled_test.csv"

# Add src for script detection if needed
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from script_detect import detect_script
except ImportError:
    detect_script = lambda x: "unknown"


st.set_page_config(
    page_title="Darija Sentiment Labeler",
    page_icon="🇲🇦",
    layout="centered",
)

st.title("🇲🇦 Darija Sentiment Labeling Tool")
st.markdown("Annotate Moroccan Darija comments for sentiment (**Positive**, **Negative**, **Neutral**).")

# Check if to_label exists
if not TO_LABEL_FILE.exists():
    st.error(f"Input file `{TO_LABEL_FILE}` not found. Please generate `data/to_label.csv` first.")
    st.stop()

# Load data to label
df_to_label = pd.read_csv(TO_LABEL_FILE)
if "id" not in df_to_label.columns:
    df_to_label["id"] = range(len(df_to_label))

# Load or initialize labeled data
if LABELED_FILE.exists():
    df_labeled = pd.read_csv(LABELED_FILE)
else:
    df_labeled = pd.DataFrame(columns=["id", "text", "script", "label", "source"])

labeled_ids = set(df_labeled["id"].tolist())

# Filter remaining
remaining = df_to_label[~df_to_label["id"].isin(labeled_ids)].reset_index(drop=True)
total_items = len(df_to_label)
completed_items = len(labeled_ids)

# Progress metrics
progress_pct = completed_items / total_items if total_items > 0 else 1.0
st.progress(progress_pct, text=f"Progress: {completed_items} / {total_items} ({progress_pct*100:.1f}%)")

# Sidebar stats
st.sidebar.header("📊 Annotation Progress")
st.sidebar.metric("Completed", completed_items)
st.sidebar.metric("Remaining", len(remaining))

if len(df_labeled) > 0:
    st.sidebar.markdown("### Class Counts")
    counts = df_labeled["label"].value_counts()
    for lbl in ["positive", "negative", "neutral", "skipped"]:
        st.sidebar.write(f"- **{lbl.capitalize()}**: {counts.get(lbl, 0)}")

    if "script" in df_labeled.columns:
        st.sidebar.markdown("### Script Counts")
        s_counts = df_labeled["script"].value_counts()
        for sc in ["arabic", "arabizi", "mixed"]:
            st.sidebar.write(f"- **{sc.capitalize()}**: {s_counts.get(sc, 0)}")

# Check completion
if len(remaining) == 0:
    st.success("🎉 All comments have been annotated! You can now run evaluation on `data/labeled_test.csv`.")
    st.balloons()
    st.stop()

# Current item
current_row = remaining.iloc[0]
current_id = current_row["id"]
current_text = current_row["text"]
current_script = current_row.get("script", detect_script(str(current_text)))
current_source = current_row.get("source", "scraped")

# Display card
st.markdown("---")
col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.caption(f"**Item ID:** #{current_id}")
with col_meta2:
    badge_color = "#2b5c8f" if current_script == "arabic" else "#e26d5c" if current_script == "arabizi" else "#38b000"
    st.markdown(f"**Detected Script:** <span style='background-color:{badge_color};color:white;padding:3px 8px;border-radius:4px;'>{current_script.upper()}</span>", unsafe_allow_html=True)

# Comment text display with large font and RTL support if Arabic
is_arabic = (current_script == "arabic")
direction_style = "direction: rtl; text-align: right;" if is_arabic else "direction: ltr; text-align: left;"

st.markdown(
    f"""
    <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 24px; margin: 16px 0; font-size: 20px; line-height: 1.6; {direction_style}">
        {current_text}
    </div>
    """,
    unsafe_allow_html=True
)

# Helper function to save label
def save_annotation(label):
    new_entry = pd.DataFrame([{
        "id": current_id,
        "text": current_text,
        "script": current_script,
        "label": label,
        "source": current_source
    }])
    if LABELED_FILE.exists():
        new_entry.to_csv(LABELED_FILE, mode="a", header=False, index=False, encoding="utf-8")
    else:
        new_entry.to_csv(LABELED_FILE, index=False, encoding="utf-8")
    st.rerun()

# Annotation buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("👍 Positive", use_container_width=True, type="primary"):
        save_annotation("positive")

with col2:
    if st.button("👎 Negative", use_container_width=True):
        save_annotation("negative")

with col3:
    if st.button("😐 Neutral", use_container_width=True):
        save_annotation("neutral")

with col4:
    if st.button("⏭️ Skip", use_container_width=True):
        save_annotation("skipped")

st.markdown("---")
with st.expander("📖 Quick Labeling Guidelines"):
    st.markdown("""
    - **Positive**: Approval, praise, gratitude, love, enthusiasm (*Tbarkllah*, *Zwin bezzaf*, *Top*).
    - **Negative**: Disapproval, anger, criticism, hate, complaint (*Khayeb*, *Hchouma*, *Makayn walo*).
    - **Neutral**: Factual question, location, off-topic, balanced mixed opinion (*Chhal taman?*, *Ana mn casa*).
    - **Sarcasm**: Label the **intended** sentiment (e.g., *'Waaaw chhal dki 🙄'* → Negative).
    - **Mixed**: Code-switching (French + Darija) should be labeled by overall emotion.
    """)

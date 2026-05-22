"""
app.py — Streamlit Web App for CNN Sentiment Analysis
======================================================
Run:  streamlit run app.py
"""

import re
import pickle
import numpy as np
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis using CNN",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    text-align: center;
    padding: 2rem 0 1.2rem;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p { color: #64748b; font-size: 1rem; margin: 0; }

/* ── Example buttons ── */
.ex-btn-wrap {
    display: flex; flex-wrap: wrap; gap: 0.7rem;
    justify-content: center; margin: 0.8rem 0 1.2rem;
}
.ex-btn {
    display: inline-block; padding: 0.55rem 1.1rem;
    border-radius: 12px; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; border: none; color: #fff;
    box-shadow: 0 6px 0 rgba(0,0,0,0.25), 0 8px 16px rgba(0,0,0,0.2);
    transition: transform 0.12s, box-shadow 0.12s;
    letter-spacing: 0.02em;
}
.ex-btn:hover {
    transform: translateY(3px);
    box-shadow: 0 3px 0 rgba(0,0,0,0.25), 0 4px 8px rgba(0,0,0,0.2);
}
.ex-btn-1 { background: linear-gradient(145deg, #6366f1, #4f46e5); box-shadow: 0 6px 0 #312e81, 0 8px 16px rgba(99,102,241,0.35); }
.ex-btn-2 { background: linear-gradient(145deg, #f43f5e, #e11d48); box-shadow: 0 6px 0 #9f1239, 0 8px 16px rgba(244,63,94,0.35); }
.ex-btn-3 { background: linear-gradient(145deg, #10b981, #059669); box-shadow: 0 6px 0 #064e3b, 0 8px 16px rgba(16,185,129,0.35); }
.ex-btn-4 { background: linear-gradient(145deg, #f59e0b, #d97706); box-shadow: 0 6px 0 #78350f, 0 8px 16px rgba(245,158,11,0.35); }
.ex-btn-5 { background: linear-gradient(145deg, #8b5cf6, #7c3aed); box-shadow: 0 6px 0 #4c1d95, 0 8px 16px rgba(139,92,246,0.35); }

/* ── Result card ── */
.result-card {
    border-radius: 16px; padding: 2rem; text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}
.result-positive {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #34d399;
}
.result-negative {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #f87171;
}
.result-label { font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; }
.positive-label { color: #065f46; }
.negative-label { color: #991b1b; }
.conf-text { font-size: 1rem; color: #475569; font-weight: 500; }

/* ── Score buttons ── */
.score-btn-wrap {
    display: flex; gap: 1.2rem; justify-content: center; margin-top: 1.4rem;
}
.score-btn {
    display: flex; flex-direction: column; align-items: center;
    padding: 1rem 2.2rem; border-radius: 14px; min-width: 150px;
    cursor: default; transition: transform 0.15s;
}
.score-btn-neg {
    background: linear-gradient(145deg, #ff6b6b, #c0392b);
    box-shadow: 0 8px 0 #7b1a1a, 0 10px 20px rgba(192,57,43,0.4);
}
.score-btn-pos {
    background: linear-gradient(145deg, #2ecc71, #1a8a4a);
    box-shadow: 0 8px 0 #0e5c30, 0 10px 20px rgba(30,180,80,0.4);
}
.score-btn:hover { transform: translateY(3px); }
.score-btn .score-title {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: rgba(255,255,255,0.85); margin-bottom: 0.3rem;
}
.score-btn .score-value { font-size: 2rem; font-weight: 800; color: #fff; }

/* ── About section ── */
.about-wrap {
    margin-top: 3rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}
.about-header {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    padding: 1rem 1.5rem;
    color: #fff; font-weight: 700; font-size: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.about-body { padding: 1.5rem; }
.layer-grid {
    display: grid; grid-template-columns: 1fr 2fr;
    gap: 0.5rem 1rem; margin: 1rem 0;
}
.layer-name {
    background: #f1f5f9; border-radius: 8px;
    padding: 0.4rem 0.8rem; font-size: 0.82rem;
    font-weight: 600; color: #4f46e5;
}
.layer-desc {
    padding: 0.4rem 0; font-size: 0.82rem; color: #475569;
    display: flex; align-items: center;
}
.stat-chips {
    display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1rem;
}
.chip {
    background: #f1f5f9; border-radius: 20px;
    padding: 0.35rem 0.9rem; font-size: 0.78rem;
    font-weight: 600; color: #334155;
    border: 1px solid #e2e8f0;
}
.chip-purple { background: #ede9fe; color: #5b21b6; border-color: #c4b5fd; }
.chip-green  { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
.chip-blue   { background: #dbeafe; color: #1e40af; border-color: #93c5fd; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
VOCAB_SIZE  = 10_000
MAX_LEN     = 200
MODEL_PATH  = "model/sentiment_cnn_final.keras"
WINDEX_PATH = "model/word_index.pkl"

# ── Load model (cached) ────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(WINDEX_PATH, "rb") as f:
        word_index = pickle.load(f)
    return model, word_index

# ── Preprocessing ──────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        stemmer.stem(w) for w in text.split()
        if w not in stop_words and len(w) > 1
    )

def predict_sentiment(text, model, word_index):
    cleaned = clean_text(text)
    offset  = 3
    seq     = [
        word_index.get(w, 2) + offset
        for w in cleaned.split()
        if word_index.get(w, 2) + offset < VOCAB_SIZE
    ]
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="post", truncating="post")
    prob   = float(model.predict(padded, verbose=0)[0][0])
    return prob

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 Sentiment Analysis using CNN</h1>
    <p>Binary Sentiment Classifier &nbsp;|&nbsp; CNN + Keras + NLP</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────
try:
    model, word_index = load_resources()
except Exception as e:
    st.error(f"❌ Model not found. Run `python train.py` first!\n\n{e}")
    st.stop()

# ── Examples ───────────────────────────────────────────────────
EXAMPLES = [
    ("🌟 Brilliant film", "This movie was absolutely brilliant! The acting was superb and the story was gripping.", "ex-btn-1"),
    ("💀 Terrible film", "Terrible film. Boring plot, horrible acting and a complete waste of time.", "ex-btn-2"),
    ("🏆 Best ever",     "One of the best movies I have ever seen. Highly recommend!", "ex-btn-3"),
    ("😴 Couldn't finish","I couldn't finish it. Dull, predictable and poorly directed.", "ex-btn-4"),
    ("😐 Not bad",       "Not bad, but could have been much better with a stronger script.", "ex-btn-5"),
]

st.markdown("""
<div style="text-align:center; margin: 1.5rem 0 1rem;">
    <div style="font-size:5rem; line-height:1; filter: drop-shadow(0 8px 16px rgba(0,0,0,0.35));">🎬</div>
    <div style="font-size:0.85rem; color:#94a3b8; margin-top:0.4rem; letter-spacing:0.05em; text-transform:uppercase; font-weight:600;">
        Lights · Camera · Sentiment
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📝 Enter a Movie Review")
st.markdown("**Try an example:**")

# Render example buttons as styled HTML links that trigger Streamlit buttons underneath
cols = st.columns(len(EXAMPLES))
for i, (col, (label, ex, _)) in enumerate(zip(cols, EXAMPLES)):
    if col.button(label, key=f"ex_{i}", use_container_width=True):
        st.session_state["review_text"] = ex
        st.session_state["result"] = None
        st.rerun()

# Style the Streamlit buttons to look like the colored 3D buttons
btn_colors = [
    ("#4f46e5", "#312e81"),
    ("#e11d48", "#9f1239"),
    ("#059669", "#064e3b"),
    ("#d97706", "#78350f"),
    ("#7c3aed", "#4c1d95"),
]
btn_style = ""
for i, (bg, shadow) in enumerate(btn_colors):
    btn_style += f"""
    div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button {{
        background: linear-gradient(145deg, {bg}cc, {bg}) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 5px 0 {shadow}, 0 7px 14px {bg}55 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: transform 0.12s, box-shadow 0.12s !important;
    }}
    div[data-testid="stHorizontalBlock"] > div:nth-child({i+1}) button:hover {{
        transform: translateY(3px) !important;
        box-shadow: 0 2px 0 {shadow}, 0 4px 8px {bg}55 !important;
    }}
    """
st.markdown(f"<style>{btn_style}</style>", unsafe_allow_html=True)

# ── Text area ──────────────────────────────────────────────────
review = st.text_area(
    label="Your review",
    value=st.session_state.get("review_text", ""),
    placeholder="Type a movie review here…",
    height=130,
    label_visibility="collapsed",
)

analyse_btn = st.button("🔍 Analyse Sentiment", use_container_width=True, type="primary")

if analyse_btn:
    text = review.strip()
    if text:
        with st.spinner("Analysing…"):
            prob = predict_sentiment(text, model, word_index)
        st.session_state["result"] = {"prob": prob, "text": text}
        st.session_state["review_text"] = text
    else:
        st.warning("Please enter a review first.")

# ── Prediction result ──────────────────────────────────────────
result = st.session_state.get("result")
if result:
    prob        = result["prob"]
    is_positive = prob >= 0.5
    label       = "POSITIVE 😊" if is_positive else "NEGATIVE 😞"
    conf        = prob if is_positive else 1 - prob
    card_class  = "result-positive" if is_positive else "result-negative"
    label_class = "positive-label" if is_positive else "negative-label"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="conf-text">Predicted Sentiment</div>
        <div class="result-label {label_class}">{label}</div>
        <div class="conf-text">Confidence: <strong>{conf*100:.1f}%</strong>
            &nbsp;|&nbsp; Raw score: {prob:.4f}</div>
    </div>
    <div class="score-btn-wrap">
        <div class="score-btn score-btn-neg">
            <span class="score-title">😞 Negative</span>
            <span class="score-value">{(1-prob)*100:.1f}%</span>
        </div>
        <div class="score-btn score-btn-pos">
            <span class="score-title">😊 Positive</span>
            <span class="score-value">{prob*100:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── About ──────────────────────────────────────────────────────
st.markdown("<div style='margin-top:3.5rem;'></div>", unsafe_allow_html=True)

with st.expander("ℹ️ About this Model — How it works"):
    st.markdown("""
<div class="about-body">

### 🧠 Model Architecture

This app uses a **1D Convolutional Neural Network (CNN)** trained on 50,000 IMDB movie reviews to classify text as **Positive** or **Negative**.

<div class="layer-grid">
  <div class="layer-name">🔤 Embedding</div>
  <div class="layer-desc">Converts each word into a 64-dim vector — the model learns word meaning from context</div>
  <div class="layer-name">🔍 Conv1D × 2</div>
  <div class="layer-desc">Scans the text with filters to detect sentiment patterns (like "not bad", "absolutely brilliant")</div>
  <div class="layer-name">📊 GlobalMaxPool</div>
  <div class="layer-desc">Picks the strongest signal from the entire review — collapses sequence to a fixed vector</div>
  <div class="layer-name">🧩 Dense + Dropout</div>
  <div class="layer-desc">64-unit fully connected layer with 40% dropout to prevent overfitting</div>
  <div class="layer-name">🎯 Output</div>
  <div class="layer-desc">Single sigmoid neuron → score between 0 (Negative) and 1 (Positive)</div>
</div>

### 📦 Dataset & Training

<div class="stat-chips">
  <span class="chip chip-purple">📚 IMDB — 50,000 reviews</span>
  <span class="chip chip-blue">🏋️ 25K train / 25K test</span>
  <span class="chip chip-green">✅ ~88–91% test accuracy</span>
  <span class="chip">⚙️ Adam optimizer</span>
  <span class="chip">🛑 Early Stopping</span>
  <span class="chip">📉 ReduceLROnPlateau</span>
</div>

### 🔧 Text Preprocessing Pipeline

1. **Lowercase** all text
2. **Strip HTML** tags (common in IMDB reviews)
3. **Remove punctuation** and special characters
4. **Remove stopwords** — but keep `not`, `no`, `never` (they flip sentiment!)
5. **Porter Stemming** — reduces words to their root form (e.g. *running → run*)
6. **Tokenize** using the IMDB vocabulary (top 10,000 words)

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:2rem;'>
    Built with ❤️ using Keras + Streamlit &nbsp;|&nbsp; Singamsetti Keerthi
</div>
""", unsafe_allow_html=True)

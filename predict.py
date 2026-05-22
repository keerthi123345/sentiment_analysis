"""
predict.py — Run sentiment prediction on new text
===================================================
Usage:
    python predict.py                          # interactive mode
    python predict.py "This movie was great!"  # single review via CLI
"""

import sys
import re
import pickle
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)

# ── Config (must match train.py) ───────────────────────────────
VOCAB_SIZE = 10_000
MAX_LEN    = 200
MODEL_PATH = "model/sentiment_cnn_final.keras"
WINDEX_PATH = "model/word_index.pkl"

# ── Preprocessing ──────────────────────────────────────────────
stemmer   = PorterStemmer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never", "but"}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        stemmer.stem(w)
        for w in text.split()
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(tokens)

def text_to_sequence(text: str, word_index: dict) -> list:
    """Convert cleaned text to integer sequence using IMDB word index."""
    # IMDB index is offset by 3 (0=pad, 1=start, 2=unknown, 3=unused)
    offset = 3
    seq = []
    for word in text.split():
        idx = word_index.get(word, 2) + offset   # 2 = <UNK>
        if idx < VOCAB_SIZE:
            seq.append(idx)
    return seq

def predict(text: str, model, word_index: dict) -> dict:
    cleaned  = clean_text(text)
    seq      = text_to_sequence(cleaned, word_index)
    padded   = pad_sequences([seq], maxlen=MAX_LEN, padding="post", truncating="post")
    prob     = float(model.predict(padded, verbose=0)[0][0])
    label    = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😞"
    conf     = prob if prob >= 0.5 else 1 - prob
    return {"label": label, "confidence": conf, "raw_score": prob}


# ── Main ───────────────────────────────────────────────────────
def main():
    print("\n🔮 Sentiment Analysis — CNN Model")
    print("=" * 45)

    # Load model and word index
    try:
        model      = load_model(MODEL_PATH)
        with open(WINDEX_PATH, "rb") as f:
            word_index = pickle.load(f)
        print("✅ Model loaded successfully.\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Run  python train.py  first!")
        return

    # CLI mode
    if len(sys.argv) > 1:
        text   = " ".join(sys.argv[1:])
        result = predict(text, model, word_index)
        print(f"Review    : {text}")
        print(f"Sentiment : {result['label']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        return

    # Interactive mode
    sample_reviews = [
        "This movie was absolutely brilliant! The acting was superb.",
        "Terrible film. Boring plot and horrible acting.",
        "Not bad, but could have been much better.",
        "One of the best movies I have ever seen!",
        "I wasted two hours of my life watching this garbage.",
    ]

    print("📋 Sample Reviews (auto-testing):")
    print("-" * 45)
    for review in sample_reviews:
        result = predict(review, model, word_index)
        print(f"  {result['label']}  ({result['confidence']*100:.1f}%)")
        print(f"  » {review[:70]}{'…' if len(review)>70 else ''}")
        print()

    print("\n" + "=" * 45)
    print("💬 Type your own review (or 'quit' to exit):")
    print("=" * 45)
    while True:
        review = input("\n> ").strip()
        if review.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break
        if not review:
            continue
        result = predict(review, model, word_index)
        print(f"  Sentiment  : {result['label']}")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Raw score  : {result['raw_score']:.4f}  (>0.5 = Positive)")


if __name__ == "__main__":
    main()

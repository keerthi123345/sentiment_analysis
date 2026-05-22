"""
Sentiment Analysis Using CNN
==============================
Binary sentiment classification (Positive / Negative) on movie reviews.
Uses a 1D Convolutional Neural Network built with Keras.

Tech Stack: Python, Keras, NLP Tools (NLTK)
Author: Singamsetti Keerthi
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# ── NLP ────────────────────────────────────────────────────────
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Keras ──────────────────────────────────────────────────────
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

# ── sklearn ────────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

# ── Config ─────────────────────────────────────────────────────
VOCAB_SIZE      = 10_000   # top-N words to keep
MAX_LEN         = 200      # pad/truncate every review to this length
EMBEDDING_DIM   = 64       # word vector size
NUM_FILTERS     = 128      # CNN filter count
KERNEL_SIZE     = 5        # CNN kernel (n-gram window)
DENSE_UNITS     = 64
DROPOUT_RATE    = 0.4
BATCH_SIZE      = 64
EPOCHS          = 15
LEARNING_RATE   = 1e-3

MODEL_DIR       = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════
def load_data():
    print("\n[1/5] Loading IMDB dataset …")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"      Train samples : {len(X_train):,}")
    print(f"      Test  samples : {len(X_test):,}")
    print(f"      Positive train: {y_train.sum():,}  |  "
          f"Negative train: {(1-y_train).sum():,}")

    X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding="post", truncating="post")

    # Save word index for the inference script
    word_index = imdb.get_word_index()
    with open(os.path.join(MODEL_DIR, "word_index.pkl"), "wb") as f:
        pickle.dump(word_index, f)

    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════
# 2.  MODEL
# ══════════════════════════════════════════════════════════════
def build_model():
    print("\n[2/5] Building CNN model …")
    model = Sequential([
        # ── Embedding layer ───────────────────────────────────
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN,
            name="embedding"
        ),

        # ── Conv block 1 ──────────────────────────────────────
        Conv1D(NUM_FILTERS, KERNEL_SIZE,
               activation="relu", padding="same", name="conv1"),
        BatchNormalization(name="bn1"),
        Dropout(DROPOUT_RATE, name="drop1"),

        # ── Conv block 2 (deeper features) ────────────────────
        Conv1D(NUM_FILTERS // 2, 3,
               activation="relu", padding="same", name="conv2"),
        BatchNormalization(name="bn2"),

        # ── Global pooling (collapses sequence → fixed vector) ─
        GlobalMaxPooling1D(name="global_pool"),

        # ── Dense head ────────────────────────────────────────
        Dense(DENSE_UNITS, activation="relu", name="dense1"),
        Dropout(DROPOUT_RATE, name="drop2"),
        Dense(1, activation="sigmoid", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ══════════════════════════════════════════════════════════════
# 3.  TRAIN
# ══════════════════════════════════════════════════════════════
def train_model(model, X_train, y_train):
    print("\n[3/5] Training …")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=2, verbose=1, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ══════════════════════════════════════════════════════════════
# 4.  EVALUATE
# ══════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test):
    print("\n[4/5] Evaluating on test set …")

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)

    print(f"\n  ✅ Test Accuracy : {acc*100:.2f}%")
    print(f"  ✅ ROC-AUC Score : {auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Negative", "Positive"]))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"            Pred Neg  Pred Pos")
    print(f"  True Neg  {cm[0,0]:>8d}  {cm[0,1]:>8d}")
    print(f"  True Pos  {cm[1,0]:>8d}  {cm[1,1]:>8d}")

    return y_pred_prob, y_pred


# ══════════════════════════════════════════════════════════════
# 5.  PLOTS
# ══════════════════════════════════════════════════════════════
def save_plots(history, y_test, y_pred_prob):
    print("\n[5/5] Saving plots …")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Sentiment Analysis — Training Results", fontsize=14, fontweight="bold")

    # ── Accuracy curve ──
    ax = axes[0]
    ax.plot(history.history["accuracy"],     label="Train", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Validation", linewidth=2, linestyle="--")
    ax.set_title("Model Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Loss curve ──
    ax = axes[1]
    ax.plot(history.history["loss"],     label="Train", linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    ax.set_title("Model Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    nltk.download("stopwords", quiet=True)

    X_train, y_train, X_test, y_test = load_data()
    model                            = build_model()
    history                          = train_model(model, X_train, y_train)
    y_pred_prob, y_pred              = evaluate_model(model, X_test, y_test)
    save_plots(history, y_test, y_pred_prob)

    # Save final model
    model.save(os.path.join(MODEL_DIR, "sentiment_cnn_final.keras"))
    print("\n  Model saved → model/sentiment_cnn_final.keras")
    print("\n✅ Done! Run  python predict.py  to classify new reviews.")

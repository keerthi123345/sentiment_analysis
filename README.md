# 🧠 Sentiment Analysis Using CNN

> Binary sentiment classification on IMDB movie reviews using a 1D Convolutional Neural Network.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-~90%25-brightgreen?style=flat-square)]()

---

## 📌 Overview

This project builds a **1D CNN model** to classify movie reviews as **Positive** or **Negative** using the IMDB dataset (50,000 reviews). The model learns text patterns through convolutional filters — similar to how CNNs detect image features, but applied to word sequences.

**Key skills demonstrated:** Text preprocessing · Tokenization · Sequence padding · CNN architecture design · Model tuning · Evaluation metrics · Streamlit deployment

---

## 🏗️ Model Architecture

```
Input (text sequence, padded to 200 tokens)
        ↓
Embedding Layer (10,000 vocab × 64 dims)
        ↓
Conv1D (128 filters, kernel=5) + BatchNorm + Dropout
        ↓
Conv1D (64 filters,  kernel=3) + BatchNorm
        ↓
GlobalMaxPooling1D
        ↓
Dense (64, ReLU) + Dropout
        ↓
Dense (1, Sigmoid) → 0 = Negative | 1 = Positive
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/keerthi123345/sentiment_analysis.git
cd sentiment_analysis

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

Downloads the IMDB dataset automatically, trains the CNN, and saves:
- `model/sentiment_cnn_final.keras`
- `model/word_index.pkl`
- `model/training_curves.png`

### 3. Predict from Command Line

```bash
# Single review
python predict.py "This movie was absolutely brilliant!"

# Interactive mode
python predict.py
```

### 4. Run the Web App

```bash
streamlit run app.py
```

### 5. Open the Notebook

```bash
jupyter notebook notebooks/Sentiment_Analysis_CNN.ipynb
```

---

## 📁 Project Structure

```
sentiment-analysis-cnn/
│
├── train.py                          # Train the CNN model
├── predict.py                        # CLI inference tool
├── app.py                            # Streamlit web app
├── requirements.txt
│
├── notebooks/
│   └── Sentiment_Analysis_CNN.ipynb  # Full step-by-step notebook
│
└── model/                            # Created after training
    ├── sentiment_cnn_final.keras
    ├── word_index.pkl
    └── training_curves.png
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~88–91% |
| ROC-AUC | ~0.95+ |
| Dataset | IMDB (25k train / 25k test) |
| Training time | ~3–5 min (CPU) |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Keras / TensorFlow** — CNN model
- **NLTK** — stopword removal, stemming
- **scikit-learn** — evaluation metrics
- **Streamlit** — web interface
- **Matplotlib / Seaborn** — visualisation

---
## 📄 Related Work 
<p><small>🔬 <b>Related Work:</b> <a href="https://doi.org/10.5281/zenodo.19596702">HOLOSIGN: Universal Gesture Communication System</a> — Published in JCES, April 2026</small></p>

---

## 👩‍💻 Author

**Singamsetti Keerthi** — B.Tech CSE | AI/ML Enthusiast  
📧 keerthisingamsetty093@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/singamsetti-keerthi) · [GitHub](https://github.com/keerthi123345)

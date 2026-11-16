"""
Prediction module for Fake News Detection.
Implements:
- FR-18 to FR-20: Text classification and confidence scoring.
- FR-21: Top 5 explainability keywords using Logistic Regression coefficients.
- FR-22: Input validation (minimum 20 words requirement).
- FR-25: Standard structured prediction output.
"""

import os
import sys
import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")

# Cache loaded models in memory (loaded once at startup)
_LOADED_MODEL = None
_LOADED_VECTORIZER = None
_LOADED_LR = None


def load_artifacts():
    """Loads the model and vectorizer from disk once."""
    global _LOADED_MODEL, _LOADED_VECTORIZER, _LOADED_LR
    if _LOADED_MODEL is None or _LOADED_VECTORIZER is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                "Trained model or vectorizer not found. Please run 'python src/train.py' first."
            )
        _LOADED_MODEL = joblib.load(MODEL_PATH)
        _LOADED_VECTORIZER = joblib.load(VECTORIZER_PATH)
        if os.path.exists(LR_MODEL_PATH):
            _LOADED_LR = joblib.load(LR_MODEL_PATH)
        else:
            _LOADED_LR = _LOADED_MODEL
    return _LOADED_MODEL, _LOADED_VECTORIZER, _LOADED_LR


def extract_top_words(vec, lr_model, vectorizer, predicted_class: int, top_n: int = 5) -> list:
    """
    FR-21: Explains prediction by finding the top words that pushed the prediction
    towards REAL or FAKE using Logistic Regression coefficients.
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
        # Logistic Regression binary coef shape: (1, n_features)
        coef = lr_model.coef_[0] if hasattr(lr_model, "coef_") else None
        if coef is None:
            return []

        # Get non-zero indices and tfidf values for this specific input
        row, col = vec.nonzero()
        if len(col) == 0:
            return []

        # Compute contribution of each present word: coef * tf-idf
        contributions = []
        for idx in col:
            word = feature_names[idx]
            tfidf_val = vec[0, idx]
            weight = coef[idx] * tfidf_val
            contributions.append((word, weight))

        # Class 1 = REAL (positive coefficients push towards REAL)
        # Class 0 = FAKE (negative coefficients push towards FAKE)
        if predicted_class == 1:
            contributions.sort(key=lambda x: x[1], reverse=True)
            top_words = [word for word, w in contributions if w > 0][:top_n]
        else:
            contributions.sort(key=lambda x: x[1])
            top_words = [word for word, w in contributions if w < 0][:top_n]

        # If not enough signed words, take strongest magnitude
        if not top_words:
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_words = [word for word, _ in contributions][:top_n]

        return top_words
    except Exception:
        return []


def predict_news(text: str) -> dict:
    """
    Predicts whether a given news article/headline is REAL or FAKE.

    Args:
        text (str): News article or headline text.

    Returns:
        dict: {
            "label": "REAL" | "FAKE",
            "confidence": float (percentage, e.g. 94.2),
            "top_words": list[str] (top 5 explanatory keywords)
        }

    Raises:
        ValueError: If input is empty or has fewer than 20 words (FR-22).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a news article. The input text cannot be empty.")

    # FR-22: Word count validation
    words = text.strip().split()
    if len(words) < 20:
        raise ValueError(
            f"Input text has only {len(words)} word(s). Please enter at least 20 words for an accurate and reliable prediction."
        )

    model, vectorizer, lr_model = load_artifacts()

    # Preprocess text (FR-10: shared cleaning)
    cleaned = clean_text(text)
    if not cleaned.strip():
        raise ValueError("Input text contained no usable words after filtering symbols and stopwords.")

    # Vectorize
    vec = vectorizer.transform([cleaned])

    # Predict class
    pred = int(model.predict(vec)[0])
    label = "REAL" if pred == 1 else "FAKE"

    # Calculate confidence percentage
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = round(float(np.max(proba)) * 100, 2)
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        # Logistic sigmoid for distance score
        conf_raw = 1.0 / (1.0 + np.exp(-abs(score)))
        confidence = round(float(conf_raw) * 100, 2)
    else:
        confidence = 90.0

    # Extract top 5 explainability words (FR-21)
    top_words = extract_top_words(vec, lr_model, vectorizer, pred, top_n=5)

    return {
        "label": label,
        "confidence": confidence,
        "top_words": top_words
    }

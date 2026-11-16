"""
Pre-processing module for Fake News Detection.
Contains text cleaning, normalization, stopword removal, and lemmatization.
Used consistently in both training and inference pipelines.
"""

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Add local project nltk_data directory if present for offline execution
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_NLTK_DATA = os.path.join(_BASE_DIR, "nltk_data")
if os.path.exists(_LOCAL_NLTK_DATA) and _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

def ensure_nltk_resources():
    """Ensure required NLTK data files are available locally without failing if offline."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, download_dir=_LOCAL_NLTK_DATA, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

# Initialize stop words and lemmatizer once
_LEMMATIZER = WordNetLemmatizer()
try:
    _STOP_WORDS = set(stopwords.words('english'))
except Exception:
    _STOP_WORDS = set()

def clean_text(text: str) -> str:
    """
    Cleans raw text input:
    1. Lowercase text (FR-6)
    2. Strip URLs, HTML tags, punctuation, numbers, special characters (FR-7)
    3. Remove Reuters source tags and data leakage patterns (FR-5)
    4. Remove English stopwords (FR-8)
    5. Apply lemmatization (FR-9)

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase (FR-6)
    text = text.lower()

    # Remove Reuters source patterns (e.g., 'washington (reuters) -', 'reuters')
    text = re.sub(r'^[a-z\s,]+(?:\(reuters\))?\s*[-–—]\s*', ' ', text)
    text = re.sub(r'\breuters\b', ' ', text)

    # Remove URLs (FR-7)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove HTML tags (FR-7)
    text = re.sub(r'<.*?>', ' ', text)

    # Remove numbers, punctuation, and non-alphabetic characters (FR-7)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords and apply lemmatization (FR-8, FR-9)
    cleaned_tokens = [
        _LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in _STOP_WORDS and len(word) > 1
    ]

    return " ".join(cleaned_tokens)

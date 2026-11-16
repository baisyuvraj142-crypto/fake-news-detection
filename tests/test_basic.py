"""
Automated unit and integration test suite for Fake News Detection project.
Covers:
- TC-1: clean_text cleaning functionality
- TC-4: Empty text validation
- TC-5: Input with fewer than 20 words
- TC-6: Real news sample prediction
- TC-7: Fake news sample prediction
- TC-8: REST API /api/predict valid JSON
- TC-9: REST API /api/predict missing/invalid text
- TC-10: /performance page rendering
"""

import os
import sys
import pytest

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text
from src.predict import predict_news
from app import app


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# --- TC-1: Text Preprocessing ---
def test_tc1_clean_text():
    """TC-1: clean_text removes URLs, punctuation, and converts to lowercase."""
    sample_input = "Visit http://x.com NOW!!!"
    cleaned = clean_text(sample_input)
    assert "http" not in cleaned, "URL was not removed"
    assert "!" not in cleaned, "Punctuation was not removed"
    assert cleaned.islower() or cleaned == "", "Output is not lowercase"
    assert "visit" in cleaned, "'visit' should remain after stopword filtering"


# --- TC-4: Empty Text Handling ---
def test_tc4_empty_text():
    """TC-4: Empty text submission raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        predict_news("")
    assert "cannot be empty" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo_spaces:
        predict_news("     ")
    assert "cannot be empty" in str(excinfo_spaces.value)


# --- TC-5: Short Text Handling (< 20 words) ---
def test_tc5_fewer_than_20_words():
    """TC-5: Submission with fewer than 20 words raises ValueError."""
    short_text = "Breaking news: A major event occurred today in the capital city."
    words = short_text.split()
    assert len(words) < 20, "Test fixture must have fewer than 20 words"

    with pytest.raises(ValueError) as excinfo:
        predict_news(short_text)
    assert "at least 20 words" in str(excinfo.value)


# --- TC-6: Real News Classification ---
def test_tc6_real_news_prediction():
    """TC-6: Real-style journalistic article predicts REAL with confidence."""
    real_sample = (
        "WASHINGTON (Reuters) - The United States Senate approved a sweeping infrastructure package "
        "following weeks of bipartisan negotiations. Lawmakers voted in favor of allocating federal "
        "funding for highway repairs, public transit upgrades, bridge restorations, and clean drinking water initiatives across several states."
    )
    res = predict_news(real_sample)
    assert res["label"] == "REAL"
    assert isinstance(res["confidence"], float)
    assert res["confidence"] > 50.0
    assert isinstance(res["top_words"], list)


# --- TC-7: Fake News Classification ---
def test_tc7_fake_news_prediction():
    """TC-7: Sensational clickbait article predicts FAKE with confidence."""
    fake_sample = (
        "SHOCKING BOMBSHELL: Secret whistleblowers have leaked classified proof exposing a hidden "
        "global cabal manipulating weather and seismic tremors. Mainstream media is totally silent "
        "and refusing to cover this explosive truth while patriots wake up! Share this breaking report immediately!"
    )
    res = predict_news(fake_sample)
    assert res["label"] == "FAKE"
    assert isinstance(res["confidence"], float)
    assert res["confidence"] > 50.0
    assert isinstance(res["top_words"], list)
    assert len(res["top_words"]) > 0


# --- TC-8: API Valid Request ---
def test_tc8_api_predict_valid(client):
    """TC-8: POST /api/predict with valid JSON returns 200, label, confidence, top_words."""
    payload = {
        "text": (
            "The United States Senate approved a sweeping infrastructure package following weeks "
            "of bipartisan negotiations. Lawmakers voted in favor of allocating federal funding for highway repairs, "
            "public transit upgrades, bridge restorations, and clean drinking water initiatives."
        )
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert data["label"] in ["REAL", "FAKE"]
    assert "confidence" in data
    assert "top_words" in data
    assert isinstance(data["top_words"], list)


# --- TC-9: API Missing / Invalid Request ---
def test_tc9_api_predict_missing_text(client):
    """TC-9: POST /api/predict with missing text returns HTTP 400 and error message."""
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

    response_short = client.post("/api/predict", json={"text": "Too short headline."})
    assert response_short.status_code == 400
    data_short = response_short.get_json()
    assert "error" in data_short


# --- TC-10: Web Pages Load Correctly ---
def test_tc10_web_pages_load(client):
    """TC-10: Verifies / and /performance and /about endpoints load with HTTP 200."""
    home_resp = client.get("/")
    assert home_resp.status_code == 200
    assert b"News Authenticity Detector" in home_resp.data

    perf_resp = client.get("/performance")
    assert perf_resp.status_code == 200
    assert b"Model Performance" in perf_resp.data

    about_resp = client.get("/about")
    assert about_resp.status_code == 200
    assert b"Viva Quick-Reference Guide" in about_resp.data

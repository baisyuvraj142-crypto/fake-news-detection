"""
Flask web application for Fake News Detection using NLP.
Author: Yuvraj Singh Bais (LNCT&S Bhopal)
Routes:
- /            : Home page for interactive text classification
- /performance : Model evaluation metrics, comparison table, and confusion matrix
- /about       : Project documentation, pipeline architecture, and limitations
- /api/predict : REST API endpoint returning JSON predictions
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from src.predict import predict_news, load_artifacts

app = Flask(__name__)

# Preload model artifacts on boot for instant sub-second responses
try:
    load_artifacts()
except Exception as e:
    print(f"Warning: Preloading artifacts failed: {e}")


# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")


def get_metrics_data():
    """Helper to load model performance metrics from models/metrics.json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    """
    FR-18 to FR-22: Home page route.
    Renders input form, validates word count, and displays prediction card.
    """
    prediction = None
    error_message = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "")
        # NFR-4: Cap overly long inputs (>10,000 words) safely
        words = input_text.strip().split()
        if len(words) > 10000:
            input_text = " ".join(words[:10000])

        try:
            prediction = predict_news(input_text)
        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            error_message = f"An unexpected error occurred while analyzing the text: {str(e)}"

    return render_template(
        "index.html",
        input_text=input_text,
        prediction=prediction,
        error_message=error_message
    )


@app.route("/performance")
def performance():
    """
    FR-23: Displays model comparison metrics table, confusion matrix image,
    and dataset distribution statistics.
    """
    metrics = get_metrics_data()
    return render_template("performance.html", metrics=metrics)


@app.route("/about")
def about():
    """
    FR-24: Explains NLP/ML methodology, dataset, explainability, and limitations.
    """
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    FR-25: REST API endpoint accepting JSON:
    POST /api/predict
    { "text": "..." }

    Returns:
    { "label": "FAKE"|"REAL", "confidence": 94.2, "top_words": [...] }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON with 'Content-Type: application/json'"}), 400

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' key in JSON request body"}), 400

    text = data.get("text", "")
    try:
        result = predict_news(text)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal analysis error: {str(e)}"}), 500


if __name__ == "__main__":
    # NFR-7: Turn off debug mode in production/final version
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

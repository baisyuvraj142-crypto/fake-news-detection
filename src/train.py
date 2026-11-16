"""
Model training and evaluation pipeline for Fake News Detection.
Implements:
- FR-1: Load Fake.csv and True.csv
- FR-2: Add label (0=Fake, 1=Real), merge, shuffle
- FR-3: Combine title and text into 'content'
- FR-4: Drop empty and duplicate rows
- FR-5: Data-leakage removal (no subject/date, strip Reuters giveaways)
- FR-11: TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
- FR-12: Save vectorizer
- FR-13: 80/20 stratified split with random_state=42
- FR-14: Train Logistic Regression, Multinomial Naive Bayes, Passive Aggressive
- FR-15: Evaluate accuracy, precision, recall, F1
- FR-16: Save best model to models/model.pkl
- FR-17: Save models/metrics.json and static/confusion_matrix.png
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating images
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text


def load_and_prepare_data(data_dir: str):
    """
    Loads True.csv and Fake.csv, assigns labels, merges, and cleans duplicates.
    Returns (DataFrame, error_message).
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    # TC-3 / Section 7.1: Check if dataset CSV files exist
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        missing = []
        if not os.path.exists(fake_path):
            missing.append("Fake.csv")
        if not os.path.exists(true_path):
            missing.append("True.csv")
        error_msg = (
            f"Error: Missing dataset file(s): {', '.join(missing)} in '{data_dir}'.\n"
            "Please download the 'Fake and Real News Dataset' (ISOT) from Kaggle:\n"
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"and place 'Fake.csv' and 'True.csv' inside '{data_dir}'."
        )
        return None, error_msg

    print(f"[1/6] Loading datasets from '{data_dir}'...")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # FR-2: Add label column: 0 for fake, 1 for real
    df_fake['label'] = 0
    df_true['label'] = 1

    # Merge datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)

    # FR-2: Shuffle dataset with fixed seed for reproducibility (NFR-8)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # FR-3: Combine title and text into single content column
    # Notice: Subject and Date columns are deliberately ignored to prevent data leakage (FR-5)
    title_col = df['title'].fillna('') if 'title' in df.columns else ''
    text_col = df['text'].fillna('') if 'text' in df.columns else ''
    df['content'] = title_col + " " + text_col

    # FR-4: Drop empty rows and duplicates
    initial_len = len(df)
    df.drop_duplicates(subset=['content'], inplace=True)
    df = df[df['content'].str.strip() != '']
    print(f"      Loaded {initial_len} rows, {len(df)} remaining after deduplication.")

    return df, None


def train_pipeline(data_dir: str = None, models_dir: str = None, static_dir: str = None):
    """
    Executes the full training, evaluation, model selection, and artifact saving pipeline.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    if static_dir is None:
        static_dir = os.path.join(PROJECT_ROOT, "static")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    df, err = load_and_prepare_data(data_dir)
    if err:
        print(err)
        return False

    # FR-6 to FR-10: Preprocess text using clean_text()
    print("[2/6] Preprocessing text (cleaning, stopwords removal, lemmatization)...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    # Remove any rows where cleaned content became empty
    df = df[df['cleaned_content'].str.strip() != '']

    X = df['cleaned_content']
    y = df['label']

    real_count = int((y == 1).sum())
    fake_count = int((y == 0).sum())
    total_samples = len(df)
    print(f"      Class distribution -> Real (1): {real_count}, Fake (0): {fake_count}")

    # FR-13: Split data 80% train / 20% test with stratification and random_state=42
    print("[3/6] Splitting dataset (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # FR-11: TF-IDF Feature Extraction (max_features=5000, ngram_range=(1,2))
    print("[4/6] Extracting TF-IDF features (max_features=5000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # FR-14 & 7.3: Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=50, random_state=42)
    }

    print("[5/6] Training and evaluating candidate models...")
    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    best_y_pred = None
    logistic_regression_model = None

    print("\n" + "=" * 78)
    print(f"{'Model Name':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 78)

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }

        print(f"{name:<32} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}%")

        if name == "Logistic Regression":
            logistic_regression_model = clf

        # FR-16: Select best model based on F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = clf
            best_y_pred = y_pred

    print("=" * 78 + "\n")
    print(f"Selected Best Model: {best_model_name} (F1-score: {best_f1 * 100:.2f}%)")

    # FR-16: Save best model and vectorizer using joblib
    print("[6/6] Saving trained artifacts...")
    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    lr_model_path = os.path.join(models_dir, "lr_model.pkl")

    joblib.dump(best_model_obj, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    if logistic_regression_model is not None:
        joblib.dump(logistic_regression_model, lr_model_path)
    print(f"      Saved model to: {model_path}")
    print(f"      Saved vectorizer to: {vectorizer_path}")

    # FR-17: Save evaluation metrics as JSON
    metrics_summary = {
        "best_model": best_model_name,
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "real_count": real_count,
        "fake_count": fake_count,
        "models": results
    }
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"      Saved metrics to: {metrics_path}")

    # FR-17: Generate and save confusion matrix chart
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['FAKE (0)', 'REAL (1)'],
        yticklabels=['FAKE (0)', 'REAL (1)']
    )
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()

    cm_path = os.path.join(static_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"      Saved confusion matrix image to: {cm_path}")

    print("\nTraining completed successfully!")
    return True


if __name__ == "__main__":
    success = train_pipeline()
    if not success:
        sys.exit(0)

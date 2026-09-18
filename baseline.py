"""Bag-of-words baseline: TF-IDF + logistic regression, no active feedback.

Trains one model per data/<n>.<n> folder, scores each on data/test, and writes
results/baseline_sweep.csv (accuracy vs. training-set size).

    python baseline.py
"""

import argparse

import joblib
import pandas as pd
from common import DATA, ROOT, TEST_DIR, load_data, preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_model(folder):
    texts, labels = load_data(folder)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(preprocess_data(texts))
    # 80% of the folder trains; the held-out 20% was only used for per-run stats
    X_train, _, y_train, _ = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, folder / "model.pkl")
    joblib.dump(vectorizer, folder / "vectorizer.pkl")
    return model, vectorizer


def test_model(model, vectorizer, test_texts, test_labels):
    y_pred = model.predict(vectorizer.transform(test_texts))
    report = classification_report(test_labels, y_pred, output_dict=True)
    return accuracy_score(test_labels, y_pred), report


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default=ROOT / "results" / "baseline_sweep.csv")
    args = p.parse_args()

    test_texts, test_labels = load_data(TEST_DIR)
    test_texts = preprocess_data(test_texts)

    folders = sorted(
        (f for f in DATA.glob("*.*") if f.is_dir()),
        key=lambda f: int(f.name.split(".")[0]),
    )
    if not folders:
        raise SystemExit(
            "No data/<n>.<n> folders. Run: python prepare_data.py train <topic>"
        )

    rows = []
    for folder in folders:
        n = int(folder.name.split(".")[0])
        accuracy, report = test_model(*train_model(folder), test_texts, test_labels)
        print(f"{n:>5} per class: accuracy {accuracy:.4f}")
        for label in ("0", "1"):
            rows.append(
                {
                    "samples_per_class": n,
                    "accuracy": accuracy,
                    "label": int(label),
                    **report[label],
                }
            )

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")

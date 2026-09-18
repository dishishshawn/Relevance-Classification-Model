"""Score every trained feedback model on data/test.

Skips any model that hasn't been trained yet. Appends one row per model to
results/model_comparison.csv.

    python evaluate.py
"""

import joblib
import numpy as np
import pandas as pd
from common import (
    BERT_DIR,
    BOW_DIR,
    LSTM_DIR,
    ROOT,
    TEST_DIR,
    load_data,
    preprocess_data,
)
from sklearn.metrics import accuracy_score, classification_report

RESULTS = ROOT / "results" / "model_comparison.csv"


def feedback_count(folder):
    path = folder / "labeled_data.csv"
    return len(pd.read_csv(path)) if path.exists() else 0


def predict_bow(texts):
    model = joblib.load(BOW_DIR / "model.pkl")
    vectorizer = joblib.load(BOW_DIR / "vectorizer.pkl")
    return model.predict(vectorizer.transform(texts))


def predict_lstm(texts):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model = load_model(LSTM_DIR / "sequential_model.h5")
    tokenizer = joblib.load(LSTM_DIR / "tokenizer.pkl")
    X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=100)
    return (model.predict(X, verbose=0).flatten() > 0.5).astype(int)


def predict_bert(texts):
    import tensorflow as tf
    from transformers import BertTokenizer, TFBertForSequenceClassification

    model = TFBertForSequenceClassification.from_pretrained(BERT_DIR / "bert_model")
    tokenizer = BertTokenizer.from_pretrained(BERT_DIR / "bert_tokenizer")
    preds = []
    for i in range(0, len(texts), 64):
        enc = tokenizer(
            texts[i : i + 64],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf",
        )
        probs = tf.nn.softmax(
            model.predict(dict(enc), verbose=0).logits, axis=1
        ).numpy()
        preds.append((probs[:, 1] > 0.5).astype(int))
    return np.concatenate(preds)


MODELS = [
    ("AL-BERT", BERT_DIR, BERT_DIR / "bert_model", predict_bert),
    ("LSTM", LSTM_DIR, LSTM_DIR / "sequential_model.h5", predict_lstm),
    ("Bag-of-Words", BOW_DIR, BOW_DIR / "model.pkl", predict_bow),
]

if __name__ == "__main__":
    texts, labels = load_data(TEST_DIR)
    texts = preprocess_data(texts)

    rows = []
    for name, folder, artifact, predict in MODELS:
        if not artifact.exists():
            print(f"{name}: not trained yet, skipping")
            continue
        y_pred = predict(texts)
        accuracy = accuracy_score(labels, y_pred)
        count = feedback_count(folder)
        print(f"\n{name} (trained on {count} labels): accuracy {accuracy:.4f}")
        print(
            classification_report(
                labels,
                y_pred,
                target_names=["Not Relevant", "Relevant"],
                zero_division=0,
            )
        )
        rows.append(
            {
                "model": name,
                "feedback_count": count,
                "accuracy": accuracy,
                "timestamp": pd.Timestamp.now(),
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS, mode="a", header=not RESULTS.exists(), index=False
        )
        print(f"Appended to {RESULTS}")

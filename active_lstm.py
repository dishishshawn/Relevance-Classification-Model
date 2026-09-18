"""LSTM (Keras Sequential) with a relevance feedback loop.

Starts from one hand-written relevant example, then keeps showing you the
tweet it thinks is most relevant. Each y/n answer is a training step.
Progress is saved to data/feedback_lstm/ after every answer, so you can stop
("stop") and pick up where you left off.

    python active_lstm.py
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from common import LSTM_DIR, load_sentiment140, preprocess_data
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_LEN = 100
INITIAL_TEXT = "I love watching football! The game was amazing, great match today!"


def main(num_samples, threshold):
    model_path = LSTM_DIR / "sequential_model.h5"
    tokenizer_path = LSTM_DIR / "tokenizer.pkl"
    labeled_path = LSTM_DIR / "labeled_data.csv"
    LSTM_DIR.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and tokenizer_path.exists():
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
    else:
        tokenizer = Tokenizer(num_words=10000)
        model = Sequential(
            [
                Embedding(10000, 128, input_length=MAX_LEN),
                LSTM(128, dropout=0.3, recurrent_dropout=0.3),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        tokenizer.fit_on_texts([INITIAL_TEXT])
        X_initial = pad_sequences(
            tokenizer.texts_to_sequences([INITIAL_TEXT]), maxlen=MAX_LEN
        )
        model.fit(X_initial, np.array([1]), epochs=3, verbose=0)

    texts = load_sentiment140(num_samples, random_state=42)["text"].tolist()
    if labeled_path.exists():
        labeled = pd.read_csv(labeled_path)
        seen = set(labeled["text"])
        texts = [t for t in texts if t not in seen]
    else:
        labeled = pd.DataFrame(columns=["text", "label"])

    tokenizer.fit_on_texts(preprocess_data(texts))

    while texts:
        X_pool = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN)
        scores = model.predict(X_pool, verbose=0).flatten()
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < threshold:
            print(f"(nothing above {threshold:.0%} confidence, showing the best guess)")

        print(f"\nPost: {texts[best_idx]}")
        print(f"Confidence of relevance: {scores[best_idx]:.2%}")
        answer = input("Relevant? (y/n/stop): ").strip().lower()
        if answer == "stop":
            break
        if answer not in ("y", "n"):
            continue

        label = 1 if answer == "y" else 0
        model.fit(
            X_pool[best_idx : best_idx + 1], np.array([label]), epochs=5, verbose=0
        )
        labeled = pd.concat(
            [labeled, pd.DataFrame({"text": [texts.pop(best_idx)], "label": [label]})],
            ignore_index=True,
        )

        labeled.to_csv(labeled_path, index=False)
        model.save(model_path)
        joblib.dump(tokenizer, tokenizer_path)
        print(
            f"Relevant: {(labeled['label'] == 1).sum()}, Not relevant: {(labeled['label'] == 0).sum()}"
        )
    else:
        print("\nAll samples have been processed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--samples", type=int, default=1000, help="tweets in the candidate pool"
    )
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    main(args.samples, args.threshold)

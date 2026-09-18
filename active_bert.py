"""AL-BERT: bert-base-uncased fine-tuned with an active feedback loop.

First run fine-tunes BERT on a small labeled folder. Then it walks through
Sentiment140 in batches of 32 and, from each batch, shows you the tweet with
the highest score (uncertainty + 0.5 x relevance probability). It retrains
every 10 labels, scores itself on data/test every 20, and saves everything to
data/feedback_bert/. Type "stop" (or Ctrl-C) to save and quit; rerun to resume.

    python active_bert.py --seed data/25.25
"""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from common import BERT_DIR, DATA, TEST_DIR, load_data, load_sentiment140
from transformers import BertTokenizer, TFBertForSequenceClassification

MODEL_PATH = BERT_DIR / "bert_model"
TOKENIZER_PATH = BERT_DIR / "bert_tokenizer"
LABELED_PATH = BERT_DIR / "labeled_data.csv"
RESULTS_PATH = BERT_DIR / "bert_feedback_results.csv"


def tokenize_texts(tokenizer, texts, max_length=128):
    return tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="tf"
    )


def train_model(model, tokenizer, labeled_data):
    texts = labeled_data["text"].tolist()
    labels = labeled_data["label"].astype(int).tolist()

    # balance the two classes so a lopsided label set doesn't swamp the model
    n = len(labels)
    n_relevant = sum(labels)
    class_weight = {
        0: n / (2 * max(n - n_relevant, 1)),
        1: n / (2 * max(n_relevant, 1)),
    }

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (dict(tokenize_texts(tokenizer, texts)), labels)
        )
        .shuffle(1000)
        .batch(16)
    )
    compile_model(model)
    model.fit(dataset, epochs=1, class_weight=class_weight)


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def evaluate(model, tokenizer, test_texts, test_labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(tokenize_texts(tokenizer, test_texts)), test_labels)
    ).batch(16)
    _, accuracy = model.evaluate(dataset, verbose=0)
    return accuracy


def save(model, tokenizer, labeled_data):
    labeled_data.to_csv(LABELED_PATH, index=False)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_PATH)


def feedback_loop(
    model, tokenizer, labeled_data, test_texts, test_labels, batch_size=32
):
    seen = set(labeled_data["text"])
    texts = [t for t in load_sentiment140()["text"] if t not in seen]
    counts = {"relevant": 0, "irrelevant": 0}

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logits = model.predict(dict(tokenize_texts(tokenizer, batch)), verbose=0).logits
        confidences = tf.nn.softmax(logits, axis=1).numpy()

        uncertainty = 1 - np.max(confidences, axis=1)
        scores = uncertainty + 0.5 * confidences[:, 1]
        top = int(np.argmax(scores))

        print(f"\nPost: {batch[top]}")
        print(
            f"Confidence: {confidences[top][1]:.2f} relevant, {confidences[top][0]:.2f} irrelevant"
        )
        print(
            f"This session - Relevant: {counts['relevant']}, Irrelevant: {counts['irrelevant']}"
        )
        answer = input("Is this relevant? (y/n/stop): ").strip().lower()
        if answer == "stop":
            return labeled_data
        if answer not in ("y", "n"):
            continue

        label = 1 if answer == "y" else 0
        counts["relevant" if label else "irrelevant"] += 1
        labeled_data = pd.concat(
            [labeled_data, pd.DataFrame({"text": [batch[top]], "label": [label]})],
            ignore_index=True,
        )
        labeled_data.to_csv(LABELED_PATH, index=False)

        if len(labeled_data) % 10 == 0:
            print("\nRetraining model on updated dataset...")
            train_model(model, tokenizer, labeled_data)
            save(model, tokenizer, labeled_data)

        if len(labeled_data) % 20 == 0:
            accuracy = evaluate(model, tokenizer, test_texts, test_labels)
            print(f"Test accuracy after {len(labeled_data)} labels: {accuracy:.4f}")
            row = pd.DataFrame(
                {
                    "feedback_count": [len(labeled_data)],
                    "accuracy": [accuracy],
                    "timestamp": [pd.Timestamp.now()],
                }
            )
            row.to_csv(
                RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False
            )
    return labeled_data


def main(seed_folder, test_size):
    BERT_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and TOKENIZER_PATH.exists():
        print("Loading saved model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
        compile_model(model)
        labeled_data = pd.read_csv(LABELED_PATH)
    else:
        print(
            f"No saved model found. Fine-tuning bert-base-uncased on {seed_folder}..."
        )
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        texts, labels = load_data(seed_folder)
        labeled_data = pd.DataFrame({"text": texts, "label": labels})
        train_model(model, tokenizer, labeled_data)
        save(model, tokenizer, labeled_data)

    # a fixed subset keeps the periodic check fast on CPU
    test_texts, test_labels = load_data(TEST_DIR)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test_texts), min(test_size, len(test_texts)), replace=False)
    test_texts = [test_texts[i] for i in idx]
    test_labels = [test_labels[i] for i in idx]
    print(
        f"Starting test accuracy: {evaluate(model, tokenizer, test_texts, test_labels):.4f}"
    )

    try:
        labeled_data = feedback_loop(
            model, tokenizer, labeled_data, test_texts, test_labels
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        labeled_data = pd.read_csv(LABELED_PATH)
    save(model, tokenizer, labeled_data)
    print(f"Saved to {BERT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--seed", default=DATA / "25.25", help="labeled folder for the first fine-tune"
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="test tweets used for the periodic check",
    )
    args = p.parse_args()
    main(args.seed, args.test_size)

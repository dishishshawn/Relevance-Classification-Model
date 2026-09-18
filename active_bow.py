"""Bag-of-words + logistic regression with an active feedback loop.

Starts from a small labeled folder, then repeatedly shows you the tweet the
model is least sure about (uncertainty sampling). Your y/n answer is taught
to the model. Type "stop" to save to data/feedback_bow/.

    python active_bow.py --seed data/25.25
"""

import argparse

import joblib
import pandas as pd
from common import BOW_DIR, DATA, load_data, load_sentiment140, preprocess_data
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main(seed_folder, pool_size):
    texts, labels = load_data(seed_folder)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(preprocess_data(texts))
    learner = ActiveLearner(
        estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
        query_strategy=uncertainty_sampling,
        X_training=X,
        y_training=labels,
    )

    tweets = load_sentiment140()["text"]
    labeled = [{"text": t, "label": l} for t, l in zip(texts, labels)]  # seed + feedback
    while True:
        # a fresh random pool each round; the learner picks the one it's least sure of
        pool = tweets.sample(pool_size).tolist()
        X_pool = vectorizer.transform(preprocess_data(pool))
        query_idx, _ = learner.query(X_pool)
        text = pool[query_idx[0]]

        print(f"\nPost: {text}")
        answer = input("Relevant? (y/n/stop): ").strip().lower()
        if answer == "stop":
            break
        if answer not in ("y", "n"):
            continue
        label = 1 if answer == "y" else 0
        learner.teach(X_pool[query_idx], [label])
        labeled.append({"text": text, "label": label})
        print(f"{len(labeled)} labels so far")

    BOW_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(learner, BOW_DIR / "model.pkl")
    joblib.dump(vectorizer, BOW_DIR / "vectorizer.pkl")
    pd.DataFrame(labeled, columns=["text", "label"]).to_csv(
        BOW_DIR / "labeled_data.csv", index=False
    )
    print(f"Saved to {BOW_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--seed", default=DATA / "25.25", help="labeled folder to start from"
    )
    p.add_argument("--pool", type=int, default=10, help="tweets sampled per query")
    args = p.parse_args()
    main(args.seed, args.pool)

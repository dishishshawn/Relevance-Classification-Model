"""Build labeled folders from Sentiment140 by keyword match.

A tweet is "relevant" if it contains one of the topic keywords. Each output
folder holds relevant/<tweet_id>.txt and irrelevant/<tweet_id>.txt, cleaned.

    python prepare_data.py test football     # data/test (1,000 tweets), run this first
    python prepare_data.py train football    # data/25.25 ... data/1000.1000, skips test tweets
"""

import argparse
from pathlib import Path

from common import DATA, TEST_DIR, clean_text, load_sentiment140


def split_by_topic(df, topics):
    topics = [t.lower() for t in topics]
    hit = df["text"].str.lower().apply(lambda s: any(t in s for t in topics))
    return df[hit], df[~hit]


def export(relevant, irrelevant, folder):
    for sub, rows in (("relevant", relevant), ("irrelevant", irrelevant)):
        out = Path(folder) / sub
        out.mkdir(parents=True, exist_ok=True)
        for tweet_id, text in zip(rows["id"], rows["text"]):
            (out / f"{tweet_id}.txt").write_text(clean_text(text), encoding="utf-8")
    print(f"{folder}: {len(relevant)} relevant, {len(irrelevant)} irrelevant")


def make_train_sets(df, topics, start, step, stop, seed):
    """One balanced folder per size, data/<n>.<n>, until the relevant tweets run out."""
    # keep test tweets out of training so the test accuracy is honest
    test_ids = {int(f.stem) for f in TEST_DIR.glob("*/*.txt")}
    relevant, irrelevant = split_by_topic(df[~df["id"].isin(test_ids)], topics)
    for n in range(start, stop + 1, step):
        if n > len(relevant):
            print(f"Stopped at {n}: only {len(relevant)} relevant tweets exist.")
            break
        export(
            relevant.sample(n, random_state=seed),
            irrelevant.sample(n, random_state=seed),
            DATA / f"{n}.{n}",
        )


def make_test_set(df, topics, size, seed):
    relevant, irrelevant = split_by_topic(df, topics)
    half = size // 2
    if len(relevant) < half:
        print(f"Only {len(relevant)} relevant tweets found, using all of them.")
    export(
        relevant.sample(min(half, len(relevant)), random_state=seed),
        irrelevant.sample(half, random_state=seed),
        TEST_DIR,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="balanced training folders of increasing size")
    t.add_argument("topics", nargs="+")
    t.add_argument("--start", type=int, default=25)
    t.add_argument("--step", type=int, default=25)
    t.add_argument("--stop", type=int, default=1000)

    s = sub.add_parser("test", help="held-out test set in data/test")
    s.add_argument("topics", nargs="+")
    s.add_argument("--size", type=int, default=1000)

    args = p.parse_args()
    df = load_sentiment140()
    if args.cmd == "train":
        make_train_sets(df, args.topics, args.start, args.step, args.stop, args.seed)
    else:
        make_test_set(df, args.topics, args.size, args.seed)

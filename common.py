"""Shared paths, dataset loading, and text preprocessing."""

import re
import urllib.request
import zipfile
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SENTIMENT140 = DATA / "raw" / "training.1600000.processed.noemoticon.csv"
SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
TEST_DIR = DATA / "test"
BOW_DIR = DATA / "feedback_bow"
LSTM_DIR = DATA / "feedback_lstm"
BERT_DIR = DATA / "feedback_bert"

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def ensure_sentiment140():
    """Download and unzip Sentiment140 (~80 MB) into data/raw/ if it isn't there yet."""
    if SENTIMENT140.exists():
        return SENTIMENT140
    SENTIMENT140.parent.mkdir(parents=True, exist_ok=True)
    zip_path = SENTIMENT140.parent / "sentiment140.zip"
    print(f"Downloading Sentiment140 from {SENTIMENT140_URL} ...")
    urllib.request.urlretrieve(SENTIMENT140_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(SENTIMENT140.name, SENTIMENT140.parent)
    zip_path.unlink()
    return SENTIMENT140


def load_sentiment140(num_samples=None, random_state=None):
    """Raw tweets as a DataFrame with columns target, id, date, flag, user, text."""
    df = pd.read_csv(
        ensure_sentiment140(),
        encoding="ISO-8859-1",
        names=["target", "id", "date", "flag", "user", "text"],
    )
    if num_samples:
        df = df.sample(n=num_samples, random_state=random_state)
    return df


def load_data(folder):
    """Read <folder>/relevant/*.txt (label 1) and <folder>/irrelevant/*.txt (label 0)."""
    texts, labels = [], []
    for label, sub in ((1, "relevant"), (0, "irrelevant")):
        for f in sorted((Path(folder) / sub).glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8").strip())
            labels.append(label)
    if not texts:
        raise SystemExit(
            f"No data in {folder}. Run prepare_data.py first (see README)."
        )
    return texts, labels


# Clean the text by removing URLs, non-alphabetic characters, and stopwords
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()  # lowercase so the same word always maps to the same token
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def preprocess_data(texts):
    return [clean_text(t) for t in texts]

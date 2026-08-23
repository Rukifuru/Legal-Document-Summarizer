"""
Legal Document Summarization - Data Collection & Preprocessing
Assignment 2, PS-13

Dataset: BillSum (US Congressional & California bills + human summaries)
Fallback: If billsum fails to load, swap in IN-Abs (Indian Supreme Court judgments)
or any local .txt/.csv of (document, summary) pairs -- see load_fallback_dataset().

Run: python 01_data_preprocessing.py
Outputs: data/train.csv, data/val.csv, data/test.csv
"""

import os
import re
import pandas as pd
from datasets import load_dataset

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

os.makedirs("data", exist_ok=True)

# Increased subset sizes — RTX 3060 can handle more training data
TRAIN_SUBSET = 5000        # was 3000; more data = better fine-tuning
VAL_SUBSET = 500           # was 300
TEST_SUBSET = 500          # was 300; more eval samples = better ROUGE stats


def clean_text(text: str) -> str:
    """Basic normalization: strip boilerplate, extra whitespace, control chars."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)          # drop non-ascii artifacts
    text = re.sub(r"SEC(TION)?\.?\s*\d+[A-Za-z]?\.", " ", text)  # legal section numbering noise
    return text.strip()


def preprocess_example(doc: str, summary: str) -> dict:
    """
    Clean and tokenize, but DO NOT TRUNCATE — keep full text for chunked summarization in app.py.
    Only compute stats (sentences, tokens) for reporting purposes.
    """
    doc_clean = clean_text(doc)
    summary_clean = clean_text(summary)

    sentences = sent_tokenize(doc_clean)          # sentence tokenization (required by assignment)
    tokens = word_tokenize(doc_clean)             # word tokenization (required by assignment)

    # NO TRUNCATION — full document and summary preserved
    return {
        "document": doc_clean,                     # full cleaned document, no truncation
        "summary": summary_clean,                  # full cleaned summary, no truncation
        "num_sentences": len(sentences),
        "num_tokens": len(tokens),
    }


def load_billsum():
    print("Loading BillSum from Hugging Face...")
    ds = load_dataset("FiscalNote/billsum")
    return ds["train"], ds["test"]


def load_fallback_dataset(csv_path="data/legal_docs_fallback.csv"):
    """
    Use this if billsum link/download fails (per assignment instructions,
    note in your report that a substitute dataset was used).
    Expects a CSV with columns: text, summary
    """
    print(f"Loading fallback dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    return df


def build_splits():
    try:
        train_raw, test_raw = load_billsum()
        train_df = pd.DataFrame(train_raw)[["text", "summary"]]
        test_df = pd.DataFrame(test_raw)[["text", "summary"]]
    except Exception as e:
        print(f"billsum load failed ({e}); falling back to local CSV.")
        full_df = load_fallback_dataset()
        train_df = full_df.sample(frac=0.85, random_state=42)
        test_df = full_df.drop(train_df.index)

    train_df = train_df.rename(columns={"text": "document"})
    test_df = test_df.rename(columns={"text": "document"})

    # Larger subsets now — GPU can handle it
    train_df = train_df.sample(n=min(TRAIN_SUBSET, len(train_df)), random_state=42).reset_index(drop=True)
    val_df = train_df.sample(n=min(VAL_SUBSET, len(train_df)), random_state=7).reset_index(drop=True)
    train_df = train_df.drop(val_df.index, errors="ignore").reset_index(drop=True)
    test_df = test_df.sample(n=min(TEST_SUBSET, len(test_df)), random_state=42).reset_index(drop=True)

    def process_df(df):
        records = [preprocess_example(row["document"], row["summary"]) for _, row in df.iterrows()]
        return pd.DataFrame(records)

    print("Preprocessing train split...")
    train_processed = process_df(train_df)
    print("Preprocessing val split...")
    val_processed = process_df(val_df)
    print("Preprocessing test split...")
    test_processed = process_df(test_df)

    train_processed.to_csv("data/train.csv", index=False)
    val_processed.to_csv("data/val.csv", index=False)
    test_processed.to_csv("data/test.csv", index=False)

    print(f"Saved: train={len(train_processed)}, val={len(val_processed)}, test={len(test_processed)}")
    print("\nSample row stats (no truncation applied):")
    sample = train_processed.iloc[0]
    print(f"  Document words: {len(sample['document'].split())}")
    print(f"  Summary words: {len(sample['summary'].split())}")
    print(f"  Sentences: {sample['num_sentences']}, Tokens: {sample['num_tokens']}")


if __name__ == "__main__":
    build_splits()
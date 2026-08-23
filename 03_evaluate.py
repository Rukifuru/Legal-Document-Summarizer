"""
Legal Document Summarization - Evaluation and Demonstration
Computes ROUGE-1, ROUGE-2, ROUGE-L between generated and reference summaries.

Run: python 03_evaluate.py
Outputs: data/eval_results.csv, prints average ROUGE scores
"""

import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_DIR = "model"          # fine-tuned checkpoint from 02_train_model.py
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
N_SAMPLES_TO_SHOW = 5         # for report screenshots / observations table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def generate_summary(document: str) -> str:
    input_text = "summarize: " + document
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=MAX_TARGET_LEN,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def evaluate():
    test_df = pd.read_csv("data/test.csv").dropna(subset=["document", "summary"])

    results = []
    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows()):
        generated = generate_summary(row["document"])
        scores = scorer.score(row["summary"], generated)
        results.append({
            "document_snippet": row["document"][:200],
            "reference_summary": row["summary"],
            "generated_summary": generated,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        })
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/eval_results.csv", index=False)

    print("\nAverage ROUGE scores on test set:")
    print(f"  ROUGE-1: {results_df['rouge1'].mean():.4f}")
    print(f"  ROUGE-2: {results_df['rouge2'].mean():.4f}")
    print(f"  ROUGE-L: {results_df['rougeL'].mean():.4f}")

    print(f"\nSample outputs (first {N_SAMPLES_TO_SHOW}) - copy these into your report:")
    for i, row in results_df.head(N_SAMPLES_TO_SHOW).iterrows():
        print(f"\n--- Example {i+1} ---")
        print(f"Reference : {row['reference_summary']}")
        print(f"Generated : {row['generated_summary']}")
        print(f"ROUGE-L   : {row['rougeL']:.4f}")


if __name__ == "__main__":
    evaluate()
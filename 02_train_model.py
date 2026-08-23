"""
Legal Document Summarization - Model Development
Encoder-Decoder (Transformer, Seq2Seq) fine-tuning using T5-small.

Why T5-small: is a genuine Encoder-Decoder
Transformer (satisfies the assignment requirement), and fine-tunes in a
reasonable time on a few thousand examples.

Run: python 02_train_model.py
Outputs: model/ (fine-tuned checkpoint)
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
OUTPUT_DIR = "model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)


def load_split(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["document", "summary"])
    return Dataset.from_pandas(df[["document", "summary"]])


def preprocess_batch(batch):
    inputs = ["summarize: " + doc for doc in batch["document"]]
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LEN, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["summary"], max_length=MAX_TARGET_LEN, truncation=True, padding="max_length"
    )
    labels["input_ids"] = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_tokenized_datasets():
    train_ds = load_split("data/train.csv")
    val_ds = load_split("data/val.csv")

    train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=["document", "summary"])
    val_tok = val_ds.map(preprocess_batch, batched=True, remove_columns=["document", "summary"])
    return train_tok, val_tok


def train():
    train_tok, val_tok = build_tokenized_datasets()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_steps=50,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_DIR}/")


def run_zero_shot_demo():
    """No-training baseline fallback: use pretrained t5-small directly."""
    sample = pd.read_csv("data/test.csv").iloc[0]["document"]
    input_text = "summarize: " + sample
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"], max_length=MAX_TARGET_LEN, num_beams=4, early_stopping=True
    )
    print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    train()
    # run_zero_shot_demo()  # uncomment to test without fine-tuning

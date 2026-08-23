"""
Legal Document Summarization - Streamlit Web Application
Assignment 2, PS-13 (BITS WILP AIMLCZG530)

Three input modes:
  1. Paste text
  2. Upload document (PDF/TXT)
  3. Load default/sample test data (from data/test.csv, used for demo + evaluation)

Shows the full pipeline on screen:
  Original document -> Preprocessing (tokenization/cleaning/normalization) -> Encoder-Decoder summary

Supports long documents via chunking: splits input into ~500-word chunks,
summarizes each chunk separately, then concatenates chunk summaries.

Run: streamlit run app.py
"""

import re
import streamlit as st
import pandas as pd
import pdfplumber
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

MODEL_DIR = "model"          # fine-tuned checkpoint; falls back to base t5-small if missing
MAX_INPUT_LEN = 512          # T5-small's native context window
MAX_TARGET_LEN = 128
DEFAULT_TEST_PATH = "data/test.csv"
CHUNK_SIZE_WORDS = 400       # slightly under 512 tokens to leave margin for subword tokenization
CHUNK_OVERLAP_WORDS = 0     # overlap between consecutive chunks to preserve context across boundaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
        source = f"{MODEL_DIR} (fine-tuned)"
    except Exception:
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        source = "t5-small (base, no fine-tuned checkpoint found)"
    model.eval()
    return tokenizer, model, source


@st.cache_data
def load_default_samples():
    try:
        df = pd.read_csv(DEFAULT_TEST_PATH)
        return df.dropna(subset=["document"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["document", "summary"])


tokenizer, model, model_source = load_model()
default_samples = load_default_samples()


def clean_text(text: str) -> str:
    """Normalization step: strip boilerplate, control chars, extra whitespace."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"SEC(TION)?\.?\s*\d+[A-Za-z]?\.", " ", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 400) -> list[str]:
    """
    Split text at sentence boundaries. No sentence is cut in half,
    which prevents incomplete clauses at chunk boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_chunk and current_words + sentence_words > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += sentence_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def preprocess_pipeline(raw_text: str) -> dict:
    """Runs tokenization, cleaning, and normalization; returns stats + cleaned text."""
    cleaned = clean_text(raw_text)
    sentences = sent_tokenize(cleaned)
    tokens = word_tokenize(cleaned)
    return {
        "cleaned_text": cleaned,
        "num_sentences": len(sentences),
        "num_tokens": len(tokens),
        "num_words": len(cleaned.split()),
        "sentences_preview": sentences[:3],
    }


def extract_text_from_pdf(uploaded_file) -> str:
    text_chunks = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


def summarize_chunk(chunk_text: str, num_beams: int = 3) -> str:
    """Summarize a single chunk of text using the fine-tuned T5 model."""
    input_text = "summarize: " + chunk_text
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True
    ).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=96,              # reduced from 128 to force conciseness
            min_length=25,              # prevent overly short outputs
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,     # prevents 3-gram repetition
            length_penalty=3.0,         # stronger penalty for long outputs (was 2.0)
            repetition_penalty=1.5,     # penalizes repeated content
            do_sample=False,            # deterministic beam search
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_full_document(cleaned_text: str, num_beams: int = 4) -> tuple:
    """
    Summarize the entire document by chunking if needed.
    Returns (full_summary, chunk_summaries_list, num_chunks).
    """
    chunks = chunk_text(cleaned_text)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        chunk_sum = summarize_chunk(chunk, num_beams=num_beams)
        chunk_summaries.append(chunk_sum)

    full_summary = " ".join(chunk_summaries)
    full_summary = re.sub(r"\s+", " ", full_summary).strip()
    return full_summary, chunk_summaries, len(chunks)


st.set_page_config(page_title="Legal Document Summarizer", layout="wide")
st.title("Legal Document Summarization System")
st.caption(f"Encoder-Decoder model in use: {model_source}")
st.info(
    "Note: Documents longer than ~450 words are automatically split into chunks "
    "and summarized section-by-section to handle the model's 512-token context limit."
)
st.write(
    "Upload a legal document, paste text, or load a sample from the test set to see "
    "preprocessing and an abstractive summary generated by an Encoder-Decoder model."
)

st.markdown("### 1. Choose Input")
input_mode = st.radio(
    "Input method:",
    ["Load default sample data", "Paste text", "Upload document"],
    horizontal=True,
)

document_text = ""
reference_summary = None

if input_mode == "Load default sample data":
    if default_samples.empty:
        st.error("No default data found. Run 01_data_preprocessing.py first to generate data/test.csv.")
    else:
        idx = st.selectbox(
            "Pick a sample document from the test set:",
            options=list(range(len(default_samples))),
            format_func=lambda i: f"Sample {i+1}",
        )
        document_text = default_samples.loc[idx, "document"]
        reference_summary = default_samples.loc[idx, "summary"] if "summary" in default_samples.columns else None

elif input_mode == "Paste text":
    document_text = st.text_area("Paste legal document text here:", height=250)

else:
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            document_text = extract_text_from_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode("utf-8", errors="ignore")

num_beams = st.slider("Beam search width (higher = slower, often better quality)", 1, 8, 4)

if document_text.strip():
    st.markdown("### 2. Original Document")
    with st.expander("View original document", expanded=True):
        st.write(document_text[:3000] + ("..." if len(document_text) > 3000 else ""))
        st.caption(f"{len(document_text.split())} raw words")

    if st.button("Run Preprocessing + Summarize", type="primary"):
        with st.spinner("Preprocessing (tokenization, cleaning, normalization)..."):
            prep = preprocess_pipeline(document_text)

        st.markdown("### 3. Preprocessing Output")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentences", prep["num_sentences"])
        c2.metric("Word Tokens", prep["num_tokens"])
        c3.metric("Words (cleaned)", prep["num_words"])

        with st.expander("View cleaned/normalized text used for the model"):
            st.write(prep["cleaned_text"][:3000] + ("..." if len(prep["cleaned_text"]) > 3000 else ""))

        with st.expander("Sample tokenized sentences (first 3)"):
            for i, s in enumerate(prep["sentences_preview"], 1):
                st.write(f"{i}. {s}")

        with st.spinner("Generating summary with Encoder-Decoder model..."):
            full_summary, chunk_summaries, num_chunks = summarize_full_document(
                prep["cleaned_text"], num_beams=num_beams
            )

        st.markdown("### 4. Generated Summary")
        st.success(full_summary)

        if num_chunks > 1:
            st.caption(f"Document was split into {num_chunks} chunks for summarization due to length.")
            with st.expander("View individual chunk summaries"):
                for i, cs in enumerate(chunk_summaries, 1):
                    st.write(f"**Chunk {i} summary:** {cs}")

        st.caption(f"Summary length: {len(full_summary.split())} words "
                   f"(compressed from {prep['num_words']} words, "
                   f"{prep['num_words'] / max(len(full_summary.split()), 1):.1f}x compression)")

        if reference_summary:
            st.markdown("### 5. Reference Summary (ground truth, for comparison)")
            with st.expander("View full reference summary", expanded=True):
                st.write(reference_summary)
else:
    st.info("Choose an input method above to begin.")

st.markdown("---")
st.caption("PS-13: Legal Document Text Summarization | BITS WILP M.Tech AIML - NLP Assignment 2")
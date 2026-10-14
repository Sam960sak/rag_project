# RAG Dynamic Parsing and Dynamic Chunking

A working prototype that compares three chunking strategies for retrieval, with
format-aware parsing of TXT, PDF and DOCX files.

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the full write-up (architecture,
test cases, measured results, limitations).

## Features

- **Dynamic parsing**: format-aware extraction (TXT/PDF/DOCX); DOCX tables and
  paragraphs; page markers stripped from embed text and kept as page metadata;
  helpful errors for empty/scanned PDFs and unsupported files.
- **Dynamic chunking**: fixed-size (sentence-snapped), sentence-based, and
  structure-aware/recursive (headings stay with their paragraphs, oversized
  sections split at sentence boundaries, tiny chunks merged).
- **Retrieval**: `sentence-transformers` (`all-MiniLM-L6-v2`) with FAISS/NumPy
  cosine similarity; the model is cached and shared across strategies.
- **Evaluation**: gold-labelled Recall@1, Recall@3 and MRR, plus a defined,
  auditable term-overlap metric (kept separate from similarity and accuracy).
- **Streamlit UI**: side-by-side expandable result cards showing source filename,
  page number, similarity, term overlap, chunk size and retrieval time.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run run_app.py
```

Choose **Use sample documents**, enter a question, and click **Retrieve & Compare**.

## Test

```powershell
python -m pytest -v
```

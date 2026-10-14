# RAG Dynamic Parsing and Dynamic Chunking

Complete working prototype comparing 3 chunking strategies for retrieval.

## Dynamic Parsing
Adapts extraction by file type (TXT/PDF/DOCX), preserving basic structure/page info.

## Dynamic Chunking
Uses three strategies: fixed-size, sentence-based, structure-aware/recursive.

## Setup
`ash
pip install -r requirements.txt
streamlit run run_app.py
`

## Features
- Multi-format ingestion
- 3 chunking strategies with counts/text/sizes
- sentence-transformers + FAISS/NumPy cosine similarity
- Top-3 retrieval per strategy
- Streamlit UI with side-by-side/tabs + similarity scores
- Evaluation with Recall@3 on 5+ sample questions
- Evidence-only (no LLM required)

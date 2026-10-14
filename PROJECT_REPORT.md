# RAG Dynamic Parsing and Dynamic Chunking — Project Report

_A comparative study of three chunking strategies for retrieval-augmented
generation, with dynamic, format-aware parsing._

> **Before you submit:** replace the placeholders marked **[EDIT]** (GitHub URL,
> author, dates) and insert your **own** screenshots where indicated. The
> measured numbers below were produced by running this code on the bundled
> sample documents; re-run the commands in section 7 to reproduce them.

---

## 1. Overview

The project is a working prototype that ingests `.txt`, `.pdf` and `.docx`
documents, cleans and structures the extracted text, splits it into chunks using
three strategies, embeds the chunks, and compares retrieval behaviour.

Three concepts are deliberately kept separate throughout the system:

| Concept | Meaning | Where it comes from |
|---|---|---|
| **Similarity** | Semantic closeness of query and chunk | Embedding cosine score (`all-MiniLM-L6-v2`) |
| **Term overlap** | Share of the question's *whole-word* terms present in a chunk, IDF-weighted | `Evaluator.query_term_overlap` |
| **Answer accuracy** | Whether the chunk actually contains the labelled answer | Gold-labelled evaluation (`Recall@1/@3`, `MRR`) |

A higher similarity score does **not** imply the chunk answers the question, and
term overlap is not answer accuracy. The UI and report state this explicitly.

---

## 2. Architecture

```
                ┌─────────────────────────────────────────────┐
   .txt/.pdf/   │  DocumentParser                             │
   .docx   ───► │  • format-aware extraction                  │
                │  • DOCX tables + paragraphs (document order)│
                │  • page markers stripped; pages kept as meta│
                │  • empty/scanned PDF → helpful error        │
                └───────────────┬─────────────────────────────┘
                                │  pages: [{page, text}], metadata {file_name,...}
                                ▼
                ┌─────────────────────────────────────────────┐
                │  Chunker  (per page, metadata-aware)        │
                │  1. fixed_size      (sentence-snapped)      │
                │  2. sentence_based                          │
                │  3. structure_aware (heading + paragraph)   │
                │  shared: min-size merge, sentence splitting │
                └───────────────┬─────────────────────────────┘
                                │  chunk {text, size, metadata{source,page,...}}
                                ▼
                ┌─────────────────────────────────────────────┐
                │  Retriever  (cached SentenceTransformer)    │
                │  • encode + normalize, FAISS Inner Product  │
                │  • returns top-k with similarity + metadata │
                └───────────────┬─────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │  Streamlit UI   +   Evaluator               │
                │  • expandable result cards                  │
                │  • similarity vs term overlap vs accuracy   │
                │  • gold-labelled Recall@1/@3, MRR           │
                └─────────────────────────────────────────────┘
```

**Module map**

| File | Responsibility |
|---|---|
| `src/parsers/document_parser.py` | Format-aware parsing, cleaning, DOCX tables, page metadata |
| `src/chunking/chunkers.py` | Three chunking strategies + shared boundary logic |
| `src/retrieval/retriever.py` | Cached embeddings + FAISS/NumPy retrieval |
| `src/evaluation/evaluator.py` | Gold labels, Recall@1/@3, MRR, term overlap |
| `src/app/app.py` | Streamlit UI and orchestration |
| `run_app.py` | Application entry point |
| `tests/` | 17 automated tests |

---

## 3. Implemented Features

### 3.1 Dynamic parsing
- TXT, PDF and DOCX support selected by extension.
- **DOCX tables are extracted** as well as paragraphs, in document order
  (`Metric | Value`, `Recall@3 | 0.90`, …).
- **Page markers are removed** from the text used for embedding; page numbers
  are stored as metadata instead (`_parse_pdf` adds `{'page': n}` per page).
- **Empty or image-only PDFs** raise `UnreadableDocumentError` with an actionable
  message ("…run OCR (e.g. ocrmypdf) first…"), surfaced in the UI as an error.
- Unsupported types raise `UnsupportedFileTypeError` with the supported list.
- Every chunk keeps `source` (filename) and `page`.

### 3.2 Dynamic chunking
- **Fixed-size**: cuts near a sentence terminator when possible, otherwise at a
  word boundary — words are never split.
- **Sentence-based**: packs sentences up to a target size with sentence overlap.
- **Structure-aware / recursive**: attaches each heading to the paragraph that
  explains it, and splits oversized sections at sentence boundaries (the heading
  is repeated on each sub-chunk so context is not lost).
- **Minimum chunk size**: chunks below 100 characters are merged into a
  neighbour, preventing fragments such as a lone 25-character code line.
- All strategies receive `metadata` and stamp each chunk with `source` + `page`.

### 3.3 Retrieval
- `all-MiniLM-L6-v2` sentence embeddings, L2-normalised.
- FAISS `IndexFlatIP` cosine search with a NumPy fallback.
- **Model caching**: the transformer is loaded once and shared by all three
  strategies (previously it was loaded three times per query).

### 3.4 Evaluation
- **Gold-labelled** questions (`GOLD_QA`): a chunk counts as relevant only if it
  contains a verbatim gold answer snippet, not merely a keyword.
- Metrics: **Recall@1**, **Recall@3**, **MRR**, plus per-question first-relevant
  rank.
- **Term overlap** is a clearly defined, auditable lexical metric. It matches
  whole words after light stemming (so `supervised` never matches
  `unsupervised`) and IDF-weights terms against the corpus, so chunks that merely
  share common words cannot score 100%.

### 3.5 UI
- Full strategy names: **Fixed-size**, **Sentence-based**,
  **Structure-aware / recursive**.
- Expandable result cards showing similarity, term overlap, source filename,
  page number, chunk size, split type, and the **complete** chunk text.
- Consistent chunk count and per-strategy retrieval time (ms).
- Three separate tabs: side-by-side comparison, comparison summary, labelled
  evaluation.

---

## 4. Repository & Running the Project  **[EDIT]**

Repository: `https://github.com/<your-username>/<your-repo>`  **[EDIT — replace]**

> This folder is not yet a git repository. To publish it:
> ```powershell
> git init
> git add .
> git commit -m "RAG dynamic parsing and chunking prototype"
> git branch -M main
> git remote add origin https://github.com/<your-username>/<your-repo>.git
> git push -u origin main
> ```

### 4.1 Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2 Run the app
```powershell
streamlit run run_app.py
```
Then choose **Use sample documents**, type a question, and click
**Retrieve & Compare**. (The first run downloads the `all-MiniLM-L6-v2` model.)

### 4.3 Run the tests
```powershell
python -m pytest -v
```

---

## 5. Test Cases and Actual Outcomes

Test run: `python -m pytest -v` → **17 passed** in ~24 s.

| # | Test | Scenario | Result |
|---|---|---|---|
| 1 | `test_parse_txt` | TXT upload | PASS |
| 2 | `test_parse_docx_includes_table` | DOCX paragraphs **and table cells** | PASS |
| 3 | `test_parse_pdf` | PDF upload | PASS |
| 4 | `test_unreadable_pdf_raises_helpful_error` | Blank/scanned PDF | PASS (helpful OCR message) |
| 5 | `test_unsupported_file_raises` | Unsupported `.csv` | PASS (raises) |
| 6 | `test_chunks_carry_source_and_page` | Metadata on all three strategies | PASS |
| 7 | `test_retrieval_all_strategies` | Retrieval works for all three strategies; empty query safe | PASS |
| 8 | `test_full_app_sample_mode_ui` | Whole app via Streamlit `AppTest`: empty question → info, real question → results, no exception | PASS |
| 9 | `test_parser_txt` | Parser returns text and page 1 | PASS |
| 10 | `test_page_markers_removed` | `--- PAGE 3 ---` stripped | PASS |
| 11 | `test_fixed_chunks_word_safe_and_metadata` | No split words; source/page preserved | PASS |
| 12 | `test_structure_keeps_heading_with_paragraph` | Heading stays with explanation | PASS |
| 13 | `test_structure_splits_oversized_at_sentences` | Oversized split at sentence boundaries | PASS |
| 14 | `test_tiny_chunks_are_merged` | Tiny fragments merged | PASS |
| 15 | `test_query_term_overlap_no_substring_false_positive` | `unsupervised` ≠ `supervised` | PASS |
| 16 | `test_query_term_overlap_idf_downweights_common_terms` | Common terms cannot inflate score | PASS |
| 17 | `test_gold_recall_and_mrr` | Recall@1, Recall@3, MRR correctness | PASS |

---

## 6. Measured Results

Corpus: the three bundled sample documents (`ml_intro.txt`, `nlp_basics.txt`,
`vectordb_overview.txt`). Chunk counts:

| Strategy | Chunks |
|---|---|
| Fixed-size | 6 |
| Sentence-based | 8 |
| Structure-aware / recursive | 9 |

### 6.1 Labelled retrieval accuracy (10 questions)

| Strategy | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| Fixed-size | 0.90 | 1.00 | 0.95 |
| Sentence-based | 0.80 | 1.00 | 0.90 |
| Structure-aware / recursive | 0.90 | 1.00 | 0.95 |

> **Reading this honestly.** Recall@3 is saturated at 1.00 for every strategy
> because the corpus is tiny (each top-3 covers a large share of a small corpus).
> Recall@1 and MRR are the informative columns here. On the bundled samples the
> three strategies are close; the differences become clearer on the questions
> below and on a larger corpus.

### 6.2 Per-question first-relevant rank (0 = miss)

| Question | Fixed | Sentence | Structure |
|---|---|---|---|
| What is supervised learning? | 1 | 1 | 1 |
| How does reinforcement learning work? | 1 | 1 | 1 |
| What does tokenization do? | 1 | 1 | 1 |
| What are transformers in NLP? | 1 | 1 | 1 |
| Why use vector databases? | 1 | 1 | 1 |
| What is unsupervised learning used for? | 1 | 1 | 1 |
| How is machine learning applied in healthcare? | 2 | 2 | **1** |
| What are common NLP applications? | 1 | 2 | 2 |
| What is retrieval augmented generation? | 1 | 1 | 1 |
| What is sentiment analysis? | 1 | 1 | 1 |

### 6.3 Similarity vs term overlap (top-3, `similarity / overlap`)

| Question | Fixed-size | Sentence-based | Structure-aware |
|---|---|---|---|
| What is supervised learning? | 0.682/1.00, 0.497/0.39, 0.401/0.39 | 0.697/1.00, 0.674/1.00, 0.494/0.39 | 0.732/1.00, 0.510/0.39, 0.435/0.39 |
| What are common NLP applications? | 0.585/0.70, 0.411/0.00, 0.288/0.30 | 0.568/0.41, 0.514/0.30, 0.459/0.00 | 0.568/0.41, **0.531/0.70**, 0.507/0.30 |
| What is retrieval augmented generation? | 0.285/1.00, … | 0.285/1.00, … | **0.537/1.00**, … |

These rows show why similarity alone is not enough: for the RAG question all
three place the answer chunk first, but structure-aware rates it much higher
(0.537 vs 0.285), and for the NLP-applications question structure-aware surfaces
the relevant chunk at #2 with 0.70 term overlap versus 0.30 for sentence-based.

---

## 7. Reproducing the Numbers

```powershell
python -m pytest -v            # 17 tests
```
The accuracy and similarity/overlap tables above come from running the app's
retrieval pipeline over `GOLD_QA`; the same figures are shown live in the
**Evaluation (labelled)** and **Comparison Summary** tabs of the app.

---

## 8. Screenshots  **[EDIT — capture these yourself]**

Genuine screenshots must be captured from your own running app. Suggested set
(create a `docs/screenshots/` folder and add each):

1. `01_home.png` — landing page with sample documents selected.
2. `02_results.png` — side-by-side results with expandable cards open, showing
   source filename, page, similarity and term overlap.
3. `03_summary.png` — comparison summary table and the similarity/term-overlap caveat.
4. `04_evaluation.png` — labelled evaluation table (ranks + Recall@1/@3/MRR).
5. `05_docx_table.png` — a DOCX with a table retrieved correctly.
6. `06_unreadable_pdf.png` — the helpful error for a scanned/empty PDF.

<!-- Example once captured:
![Side-by-side results](docs/screenshots/02_results.png)
-->

---

## 9. Limitations

- **Corpus size.** With only three short documents, Recall@3 saturates at 1.00;
  Recall@1/MRR are more informative, and a larger corpus is needed for stable
  conclusions.
- **Gold labels are hand-written** for the ten sample questions; they do not
  cover every possible question.
- **Term overlap is lexical**, not semantic — it is a transparent aid, not a
  correctness measure.
- **Fixed-size chunking is a baseline.** It can span topic boundaries, so its
  chunks may contain context that belongs to a different section.
- **PDF extraction** relies on `PyPDF2`; complex layouts (multi-column, tables)
  are not reconstructed. Scanned PDFs require external OCR.
- **No LLM generation**: the tool returns evidence only and does not synthesise
  an answer.
- **`PyPDF2` is deprecated**; migrating to `pypdf` is recommended (a warning is
  emitted).

---

## 10. Conclusion

The prototype now separates three ideas that are often conflated: embedding
**similarity**, lexical **term overlap**, and labelled **answer accuracy**.
Parsing and chunking were fixed first, which improved both retrieval quality and
the honesty of the comparison. On the sample corpus the strategies perform
comparably, with structure-aware chunking showing an advantage on
section-specific questions (higher similarity and term overlap for the
retrieval-augmented-generation and NLP-application queries). The central finding
is methodological: **the highest similarity score alone does not prove a
strategy is best — the retrieved text must actually answer the question**, which
is why the tool exposes labelled metrics and the full expanded chunks for
inspection.

The main next step is to repeat the evaluation on a larger, more realistic
corpus (e.g. the project report itself) where Recall@3 no longer saturates and
the strategies can be distinguished with confidence.

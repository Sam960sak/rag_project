import os
import sys
import time
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.parsers.document_parser import (
    DocumentParser, UnreadableDocumentError, UnsupportedFileTypeError)
from src.chunking.chunkers import Chunker
from src.retrieval.retriever import Retriever
from src.evaluation.evaluator import GOLD_QA, Evaluator

SAMPLE_DIR = Path('sample_docs')

STRATEGIES = {
    'fixed': Chunker.fixed_size,
    'sentence': Chunker.sentence_based,
    'structure': Chunker.structure_aware,
}

STRATEGY_NAMES = {
    'fixed': 'Fixed-size',
    'sentence': 'Sentence-based',
    'structure': 'Structure-aware / recursive',
}


def get_sample_files():
    if SAMPLE_DIR.exists():
        return sorted([str(SAMPLE_DIR / f) for f in os.listdir(SAMPLE_DIR)
                       if f.endswith(('.txt', '.pdf', '.docx'))])
    return []


def load_docs(paths):
    parser = DocumentParser()
    docs = []
    for p in paths:
        try:
            docs.append(parser.parse_file(p))
        except UnreadableDocumentError as e:
            st.error(f'{Path(p).name}: {e}')
        except UnsupportedFileTypeError as e:
            st.error(f'{Path(p).name}: {e}')
        except Exception as e:  # corrupted file, permissions, etc.
            st.error(f'Failed to parse {Path(p).name}: {e}')
    return docs


def chunk_docs(docs):
    """Chunk every page of every document, tagging each chunk with source + page."""
    out = {name: [] for name in STRATEGIES}
    for doc in docs:
        source = doc['metadata'].get('file_name', 'document')
        pages = doc.get('pages') or [{'page': 1, 'text': doc.get('text', '')}]
        for pg in pages:
            base = {'source': source, 'page': pg.get('page', 1)}
            for name, fn in STRATEGIES.items():
                out[name].extend(fn(pg['text'], metadata=base))
    for chunks in out.values():
        for i, c in enumerate(chunks):
            c['chunk_id'] = i
    return out


def render_results(res, chunks, times, question, corpus):
    cols = st.columns(3)
    for i, (name, items) in enumerate(res.items()):
        with cols[i]:
            st.subheader(STRATEGY_NAMES[name])
            st.caption(f'{len(chunks[name])} chunks · retrieved in {times[name]:.1f} ms')
            if not items:
                st.info('No chunks retrieved.')
                continue
            for j, cr in enumerate(items):
                meta = cr.get('metadata', {})
                source = meta.get('source', 'unknown')
                page = meta.get('page', '?')
                overlap = Evaluator.query_term_overlap(question, cr['text'], corpus)
                label = (f"#{j + 1} · similarity {cr['similarity']:.3f} · "
                         f"term overlap {overlap['score']:.0%} · {source} (p.{page})")
                with st.expander(label):
                    st.caption(f"source: {source} · page: {page} · "
                               f"{cr['size']} chars · split: {meta.get('split', cr['type'])}")
                    st.markdown(f"**Matched query terms:** {', '.join(overlap['matched']) or '—'}")
                    if overlap['missing']:
                        st.markdown(f"**Missing query terms:** {', '.join(overlap['missing'])}")
                    st.write(cr['text'])


def render_summary(res, chunks, times, question, corpus):
    st.write('Chunk counts:', {STRATEGY_NAMES[k]: len(v) for k, v in chunks.items()})
    rows = []
    for name, items in res.items():
        avg_sim = (sum(x['similarity'] for x in items) / len(items)) if items else 0.0
        avg_overlap = (sum(Evaluator.query_term_overlap(question, x['text'], corpus)['score']
                           for x in items) / len(items)) if items else 0.0
        rows.append({
            'Strategy': STRATEGY_NAMES[name],
            'Chunks': len(chunks[name]),
            'Avg similarity': round(avg_sim, 3),
            'Avg term overlap': round(avg_overlap, 3),
            'Retrieval (ms)': round(times[name], 1),
        })
    st.table(rows)

    top1 = {name: (items[0]['text'] if items else None) for name, items in res.items()}
    same_ss = top1.get('sentence') is not None and top1.get('sentence') == top1.get('structure')
    st.markdown(f"Top-1 chunk identical between **Sentence-based** and "
                f"**Structure-aware**: **{'yes' if same_ss else 'no'}**.")
    st.caption(
        'Three different things are shown. **Similarity** = embedding cosine score '
        '(semantic closeness). **Term overlap** = IDF-weighted share of the question\'s '
        'whole-word terms present in the chunk (a defined lexical metric). **Answer '
        'accuracy** = whether the chunk actually contains the labelled answer, which is '
        'only measured in the Evaluation tab. None of the three is interchangeable; read '
        'the expanded chunks before judging a strategy.')


def render_evaluation(retrievers):
    results_per_q = []
    for qitem in GOLD_QA:
        results_per_q.append({name: r.search(qitem['question'], top_k=3)
                              for name, r in retrievers.items()})

    table_rows = []
    for t, qitem in enumerate(GOLD_QA):
        row = {'Question': qitem['question']}
        for name in retrievers:
            ranks = Evaluator.ranks_per_question([results_per_q[t][name]], qa=[qitem])
            row[STRATEGY_NAMES[name]] = f"rank {ranks[0]}" if ranks[0] else 'miss'
        table_rows.append(row)
    st.table(table_rows)

    scores = {}
    for name in retrievers:
        per_q = [results_per_q[t][name] for t in range(len(GOLD_QA))]
        scores[name] = {
            'Recall@3': Evaluator.recall_at_k(per_q, qa=GOLD_QA, k=3),
            'Recall@1': Evaluator.recall_at_1(per_q, qa=GOLD_QA),
            'MRR': Evaluator.reciprocal_rank(per_q, qa=GOLD_QA),
        }
    order = ['fixed', 'sentence', 'structure']
    st.table({
        'Strategy': [STRATEGY_NAMES[n] for n in order],
        'Recall@3': [round(scores[n]['Recall@3'], 2) for n in order],
        'Recall@1': [round(scores[n]['Recall@1'], 2) for n in order],
        'MRR': [round(scores[n]['MRR'], 2) for n in order],
    })
    st.caption(
        f'Measured over {len(GOLD_QA)} questions using labelled gold answer snippets. '
        'A chunk counts as relevant only if it actually contains the labelled answer. '
        'Recall@3 can saturate on a small corpus; Recall@1 and MRR show how highly the '
        'correct chunk is ranked. No strategy is hardcoded as the winner.')


def main():
    st.set_page_config(page_title='RAG Dynamic Parsing & Dynamic Chunking', layout='wide')
    st.title('RAG Dynamic Parsing and Dynamic Chunking')
    st.caption('Compare fixed-size, sentence-based, and structure-aware chunking for retrieval')

    mode = st.radio('Document source', ['Use sample documents', 'Upload documents'])
    paths = []
    if mode == 'Use sample documents':
        paths = get_sample_files()
        st.write('Using:', paths)
        st.caption('Start with the sample ML, NLP, and vector database documents. '
                   'Upload your project report only when you want to search that report.')
    else:
        files = st.file_uploader('Upload TXT/PDF/DOCX', type=['txt', 'pdf', 'docx'],
                                 accept_multiple_files=True)
        for f in files:
            tmp = Path('temp') / f.name
            tmp.parent.mkdir(exist_ok=True)
            with open(tmp, 'wb') as out:
                out.write(f.read())
            paths.append(str(tmp))

    question = st.text_input('Enter your question')
    if st.button('Retrieve & Compare'):
        if not paths:
            st.info('Please select or upload documents first.')
        elif not question or not question.strip():
            st.info('Please enter a question.')
        else:
            docs = load_docs(paths)
            if not docs:
                st.stop()
            ch = chunk_docs(docs)
            corpus = [c['text'] for name in ch for c in ch[name]]
            retrievers = {}
            for name, chunks in ch.items():
                retrievers[name] = Retriever('all-MiniLM-L6-v2').build(chunks)

            res, times = {}, {}
            for name, r in retrievers.items():
                t0 = time.perf_counter()
                res[name] = r.search(question, top_k=3)
                times[name] = (time.perf_counter() - t0) * 1000.0

            tabs = st.tabs(['Side by side comparison', 'Comparison Summary',
                            'Evaluation (labelled)'])
            with tabs[0]:
                render_results(res, ch, times, question, corpus)
            with tabs[1]:
                render_summary(res, ch, times, question, corpus)
            with tabs[2]:
                render_evaluation(retrievers)


if __name__ == '__main__':
    main()

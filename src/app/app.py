import os
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.parsers.document_parser import DocumentParser
from src.chunking.chunkers import Chunker
from src.retrieval.retriever import Retriever
from src.evaluation.evaluator import SAMPLE_QA, Evaluator

SAMPLE_DIR = Path('sample_docs')

def get_sample_files():
    if SAMPLE_DIR.exists():
        return sorted([str(SAMPLE_DIR / f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('.txt','.pdf','.docx'))])
    return []

def load_docs(paths):
    parser = DocumentParser()
    docs = []
    for p in paths:
        try:
            docs.append(parser.parse_file(p))
        except Exception as e:
            st.warning(f'Failed to parse {p}: {e}')
    return docs

def chunk_all(text):
    return {
        'fixed': Chunker.fixed_size(text),
        'sentence': Chunker.sentence_based(text),
        'structure': Chunker.structure_aware(text),
    }

def main():
    st.set_page_config(page_title='RAG Dynamic Parsing & Dynamic Chunking', layout='wide')
    st.title('RAG Dynamic Parsing and Dynamic Chunking')
    st.caption('Compare fixed-size, sentence-based, and structure-aware chunking for retrieval')

    mode = st.radio('Document source', ['Use sample documents', 'Upload documents'])
    paths = []
    if mode == 'Use sample documents':
        paths = get_sample_files()
        st.write('Using:', paths)
    else:
        files = st.file_uploader('Upload TXT/PDF/DOCX', type=['txt','pdf','docx'], accept_multiple_files=True)
        for f in files:
            tmp = Path('temp') / f.name
            tmp.parent.mkdir(exist_ok=True)
            with open(tmp,'wb') as out:
                out.write(f.read())
            paths.append(str(tmp))

    question = st.text_input('Enter your question')
    if st.button('Retrieve & Compare') and paths and question:
        docs = load_docs(paths)
        all_text = '\n\n'.join([d['text'] for d in docs])
        ch = chunk_all(all_text)
        retrievers = {}
        for name, chunks in ch.items():
            r = Retriever('all-MiniLM-L6-v2')
            r.build(chunks)
            retrievers[name] = r
        res = {}
        for name, r in retrievers.items():
            res[name] = r.search(question, top_k=3)

        tabs = st.tabs(['Side by side comparison', 'Comparison Summary', 'Evaluation (Recall@3)'])
        with tabs[0]:
            cols = st.columns(3)
            for i,(name, chunks_res) in enumerate(res.items()):
                with cols[i]:
                    st.subheader(name)
                    st.write(f'Chunks count (from combined text): {len(ch[name])}')
                    for j, cr in enumerate(chunks_res):
                        st.markdown(f'**#{j+1} | sim={cr["similarity"]:.3f} | size={cr["size"]}**')
                        st.text_area('', cr['text'], height=120, key=f'{name}_{j}')
        with tabs[1]:
            st.write('Total chunks:', {k: len(v) for k,v in ch.items()})
            for name, chunks_res in res.items():
                st.write(f'{name}: top3 sims = ' + ', '.join([f'{x["similarity"]:.3f}' for x in chunks_res]))
        with tabs[2]:
            results_per_q = []
            for qitem in SAMPLE_QA:
                row = {}
                for name, r in retrievers.items():
                    row[name] = r.search(qitem['question'], top_k=3)
                results_per_q.append(row)
                    
            # simpler compute
            c1 = Evaluator.recall_at_k([results_per_q[t]['fixed'] for t in range(len(SAMPLE_QA))], qa=SAMPLE_QA)
            c2 = Evaluator.recall_at_k([results_per_q[t]['sentence'] for t in range(len(SAMPLE_QA))], qa=SAMPLE_QA)
            c3 = Evaluator.recall_at_k([results_per_q[t]['structure'] for t in range(len(SAMPLE_QA))], qa=SAMPLE_QA)
            st.table({'Strategy': ['fixed_size','sentence_based','structure_aware'], 'Recall@3': [c1,c2,c3]})
            st.caption('Calculated from actual retrieval; no strategy is hardcoded as winner.')
    else:
        if question and not paths:
            st.info('Please select/upload documents first.')
        elif not question and paths:
            st.info('Enter a question to compare.')

if __name__ == '__main__':
    main()

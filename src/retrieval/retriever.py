import numpy as np
from typing import List, Dict, Any
try:
    import faiss
except Exception:
    faiss = None

_MODEL_CACHE: Dict[str, Any] = {}


def get_model(model_name: str = 'all-MiniLM-L6-v2'):
    """Load each SentenceTransformer once and reuse it across retrievers."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = get_model(model_name)
        self.chunks = []
        self.embeddings = None
        self.index = None

    def build(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        texts = [c['text'] for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype('float32'))
        return self

    def search(self, query: str, top_k=3):
        if not self.chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        if self.index is not None:
            scores, idxs = self.index.search(q.astype('float32'), top_k)
            results = []
            for s, i in zip(scores[0], idxs[0]):
                if i < 0: continue
                c = self.chunks[i].copy()
                c['similarity'] = float(s)
                results.append(c)
            return results
        sims = np.dot(self.embeddings, q.T).flatten()
        top = sims.argsort()[::-1][:top_k]
        results = []
        for i in top:
            c = self.chunks[i].copy()
            c['similarity'] = float(sims[i])
            results.append(c)
        return results

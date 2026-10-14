import math
import re
from typing import List, Dict, Any, Set, Tuple

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'of', 'to', 'in', 'on', 'for', 'with', 'and', 'or', 'as', 'at', 'by',
    'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'how', 'why', 'when', 'where', 'does', 'do', 'did', 'can', 'could',
    'would', 'should', 'will', 'from', 'into', 'than', 'then', 'so',
}

# Each question is paired with one or more GOLD snippets copied verbatim from
# the source documents. A retrieved chunk is counted as relevant ONLY if it
# actually contains a gold answer snippet. This is a labelled relevance
# judgement, not a keyword/substring guess.
GOLD_QA = [
    {'question': 'What is supervised learning?',
     'gold': ['Supervised learning uses labeled datasets to train algorithms.']},
    {'question': 'How does reinforcement learning work?',
     'gold': ['Reinforcement learning involves agents learning through interaction with an environment.']},
    {'question': 'What does tokenization do?',
     'gold': ['Tokenization splits text into words or sentences.']},
    {'question': 'What are transformers in NLP?',
     'gold': ['Transformers have revolutionized NLP.']},
    {'question': 'Why use vector databases?',
     'gold': ['Vector databases enable similarity search at scale.']},
    {'question': 'What is unsupervised learning used for?',
     'gold': ['Unsupervised learning analyzes unlabeled data to find hidden patterns.']},
    {'question': 'How is machine learning applied in healthcare?',
     'gold': ['Healthcare uses it for diagnosis and drug discovery.']},
    {'question': 'What are common NLP applications?',
     'gold': ['Applications include translation, chatbots, and summarization.']},
    {'question': 'What is retrieval augmented generation?',
     'gold': ['Retrieval augmented generation relies on vector databases.']},
    {'question': 'What is sentiment analysis?',
     'gold': ['Sentiment analysis determines emotional tone.']},
]


class Evaluator:
    # ---- lexical metric ---------------------------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r'\s+', ' ', (text or '').lower()).strip()

    @staticmethod
    def _stem(word: str) -> str:
        """Light suffix stripping so plurals/verb forms match (whole words only)."""
        w = word.lower().strip("'")
        if len(w) > 4 and w.endswith('ies'):
            w = w[:-3] + 'y'
        elif len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        if len(w) > 5 and w.endswith('ing'):
            w = w[:-3]
        elif len(w) > 4 and w.endswith('ed'):
            w = w[:-2]
        return w

    @staticmethod
    def _terms(text: str) -> List[Tuple[str, str]]:
        words = re.findall(r"[a-zA-Z0-9']+", (text or '').lower())
        return [(w, Evaluator._stem(w)) for w in words
                if w not in STOPWORDS and len(w) > 2]

    @staticmethod
    def query_term_overlap(query: str, text: str, corpus: List[str] = None) -> Dict[str, Any]:
        """Defined lexical metric (distinct from similarity and from accuracy).

        score = (sum of IDF weights of the question's whole-word stemmed terms
                 that appear in the chunk)
                / (sum of IDF weights of all the question's terms)

        IDF is computed over the supplied corpus, so terms that appear in most
        chunks carry little weight. This stops a chunk that merely shares common
        words from scoring 100%. Matching is whole-word after light stemming, so
        'supervised' never matches 'unsupervised'. Returns which terms matched
        and which are missing so the value is auditable.
        """
        q_pairs = Evaluator._terms(query)
        if not q_pairs:
            return {'score': 0.0, 'matched': [], 'missing': []}
        qmap: Dict[str, Set[str]] = {}
        for orig, stem in q_pairs:
            qmap.setdefault(stem, set()).add(orig)
        chunk_stems = {stem for _, stem in Evaluator._terms(text)}
        q_stems = set(qmap)
        matched = q_stems & chunk_stems
        missing = q_stems - chunk_stems

        if corpus:
            docs = [{s for _, s in Evaluator._terms(t)} for t in corpus]
            n = max(1, len(docs))

            def idf(term: str) -> float:
                df = sum(1 for d in docs if term in d)
                return math.log((n + 1) / (df + 1)) + 1.0
        else:
            def idf(term: str) -> float:
                return 1.0

        total = sum(idf(t) for t in q_stems)
        got = sum(idf(t) for t in matched)
        return {
            'score': got / total if total else 0.0,
            'matched': sorted(o for s in matched for o in qmap[s]),
            'missing': sorted(o for s in missing for o in qmap[s]),
        }

    # ---- labelled relevance accuracy -------------------------------------
    @staticmethod
    def is_relevant(chunk_text: str, gold: List[str]) -> bool:
        hay = Evaluator._norm(chunk_text)
        return any(Evaluator._norm(g) in hay for g in gold)

    @staticmethod
    def first_relevant_rank(results: List[Dict[str, Any]], gold: List[str]) -> int:
        for rank, r in enumerate(results, start=1):
            if Evaluator.is_relevant(r['text'], gold):
                return rank
        return 0

    @staticmethod
    def hits_per_question(results_per_q: List[List[Dict[str, Any]]],
                          qa: List[Dict[str, Any]] = GOLD_QA, k: int = 3) -> List[bool]:
        return [any(Evaluator.is_relevant(r['text'], qitem['gold']) for r in res[:k])
                for res, qitem in zip(results_per_q, qa)]

    @staticmethod
    def recall_at_k(results_per_q: List[List[Dict[str, Any]]],
                    qa: List[Dict[str, Any]] = GOLD_QA, k: int = 3) -> float:
        hits = Evaluator.hits_per_question(results_per_q, qa, k)
        return sum(hits) / len(hits) if hits else 0.0

    @staticmethod
    def recall_at_1(results_per_q: List[List[Dict[str, Any]]],
                    qa: List[Dict[str, Any]] = GOLD_QA) -> float:
        return Evaluator.recall_at_k(results_per_q, qa, k=1)

    @staticmethod
    def reciprocal_rank(results_per_q: List[List[Dict[str, Any]]],
                        qa: List[Dict[str, Any]] = GOLD_QA) -> float:
        total = 0.0
        for res, qitem in zip(results_per_q, qa):
            rank = Evaluator.first_relevant_rank(res, qitem['gold'])
            if rank:
                total += 1.0 / rank
        return total / len(qa) if qa else 0.0

    @staticmethod
    def ranks_per_question(results_per_q: List[List[Dict[str, Any]]],
                           qa: List[Dict[str, Any]] = GOLD_QA) -> List[int]:
        return [Evaluator.first_relevant_rank(res, qitem['gold'])
                for res, qitem in zip(results_per_q, qa)]

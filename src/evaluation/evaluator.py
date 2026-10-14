from typing import List, Dict, Any

SAMPLE_QA = [
    {'question': 'What is supervised learning?', 'expected': ['supervised learning', 'labeled datasets', 'classification', 'regression']},
    {'question': 'What is reinforcement learning?', 'expected': ['reinforcement learning', 'rewards', 'penalties', 'agent']},
    {'question': 'What does tokenization do?', 'expected': ['tokenization', 'splits text', 'words', 'sentences']},
    {'question': 'What are transformers in NLP?', 'expected': ['transformers', 'bert', 'gpt', 'revolutionized']},
    {'question': 'Why use vector databases?', 'expected': ['vector databases', 'semantic search', 'similarity search', 'embeddings']},
]

class Evaluator:
    @staticmethod
    def recall_at_k(results_per_q: List[List[Dict[str, Any]]], qa=SAMPLE_QA, k=3):
        hits = 0
        for res, qitem in zip(results_per_q, qa):
            found = False
            topk = res[:k]
            text_all = ' '.join([r['text'].lower() for r in topk])
            for exp in qitem['expected']:
                if exp.lower() in text_all:
                    found = True
                    break
            if found:
                hits += 1
        return hits / len(qa) if qa else 0.0

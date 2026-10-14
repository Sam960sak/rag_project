import re
from typing import List, Dict, Any


class Chunker:
    @staticmethod
    def fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        chunks = []
        if len(text) == 0:
            return chunks
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({'chunk_id': idx, 'text': chunk_text, 'size': len(chunk_text), 'type': 'fixed_size', 'metadata': {'start': start, 'end': min(end, len(text))}})
            idx += 1
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def sentence_based(text: str, target_size: int = 400, overlap_sent: int = 1) -> List[Dict[str, Any]]:
        chunks = []
        if not text:
            return chunks
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        sents = [s for s in sents if s.strip()]
        if not sents:
            return Chunker.fixed_size(text, target_size, int(target_size*0.1))
        cur = []
        cur_len = 0
        idx = 0
        for s in sents:
            sl = len(s)
            if cur_len + sl > target_size and cur:
                chunk_text = ' '.join(cur)
                chunks.append({'chunk_id': idx, 'text': chunk_text, 'size': len(chunk_text), 'type': 'sentence_based', 'metadata': {'sent_count': len(cur)}})
                idx += 1
                if overlap_sent > 0:
                    cur = cur[-overlap_sent:]
                    cur_len = sum(len(x) for x in cur)
                else:
                    cur = []
                    cur_len = 0
            cur.append(s)
            cur_len += sl
        if cur:
            chunk_text = ' '.join(cur)
            chunks.append({'chunk_id': idx, 'text': chunk_text, 'size': len(chunk_text), 'type': 'sentence_based', 'metadata': {'sent_count': len(cur)}})
        return chunks

    @staticmethod
    def structure_aware(text: str, max_size: int = 450) -> List[Dict[str, Any]]:
        chunks = []
        if not text:
            return chunks
        parts = re.split(r'\n\s*\n', text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1:
            parts = re.split(r'\n(?=[A-Z][A-Za-z\s]{2,60})', text)
            parts = [p.strip() for p in parts if p.strip()]
        idx = 0
        for p in parts:
            if len(p) <= max_size:
                chunks.append({'chunk_id': idx, 'text': p, 'size': len(p), 'type': 'structure_aware', 'metadata': {'split': 'para'}})
                idx += 1
            else:
                subs = Chunker.sentence_based(p, target_size=max_size-20)
                for s in subs:
                    s['chunk_id'] = idx
                    s['type'] = 'structure_aware'
                    idx += 1
                    chunks.append(s)
        return chunks

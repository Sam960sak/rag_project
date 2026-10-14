import re
from typing import List, Dict, Any, Optional

DEFAULT_MIN_SIZE = 100


class Chunker:
    @staticmethod
    def _meta(metadata: Optional[Dict[str, Any]], extra: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(metadata or {})
        meta.update(extra)
        return meta

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p.strip()]

    @staticmethod
    def _is_heading(para: str) -> bool:
        p = para.strip()
        if not p or '\n' in p or len(p) > 80 or len(p.split()) > 12:
            return False
        if p[-1] in '.!?;:,':
            return False
        words = p.split()
        caps = sum(1 for w in words if w[:1].isupper())
        return p.isupper() or caps >= max(1, len(words) - 1)

    @staticmethod
    def _merge_small(chunks: List[Dict[str, Any]], min_size: int) -> List[Dict[str, Any]]:
        """Fold chunks shorter than min_size into a neighbour.

        Prevents tiny fragments (e.g. a lone 'x = 5' line) from becoming
        standalone chunks with no context.
        """
        if not chunks or min_size <= 0:
            return chunks
        out: List[Dict[str, Any]] = []
        for c in chunks:
            c = dict(c)
            c['metadata'] = dict(c.get('metadata', {}))
            if out and (c['size'] < min_size or out[-1]['size'] < min_size):
                prev = out[-1]
                prev['text'] = (prev['text'].rstrip() + '\n\n' + c['text'].lstrip()).strip()
                prev['size'] = len(prev['text'])
                if 'end' in c['metadata'] and 'end' in prev['metadata']:
                    prev['metadata']['end'] = c['metadata']['end']
                if 'sent_count' in c['metadata']:
                    prev['metadata']['sent_count'] = prev['metadata'].get('sent_count', 0) + c['metadata']['sent_count']
            else:
                out.append(c)
        if len(out) > 1 and out[0]['size'] < min_size:
            head = out.pop(0)
            out[0]['text'] = (head['text'].rstrip() + '\n\n' + out[0]['text'].lstrip()).strip()
            out[0]['size'] = len(out[0]['text'])
        for i, c in enumerate(out):
            c['chunk_id'] = i
        return out

    @staticmethod
    def fixed_size(text: str, chunk_size: int = 500, overlap: int = 50,
                   metadata: Optional[Dict[str, Any]] = None,
                   min_size: int = DEFAULT_MIN_SIZE) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        text = text or ''
        n = len(text)
        if n == 0:
            return chunks
        start = 0
        idx = 0
        while start < n:
            end = min(start + chunk_size, n)
            if end < n:
                # Prefer cutting after a sentence terminator, so sentences and
                # headings stay intact where possible.
                cut = None
                search_from = start + max(1, chunk_size // 2)
                for m in re.finditer(r'[.!?]\s', text[search_from:end]):
                    cut = search_from + m.end()
                if cut:
                    end = cut
                else:
                    snap = text.rfind(' ', start, end)
                    if snap > start:
                        end = snap
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'chunk_id': idx,
                    'text': chunk_text,
                    'size': len(chunk_text),
                    'type': 'fixed_size',
                    'metadata': Chunker._meta(metadata, {'start': start, 'end': end}),
                })
                idx += 1
            if end >= n:
                break
            new_start = end - overlap
            if new_start <= start:
                new_start = end
            space = text.find(' ', new_start)
            if space != -1 and space < end:
                new_start = space + 1
            start = new_start
        return Chunker._merge_small(chunks, min_size)

    @staticmethod
    def sentence_based(text: str, target_size: int = 400, overlap_sent: int = 1,
                       metadata: Optional[Dict[str, Any]] = None,
                       min_size: int = DEFAULT_MIN_SIZE) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not text:
            return chunks
        sents = Chunker._split_sentences(text)
        if not sents:
            return Chunker.fixed_size(text, target_size, int(target_size * 0.1),
                                      metadata, min_size)
        cur: List[str] = []
        cur_len = 0
        idx = 0
        for s in sents:
            sl = len(s)
            if cur_len + sl > target_size and cur:
                chunk_text = ' '.join(cur)
                chunks.append({
                    'chunk_id': idx, 'text': chunk_text, 'size': len(chunk_text),
                    'type': 'sentence_based',
                    'metadata': Chunker._meta(metadata, {'sent_count': len(cur)}),
                })
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
            chunks.append({
                'chunk_id': idx, 'text': chunk_text, 'size': len(chunk_text),
                'type': 'sentence_based',
                'metadata': Chunker._meta(metadata, {'sent_count': len(cur)}),
            })
        return Chunker._merge_small(chunks, min_size)

    @staticmethod
    def structure_aware(text: str, max_size: int = 450,
                        metadata: Optional[Dict[str, Any]] = None,
                        min_size: int = DEFAULT_MIN_SIZE) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not text:
            return chunks
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if len(paras) <= 1:
            paras = [p.strip() for p in re.split(r'\n(?=[A-Z][A-Za-z\s]{2,60})', text) if p.strip()]

        # Attach each heading to the paragraph that explains it.
        blocks: List[str] = []
        i = 0
        while i < len(paras):
            if Chunker._is_heading(paras[i]) and i + 1 < len(paras):
                blocks.append(paras[i] + '\n\n' + paras[i + 1])
                i += 2
            else:
                blocks.append(paras[i])
                i += 1

        idx = 0
        for block in blocks:
            heading = None
            body = block
            if '\n\n' in block:
                first, rest = block.split('\n\n', 1)
                if Chunker._is_heading(first):
                    heading, body = first, rest
            if len(block) <= max_size:
                chunks.append({
                    'chunk_id': idx, 'text': block, 'size': len(block),
                    'type': 'structure_aware',
                    'metadata': Chunker._meta(metadata, {'split': 'para', 'heading': heading}),
                })
                idx += 1
            else:
                budget = max(120, max_size - (len(heading) + 2 if heading else 0))
                subs = Chunker.sentence_based(body, target_size=budget, metadata=metadata)
                for s in subs:
                    txt = f'{heading}\n\n{s["text"]}' if heading else s['text']
                    s['chunk_id'] = idx
                    s['text'] = txt
                    s['size'] = len(txt)
                    s['type'] = 'structure_aware'
                    s['metadata'] = Chunker._meta(s.get('metadata'),
                                                  {'split': 'sentence', 'heading': heading})
                    idx += 1
                    chunks.append(s)
        return Chunker._merge_small(chunks, min_size)

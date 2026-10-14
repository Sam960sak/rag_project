from src.parsers.document_parser import DocumentParser
from src.chunking.chunkers import Chunker

def test_parser_txt():
    dp = DocumentParser()
    res = dp.parse_file('sample_docs/ml_intro.txt')
    assert 'text' in res and len(res['text']) > 0

def test_fixed_chunks():
    c = Chunker.fixed_size('hello world ' * 10, chunk_size=20, overlap=5)
    assert len(c) > 0
    assert all('text' in x for x in c)

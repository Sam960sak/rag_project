from src.parsers.document_parser import DocumentParser
from src.chunking.chunkers import Chunker
from src.evaluation.evaluator import Evaluator


def test_parser_txt():
    dp = DocumentParser()
    res = dp.parse_file('sample_docs/ml_intro.txt')
    assert 'text' in res and len(res['text']) > 0
    assert res['pages'] and res['pages'][0]['page'] == 1


def test_page_markers_removed():
    cleaned = DocumentParser._clean_text('--- PAGE 3 ---\nHello world')
    assert 'PAGE' not in cleaned
    assert 'Hello world' in cleaned


def test_fixed_chunks_word_safe_and_metadata():
    text = 'word ' * 100
    c = Chunker.fixed_size(text, chunk_size=20, overlap=5,
                           metadata={'source': 'f.txt', 'page': 2})
    assert len(c) > 0
    assert all(set(x['text'].split()) == {'word'} for x in c)
    assert all(x['metadata']['source'] == 'f.txt' for x in c)
    assert all(x['metadata']['page'] == 2 for x in c)


def test_structure_keeps_heading_with_paragraph():
    text = 'Supervised Learning\n\nSupervised learning uses labeled datasets. ' \
           'Common examples include classification and regression.'
    c = Chunker.structure_aware(text)
    assert len(c) == 1
    assert c[0]['text'].startswith('Supervised Learning')
    assert 'labeled datasets' in c[0]['text']


def test_structure_splits_oversized_at_sentences():
    body = ' '.join(f'Sentence number {i} contains useful detail.' for i in range(40))
    text = 'Big Heading\n\n' + body
    c = Chunker.structure_aware(text, max_size=200)
    assert len(c) > 1
    assert all(x['text'].startswith('Big Heading') for x in c)
    assert all(x['text'].rstrip().endswith(('.', '!', '?')) for x in c)


def test_tiny_chunks_are_merged():
    text = 'Heading One\n\nx = 5\n\nHeading Two\n\nThis is a much longer explanatory ' \
           'paragraph that clearly exceeds the minimum chunk size threshold.'
    c = Chunker.structure_aware(text, min_size=100)
    assert all(x['size'] >= 20 for x in c)
    assert 'x = 5' in c[0]['text']


def test_query_term_overlap_no_substring_false_positive():
    q = 'What is supervised learning?'
    sup = Evaluator.query_term_overlap(q, 'Supervised learning uses labeled datasets.', None)
    unsup = Evaluator.query_term_overlap(q, 'Unsupervised learning analyzes unlabeled data.', None)
    assert sup['score'] == 1.0
    assert unsup['score'] < 1.0
    assert 'supervised' not in unsup['matched']


def test_query_term_overlap_idf_downweights_common_terms():
    corpus = [
        'Machine learning is used everywhere.',
        'Machine learning uses labeled datasets for supervised tasks.',
        'Vector databases store embeddings.',
    ]
    cov = Evaluator.query_term_overlap(
        'What is supervised learning?', 'Machine learning is used everywhere.', corpus)
    assert cov['score'] < 1.0
    assert 'learning' in cov['matched']
    assert 'supervised' in cov['missing']


def test_gold_recall_and_mrr():
    qa = [{'question': 'q', 'gold': ['the exact gold sentence.']}]
    rank1 = [[{'text': 'intro the exact gold sentence. more'},
              {'text': 'other'}]]
    rank2 = [[{'text': 'other'}, {'text': 'the exact gold sentence.'}]]
    miss = [[{'text': 'other'}, {'text': 'nope'}]]
    assert Evaluator.recall_at_1(rank1, qa=qa) == 1.0
    assert Evaluator.reciprocal_rank(rank1, qa=qa) == 1.0
    assert Evaluator.recall_at_1(rank2, qa=qa) == 0.0
    assert Evaluator.recall_at_k(rank2, qa=qa, k=3) == 1.0
    assert Evaluator.reciprocal_rank(rank2, qa=qa) == 0.5
    assert Evaluator.recall_at_k(miss, qa=qa, k=3) == 0.0

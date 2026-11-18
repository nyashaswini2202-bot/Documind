from app.bm25_index import BM25Index, tokenize
from app.retriever import rrf


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Quick brown FOX is here") == ["quick", "brown", "fox", "here"]


def test_bm25_ranks_matching_document_first():
    docs = ["cats purr and sleep", "dogs bark loudly at night", "the stock market fell sharply"]
    index = BM25Index(docs)
    top = index.search("stock market", 3)
    assert top[0][0] == 2 and top[0][1] > 0
    assert index.search("zebra", 3) == []


def test_bm25_prefers_rarer_terms():
    docs = ["apple banana", "apple cherry", "apple durian"]
    index = BM25Index(docs)
    assert index.search("apple durian", 3)[0][0] == 2


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    ranked = sorted(fused, key=fused.get, reverse=True)
    assert ranked[:2] == [1, 3]  # appear in both lists
    assert set(ranked) == {1, 2, 3, 4}

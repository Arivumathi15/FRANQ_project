from franq_ext.retrieval import LexicalRetriever


def test_lexical_retriever_ranks_relevant_passage_first():
    corpus = [
        "Albert Einstein was born in 1879 in Ulm, Germany.",
        "Isaac Newton formulated the laws of motion.",
        "Marie Curie researched radioactivity.",
    ]
    r = LexicalRetriever()
    r.index(corpus)
    hits = r.retrieve("Where was Einstein born?", top_k=2)
    assert hits, "expected at least one hit"
    assert "Einstein" in hits[0][0]
    assert hits[0][1] > 0.0


def test_empty_corpus_returns_nothing():
    r = LexicalRetriever()
    r.index([])
    assert r.retrieve("anything") == []

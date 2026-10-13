import asyncio
from ownllm.knowledge.retrievers import BM25Retriever


def test_bm25():
    r = BM25Retriever([
        {"id": "1", "text": "alpha beta"},
        {"id": "2", "text": "gamma delta"},
    ])
    out = asyncio.run(r.search("alpha", 2))
    assert out[0]["id"] == "1"

from ownllm.knowledge.fusion import reciprocal_rank_fusion, Reranker


def test_rrf():
    a = [{"id": "1", "text": "a", "score": 1.0}, {"id": "2", "text": "b", "score": 0.5}]
    b = [{"id": "2", "text": "b", "score": 1.0}, {"id": "3", "text": "c", "score": 0.4}]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0]["id"] in {"1", "2"}
    ranked = Reranker().rerank("q", fused, limit=2)
    assert len(ranked) == 2

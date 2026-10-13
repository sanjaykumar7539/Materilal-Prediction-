import numpy as np
from ownllm.knowledge.vector_store import LocalVectorStore


def test_vector(tmp_path):
    s = LocalVectorStore(tmp_path / "v")
    s.build([[1, 0], [0, 1]], [{"id": "a"}, {"id": "b"}])
    assert s.search(np.array([1.0, 0.0]), 1)[0]["id"] == "a"

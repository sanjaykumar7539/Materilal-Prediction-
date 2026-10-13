from ownllm.knowledge.kag import KnowledgeGraph


def test_graph():
    g = KnowledgeGraph()
    g.add("a", "r", "b")
    g.add("b", "r", "c")
    assert len(g.traverse("a", 2)) == 2

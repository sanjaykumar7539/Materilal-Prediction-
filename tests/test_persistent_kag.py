from ownllm.knowledge.persistent_kag import PersistentKAG


def test_pkag(tmp_path):
    g = PersistentKAG(tmp_path / "g.db")
    g.add("a", "r", "b")
    g.add("b", "r", "c")
    assert len(g.traverse("a", 2)) == 2
    g.close()

from ownllm.data.memmap import write_token_shard, MemmapTokenDataset


def test_memmap(tmp_path):
    p = tmp_path / "a.bin"
    write_token_shard(list(range(101)), p)
    d = MemmapTokenDataset([p], 10)
    x, y = d[0]
    assert len(x) == 10 and int(y[0]) == 1

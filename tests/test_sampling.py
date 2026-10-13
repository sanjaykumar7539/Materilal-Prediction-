import torch
from ownllm.inference.generate import sample


def test_sample_greedy():
    x = torch.tensor([[0.0, 1.0, 2.0]])
    assert sample(x, temperature=0).item() == 2


def test_sample_topk():
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    out = sample(x, temperature=1.0, top_k=2, top_p=1.0)
    assert out.item() in (2, 3)

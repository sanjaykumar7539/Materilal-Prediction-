import torch
from ownllm import OwnLLM, ModelConfig


def test_forward():
    c = ModelConfig(
        vocab_size=128, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_hidden=128, max_seq_len=64,
    )
    m = OwnLLM(c)
    x = torch.randint(0, 128, (2, 16))
    logits, loss, caches = m(x, x)
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(loss)
    assert len(caches) == 2


def test_kv_cache_decode():
    c = ModelConfig(
        vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, max_seq_len=32,
    )
    m = OwnLLM(c).eval()
    prompt = torch.randint(0, 64, (1, 4))
    with torch.inference_mode():
        logits, _, caches = m(prompt, caches=None)
        nxt = logits[:, -1:].argmax(-1)
        logits2, _, caches2 = m(nxt, caches=caches)
    assert logits2.shape[1] == 1
    assert caches2[0][0].shape[2] == 5


def test_num_parameters():
    c = ModelConfig(vocab_size=100, dim=32, n_layers=1, n_heads=4, n_kv_heads=2, ffn_hidden=64)
    m = OwnLLM(c)
    assert m.num_parameters() > 0


def test_cached_decoding_matches_full_sequence_logits():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=64,dim=32,n_layers=2,n_heads=4,n_kv_heads=2,
                      ffn_hidden=64,max_seq_len=32)
    model = OwnLLM(cfg).eval()
    ids = torch.randint(0,64,(1,9))
    with torch.inference_mode():
        full,_,_ = model(ids)
        _,_,cache = model(ids[:,:5])
        chunk,_,cache = model(ids[:,5:8],caches=cache)
        last,_,_ = model(ids[:,8:],caches=cache)
    torch.testing.assert_close(chunk,full[:,5:8],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(last,full[:,8:],atol=1e-6,rtol=1e-5)

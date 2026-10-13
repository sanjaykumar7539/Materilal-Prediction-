"""Regression: chunked prefill (cache + multi-token chunk) must stay causal.

Before the fix, Attention used is_causal=False whenever a cache was present,
letting tokens attend to future positions within the new chunk. A naive
is_causal=True would instead mask out all past cache keys. This test pins the
correct behavior: past keys fully visible, future positions in the new chunk
masked.
"""
import torch

from ownllm import ModelConfig, OwnLLM


def _tiny():
    c = ModelConfig(
        vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, max_seq_len=32,
    )
    return OwnLLM(c).eval()


def test_chunked_prefill_matches_full_forward():
    m = _tiny()
    torch.manual_seed(0)
    ids = torch.randint(0, 64, (1, 8))
    with torch.inference_mode():
        # Full forward in one shot
        full_logits, _, _ = m(ids)
        # Same ids processed in two chunks: first 5 tokens, then remaining 3
        logits_a, _, caches = m(ids[:, :5])
        logits_b, _, _ = m(ids[:, 5:], caches=caches)
    torch.testing.assert_close(logits_b, full_logits[:, 5:], atol=1e-5, rtol=1e-4)


def test_chunked_prefill_cache_grows():
    m = _tiny()
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (1, 6))
    with torch.inference_mode():
        _, _, caches = m(ids[:, :4])
        _, _, caches2 = m(ids[:, 4:], caches=caches)
    assert caches2[0][0].shape[2] == 6  # K cache holds all 6 positions

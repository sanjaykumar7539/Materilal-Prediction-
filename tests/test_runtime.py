"""Smoke test for ModelRuntime with a tiny model and mock tokenizer."""
import asyncio
import torch
from ownllm import OwnLLM, ModelConfig
from ownllm.runtime import ModelRuntime


class MockTokenizer:
    def encode(self, text):
        class Enc:
            ids = [1, 2, 3]

        return Enc()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def token_to_id(self, tok):
        return 0 if tok == "<eos>" else None


def test_runtime_generate():
    c = ModelConfig(
        vocab_size=32, dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        ffn_hidden=64, max_seq_len=32,
    )
    m = OwnLLM(c)
    rt = ModelRuntime(m, MockTokenizer(), device="cpu", max_concurrency=1)

    async def run():
        return await rt.generate_text("hi", max_new_tokens=4, temperature=0)

    text = asyncio.run(run())
    assert isinstance(text, str) and len(text) > 0

"""BPE tokenizer training from local text files (offline)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def train_bpe(
    files: Sequence[str | Path],
    out_dir: str | Path,
    vocab_size: int = 32000,
    special_tokens: Sequence[str] | None = None,
):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([str(x) for x in files], trainer)
    tok.save(str(out / "tokenizer.json"))
    return tok

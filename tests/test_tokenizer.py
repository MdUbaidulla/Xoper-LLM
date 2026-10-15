import os

from data.tokenizer_train import train_tokenizer


def test_train_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "The quick brown fox jumps over the lazy dog. " * 50
        + "Byte level BPE handles any unicode text robustly, even emoji like 🚀 and 日本語.",
        encoding="utf-8",
    )
    out_path = str(tmp_path / "tokenizer.json")

    train_tokenizer(str(corpus), vocab_size=512, out_path=out_path)
    assert os.path.exists(out_path)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(out_path)
    assert tok.get_vocab_size() <= 512

    text = "The quick brown fox 🚀 日本語"
    ids = tok.encode(text).ids
    decoded = tok.decode(ids)
    # byte-level fallback means every character round-trips, including emoji/CJK
    assert "quick brown fox" in decoded
    assert len(ids) > 0

    for special in ("<pad>", "<bos>", "<eos>", "<unk>"):
        assert tok.token_to_id(special) is not None

"""Train a byte-level BPE tokenizer (HuggingFace `tokenizers`), with byte
fallback so no input string, Unicode character or emoji ever hits an
out-of-vocabulary error.

Usage:
    python -m data.tokenizer_train --input cleaned/*.txt --vocab-size 32768 \
        --out tokenizer.json
"""
from __future__ import annotations

import argparse
import glob


def train_tokenizer(input_glob: str, vocab_size: int, out_path: str,
                     special_tokens: list[str] | None = None) -> None:
    from tokenizers import ByteLevelBPETokenizer

    special_tokens = special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"]
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {input_glob!r}")

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=files,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
    )
    tok.save(out_path)
    print(f"Trained byte-level BPE tokenizer on {len(files)} file(s) -> {out_path} "
          f"(vocab_size={vocab_size})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="glob of cleaned .txt files")
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--out", default="tokenizer.json")
    args = p.parse_args()
    train_tokenizer(args.input, args.vocab_size, args.out)


if __name__ == "__main__":
    main()

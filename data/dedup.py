"""MinHash + LSH near-duplicate detection, run across shards before tokenizing.

Pure-Python, dependency-free implementation (no `datasketch` required)
so it runs anywhere this repo does. At real (billion-document) scale,
swap this for a distributed MinHashLSH job (e.g. `datasketch` + Spark,
or `text-dedup`), keeping the same shingle/hash/band parameters.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+")

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _shingles(text: str, n: int = 5) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _sha_to_int(s: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{s}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & _MAX_HASH


@dataclass
class MinHasher:
    num_perm: int = 64
    shingle_size: int = 5

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = _shingles(text, self.shingle_size)
        if not shingles:
            return tuple(_MAX_HASH for _ in range(self.num_perm))
        return tuple(min(_sha_to_int(sh, seed) for sh in shingles) for seed in range(self.num_perm))


@dataclass
class LSHIndex:
    """Banded LSH: a document is a near-duplicate candidate of another if any band matches."""

    num_perm: int = 64
    bands: int = 16
    _rows: int = field(init=False)
    _buckets: dict[tuple[int, tuple[int, ...]], list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_perm % self.bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self._rows = self.num_perm // self.bands

    def _band_keys(self, sig: tuple[int, ...]):
        for b in range(self.bands):
            yield (b, sig[b * self._rows : (b + 1) * self._rows])

    def query_candidates(self, sig: tuple[int, ...]) -> set[str]:
        candidates: set[str] = set()
        for key in self._band_keys(sig):
            candidates.update(self._buckets.get(key, []))
        return candidates

    def insert(self, doc_id: str, sig: tuple[int, ...]) -> None:
        for key in self._band_keys(sig):
            self._buckets.setdefault(key, []).append(doc_id)


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


def deduplicate(
    documents: list[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 16,
    shingle_size: int = 5,
) -> list[int]:
    """Returns the indices of `documents` to KEEP (first occurrence of each near-duplicate cluster)."""
    hasher = MinHasher(num_perm, shingle_size)
    index = LSHIndex(num_perm, bands)
    signatures: dict[str, tuple[int, ...]] = {}
    keep: list[int] = []

    for i, doc in enumerate(documents):
        doc_id = str(i)
        sig = hasher.signature(doc)
        is_dup = False
        for cand_id in index.query_candidates(sig):
            if jaccard_estimate(sig, signatures[cand_id]) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
            signatures[doc_id] = sig
            index.insert(doc_id, sig)
    return keep

"""Heuristic filtering: strip boilerplate, near-gibberish and low-signal lines/docs.

Mirrors stage 1 of the data pipeline: drop lines with low word counts,
excessive punctuation, or repetitive character strings, before any
language-ID or dedup work runs on them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCT_RUN = re.compile(r"[^\w\s]{4,}")
_REPEATED_CHAR = re.compile(r"(.)\1{6,}")
_WORD = re.compile(r"\b\w+\b")


@dataclass
class FilterConfig:
    min_words_per_line: int = 3
    min_avg_word_len: float = 2.0
    max_punct_run: int = 4
    max_symbol_ratio: float = 0.3


def is_low_quality_line(line: str, cfg: FilterConfig = FilterConfig()) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    words = _WORD.findall(stripped)
    if len(words) < cfg.min_words_per_line:
        return True
    if _PUNCT_RUN.search(stripped):
        return True
    if _REPEATED_CHAR.search(stripped):
        return True
    avg_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_len < cfg.min_avg_word_len:
        return True
    symbol_chars = sum(1 for c in stripped if not (c.isalnum() or c.isspace()))
    if symbol_chars / max(len(stripped), 1) > cfg.max_symbol_ratio:
        return True
    return False


def filter_document(text: str, cfg: FilterConfig = FilterConfig()) -> str:
    """Drops low-quality lines and returns the cleaned document (possibly empty)."""
    kept = [ln for ln in text.splitlines() if not is_low_quality_line(ln, cfg)]
    return "\n".join(kept)


def document_survives(text: str, cfg: FilterConfig = FilterConfig(), min_kept_ratio: float = 0.3) -> bool:
    """A whole document is dropped if filtering would gut most of its lines (likely boilerplate-heavy)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    kept = sum(1 for ln in lines if not is_low_quality_line(ln, cfg))
    return (kept / len(lines)) >= min_kept_ratio

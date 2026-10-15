"""Lightweight language identification.

Uses a fast character-script heuristic by default (no model download
required) and optionally defers to `fasttext`'s lid.176 model when
installed and a model path is provided, matching the "fast text
classifier" stage described in the architecture doc.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_SCRIPT_RANGES = {
    "bn": [(0x0980, 0x09FF)],   # Bengali
    "ar": [(0x0600, 0x06FF)],   # Arabic
    "zh": [(0x4E00, 0x9FFF)],   # CJK
    "ja": [(0x3040, 0x30FF)],   # Hiragana/Katakana
    "ko": [(0xAC00, 0xD7A3)],   # Hangul
    "ru": [(0x0400, 0x04FF)],   # Cyrillic
    "hi": [(0x0900, 0x097F)],   # Devanagari
}

_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if ch.isspace():
            continue
        matched = False
        for lang, ranges in _SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[lang] += 1
                matched = True
                break
        if not matched and unicodedata.category(ch).startswith("L"):
            counts["en"] += 1  # Latin / other -> treat as English-ish bucket
    return counts


def detect_language(text: str, min_confidence: float = 0.5) -> tuple[str | None, float]:
    """Returns (lang_code, confidence) using script majority; (None, 0.0) if too short/mixed."""
    counts = _script_counts(text[:2000])
    total = sum(counts.values())
    if total < 20:
        return None, 0.0
    lang, n = counts.most_common(1)[0]
    conf = n / total
    if conf < min_confidence:
        return None, conf
    return lang, conf


class FastTextLangID:
    """Optional wrapper around a real fastText lid.176.bin model, if available."""

    def __init__(self, model_path: str) -> None:
        import fasttext  # type: ignore

        self.model = fasttext.load_model(model_path)

    def detect(self, text: str) -> tuple[str, float]:
        labels, probs = self.model.predict(text.replace("\n", " ")[:2000], k=1)
        lang = labels[0].replace("__label__", "")
        return lang, float(probs[0])

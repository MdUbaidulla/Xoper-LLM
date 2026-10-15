"""Regex-based PII scrubbing: emails, phone numbers, and ID-like number strings.

Not a substitute for a trained PII model at real scale, but catches
the common, cheap-to-match cases before text reaches training shards.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{3,4}(?!\d)")
_LONG_DIGIT_RUN = re.compile(r"\b\d{9,}\b")  # catches SSN/ID-like runs
_URL_WITH_CREDS = re.compile(r"[a-zA-Z][\w+.-]*://[^/\s:]+:[^/\s@]+@")


def scrub_pii(text: str) -> str:
    text = _URL_WITH_CREDS.sub("[REDACTED_URL]", text)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _LONG_DIGIT_RUN.sub("[REDACTED_ID]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    return text

from data.dedup import deduplicate
from data.filters import FilterConfig, document_survives, filter_document, is_low_quality_line
from data.lang_id import detect_language
from data.pii import scrub_pii


def test_low_quality_line_detection():
    assert is_low_quality_line("")
    assert is_low_quality_line("ok")
    assert is_low_quality_line("!!!??!!***###")
    assert is_low_quality_line("aaaaaaaaaaaaaaaaaaa i i i i")
    assert not is_low_quality_line("The quick brown fox jumps over the lazy dog.")


def test_filter_document_drops_boilerplate_lines():
    text = "This is a real sentence with real words.\n!!!\n\nAnother good sentence here today."
    cleaned = filter_document(text)
    assert "real sentence" in cleaned
    assert "!!!" not in cleaned


def test_document_survives_threshold():
    mostly_junk = "\n".join(["!!!"] * 9 + ["One real sentence with enough words in it."])
    assert not document_survives(mostly_junk, min_kept_ratio=0.3)

    mostly_good = "\n".join(["A perfectly normal sentence of real words."] * 9 + ["!!!"])
    assert document_survives(mostly_good, min_kept_ratio=0.3)


def test_scrub_pii_redacts_email_and_phone():
    text = "Contact me at jane.doe@example.com or call +1 415-555-0199."
    scrubbed = scrub_pii(text)
    assert "jane.doe@example.com" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed


def test_scrub_pii_redacts_long_id_numbers():
    text = "My account number is 123456789012 for reference."
    scrubbed = scrub_pii(text)
    assert "123456789012" not in scrubbed
    assert "[REDACTED_ID]" in scrubbed


def test_dedup_keeps_unique_drops_near_duplicate():
    docs = [
        "The quick brown fox jumps over the lazy dog near the river bank today.",
        "The quick brown fox jumps over the lazy dog near the river bank now.",  # near-dup
        "Completely different content about rockets and space travel exploration.",
    ]
    keep = deduplicate(docs, threshold=0.6)
    assert 0 in keep
    assert 2 in keep
    assert len(keep) == 2  # the near-duplicate of doc 0 is dropped


def test_detect_language_distinguishes_scripts():
    bengali_sample = "আমি বাংলায় কথা বলি এবং এটি আমার মাতৃভাষা।" * 5
    lang, conf = detect_language(bengali_sample)
    assert lang == "bn"
    assert conf > 0.5

    english_sample = "This is a long piece of plain English text for testing the detector." * 3
    lang, conf = detect_language(english_sample)
    assert lang == "en"

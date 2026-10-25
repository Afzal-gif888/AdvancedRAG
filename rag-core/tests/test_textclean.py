"""Regression tests for every text-hygiene rule: what it fixes AND what it must leave alone."""
import pytest

from rag_core import textclean as tc
from rag_core.parsing import normalize


# ---------------------------------------------------------------- characters

def test_ligatures_become_letters():
    assert tc.clean_chars("eﬀective oﬃce ﬁrst ﬂow diﬃculty") == "effective office first flow difficulty"


def test_invisible_characters_and_exotic_spaces():
    assert tc.clean_chars("co­mputer​ zero﻿width") == "computer zerowidth"
    assert tc.clean_chars("a b c") == "a b c"


def test_nfc_but_not_nfkc():
    # e + combining acute -> precomposed e-acute (NFC) ...
    assert tc.clean_chars("café") == "café"
    # ... but compatibility forms that carry meaning are preserved (NFKC would rewrite them)
    assert tc.clean_chars("x² ½ ①") == "x² ½ ①"


def test_code_and_math_characters_survive():
    src = "if a <= b && c != d: x = y ** 2  # ≤ ∑ √"
    assert tc.clean_chars(src) == src


# ---------------------------------------------------------------- dehyphenation

def test_dehyphenate_joins_a_split_word():
    out, n = tc.dehyphenate("The inter-\nnational market and the na-\ntional bank.")
    assert out == "The international market and the national bank." and n == 2


def test_dehyphenate_keeps_real_compounds_that_appear_elsewhere():
    text = "A well-known fact. Another well-\nknown author."
    out, n = tc.dehyphenate(text)
    assert "well-known author" in out and "wellknown" not in out


def test_dehyphenate_leaves_identifiers_urls_numbers_and_code_alone():
    for src in [
        "see https://exam-\nple.org/path",          # URL
        "ticket ABC-\n1234 was closed",            # identifier with digits
        "call snake_case-\nname now",              # underscore => identifier
        "write to dev-\nops@example.com today",    # e-mail
        "COVID-\n19 cases",                        # upper-case / digits
    ]:
        out, n = tc.dehyphenate(src)
        assert out == src and n == 0, src


def test_dehyphenate_skips_fenced_code_blocks():
    text = "```\nx = foo-\nbar\n```\nThe inter-\nnational plan."
    out, n = tc.dehyphenate(text)
    assert "x = foo-\nbar" in out and "international plan" in out and n == 1


def test_dehyphenate_does_not_touch_dashes_in_prose():
    src = "a result - however\nunexpected - arrived; pre-\n2020 data"
    out, _ = tc.dehyphenate(src)
    assert "result - however" in out


# ---------------------------------------------------------------- repeated headers / footers

WORDS = ["river", "stone", "cloud", "forest", "copper", "violet", "harbor", "meadow", "lantern", "marble", "falcon", "orchard"]


def _body(i):
    """Five different lines per page (different words, so no two pages share a body line)."""
    return "\n".join(
        f"The {WORDS[(i + j) % 12]} beside the {WORDS[(i * 3 + j) % 12]} held {WORDS[(i + 5 * j) % 12]} number {i}{j}." for j in range(5)
    )


def _pages(n):
    return [f"ACME Corp Confidential\n{_body(i)}\nPage {i} of {n}" for i in range(1, n + 1)]


def test_repeated_header_and_page_number_footer_are_removed():
    cleaned, removed = tc.remove_repeated_margins(_pages(6))
    assert removed == 12
    assert all("ACME" not in p and "Page " not in p for p in cleaned)
    assert cleaned[1].split("\n") == _body(2).split("\n")   # the body is untouched


def test_body_text_that_repeats_in_the_middle_is_kept():
    pages = [
        f"Title {chr(65 + i)}\nalpha {chr(65 + i)}\nbeta {chr(70 + i)}\nThe standard disclaimer appears here.\n"
        f"gamma {chr(75 + i)}\ndelta {chr(80 + i)}\nEnding {chr(85 + i)}"
        for i in range(6)
    ]
    cleaned, removed = tc.remove_repeated_margins(pages)
    assert removed == 0 and all("standard disclaimer" in p for p in cleaned)


def test_no_removal_for_short_documents_or_infrequent_lines():
    pages = _pages(2)
    assert tc.remove_repeated_margins(pages) == (pages, 0)
    mixed = [f"{h}\n{_body(i)}\nfooter {chr(65 + i)}" for i, h in enumerate(["Header", "Other", "Header", "Third", "Fourth"])]
    assert tc.remove_repeated_margins(mixed)[1] == 0  # "Header" is on 2 of 5 pages: below 60%


def test_short_pages_are_never_stripped():
    pages = ["Header\nonly two lines"] * 6   # fewer than 5 lines: the margins would be the entire body
    assert tc.remove_repeated_margins(pages) == (pages, 0)


# ---------------------------------------------------------------- unwrapping

def test_wrapped_paragraph_is_rejoined():
    wrapped = "This long sentence was wrapped by the page layout and\ncontinues on the next line until it finally ends."
    assert tc.unwrap_lines(wrapped) == "This long sentence was wrapped by the page layout and continues on the next line until it finally ends."


def test_lists_tables_headings_and_code_are_never_joined():
    for src in [
        "1. First item in a numbered list that is quite long indeed\n2. second item continues here",
        "- a bullet that is long enough to look like a wrapped line of text\n- next bullet",
        "Property: Battery life | Value: 14 months and more text\nproperty: weight | value: 85 grams in total",
        "Short heading\ncontinues in lower case",                       # first line too short to be a wrapped line
        "A sentence that ends properly and is long enough to count.\nnext line starts lower case",
        "while (x < 10) { y = compute_something_long_enough(x); }\nrest",
    ]:
        assert tc.unwrap_lines(src) == src, src


def test_code_fences_are_left_alone_by_unwrap():
    src = "text that wraps in the middle of a long sentence and\ncontinues here.\n\n```\nline one of the code that is long enough\nline two\n```"
    out = tc.unwrap_lines(src)
    assert "line one of the code that is long enough\nline two" in out
    assert "wraps in the middle of a long sentence and continues here." in out


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_input_is_stable(text):
    assert tc.clean_chars(text) == text
    assert tc.dehyphenate(text) == (text, 0)
    assert tc.unwrap_lines(text) == text
    assert normalize(text) == ""

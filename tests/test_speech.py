# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for speech normalisation.

These encode a real failure: the agent was explicitly told to speak American
units and avoid symbols, and still said "28 degrees Celsius ... 50% ...
6 kilometers per hour". Prompt instructions were not reliable, so the
behaviour moved into code and is pinned here.
"""

from __future__ import annotations

import pytest

from homeai.speech import chunk_for_speech, split_sentences, flatten_markdown, convert_units, expand_symbols, normalise_for_speech


# -- temperature -----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("28 degrees Celsius", "82 degrees"),
        ("28°C", "82 degrees"),
        ("28 °C", "82 degrees"),
        ("0 degrees celsius", "32 degrees"),
        ("-5 degrees Celsius", "23 degrees"),
        ("100 degrees Celsius", "212 degrees"),
    ],
)
def test_celsius_is_converted_to_fahrenheit(text, expected):
    assert convert_units(text) == expected


def test_the_actual_reported_weather_sentence():
    said = (
        "It's partly cloudy today in Lilburn, with a temperature of 28 degrees "
        "Celsius. The humidity is at 50%, and there's a light breeze from the "
        "east at 6 kilometers per hour."
    )
    out = normalise_for_speech(said)
    assert "82 degrees" in out
    assert "Celsius" not in out
    assert "50 percent" in out
    assert "%" not in out
    assert "3.7 miles per hour" in out
    assert "kilometer" not in out


# -- distance, speed, mass -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10 kilometers", "6.2 miles"),
        ("100 km", "62 miles"),
        ("6 kilometers per hour", "3.7 miles per hour"),
        ("100 km/h", "62 miles per hour"),
        ("50 kph", "31 miles per hour"),
        ("10 centimeters", "3.9 inches"),
        ("25 mm", "1 inches"),  # 0.98 rounds to a clean "1"
        ("10 kilograms", "22 pounds"),
    ],
)
def test_metric_units_convert(text, expected):
    assert convert_units(text) == expected


def test_conversion_can_be_disabled():
    out = normalise_for_speech("28 degrees Celsius", us_units=False)
    assert "Celsius" in out


# -- symbols ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("humidity is 50%", "50 percent"),
        ("it is 75°F outside", "75 degrees"),
        ("see https://example.com/x for more", "a link"),
        ("**important** point", "important"),
        ("## Heading", "Heading"),
    ],
)
def test_symbols_are_spoken_not_shown(text, fragment):
    assert fragment in expand_symbols(text)


def test_no_raw_symbols_survive():
    out = normalise_for_speech("It is 30°C with 80% humidity & rising")
    for bad in ("°", "%", "&"):
        assert bad not in out


# -- must not mangle ordinary prose ---------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The capital of France is Paris.",
        "Atlanta is warm today.",
        "A heat pump moves heat rather than creating it.",
        "Your meeting is at three o'clock.",
        "Vitamin C is good for you.",
    ],
)
def test_ordinary_prose_is_untouched(text):
    assert normalise_for_speech(text) == text


def test_standalone_c_as_a_letter_is_not_read_as_celsius():
    # "Vitamin C" must not become "Vitamin 33.8 degrees".
    assert "degrees" not in normalise_for_speech("Vitamin C is good for you.")


def test_empty_input_is_safe():
    assert normalise_for_speech("") == ""
    assert convert_units("") == ""
    assert expand_symbols("") == ""


# -- truncation limit ------------------------------------------------------


def test_long_answers_are_not_truncated_at_600_chars():
    """Regression: the old 600-char cap silently cut detailed answers in half."""
    from homeai.safety import sanitise_for_speech

    long_answer = "A heat pump moves heat rather than generating it. " * 25
    assert len(long_answer) > 1000
    out = sanitise_for_speech(long_answer)
    assert len(out) > 1000, "detailed answers must survive intact"


class TestFlattenMarkdown:
    """Markdown flattening must run before whitespace is collapsed.

    Every case here is a construct the benchmark observed the live model
    emitting despite a system prompt forbidding markdown.
    """

    def test_dash_bullets_become_sentences(self):
        out = flatten_markdown("- check the oil\n- check the tyres")
        assert out == "check the oil.\ncheck the tyres."

    def test_asterisk_and_unicode_bullets(self):
        assert flatten_markdown("* one\n\u2022 two") == "one.\ntwo."

    def test_numbered_lists(self):
        out = flatten_markdown("1. first thing\n2) second thing")
        assert out == "first thing.\nsecond thing."

    def test_existing_punctuation_not_doubled(self):
        assert flatten_markdown("- already done.") == "already done."
        assert flatten_markdown("- a question?") == "a question?"

    def test_trailing_comma_replaced_not_appended(self):
        assert flatten_markdown("- eggs,") == "eggs."

    def test_headings_become_sentences(self):
        assert flatten_markdown("## Before You Go\ntext") == "Before You Go.\ntext"

    def test_blockquote_marker_removed(self):
        assert flatten_markdown("> quoted text") == "quoted text"

    def test_table_becomes_commas(self):
        out = flatten_markdown("| a | b |\n|---|---|\n| 1 | 2 |")
        assert "|" not in out
        assert "a , b" in out.replace("  ", " ")

    def test_hyphen_mid_sentence_is_untouched(self):
        """The marker rule is line-anchored; prose hyphens must survive."""
        text = "a well-known fact - and a dash - stays put"
        assert flatten_markdown(text) == text

    def test_negative_number_at_line_start_survives(self):
        """'-5 degrees' must not be read as a bullet whose content is '5'."""
        assert flatten_markdown("-5 degrees outside") == "-5 degrees outside"

    def test_decimal_list_like_prose_untouched(self):
        assert flatten_markdown("3.5 miles to go") == "3.5 miles to go"

    def test_empty_and_plain_text(self):
        assert flatten_markdown("") == ""
        assert flatten_markdown("just a sentence.") == "just a sentence."

    def test_full_chain_produces_speakable_text(self):
        """End-to-end in the daemon's real order."""
        from homeai.safety import sanitise_for_speech

        raw = "## Trip Checklist\n- tyre pressure at 32 psi\n- coolant, 2 liters\n"
        out = normalise_for_speech(sanitise_for_speech(flatten_markdown(raw)))
        assert "-" not in out
        assert "#" not in out
        assert "Trip Checklist. tyre pressure" in out


class TestChunking:
    """Chunking is what makes a long answer interruptible.

    A single 68-second `aplay` call had no seam to stop at; these tests
    protect the seams and the things that must not become seams.
    """

    def test_splits_on_sentence_ends(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_abbreviations_do_not_split(self):
        """Splitting at 'Dr.' produces an audible stumble mid-phrase."""
        assert split_sentences("Dr. Smith left.") == ["Dr. Smith left."]
        assert split_sentences("It was 3 p.m. then.") == ["It was 3 p.m. then."]

    def test_empty_input(self):
        assert split_sentences("") == []
        assert chunk_for_speech("") == []

    def test_short_text_is_one_chunk(self):
        assert chunk_for_speech("Just a short reply.") == ["Just a short reply."]

    def test_long_text_is_split(self):
        chunks = chunk_for_speech("This is a sentence. " * 30)
        assert len(chunks) > 1

    def test_chunks_respect_target_size(self):
        for chunk in chunk_for_speech("A sentence here. " * 40, target_chars=200):
            assert len(chunk) <= 400

    def test_runaway_sentence_is_still_broken_up(self):
        """One endless sentence must not restore the uninterruptible block."""
        runaway = "word " * 400
        chunks = chunk_for_speech(runaway, target_chars=200, max_chars=300)
        assert len(chunks) > 1
        assert all(len(c) <= 300 for c in chunks)

    def test_no_text_is_lost(self):
        text = "First sentence. Second sentence. Third sentence."
        assert "".join(chunk_for_speech(text).__iter__()).replace(" ", "") == \
            text.replace(" ", "")

    def test_clause_split_prefers_punctuation(self):
        long_clause = "alpha, " * 50
        chunks = chunk_for_speech(long_clause, target_chars=100, max_chars=150)
        assert any(c.rstrip().endswith(",") for c in chunks)

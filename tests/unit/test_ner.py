"""Unit tests for taxonomy skill matching."""

import pytest

from src.features.ner import (
    ExtractionResult,
    _build_automaton,
    _is_whole_word,
    _search_with_regex,
    load_taxonomy,
)

SKILLS = {"java", "javascript", "go", "r", "react", "react-native", "sql", "c++"}

automaton_required = pytest.mark.skipif(
    _build_automaton({"probe"}) is None, reason="pyahocorasick is not installed"
)


def _search_with_automaton(text: str, skills: set[str]) -> set[str]:
    from src.features.ner import _search_with_automaton as search

    return search(text, _build_automaton(skills))


# ── Word boundaries ───────────────────────────────────────────────────────────


def test_is_whole_word_accepts_a_standalone_token():
    assert _is_whole_word("i write go daily", 8, 10)


def test_is_whole_word_rejects_an_embedded_token():
    assert not _is_whole_word("algorithms are fun", 4, 6)


def test_is_whole_word_treats_hyphens_as_part_of_the_token():
    """'react' inside 'react-native' belongs to the longer skill, not this one."""
    assert not _is_whole_word("react-native app", 0, 5)


# ── Regex matcher ─────────────────────────────────────────────────────────────


def test_regex_matcher_finds_standalone_skills():
    text = "Built a JavaScript and React frontend. Wrote services in Go."
    assert _search_with_regex(text, SKILLS) == {"javascript", "react", "go"}


def test_regex_matcher_does_not_match_substrings():
    """
    Regression: only skills of three characters or fewer were word-bounded, so
    'java' matched inside 'javascript' and 'go' inside 'algorithms'.
    """
    found = _search_with_regex("javascript algorithms", SKILLS)
    assert "java" not in found
    assert "go" not in found
    assert found == {"javascript"}


def test_regex_matcher_is_case_insensitive():
    assert "sql" in _search_with_regex("Wrote complex SQL queries", SKILLS)


def test_regex_matcher_finds_single_letter_skills():
    assert "r" in _search_with_regex("Statistical modelling in R", SKILLS)
    assert "r" not in _search_with_regex("Wrote a report", SKILLS)


def test_regex_matcher_handles_symbols_in_skill_names():
    assert "c++" in _search_with_regex("Ten years of C++ development", SKILLS)


def test_regex_matcher_returns_empty_for_unrelated_text():
    assert _search_with_regex("Pastry chef and kitchen manager", SKILLS) == set()


# ── Automaton matcher ─────────────────────────────────────────────────────────


@automaton_required
def test_automaton_matches_the_regex_matcher():
    text = "Built a JavaScript and React frontend. Wrote services in Go, stats in R."
    assert _search_with_automaton(text, SKILLS) == _search_with_regex(text, SKILLS)


@automaton_required
def test_automaton_checks_boundaries_at_the_matched_position():
    """
    Regression: the position was re-derived with str.find, which returns the
    first occurrence in the document. Here 'go' appears inside 'algorithms'
    before it appears standalone, so the boundary check ran on the wrong one and
    the real match was discarded.
    """
    text = "algorithms are fun. I write Go daily."
    assert "go" in _search_with_automaton(text, {"go"})


@automaton_required
def test_automaton_rejects_embedded_occurrences():
    assert _search_with_automaton("algorithms everywhere", {"go"}) == set()


@automaton_required
def test_automaton_returns_none_for_an_empty_taxonomy():
    assert _build_automaton(set()) is None


# ── Taxonomy ──────────────────────────────────────────────────────────────────


def test_taxonomy_loads_three_non_empty_lowercase_sets():
    technical, soft, certifications = load_taxonomy()

    assert technical and soft and certifications
    for group in (technical, soft, certifications):
        assert all(skill == skill.lower() for skill in group)


# ── ExtractionResult ──────────────────────────────────────────────────────────


def test_extraction_result_defaults_are_empty():
    result = ExtractionResult()
    assert result.technical_skills == []
    assert result.experience_years == 0.0
    assert result.education_level == "none"


def test_all_skills_concatenates_technical_and_soft():
    result = ExtractionResult(
        technical_skills=["python", "sql"], soft_skills=["communication"]
    )
    assert result.all_skills == ["python", "sql", "communication"]


def test_to_dict_exposes_every_field():
    payload = ExtractionResult(technical_skills=["python"]).to_dict()
    assert set(payload) == {
        "technical_skills",
        "soft_skills",
        "experience_years",
        "education_level",
        "job_titles",
        "certifications",
        "raw_entities",
    }

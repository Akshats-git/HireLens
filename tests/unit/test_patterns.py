"""Unit tests for education and experience extraction."""

import pytest

from src.features.patterns import (
    DEGREE_ORDER,
    DEGREE_RANK,
    detect_education_level,
    extract_experience_years,
    required_experience_years,
)


# ── Education detection ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PhD in Computer Science", "phd"),
        ("Ph.D. candidate at MIT", "phd"),
        ("Doctorate in Physics", "phd"),
        ("Master of Science in Data Science", "masters"),
        ("M.S. Statistics, 2019", "masters"),
        ("MBA, Wharton School", "masters"),
        ("Bachelor of Engineering", "bachelors"),
        ("B.S. Computer Science, State University", "bachelors"),
        ("BS Computer Science", "bachelors"),
        ("B.E. Mechanical Engineering", "bachelors"),
        ("B.Tech in CSE", "bachelors"),
        ("Associate degree in Nursing", "associate"),
        ("Diploma in Hospitality Management", "diploma"),
        ("AWS Certified Solutions Architect", "certification"),
        ("Completed a 12-week coding bootcamp", "bootcamp"),
    ],
)
def test_detects_stated_degree(text, expected):
    assert detect_education_level(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I would be happy to join your team.",
        "Worked as a senior engineer at Acme Corp.",
        "Reduced latency to 200 ms under peak load.",
        "Responsibilities included on-call rotation.",
    ],
)
def test_common_words_are_not_degrees(text):
    """
    Regression: the previous patterns allowed undotted two-letter abbreviations,
    so 'be' matched bachelors, 'as' matched associate, and 'ms' matched masters.
    """
    assert detect_education_level(text) == "none"


def test_highest_degree_wins_when_several_are_listed():
    resume = "B.S. Computer Science, 2016. M.S. Machine Learning, 2018."
    assert detect_education_level(resume) == "masters"


def test_default_is_returned_when_nothing_matches():
    assert detect_education_level("No schooling mentioned.") == "none"
    assert detect_education_level("No schooling mentioned.", default="bachelors") == (
        "bachelors"
    )


def test_degree_rank_orders_lowest_to_highest():
    assert DEGREE_RANK["none"] == 0
    assert DEGREE_RANK["phd"] == len(DEGREE_ORDER) - 1
    assert DEGREE_RANK["bachelors"] < DEGREE_RANK["masters"] < DEGREE_RANK["phd"]
    assert DEGREE_RANK["bootcamp"] < DEGREE_RANK["bachelors"]


# ── Experience extraction ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5 years of experience", 5.0),
        ("3+ years experience in Python", 3.0),
        ("3-5 years of experience", 3.0),
        ("3 to 5 years experience", 3.0),
        ("2.5 years of work", 2.5),
        ("10 years of exp", 10.0),
    ],
)
def test_extracts_stated_duration(text, expected):
    assert extract_experience_years(text) == expected


def test_returns_largest_duration_mentioned():
    resume = "2 years of work at Acme, then 7 years of experience at Globex."
    assert extract_experience_years(resume) == 7.0


def test_returns_zero_when_no_duration_is_stated():
    assert extract_experience_years("Software engineer at Acme.") == 0.0
    assert extract_experience_years("") == 0.0


def test_requirement_uses_the_first_duration_not_the_largest():
    """A posting states its requirement up front; later figures are incidental."""
    posting = (
        "3+ years of experience required. Our team has 20 years of work behind it."
    )
    assert required_experience_years(posting) == 3.0
    assert extract_experience_years(posting) == 20.0


def test_requirement_is_none_when_unstated():
    assert required_experience_years("Join our growing team.") is None

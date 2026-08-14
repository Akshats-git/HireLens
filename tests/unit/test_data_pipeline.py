"""Unit tests for text cleaning, pair construction, and evaluation-set scoring."""

import numpy as np
import pandas as pd
import pytest

from src.data.eval_dataset import (
    build_resume_text,
    compute_match_score,
    parse_skills,
    score_education_fit,
    score_experience_fit,
    score_skill_overlap,
    score_title_keyword_match,
)
from src.data.preprocessing import (
    build_training_pairs,
    clean_text,
    detect_sections,
    has_required_sections,
    split_pairs,
)


# ── Text cleaning ─────────────────────────────────────────────────────────────


def test_clean_text_returns_empty_for_blank_input():
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_clean_text_strips_contact_details():
    cleaned = clean_text(
        "Jane Doe\njane.doe@example.com\n+1 (555) 123-4567\nhttps://example.com/jane"
    )
    assert "@" not in cleaned
    assert "example.com" not in cleaned
    assert "555" not in cleaned
    assert "Jane Doe" in cleaned


def test_clean_text_normalises_unicode_punctuation():
    cleaned = clean_text("Jane’s role – building “things” • daily")
    assert "’" not in cleaned and "'" in cleaned
    assert "–" not in cleaned and "-" in cleaned
    assert "“" not in cleaned and '"' in cleaned


def test_clean_text_drops_decorative_rules():
    cleaned = clean_text("EXPERIENCE\n--------\nBuilt services")
    assert "--------" not in cleaned
    assert "EXPERIENCE" in cleaned
    assert "Built services" in cleaned


def test_clean_text_collapses_whitespace():
    assert clean_text("Python     engineer") == "Python engineer"
    assert "\n\n\n" not in clean_text("A\n\n\n\n\nB")


# ── Section detection ─────────────────────────────────────────────────────────


def test_detect_sections_assigns_content_to_its_header():
    sections = detect_sections(
        "SUMMARY\nBackend engineer.\n"
        "EXPERIENCE\nBuilt microservices at Acme.\n"
        "EDUCATION\nB.S. Computer Science.\n"
        "SKILLS\nPython, SQL, Docker."
    )
    assert "Backend engineer." in sections["summary"]
    assert "Built microservices at Acme." in sections["experience"]
    assert "B.S. Computer Science." in sections["education"]
    assert "Python, SQL, Docker." in sections["skills"]


def test_detect_sections_puts_preamble_in_other():
    sections = detect_sections("Jane Doe\nSKILLS\nPython")
    assert "Jane Doe" in sections["other"]


def test_detect_sections_always_returns_every_key():
    sections = detect_sections("")
    assert "other" in sections
    assert all(isinstance(value, str) for value in sections.values())


def test_has_required_sections_counts_populated_sections():
    populated = {"experience": "x" * 50, "education": "y" * 50, "skills": ""}
    assert has_required_sections(populated, min_sections=2)
    assert not has_required_sections(populated, min_sections=3)
    assert not has_required_sections({"experience": "tiny"})


# ── Training pairs ────────────────────────────────────────────────────────────


@pytest.fixture
def resumes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "resume_text": ["Python engineer resume.", "Chef resume."],
            "category": ["Python Developer", "Chef"],
        }
    )


@pytest.fixture
def postings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "We need a python developer for our django backend team.",
                "Hiring a flask and fastapi python developer.",
                "Seeking a chef for our culinary team.",
                "Kitchen manager and chef wanted for food service.",
            ]
        }
    )


def test_pairs_have_the_expected_columns(resumes, postings):
    pairs = build_training_pairs(resumes, postings, negatives_per_positive=1)
    assert set(pairs.columns) == {
        "resume_id",
        "resume_text",
        "job_description",
        "label",
        "resume_category",
        "pair_type",
    }


def test_each_resume_yields_one_positive_and_n_negatives(resumes, postings):
    pairs = build_training_pairs(resumes, postings, negatives_per_positive=2)
    assert (pairs["label"] == 1.0).sum() == len(resumes)
    assert (pairs["label"] == 0.0).sum() == len(resumes) * 2


def test_positive_pairs_match_the_resume_category(resumes, postings):
    pairs = build_training_pairs(resumes, postings, negatives_per_positive=1)
    positives = pairs[pairs["label"] == 1.0]

    python_pair = positives[positives["resume_category"] == "Python Developer"].iloc[0]
    assert "python developer" in python_pair["job_description"].lower()

    chef_pair = positives[positives["resume_category"] == "Chef"].iloc[0]
    assert "chef" in chef_pair["job_description"].lower()


def test_negative_pairs_come_from_a_different_domain(resumes, postings):
    pairs = build_training_pairs(resumes, postings, negatives_per_positive=1)
    negatives = pairs[(pairs["label"] == 0.0) & (pairs["resume_category"] == "Chef")]
    for description in negatives["job_description"]:
        assert "chef" not in description.lower()


def test_pair_construction_is_reproducible(resumes, postings):
    first = build_training_pairs(resumes, postings, random_seed=7)
    second = build_training_pairs(resumes, postings, random_seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_resumes_without_a_matching_posting_are_skipped(postings):
    orphan = pd.DataFrame(
        {
            "id": [9],
            "resume_text": ["Marine biologist."],
            "category": ["Marine Biology"],
        }
    )
    assert build_training_pairs(orphan, postings).empty


def test_split_partitions_every_pair_exactly_once():
    pairs = pd.DataFrame(
        {"resume_text": [f"r{i}" for i in range(100)], "label": [1.0, 0.0] * 50}
    )
    train, val, test = split_pairs(pairs, [0.8, 0.1, 0.1], random_seed=42)

    assert len(train) + len(val) + len(test) == len(pairs)
    assert len(train) == 80
    assert len(val) == len(test) == 10

    indices = set(train.index) | set(val.index) | set(test.index)
    assert len(indices) == len(pairs)


def test_split_preserves_the_label_balance():
    pairs = pd.DataFrame(
        {"resume_text": [f"r{i}" for i in range(100)], "label": [1.0, 0.0] * 50}
    )
    for part in split_pairs(pairs, [0.8, 0.1, 0.1], random_seed=42):
        assert part["label"].mean() == pytest.approx(0.5)


# ── Evaluation set ────────────────────────────────────────────────────────────


def _row(**overrides) -> pd.Series:
    defaults = {
        "Name": "Jane Doe",
        "Current_Job_Title": "Data Scientist",
        "Experience_Years": 4,
        "Previous_Job_Titles": "Analyst, Junior Analyst",
        "Education_Level": "Master's",
        "Field_of_Study": "Statistics",
        "Degrees": "M.S.",
        "Institute_Name": "State University",
        "Graduation_Year": 2019,
        "Skills": "python, sql, pandas",
        "Certifications": "AWS Certified",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def test_resume_text_includes_every_populated_section():
    text = build_resume_text(_row())
    for expected in (
        "Jane Doe",
        "PROFESSIONAL SUMMARY",
        "EXPERIENCE",
        "EDUCATION",
        "SKILLS",
        "CERTIFICATIONS",
        "2019",
    ):
        assert expected in text


def test_missing_graduation_year_does_not_raise():
    """Regression: NaN is truthy, so `int(nan)` raised ValueError on any row
    without a graduation year."""
    text = build_resume_text(_row(Graduation_Year=np.nan))
    assert "EDUCATION" in text
    assert "nan" not in text.lower()


@pytest.mark.parametrize(
    "field, value",
    [
        ("Previous_Job_Titles", np.nan),
        ("Institute_Name", "nan"),
        ("Certifications", "none"),
        ("Current_Job_Title", np.nan),
        ("Degrees", np.nan),
    ],
)
def test_missing_fields_are_omitted_rather_than_printed(field, value):
    text = build_resume_text(_row(**{field: value}))
    assert "nan" not in text.lower()
    assert "none" not in text.lower()


def test_resume_without_a_title_reads_as_a_recent_graduate():
    assert "Recent graduate" in build_resume_text(_row(Current_Job_Title=np.nan))


def test_parse_skills_normalises_to_a_lowercase_set():
    assert parse_skills("Python, SQL , pandas") == {"python", "sql", "pandas"}
    assert parse_skills(np.nan) == set()
    assert parse_skills("none") == set()


def test_skill_overlap_reflects_how_many_skills_the_posting_mentions():
    assert score_skill_overlap({"python", "sql"}, "python and sql required") == 1.0
    assert score_skill_overlap({"python", "sql"}, "python required") == 0.5
    assert score_skill_overlap({"python"}, "chef wanted") == 0.0
    assert score_skill_overlap(set(), "anything") == 0.3


def test_experience_fit_rewards_meeting_the_requirement():
    assert score_experience_fit(5.0, 3.0) == 1.0
    assert score_experience_fit(20.0, 3.0) == 0.8  # seniority mismatch
    assert score_experience_fit(1.0, 3.0) < 1.0
    assert score_experience_fit(4.0, None) == 0.6


def test_experience_fit_never_goes_negative():
    assert score_experience_fit(0.0, 50.0) == 0.0


def test_education_fit_compares_degree_levels():
    assert score_education_fit("Master's", "bachelor's degree required") == 1.0
    assert score_education_fit("Bachelor's", "phd required") < 1.0
    assert score_education_fit("Bachelor's", "no degree mentioned") == 1.0


def test_title_match_rewards_a_shared_title_and_field():
    matched = score_title_keyword_match(
        _row(), "hiring a data scientist with statistics"
    )
    unmatched = score_title_keyword_match(_row(), "hiring a pastry chef")
    assert matched > unmatched
    assert 0.0 <= unmatched <= matched <= 1.0


def test_match_score_is_bounded_and_fully_broken_down():
    score, breakdown = compute_match_score(
        _row(),
        "Seeking a data scientist with 3+ years of experience in python and sql.",
    )
    assert 0.0 <= score <= 1.0
    assert set(breakdown) == {
        "skills_match",
        "experience_relevance",
        "education_fit",
        "keyword_alignment",
    }
    assert all(0.0 <= value <= 1.0 for value in breakdown.values())


def test_a_relevant_posting_outscores_an_irrelevant_one():
    relevant, _ = compute_match_score(
        _row(), "Data scientist wanted: python, sql, pandas, statistics background."
    )
    irrelevant, _ = compute_match_score(
        _row(), "Pastry chef wanted for a busy kitchen. Culinary diploma preferred."
    )
    assert relevant > irrelevant


def test_unparseable_experience_years_defaults_to_zero():
    score, breakdown = compute_match_score(
        _row(Experience_Years="not a number"), "5 years of experience required"
    )
    assert 0.0 <= score <= 1.0

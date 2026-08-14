"""Unit tests for the four-component match scorer."""

import numpy as np
import pytest

from src.models.scorer import (
    MatchResult,
    MatchScorer,
    generate_suggestions,
    keyword_weights,
    score_education_fit,
    score_experience_relevance,
    score_keyword_alignment,
    score_skills_match,
    tokenize,
)

TOP_K = 20


def _unit_vector(*values: float) -> np.ndarray:
    """Return an L2-normalised vector, as the embedding model produces."""
    vector = np.array(values, dtype=float)
    return vector / np.linalg.norm(vector)


# ── Tokenisation ──────────────────────────────────────────────────────────────


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Quick Python and FastAPI") == ["quick", "python", "fastapi"]


def test_tokenize_keeps_technical_punctuation():
    assert "c++" in tokenize("Experienced in C++ development")
    assert "node.js" in tokenize("Built services with Node.js")


def test_tokenize_returns_empty_for_blank_input():
    assert tokenize("") == []


# ── Skills ────────────────────────────────────────────────────────────────────


def test_identical_skill_sets_score_at_the_top():
    skills = {"python", "sql", "docker"}
    score, matched, missing = score_skills_match(skills, skills)
    assert score == 1.0
    assert matched == ["docker", "python", "sql"]
    assert missing == []


def test_disjoint_skill_sets_score_zero():
    score, matched, missing = score_skills_match({"chef"}, {"python"})
    assert score == 0.0
    assert matched == []
    assert missing == ["python"]


def test_missing_skills_are_those_the_posting_wants():
    _, matched, missing = score_skills_match(
        {"python", "sql"}, {"python", "kubernetes", "terraform"}
    )
    assert matched == ["python"]
    assert missing == ["kubernetes", "terraform"]


def test_both_sides_empty_scores_neutral():
    score, matched, missing = score_skills_match(set(), set())
    assert score == 0.5
    assert (matched, missing) == ([], [])


def test_embeddings_blend_into_the_skill_score():
    skills = {"python"}
    aligned = _unit_vector(1.0, 0.0)
    without = score_skills_match(skills, skills)[0]
    with_embeddings = score_skills_match(skills, skills, aligned, aligned)[0]
    assert without == pytest.approx(1.0)
    assert with_embeddings == pytest.approx(1.0)


def test_half_supplied_embeddings_do_not_penalise_the_score():
    """
    Regression: the blend was applied whenever the resume embedding was present,
    but the semantic term was only computed when both were, so supplying one
    scored as though semantic similarity were zero.
    """
    skills = {"python", "sql"}
    resume_only = score_skills_match(skills, skills, _unit_vector(1.0, 0.0), None)[0]
    neither = score_skills_match(skills, skills)[0]
    assert resume_only == pytest.approx(neither)


def test_orthogonal_embeddings_lower_the_blended_score():
    skills = {"python"}
    orthogonal = score_skills_match(
        skills, skills, _unit_vector(1.0, 0.0), _unit_vector(0.0, 1.0)
    )[0]
    assert orthogonal == pytest.approx(0.6)


# ── Experience ────────────────────────────────────────────────────────────────


def test_experience_falls_back_to_neutral_without_embeddings():
    assert score_experience_relevance("Engineer.", "Engineer wanted.") == 0.5


def test_meeting_the_requirement_earns_a_bonus():
    aligned = _unit_vector(1.0, 0.0)
    met = score_experience_relevance(
        "5 years of experience.", "3+ years of experience required.", aligned, aligned
    )
    short = score_experience_relevance(
        "1 years of experience.", "3+ years of experience required.", aligned, aligned
    )
    assert met > short


def test_falling_short_is_penalised_but_bounded():
    resume_emb = jd_emb = _unit_vector(1.0, 0.0)
    score = score_experience_relevance(
        "1 years of experience.",
        "20 years of experience required.",
        resume_emb,
        jd_emb,
    )
    # Similarity is 1.0 and the penalty is capped at -0.2.
    assert score == pytest.approx(0.8)


def test_experience_score_stays_within_bounds():
    for resume, jd in [
        ("30 years of experience.", "1 years of experience."),
        ("1 years of experience.", "30 years of experience."),
    ]:
        score = score_experience_relevance(
            resume, jd, _unit_vector(1.0), _unit_vector(1.0)
        )
        assert 0.0 <= score <= 1.0


# ── Education ─────────────────────────────────────────────────────────────────


def test_meeting_the_education_requirement_scores_full_marks():
    assert score_education_fit("M.S. in CS", "Bachelor's degree required") == 1.0
    assert score_education_fit("B.S. in CS", "Bachelor's degree required") == 1.0


def test_education_score_tapers_with_the_shortfall():
    one_short = score_education_fit("Associate degree", "Bachelor's required")
    two_short = score_education_fit("Diploma in IT", "Bachelor's required")
    far_short = score_education_fit("Coding bootcamp", "PhD required")
    assert one_short == 0.7
    assert two_short == 0.4
    assert far_short == 0.2
    assert one_short > two_short > far_short


def test_posting_without_a_stated_requirement_assumes_a_degree():
    assert score_education_fit("B.S. in CS", "Great team, fun culture.") == 1.0
    assert score_education_fit("Coding bootcamp", "Great team, fun culture.") < 1.0


# ── Keywords ──────────────────────────────────────────────────────────────────


def test_keyword_weights_grow_with_repetition():
    weights = keyword_weights(["python", "python", "sql"])
    assert weights["python"] > weights["sql"]


def test_full_keyword_coverage_scores_one():
    text = "python fastapi docker kubernetes postgresql"
    assert score_keyword_alignment(text, text, TOP_K) == pytest.approx(1.0)


def test_no_keyword_coverage_scores_zero():
    score = score_keyword_alignment(
        "chef culinary kitchen plating", "python fastapi docker engineer", TOP_K
    )
    assert score == pytest.approx(0.0)


def test_keyword_alignment_increases_with_overlap():
    """
    Regression: two-document IDF gave shared terms zero weight, so the score fell
    as overlap rose — a strong match previously scored 0.0.
    """
    posting = "senior python engineer with fastapi docker postgresql aws experience"
    none = score_keyword_alignment("chef culinary kitchen", posting, TOP_K)
    some = score_keyword_alignment("python data analysis pandas", posting, TOP_K)
    most = score_keyword_alignment(
        "senior python engineer fastapi docker postgresql aws", posting, TOP_K
    )
    assert none < some < most
    assert most > 0.5


def test_empty_posting_scores_neutral():
    assert score_keyword_alignment("python developer", "", TOP_K) == 0.5


# ── Suggestions ───────────────────────────────────────────────────────────────


def _result(**overrides) -> MatchResult:
    defaults = dict(
        final_score=0.9,
        score_pct=90.0,
        label="Excellent",
        skills_match=0.9,
        experience_relevance=0.9,
        education_fit=0.9,
        keyword_alignment=0.9,
    )
    defaults.update(overrides)
    return MatchResult(**defaults)


def test_strong_match_gets_an_encouraging_suggestion():
    suggestions = generate_suggestions(_result(), "Python engineer wanted.")
    assert len(suggestions) == 1
    assert "confidence" in suggestions[0].lower()


def test_missing_skills_are_named_in_the_suggestion():
    result = _result(skills_match=0.2, missing_skills=["kubernetes", "terraform"])
    suggestions = generate_suggestions(result, "DevOps role.")
    assert any("kubernetes" in s for s in suggestions)


def test_each_weak_component_adds_a_suggestion():
    weak = _result(
        final_score=0.2,
        skills_match=0.1,
        experience_relevance=0.1,
        education_fit=0.1,
        keyword_alignment=0.1,
        missing_skills=["python"],
    )
    suggestions = generate_suggestions(weak, "PhD required for this research role.")
    assert len(suggestions) >= 4


def test_suggestions_are_never_empty():
    assert generate_suggestions(_result(final_score=0.6), "Some role.") != []


# ── MatchResult ───────────────────────────────────────────────────────────────


def test_to_dict_rounds_and_nests_the_breakdown():
    result = _result(skills_match=0.123456, missing_skills=[f"s{i}" for i in range(15)])
    payload = result.to_dict()

    assert payload["breakdown"]["skills_match"] == 0.1235
    assert payload["score_pct"] == 90.0
    assert set(payload["breakdown"]) == {
        "skills_match",
        "experience_relevance",
        "education_fit",
        "keyword_alignment",
    }


def test_to_dict_caps_the_missing_skills_it_reports():
    result = _result(missing_skills=[f"skill{i}" for i in range(25)])
    assert len(result.to_dict()["missing_skills"]) == 10


# ── MatchScorer ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scorer() -> MatchScorer:
    return MatchScorer()


@pytest.mark.parametrize(
    "score_pct, expected",
    [
        (100.0, "Excellent"),
        (85.0, "Excellent"),
        (84.9, "Good"),
        (70.0, "Good"),
        (69.9, "Fair"),
        (50.0, "Fair"),
        (49.9, "Poor"),
        (0.0, "Poor"),
    ],
)
def test_label_thresholds(scorer, score_pct, expected):
    assert scorer._label(score_pct) == expected


def test_score_uses_supplied_inputs_without_loading_models(scorer):
    """Every optional input is provided, so no embedding or NER model is touched."""
    embedding = _unit_vector(1.0, 0.0)
    result = scorer.score(
        resume_text="Python engineer with 5 years of experience. B.S. Computer Science.",
        jd_text="Seeking a Python engineer with 3+ years of experience. Bachelor's required.",
        resume_skills={"python"},
        jd_skills={"python"},
        resume_emb=embedding,
        jd_emb=embedding,
    )

    assert 0.0 <= result.final_score <= 1.0
    assert result.score_pct == pytest.approx(result.final_score * 100, abs=0.05)
    assert result.label in {"Excellent", "Good", "Fair", "Poor"}
    assert result.matched_skills == ["python"]
    assert result.suggestions


def test_a_strong_pair_outscores_a_weak_one(scorer):
    aligned = _unit_vector(1.0, 0.0)
    orthogonal = _unit_vector(0.0, 1.0)
    posting = (
        "Seeking a Python engineer with 3+ years of experience. Bachelor's required."
    )

    strong = scorer.score(
        "Python engineer, 5 years of experience, B.S. Computer Science.",
        posting,
        resume_skills={"python"},
        jd_skills={"python"},
        resume_emb=aligned,
        jd_emb=aligned,
    )
    weak = scorer.score(
        "Pastry chef with 2 years of experience in kitchens.",
        posting,
        resume_skills={"baking"},
        jd_skills={"python"},
        resume_emb=aligned,
        jd_emb=orthogonal,
    )
    assert strong.final_score > weak.final_score


def test_configured_weights_sum_to_one(scorer):
    assert sum(scorer._weights.values()) == pytest.approx(1.0)

"""
Synthetic resume-JD pair generation for HireLens evaluation set.

Uses the Anthropic API (claude-sonnet-4-6) to generate 500 diverse
resume / job-description pairs with explicit match scores.
These pairs are ONLY used as a held-out evaluation set — never for training.

Output: data/synthetic/synthetic_pairs.csv
Columns: resume_text, job_description, match_score, category, rationale
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic
import mlflow
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "hirelens-resume-matching")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL_ID = "claude-sonnet-4-6"
TARGET_PAIRS = 500
BATCH_SIZE = 10          # pairs per API call
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}")
logger.add(PROJECT_ROOT / "logs" / "synthetic_gen.log", rotation="50 MB", retention="14 days", level="DEBUG")


# ── Prompt construction ───────────────────────────────────────────────────────

# Covers a broad range of domains so the synthetic eval set is diverse
_DOMAIN_POOL = [
    "Data Science", "Software Engineering", "DevOps", "Product Management",
    "UX Design", "Marketing", "Finance", "Healthcare", "Legal", "Sales",
    "Machine Learning Engineering", "Cybersecurity", "Cloud Architecture",
    "Business Analysis", "HR", "Supply Chain", "Education", "Research",
]

_SCORE_BUCKETS = [
    (0.85, 1.00, "strong match"),
    (0.65, 0.84, "moderate match"),
    (0.40, 0.64, "weak match"),
    (0.00, 0.39, "poor match"),
]


def _build_prompt(domain: str, score_range: tuple[float, float], score_label: str, n: int) -> str:
    """Build the prompt for a batch of synthetic pairs."""
    low, high = score_range
    return f"""You are generating synthetic resume and job description pairs for an AI resume screening system evaluation dataset.

Generate exactly {n} resume-job description pairs for the domain: **{domain}**
Each pair must have a match score between {low:.2f} and {high:.2f} ({score_label}).

Requirements:
- Resumes: 200-400 words, realistic professional content with skills, experience, education sections
- Job descriptions: 150-300 words, realistic with requirements, responsibilities, qualifications
- Match score: float between {low:.2f} and {high:.2f} reflecting genuine resume-JD alignment
- Rationale: 1-2 sentences explaining the score

Respond with ONLY a valid JSON array, no markdown, no extra text:
[
  {{
    "resume_text": "...",
    "job_description": "...",
    "match_score": 0.XX,
    "category": "{domain}",
    "rationale": "..."
  }},
  ...
]"""


# ── API calls ─────────────────────────────────────────────────────────────────

def _call_api_with_retry(client: anthropic.Anthropic, prompt: str) -> list[dict[str, Any]]:
    """
    Call the Claude API and parse the JSON response.

    Retries up to MAX_RETRIES times on transient errors or malformed JSON.

    Args:
        client: Initialised Anthropic client.
        prompt: Full prompt string.

    Returns:
        List of parsed pair dicts.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown code fences
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                raw = raw.rstrip("`").strip()

            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError("Response is not a JSON array.")

            # Validate and coerce each record
            valid = []
            for item in pairs:
                score = float(item.get("match_score", 0.5))
                score = max(0.0, min(1.0, score))
                valid.append({
                    "resume_text": str(item.get("resume_text", "")).strip(),
                    "job_description": str(item.get("job_description", "")).strip(),
                    "match_score": score,
                    "category": str(item.get("category", "Unknown")).strip(),
                    "rationale": str(item.get("rationale", "")).strip(),
                })
            return valid

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: parse error — {e}")
        except anthropic.RateLimitError:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: rate limit hit, sleeping 30s.")
            time.sleep(30)
        except anthropic.APIError as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: API error — {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for this batch.")


# ── Generation orchestration ──────────────────────────────────────────────────

def generate_synthetic_pairs(n_pairs: int = TARGET_PAIRS) -> pd.DataFrame:
    """
    Generate n_pairs synthetic resume-JD pairs across diverse domains and score buckets.

    Distributes pairs evenly across score buckets to ensure evaluation coverage
    across the full match score spectrum.

    Args:
        n_pairs: Total number of pairs to generate.

    Returns:
        DataFrame with columns: [resume_text, job_description, match_score, category, rationale].
    """
    import re  # local import to avoid module-level dependency for type checkers

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger.info(f"Generating {n_pairs} synthetic pairs using {MODEL_ID}...")

    all_pairs: list[dict[str, Any]] = []

    # Distribute evenly across 4 score buckets
    pairs_per_bucket = n_pairs // len(_SCORE_BUCKETS)
    remainder = n_pairs % len(_SCORE_BUCKETS)

    import itertools
    domain_cycle = itertools.cycle(_DOMAIN_POOL)

    with tqdm(total=n_pairs, desc="Generating synthetic pairs") as pbar:
        for bucket_idx, (low, high, label) in enumerate(_SCORE_BUCKETS):
            bucket_target = pairs_per_bucket + (1 if bucket_idx < remainder else 0)
            generated = 0

            while generated < bucket_target:
                batch_n = min(BATCH_SIZE, bucket_target - generated)
                domain = next(domain_cycle)
                prompt = _build_prompt(domain, (low, high), label, batch_n)

                try:
                    batch = _call_api_with_retry(client, prompt)
                    all_pairs.extend(batch)
                    generated += len(batch)
                    pbar.update(len(batch))
                    logger.debug(f"Bucket '{label}': {generated}/{bucket_target} done.")
                except RuntimeError as e:
                    logger.error(f"Batch failed, skipping: {e}")
                    # Move on to next batch attempt rather than crashing the whole run
                    generated += batch_n
                    pbar.update(batch_n)

                # Polite rate-limit spacing
                time.sleep(0.5)

    df = pd.DataFrame(all_pairs)
    logger.info(f"Generated {len(df)} synthetic pairs total.")
    logger.info(f"Score distribution:\n{df['match_score'].describe()}")
    return df


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path() -> Path:
    return SYNTHETIC_DIR / "synthetic_pairs_checkpoint.json"


def load_checkpoint() -> list[dict]:
    """Load any previously generated pairs from a checkpoint file."""
    path = _checkpoint_path()
    if path.exists():
        logger.info(f"Resuming from checkpoint: {path}")
        return json.loads(path.read_text())
    return []


def save_checkpoint(pairs: list[dict]) -> None:
    """Save generated pairs to checkpoint so progress isn't lost on interruption."""
    _checkpoint_path().write_text(json.dumps(pairs, indent=2))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_synthetic_generation(n_pairs: int = TARGET_PAIRS) -> pd.DataFrame:
    """
    Full synthetic generation pipeline with checkpointing, validation, and MLflow logging.

    Args:
        n_pairs: Target number of synthetic pairs.

    Returns:
        Saved synthetic pairs DataFrame.
    """
    logger.info("=" * 60)
    logger.info("Starting synthetic data generation")
    logger.info("=" * 60)

    # Resume from checkpoint if interrupted
    existing = load_checkpoint()
    if len(existing) >= n_pairs:
        logger.info(f"Checkpoint has {len(existing)} pairs — skipping generation.")
        df = pd.DataFrame(existing)
    else:
        remaining = n_pairs - len(existing)
        logger.info(f"Checkpoint has {len(existing)} pairs, generating {remaining} more.")
        new_pairs_df = generate_synthetic_pairs(remaining)
        all_pairs = existing + new_pairs_df.to_dict("records")
        save_checkpoint(all_pairs)
        df = pd.DataFrame(all_pairs)

    # Basic quality filter
    before = len(df)
    df = df[
        (df["resume_text"].str.len() > 100) &
        (df["job_description"].str.len() > 80) &
        (df["match_score"].between(0.0, 1.0))
    ].drop_duplicates(subset=["resume_text"]).reset_index(drop=True)
    logger.info(f"Quality filter: {before} → {len(df)} pairs")

    # Save final output
    output_path = SYNTHETIC_DIR / "synthetic_pairs.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} synthetic pairs to {output_path}")

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="synthetic-data-generation"):
            mlflow.log_metrics({
                "synthetic_pairs_total": float(len(df)),
                "synthetic_score_mean": float(df["match_score"].mean()),
                "synthetic_score_std": float(df["match_score"].std()),
                "synthetic_score_min": float(df["match_score"].min()),
                "synthetic_score_max": float(df["match_score"].max()),
                "synthetic_unique_categories": float(df["category"].nunique()),
            })
            mlflow.log_artifact(str(output_path), artifact_path="synthetic")
            mlflow.set_tag("stage", "synthetic_generation")
            mlflow.set_tag("model_used", MODEL_ID)
            logger.success("Synthetic generation stats logged to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    # Remove checkpoint on successful completion
    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()

    logger.success(f"Synthetic generation complete: {len(df)} pairs saved.")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic resume-JD pairs.")
    parser.add_argument("--n-pairs", type=int, default=TARGET_PAIRS,
                        help=f"Number of pairs to generate (default: {TARGET_PAIRS})")
    args = parser.parse_args()
    run_synthetic_generation(n_pairs=args.n_pairs)

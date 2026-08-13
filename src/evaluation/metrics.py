"""
Evaluation suite.

Computes, on the held-out test pairs:
  - Precision@K and NDCG@K over each resume's ranked candidate postings
  - MRR and MAP@K
  - Pearson / Spearman correlation and AUC-ROC on pair similarity
  - Token-level precision / recall / F1 for skill extraction

Writes logs/evaluation_report.json and logs the same metrics to MLflow.

Usage:
    python -m src.evaluation.metrics
    python -m src.evaluation.metrics --model-path models/fine_tuned/hirelens_matcher
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, get_section
from src.logging_setup import configure_logging

TEST_PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "test_pairs.csv"
NER_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "eval_dataset" / "resume_dataset_1200.csv"
)
REPORT_PATH = PROJECT_ROOT / "logs" / "evaluation_report.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fine_tuned" / "hirelens_matcher"

ENCODE_BATCH_SIZE = 64


# ── Ranking metrics ───────────────────────────────────────────────────────────


def precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """
    Fraction of the top-K retrieved items that are relevant.

    Args:
        relevant: Indices of relevant items.
        retrieved: Retrieved item indices, best-first.
        k: Cutoff.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain over the first K ranked relevance scores."""
    return sum(
        relevance / np.log2(rank + 2) for rank, relevance in enumerate(relevances[:k])
    )


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """DCG@K normalised by the DCG of the ideal ranking."""
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return dcg_at_k(relevances, k) / ideal if ideal > 0 else 0.0


def reciprocal_rank(relevant: set[int], retrieved: list[int]) -> float:
    """Reciprocal of the rank of the first relevant item, or 0.0 if none is."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """
    Average precision over the top-K, normalised by the reachable number of hits.

    Args:
        relevant: Indices of relevant items.
        retrieved: Retrieved item indices, best-first.
        k: Cutoff.

    Returns:
        AP@K in [0, 1].
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)


# ── Skill extraction metrics ──────────────────────────────────────────────────


def token_level_f1(
    predictions: list[set[str]], ground_truths: list[set[str]]
) -> dict[str, float]:
    """
    Micro-averaged precision, recall, and F1 over predicted skill sets.

    Args:
        predictions: Predicted skills per document.
        ground_truths: Reference skills per document.

    Returns:
        Dict with 'precision', 'recall', and 'f1'.
    """
    true_positives = false_positives = false_negatives = 0
    for predicted, actual in zip(predictions, ground_truths):
        true_positives += len(predicted & actual)
        false_positives += len(predicted - actual)
        false_negatives += len(actual - predicted)

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── Retrieval evaluation ──────────────────────────────────────────────────────


def _load_model(model_path: str | Path | None):
    """Load the fine-tuned model, falling back to the configured base model."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = get_section("model")["base"]

    if model_path and Path(model_path).exists():
        logger.info(f"Loading fine-tuned model from {model_path}")
        model = SentenceTransformer(str(model_path), device=device)
    else:
        logger.warning(
            f"Fine-tuned model not found at {model_path} — "
            f"evaluating the base model {base_config['name']} instead."
        )
        model = SentenceTransformer(base_config["name"], device=device)

    model.max_seq_length = base_config["max_seq_length"]
    return model


def evaluate_retrieval(
    model_path: str | Path | None, eval_df: pd.DataFrame, k_values: list[int]
) -> dict[str, float]:
    """
    Evaluate reranking quality on labelled pairs.

    Pairs are grouped by resume; within each group the postings are ranked by
    cosine similarity and scored on whether positives outrank negatives.
    Correlation and AUC-ROC are computed across all pairs.

    Args:
        model_path: Fine-tuned model directory, or None for the base model.
        eval_df: Pairs with [resume_text, job_description, label].
        k_values: Cutoffs to report.

    Returns:
        Metric name to value, rounded to four decimal places.
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    model = _load_model(model_path)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    logger.info(f"Encoding {len(eval_df)} resume/posting pairs...")
    resume_embeddings = encode(eval_df["resume_text"].tolist())
    jd_embeddings = encode(eval_df["job_description"].tolist())

    # Row-wise dot product: similarity of each pair, not the full cross product.
    similarities = np.einsum("ij,ij->i", resume_embeddings, jd_embeddings)
    labels = eval_df["label"].astype(float).to_numpy()
    binary_labels = (labels >= 0.5).astype(int)

    results: dict[str, float] = {
        "pearson_cosine": float(pearsonr(similarities, labels)[0]),
        "spearman_cosine": float(spearmanr(similarities, labels)[0]),
        "auc_roc": float(
            roc_auc_score(binary_labels, similarities)
            if len(set(binary_labels)) > 1
            else 0.5
        ),
    }

    grouped = eval_df.assign(_similarity=similarities, _relevant=binary_labels)
    precision: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg: dict[int, list[float]] = {k: [] for k in k_values}
    average_precision: dict[int, list[float]] = {k: [] for k in k_values}
    reciprocal_ranks: list[float] = []

    for _, group in grouped.groupby("resume_text", sort=False):
        if len(group) < 2:
            continue

        ranked = group.sort_values("_similarity", ascending=False)
        relevances = ranked["_relevant"].tolist()
        relevant_positions = {i for i, value in enumerate(relevances) if value == 1}
        if not relevant_positions:
            continue

        positions = list(range(len(ranked)))
        for k in k_values:
            precision[k].append(precision_at_k(relevant_positions, positions, k))
            ndcg[k].append(ndcg_at_k(relevances, k))
            average_precision[k].append(
                average_precision_at_k(relevant_positions, positions, k)
            )
        reciprocal_ranks.append(reciprocal_rank(relevant_positions, positions))

    results["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    for k in k_values:
        if precision[k]:
            results[f"precision_at_{k}"] = float(np.mean(precision[k]))
            results[f"ndcg_at_{k}"] = float(np.mean(ndcg[k]))
            results[f"map_at_{k}"] = float(np.mean(average_precision[k]))

    return {name: round(value, 4) for name, value in results.items()}


# ── Skill extraction evaluation ───────────────────────────────────────────────


def evaluate_ner(eval_df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate skill extraction against the structured dataset's Skills column.

    Args:
        eval_df: Structured resume rows including a 'Skills' column.

    Returns:
        Precision, recall, and F1.
    """
    if "Skills" not in eval_df.columns:
        logger.warning("No 'Skills' column present — skipping skill extraction eval.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    from src.data.eval_dataset import build_resume_text, parse_skills
    from src.features.ner import get_extractor

    extractor = get_extractor()
    has_resume_text = "resume_text" in eval_df.columns

    predictions: list[set[str]] = []
    ground_truths: list[set[str]] = []
    for _, row in eval_df.iterrows():
        text = str(row["resume_text"]) if has_resume_text else build_resume_text(row)
        predictions.append(set(extractor.extract(text).technical_skills))
        ground_truths.append(parse_skills(row.get("Skills")))

    return token_level_f1(predictions, ground_truths)


# ── Report ────────────────────────────────────────────────────────────────────


def run_evaluation(
    model_path: str | Path | None = None, log_to_mlflow: bool = True
) -> dict[str, Any]:
    """
    Run the full evaluation suite and write logs/evaluation_report.json.

    Args:
        model_path: Fine-tuned model directory, or None for the base model.
        log_to_mlflow: Whether to log the results to MLflow.

    Returns:
        The report dict.

    Raises:
        FileNotFoundError: If the test pairs are missing.
    """
    logger.info("=" * 60)
    logger.info("HireLens evaluation suite")
    logger.info("=" * 60)

    evaluation_config = get_section("evaluation")

    if not TEST_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"Test pairs not found at {TEST_PAIRS_PATH}. Run the preprocess stage first."
        )
    eval_df = pd.read_csv(TEST_PAIRS_PATH)
    logger.info(f"Loaded {len(eval_df)} test pairs.")

    sample_size = evaluation_config["num_eval_samples"]
    if len(eval_df) > sample_size:
        eval_df = eval_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled down to {sample_size} pairs for speed.")

    logger.info("Computing retrieval metrics...")
    retrieval_metrics = evaluate_retrieval(
        model_path, eval_df, evaluation_config["k_values"]
    )
    for name, value in retrieval_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info("Computing skill extraction metrics...")
    ner_metrics: dict[str, float] = {}
    if NER_SOURCE_PATH.exists():
        ner_df = pd.read_csv(NER_SOURCE_PATH).head(
            evaluation_config["ner_eval_samples"]
        )
        ner_metrics = evaluate_ner(ner_df)
        logger.info(f"  NER F1: {ner_metrics.get('f1', 0.0):.4f}")
    else:
        logger.warning(f"Skipping skill extraction eval — {NER_SOURCE_PATH} not found.")

    report = {
        "retrieval": retrieval_metrics,
        "ner": ner_metrics,
        "target_check": _check_targets(
            retrieval_metrics, ner_metrics, evaluation_config["target_metrics"]
        ),
        "eval_samples": len(eval_df),
        "model_path": str(model_path) if model_path else "base_model",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info(f"Evaluation report written to {REPORT_PATH}")

    if log_to_mlflow:
        _log_to_mlflow(retrieval_metrics, ner_metrics)

    return report


def _check_targets(
    retrieval_metrics: dict[str, float],
    ner_metrics: dict[str, float],
    targets: dict[str, float],
) -> dict[str, dict]:
    """Compare each achieved metric against its project target and log the result."""
    checks: dict[str, dict] = {}
    for name, target in targets.items():
        key = name[len("ner_") :] if name.startswith("ner_") else name
        actual = retrieval_metrics.get(name, ner_metrics.get(key))
        if actual is None:
            continue

        passed = actual >= target
        checks[name] = {"target": target, "actual": round(actual, 4), "passed": passed}
        logger.info(
            f"  {'PASS' if passed else 'FAIL'} | {name}: {actual:.4f} (target {target})"
        )
    return checks


def _log_to_mlflow(
    retrieval_metrics: dict[str, float], ner_metrics: dict[str, float]
) -> None:
    """Record evaluation metrics to MLflow. Failures are non-fatal."""
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(get_section("mlflow")["experiment_name"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(
                {
                    **retrieval_metrics,
                    **{f"ner_{k}": v for k, v in ner_metrics.items()},
                }
            )
            mlflow.log_artifact(str(REPORT_PATH), artifact_path="evaluation")
            mlflow.set_tag("stage", "evaluation")
            logger.success("Evaluation metrics logged to MLflow.")
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HireLens evaluation suite.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Fine-tuned model directory (defaults to models/fine_tuned/hirelens_matcher)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    configure_logging(log_file="evaluation.log")
    run_evaluation(
        model_path=args.model_path or str(DEFAULT_MODEL_PATH),
        log_to_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()

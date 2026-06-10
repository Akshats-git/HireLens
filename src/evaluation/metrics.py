"""
Evaluation metrics for HireLens.

Computes:
  - Precision@K  : fraction of top-K retrieved items that are relevant
  - NDCG@K       : Normalised Discounted Cumulative Gain
  - MRR          : Mean Reciprocal Rank
  - MAP@K        : Mean Average Precision
  - NER F1       : token-level F1 for skill extraction
  - Full report  : JSON summary of all metrics

Usage:
    python -m src.evaluation.metrics
    python -m src.evaluation.metrics --model-path models/fine_tuned/hirelens_matcher
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Ranking metrics ───────────────────────────────────────────────────────────


def precision_at_k(relevant: list[int], retrieved: list[int], k: int) -> float:
    """
    Precision@K: fraction of the top-K retrieved items that are relevant.

    Args:
        relevant: List of relevant item indices (ground truth).
        retrieved: List of retrieved item indices sorted by rank.
        k: Cutoff.

    Returns:
        Precision@K in [0, 1].
    """
    top_k = set(retrieved[:k])
    rel_set = set(relevant)
    return len(top_k & rel_set) / k if k > 0 else 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    Discounted Cumulative Gain @ K.

    Args:
        relevances: Relevance scores of retrieved items in ranked order.
        k: Cutoff.

    Returns:
        DCG@K value.
    """
    relevances = relevances[:k]
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """
    Normalised DCG@K.

    Args:
        relevances: Relevance scores of retrieved items in ranked order.
        k: Cutoff.

    Returns:
        NDCG@K in [0, 1].
    """
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def reciprocal_rank(relevant: set[int], retrieved: list[int]) -> float:
    """
    Reciprocal rank: 1/rank of the first relevant item.

    Args:
        relevant: Set of relevant item indices.
        retrieved: Retrieved items sorted by rank.

    Returns:
        Reciprocal rank in (0, 1].
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """
    Average Precision@K.

    Args:
        relevant: Set of relevant item indices.
        retrieved: Retrieved items sorted by rank.
        k: Cutoff.

    Returns:
        AP@K in [0, 1].
    """
    hits = 0
    sum_precision = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / min(len(relevant), k) if relevant else 0.0


# ── NER / skill extraction F1 ─────────────────────────────────────────────────


def token_level_f1(
    predictions: list[set[str]],
    ground_truths: list[set[str]],
) -> dict[str, float]:
    """
    Compute micro-averaged token-level Precision, Recall, F1
    for skill extraction.

    Args:
        predictions: List of predicted skill sets per document.
        ground_truths: List of ground-truth skill sets per document.

    Returns:
        Dict with keys: precision, recall, f1.
    """
    tp = fp = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ── Embedding-based retrieval evaluation ─────────────────────────────────────


def evaluate_retrieval(
    model_path: str | Path | None,
    eval_df: pd.DataFrame,
    k_values: list[int],
) -> dict[str, float]:
    """
    Evaluate reranking quality on labeled test pairs.

    Groups pairs by resume_text. For each resume, ranks its associated JDs
    by cosine similarity and evaluates whether positive pairs (label=1) rank
    above negative pairs (label=0). This is a proper reranking evaluation.

    Also reports Spearman / Pearson correlation and AUC-ROC.

    Args:
        model_path: Path to fine-tuned model. Uses base model if None.
        eval_df: DataFrame with [resume_text, job_description, label].
        k_values: List of K values to compute metrics for.

    Returns:
        Dict of metric_name → value.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr, spearmanr
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path and Path(model_path).exists():
        logger.info(f"Loading fine-tuned model from {model_path}")
        model = SentenceTransformer(str(model_path), device=device)
    else:
        cfg = _load_config()
        base = cfg["model"]["base"]["name"]
        logger.info(f"Fine-tuned model not found — using base model {base}")
        model = SentenceTransformer(base, device=device)

    model.max_seq_length = 256

    # ── Encode all texts ──────────────────────────────────────────────────────
    resumes = eval_df["resume_text"].tolist()
    jds = eval_df["job_description"].tolist()
    labels = eval_df["label"].astype(float).tolist()

    logger.info(f"Encoding {len(resumes)} resumes...")
    resume_embs = model.encode(
        resumes,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info(f"Encoding {len(jds)} job descriptions...")
    jd_embs = model.encode(
        jds,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # ── Pair-level similarity scores ──────────────────────────────────────────
    sim_scores = np.einsum("ij,ij->i", resume_embs, jd_embs)  # dot product per pair

    # ── Correlation metrics ───────────────────────────────────────────────────
    pearson_r, _ = pearsonr(sim_scores, labels)
    spearman_r, _ = spearmanr(sim_scores, labels)

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    binary_labels = (np.array(labels) >= 0.5).astype(int)
    auc = (
        roc_auc_score(binary_labels, sim_scores) if len(set(binary_labels)) > 1 else 0.5
    )

    # ── Reranking within resume groups ────────────────────────────────────────
    eval_df = eval_df.copy()
    eval_df["_sim"] = sim_scores
    eval_df["_bin_label"] = binary_labels

    precision_scores: dict[int, list[float]] = {k: [] for k in k_values}
    ndcg_scores: dict[int, list[float]] = {k: [] for k in k_values}
    mrr_list: list[float] = []

    groups = eval_df.groupby("resume_text", sort=False)
    for _, group in groups:
        if len(group) < 2:
            continue
        ranked = group.sort_values("_sim", ascending=False)
        ranked_rels = ranked["_bin_label"].tolist()
        ranked_indices = list(range(len(ranked)))
        relevant_set = {i for i, r in enumerate(ranked_rels) if r == 1}

        if not relevant_set:
            continue

        for k in k_values:
            precision_scores[k].append(
                precision_at_k(list(relevant_set), ranked_indices, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_rels, k))

        mrr_list.append(reciprocal_rank(relevant_set, ranked_indices))

    results: dict[str, float] = {
        "pearson_cosine": float(pearson_r),
        "spearman_cosine": float(spearman_r),
        "auc_roc": float(auc),
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
    }
    for k in k_values:
        if precision_scores[k]:
            results[f"precision_at_{k}"] = float(np.mean(precision_scores[k]))
            results[f"ndcg_at_{k}"] = float(np.mean(ndcg_scores[k]))

    return {k: round(v, 4) for k, v in results.items()}


# ── NER evaluation ────────────────────────────────────────────────────────────


def evaluate_ner(eval_df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate NER skill extraction against the structured Kaggle eval set.

    Ground truth: 'Skills' column (comma-separated).
    Prediction: NERExtractor on resume text built from structured fields.

    Args:
        eval_df: DataFrame from resume_dataset_1200.csv with a 'Skills' column.

    Returns:
        Precision, Recall, F1 dict.
    """
    if "Skills" not in eval_df.columns:
        logger.warning("No 'Skills' column in eval_df — skipping NER evaluation.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    from src.features.ner import get_extractor
    from src.data.synthetic_gen import build_resume_text

    extractor = get_extractor()

    predictions: list[set[str]] = []
    ground_truths: list[set[str]] = []

    for _, row in eval_df.iterrows():
        # Build resume text from structured fields if resume_text not present
        if "resume_text" in eval_df.columns:
            text = str(row["resume_text"])
        else:
            text = build_resume_text(row)

        pred = set(extractor.extract(text).technical_skills)
        gt_raw = str(row.get("Skills", ""))
        gt = {s.strip().lower() for s in gt_raw.split(",") if s.strip()}
        predictions.append(pred)
        ground_truths.append(gt)

    return token_level_f1(predictions, ground_truths)


# ── Full evaluation report ────────────────────────────────────────────────────


def run_evaluation(
    model_path: str | Path | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Run the complete evaluation suite and generate a JSON report.

    Args:
        model_path: Path to fine-tuned model directory. Uses base if None.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Report dict with all metrics.
    """
    cfg = _load_config()
    k_values = cfg["evaluation"]["k_values"]

    logger.info("=" * 60)
    logger.info("HireLens Evaluation Suite")
    logger.info("=" * 60)

    # Use held-out test pairs (binary labels, proper reranking eval)
    test_path = PROJECT_ROOT / "data" / "processed" / "test_pairs.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test pairs not found at {test_path}. Run preprocessing first."
        )
    eval_df = pd.read_csv(test_path)
    logger.info(f"Loaded {len(eval_df)} test pairs.")

    # Cap to config limit for speed
    n_eval = cfg["evaluation"]["num_eval_samples"]
    if len(eval_df) > n_eval:
        eval_df = eval_df.sample(n=n_eval, random_state=42).reset_index(drop=True)
        logger.info(f"Capped to {n_eval} eval samples.")

    # Load Kaggle structured data for NER eval (has Skills column)
    ner_source = (
        PROJECT_ROOT / "data" / "raw" / "eval_dataset" / "resume_dataset_1200.csv"
    )
    ner_df = pd.read_csv(ner_source) if ner_source.exists() else pd.DataFrame()

    # ── Retrieval metrics ──────────────────────────────────────────────────────
    logger.info("Computing retrieval metrics...")
    retrieval_metrics = evaluate_retrieval(model_path, eval_df, k_values)
    for k, v in retrieval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── NER metrics ────────────────────────────────────────────────────────────
    logger.info("Computing NER metrics...")
    ner_metrics = {}
    if not ner_df.empty:
        ner_metrics = evaluate_ner(ner_df.head(200))  # sample for speed
        logger.info(f"  NER F1: {ner_metrics.get('f1', 0):.4f}")
    else:
        logger.warning("Skipping NER eval — Kaggle source not found.")

    # ── Target metric check ────────────────────────────────────────────────────
    targets = cfg["evaluation"]["target_metrics"]
    target_check = {}
    for metric, target in targets.items():
        ner_key = metric.replace("ner_", "") if metric.startswith("ner_") else metric
        actual = retrieval_metrics.get(metric, ner_metrics.get(ner_key, None))
        if actual is not None:
            passed = actual >= target
            target_check[metric] = {
                "target": target,
                "actual": round(actual, 4),
                "passed": passed,
            }
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status} | {metric}: {actual:.4f} (target: {target})")

    report = {
        "retrieval": retrieval_metrics,
        "ner": ner_metrics,
        "target_check": target_check,
        "eval_samples": len(eval_df),
        "model_path": str(model_path) if model_path else "base_model",
    }

    # Save report
    report_path = PROJECT_ROOT / "logs" / "evaluation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Evaluation report saved to {report_path}")

    # Log to MLflow
    if log_to_mlflow:
        try:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
            with mlflow.start_run(run_name="evaluation"):
                mlflow.log_metrics(
                    {
                        **retrieval_metrics,
                        **{f"ner_{k}": v for k, v in ner_metrics.items()},
                    }
                )
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")
                mlflow.set_tag("stage", "evaluation")
                logger.success("Evaluation metrics logged to MLflow.")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HireLens evaluation suite.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to fine-tuned model directory (uses base model if not provided).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    model_path = args.model_path or str(
        PROJECT_ROOT / "models" / "fine_tuned" / "hirelens_matcher"
    )

    run_evaluation(
        model_path=model_path,
        log_to_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()

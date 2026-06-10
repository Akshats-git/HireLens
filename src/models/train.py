"""
Fine-tuning script for HireLens resume-JD matcher.

Fine-tunes all-MiniLM-L6-v2 using sentence-transformers 3.x API with
CosineSimilarityLoss on the weak-supervision training pairs from Phase 2.

GPU: Targets GTX 1650 (4 GB VRAM) — fp16 enabled, batch sizes tuned accordingly.

Usage:
    python -m src.models.train
    python -m src.models.train --epochs 3 --batch-size 16 --no-fp16
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import CosineSimilarityLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}")
logger.add(PROJECT_ROOT / "logs" / "training.log", rotation="100 MB", retention="30 days", level="DEBUG")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pairs(csv_path: Path, max_samples: int | None = None) -> pd.DataFrame:
    """
    Load training/validation pairs from CSV.

    Expected columns: resume_text, job_description, label (0.0 or 1.0)

    Args:
        csv_path: Path to the pairs CSV.
        max_samples: Cap number of rows (useful for debugging).

    Returns:
        DataFrame with at least [resume_text, job_description, label].
    """
    df = pd.read_csv(csv_path)
    required = {"resume_text", "job_description", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairs CSV missing columns: {missing}")

    df = df.dropna(subset=["resume_text", "job_description", "label"])
    df["label"] = df["label"].astype(float).clip(0.0, 1.0)
    df["resume_text"] = df["resume_text"].str.strip()
    df["job_description"] = df["job_description"].str.strip()

    # Drop rows with very short texts
    df = df[
        (df["resume_text"].str.len() > 50) &
        (df["job_description"].str.len() > 50)
    ]

    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    logger.info(f"Loaded {len(df)} pairs from {csv_path.name} | "
                f"pos: {(df['label'] == 1.0).sum()}, neg: {(df['label'] == 0.0).sum()}")
    return df.reset_index(drop=True)


def df_to_hf_dataset(df: pd.DataFrame):
    """
    Convert a pairs DataFrame into a HuggingFace Dataset suitable for
    sentence-transformers 3.x CosineSimilarityLoss.

    Required columns: sentence1, sentence2, label
    """
    from datasets import Dataset
    return Dataset.from_dict({
        "sentence1": df["resume_text"].tolist(),
        "sentence2": df["job_description"].tolist(),
        "label": df["label"].astype(float).tolist(),
    })


# ── Evaluator builder ─────────────────────────────────────────────────────────

def build_evaluator(val_df: pd.DataFrame, name: str = "val") -> EmbeddingSimilarityEvaluator:
    """
    Build an EmbeddingSimilarityEvaluator from a validation DataFrame.

    Args:
        val_df: Validation pairs DataFrame.
        name: Evaluator name (used in MLflow metric keys).

    Returns:
        EmbeddingSimilarityEvaluator instance.
    """
    return EmbeddingSimilarityEvaluator(
        sentences1=val_df["resume_text"].tolist(),
        sentences2=val_df["job_description"].tolist(),
        scores=val_df["label"].astype(float).tolist(),
        main_similarity=SimilarityFunction.COSINE,
        name=name,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    fp16: bool | None = None,
    max_train_samples: int | None = None,
    run_name: str = "finetune-miniLM",
) -> SentenceTransformer:
    """
    Fine-tune all-MiniLM-L6-v2 and register the best model in MLflow.

    Args:
        epochs: Override config num_train_epochs.
        batch_size: Override config per_device_train_batch_size.
        fp16: Override config fp16 flag.
        max_train_samples: Cap training set size (for quick experiments).
        run_name: MLflow run name.

    Returns:
        Fine-tuned SentenceTransformer model.
    """
    cfg = _load_config()
    ft = cfg["model"]["fine_tuning"]
    base_model_name = cfg["model"]["base"]["name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve params (CLI args override config)
    n_epochs = epochs if epochs is not None else ft["num_train_epochs"]
    train_bs = batch_size if batch_size is not None else ft["per_device_train_batch_size"]
    use_fp16 = fp16 if fp16 is not None else (ft["fp16"] and device == "cuda")
    output_dir = PROJECT_ROOT / ft["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HireLens Model Fine-tuning")
    logger.info(f"  Base model : {base_model_name}")
    logger.info(f"  Device     : {device}")
    logger.info(f"  Epochs     : {n_epochs}")
    logger.info(f"  Batch size : {train_bs}")
    logger.info(f"  FP16       : {use_fp16}")
    logger.info("=" * 60)

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = PROJECT_ROOT / "data" / "processed" / "train_pairs.csv"
    val_path   = PROJECT_ROOT / "data" / "processed" / "val_pairs.csv"

    train_df = load_pairs(train_path, max_samples=max_train_samples)
    val_df   = load_pairs(val_path)

    train_dataset = df_to_hf_dataset(train_df)
    evaluator = build_evaluator(val_df, name="val")

    # ── Model & loss ──────────────────────────────────────────────────────────
    model = SentenceTransformer(base_model_name, device=device)
    model.max_seq_length = 256  # all-MiniLM-L6-v2 supports max 256

    loss = CosineSimilarityLoss(model)

    # ── Training arguments ────────────────────────────────────────────────────
    steps_per_epoch = len(train_df) // train_bs
    warmup = min(ft["warmup_steps"], steps_per_epoch)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=ft["per_device_eval_batch_size"],
        warmup_steps=warmup,
        weight_decay=ft["weight_decay"],
        learning_rate=ft["learning_rate"],
        lr_scheduler_type=ft["lr_scheduler_type"],
        eval_strategy=ft["evaluation_strategy"],
        save_strategy=ft["save_strategy"],
        load_best_model_at_end=ft["load_best_model_at_end"],
        metric_for_best_model=ft["metric_for_best_model"],
        fp16=use_fp16,
        dataloader_num_workers=0,  # 0 avoids CUDA fork issues on Linux
        report_to="none",           # disable HF hub reporting; we use MLflow
        logging_steps=50,
        save_total_limit=2,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params({
            "base_model": base_model_name,
            "epochs": n_epochs,
            "train_batch_size": train_bs,
            "learning_rate": ft["learning_rate"],
            "warmup_steps": warmup,
            "fp16": use_fp16,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "max_seq_length": 256,
            "loss": "CosineSimilarityLoss",
        })
        mlflow.set_tags(cfg["mlflow"]["run_tags"])
        mlflow.set_tag("stage", "fine_tuning")

        # ── Train ─────────────────────────────────────────────────────────────
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=evaluator,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.success("Training complete.")

        # ── Evaluate ──────────────────────────────────────────────────────────
        eval_results = evaluator(model)
        logger.info(f"Validation results: {eval_results}")

        # Log eval metrics
        if isinstance(eval_results, dict):
            mlflow.log_metrics({k: float(v) for k, v in eval_results.items()})
        else:
            mlflow.log_metric("val_cosine_pearson", float(eval_results))

        # ── Save model ────────────────────────────────────────────────────────
        model.save(str(output_dir))
        logger.info(f"Model saved to {output_dir}")

        # Log model path as param (avoids S3 upload when artifact store is remote)
        mlflow.log_param("model_saved_path", str(output_dir))
        try:
            mlflow.log_artifacts(str(output_dir), artifact_path="model")
            logger.success(f"Model artifacts logged to MLflow. Run ID: {run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow artifact upload skipped (no remote store configured): {e}")
            logger.info(f"Model is saved locally at: {output_dir}")

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune HireLens resume matcher.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 training")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training set for quick experiments")
    parser.add_argument("--run-name", default="finetune-miniLM")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fp16=not args.no_fp16 if args.no_fp16 else None,
        max_train_samples=args.max_train_samples,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

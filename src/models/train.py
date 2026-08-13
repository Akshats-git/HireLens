"""
Fine-tuning for the resume/posting matcher.

Fine-tunes the configured sentence-transformer with CosineSimilarityLoss on the
weak-supervision pairs from the preprocess stage. Hyperparameters come from
model.fine_tuning in configs/config.yaml; CLI flags override them.

Usage:
    python -m src.models.train
    python -m src.models.train --epochs 3 --batch-size 16 --no-fp16
"""

import argparse
import json
import os

import mlflow
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sentence_transformers.losses import CosineSimilarityLoss

from src.config import PROJECT_ROOT, get_section
from src.logging_setup import configure_logging

TRAIN_PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "train_pairs.csv"
VAL_PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "val_pairs.csv"
METRICS_PATH = PROJECT_ROOT / "logs" / "training_metrics.json"

# Pairs with either side shorter than this are extraction noise.
MIN_PAIR_TEXT_CHARS = 50

REQUIRED_COLUMNS = {"resume_text", "job_description", "label"}


# ── Data ──────────────────────────────────────────────────────────────────────


def load_pairs(csv_path, max_samples: int | None = None) -> pd.DataFrame:
    """
    Load and clean labelled training pairs.

    Args:
        csv_path: Path to a pairs CSV.
        max_samples: Optional cap on the number of rows, for quick experiments.

    Returns:
        DataFrame with [resume_text, job_description, label].

    Raises:
        ValueError: If any required column is absent.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df["label"] = df["label"].astype(float).clip(0.0, 1.0)
    df["resume_text"] = df["resume_text"].str.strip()
    df["job_description"] = df["job_description"].str.strip()
    df = df[
        (df["resume_text"].str.len() > MIN_PAIR_TEXT_CHARS)
        & (df["job_description"].str.len() > MIN_PAIR_TEXT_CHARS)
    ]

    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    logger.info(
        f"Loaded {len(df)} pairs from {csv_path.name} "
        f"({(df['label'] == 1.0).sum()} positive, {(df['label'] == 0.0).sum()} negative)"
    )
    return df.reset_index(drop=True)


def to_hf_dataset(df: pd.DataFrame):
    """Convert a pairs DataFrame to the column layout CosineSimilarityLoss expects."""
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "sentence1": df["resume_text"].tolist(),
            "sentence2": df["job_description"].tolist(),
            "label": df["label"].astype(float).tolist(),
        }
    )


def build_evaluator(val_df: pd.DataFrame, name: str = "val"):
    """Build a cosine-similarity evaluator over the validation pairs."""
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
    run_name: str = "finetune",
) -> SentenceTransformer:
    """
    Fine-tune the base model and record the run in MLflow.

    Args:
        epochs: Overrides the configured epoch count.
        batch_size: Overrides the configured training batch size.
        fp16: Overrides mixed-precision training.
        max_train_samples: Caps the training set, for quick experiments.
        run_name: MLflow run name.

    Returns:
        The fine-tuned model.
    """
    model_config = get_section("model")
    base_config = model_config["base"]
    tuning = model_config["fine_tuning"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = epochs if epochs is not None else tuning["num_train_epochs"]
    batch_size = (
        batch_size if batch_size is not None else tuning["per_device_train_batch_size"]
    )
    use_fp16 = fp16 if fp16 is not None else (tuning["fp16"] and device == "cuda")
    max_seq_length = base_config["max_seq_length"]
    output_dir = PROJECT_ROOT / tuning["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HireLens model fine-tuning")
    logger.info(f"  Base model : {base_config['name']}")
    logger.info(f"  Device     : {device}")
    logger.info(f"  Epochs     : {epochs}")
    logger.info(f"  Batch size : {batch_size}")
    logger.info(f"  FP16       : {use_fp16}")
    logger.info("=" * 60)

    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {properties.total_memory / 1e9:.1f} GB"
        )

    train_df = load_pairs(TRAIN_PAIRS_PATH, max_samples=max_train_samples)
    val_df = load_pairs(VAL_PAIRS_PATH)

    model = SentenceTransformer(base_config["name"], device=device)
    model.max_seq_length = max_seq_length
    evaluator = build_evaluator(val_df)

    steps_per_epoch = max(len(train_df) // batch_size, 1)
    warmup_steps = min(tuning["warmup_steps"], steps_per_epoch)

    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=tuning["per_device_eval_batch_size"],
        warmup_steps=warmup_steps,
        weight_decay=tuning["weight_decay"],
        learning_rate=tuning["learning_rate"],
        lr_scheduler_type=tuning["lr_scheduler_type"],
        eval_strategy=tuning["evaluation_strategy"],
        save_strategy=tuning["save_strategy"],
        load_best_model_at_end=tuning["load_best_model_at_end"],
        metric_for_best_model=tuning["metric_for_best_model"],
        fp16=use_fp16,
        # Zero workers avoids CUDA re-initialisation in forked dataloader processes.
        dataloader_num_workers=0,
        # MLflow is driven explicitly below; leave the Trainer's own reporting off.
        report_to="none",
        logging_steps=tuning["logging_steps"],
        save_total_limit=tuning["save_total_limit"],
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(get_section("mlflow")["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "base_model": base_config["name"],
                "epochs": epochs,
                "train_batch_size": batch_size,
                "learning_rate": tuning["learning_rate"],
                "warmup_steps": warmup_steps,
                "fp16": use_fp16,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "max_seq_length": max_seq_length,
                "loss": "CosineSimilarityLoss",
            }
        )
        mlflow.set_tags(get_section("mlflow")["run_tags"])
        mlflow.set_tag("stage", "fine_tuning")

        trainer = SentenceTransformerTrainer(
            model=model,
            args=arguments,
            train_dataset=to_hf_dataset(train_df),
            loss=CosineSimilarityLoss(model),
            evaluator=evaluator,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.success("Training complete.")

        metrics = _normalise_eval_results(evaluator(model))
        logger.info(f"Validation results: {metrics}")
        mlflow.log_metrics(metrics)

        _write_metrics_file(
            {
                **metrics,
                "epochs": epochs,
                "train_batch_size": batch_size,
                "learning_rate": tuning["learning_rate"],
                "train_samples": len(train_df),
                "val_samples": len(val_df),
            }
        )

        model.save(str(output_dir))
        logger.info(f"Model saved to {output_dir}")
        mlflow.log_param("model_saved_path", str(output_dir))

        try:
            mlflow.log_artifacts(str(output_dir), artifact_path="model")
            logger.success(f"Model artifacts logged to run {run.info.run_id}")
        except Exception as exc:
            logger.warning(
                f"Artifact upload skipped ({exc}) — the model is on disk at {output_dir}."
            )

    return model


def _normalise_eval_results(results) -> dict[str, float]:
    """Coerce the evaluator's return value into a flat metric dict."""
    if isinstance(results, dict):
        return {name: float(value) for name, value in results.items()}
    return {"val_cosine_pearson": float(results)}


def _write_metrics_file(metrics: dict) -> None:
    """Write the DVC-tracked training metrics file."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Training metrics written to {METRICS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the HireLens matcher.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--no-fp16", action="store_true", help="Disable mixed-precision training"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the training set, for quick experiments",
    )
    parser.add_argument("--run-name", default="finetune")
    args = parser.parse_args()

    configure_logging(log_file="training.log")
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        fp16=False if args.no_fp16 else None,
        max_train_samples=args.max_train_samples,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

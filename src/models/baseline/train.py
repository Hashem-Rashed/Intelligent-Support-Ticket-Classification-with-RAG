"""
Train TF‑IDF + Logistic Regression on merged support data (all categories).
Updated with better defaults and optional class weighting.
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.models.baseline.tfidf_logreg import TFIDFLogReg
from src.models.evaluation import evaluate_model, plot_confusion_matrix
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_merged_data(data_path: str = None) -> pd.DataFrame:
    """Load merged support data."""
    if data_path is None:
        data_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} rows from {data_path}")
    return df


def balance_dataset(df: pd.DataFrame, target_col: str = "category", samples_per_class: int = None) -> pd.DataFrame:
    """Undersample majority classes, optionally oversample minority.
    If samples_per_class is None, no balancing occurs."""
    if samples_per_class is None:
        logger.info("No balancing applied.")
        return df
    logger.info(f"Balancing dataset to {samples_per_class} samples per class")
    balanced_dfs = []
    for cat in df[target_col].unique():
        cat_df = df[df[target_col] == cat]
        if len(cat_df) >= samples_per_class:
            sampled = cat_df.sample(n=samples_per_class, random_state=42)
        else:
            sampled = cat_df.sample(n=samples_per_class, replace=True, random_state=42)
        balanced_dfs.append(sampled)
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset size: {len(balanced_df):,}")
    return balanced_df


def prepare_train_test(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "category",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split."""
    X = df[text_col].astype(str).tolist()
    y = df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: list,
    y_train: list,
    max_features: int = 15000,
    max_df: float = 0.6,
    min_df: int = 3,
    ngram_range: tuple = (1, 3),
    C: float = 1.0,
    class_weight: str = "balanced",
) -> TFIDFLogReg:
    """Instantiate and train model."""
    model = TFIDFLogReg(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        C=C,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(model: TFIDFLogReg, output_dir: Path):
    """Save model, config, and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tfidf_logreg_model.pkl"
    model.save(str(model_path))

    config = {
        "model_type": "TFIDFLogReg",
        "classes": model.label_encoder.classes_.tolist(),
        "num_classes": len(model.label_encoder.classes_),
        "max_features": model.max_features,
        "max_df": model.max_df,
        "min_df": model.min_df,
        "ngram_range": model.ngram_range,
        "C": model.C,
    }
    import json
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Model artifacts saved to {output_dir}")


def main(args):
    start_time = time.time()

    df = load_merged_data(args.data_path)
    if args.balance:
        df = balance_dataset(df, samples_per_class=args.samples_per_class)

    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=args.test_size)

    model = train_baseline_model(
        X_train, y_train,
        max_features=args.max_features,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram_range=(1, args.ngram_max),
        C=args.C,
        class_weight=args.class_weight,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Plot confusion matrix
    output_dir = Path(args.output_dir) if args.output_dir else Path(settings.PROJECT_ROOT) / "models" / "baseline"
    plot_confusion_matrix(y_test, y_pred, labels=model.label_encoder.classes_.tolist(),
                          save_path=output_dir / "confusion_matrix.png")

    save_model_artifacts(model, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"Total training time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--data-path", type=str, help="Path to merged CSV")
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--samples-per-class", type=int, default=None, help="If None, no balancing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=15000)
    parser.add_argument("--max-df", type=float, default=0.6)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    main(args)
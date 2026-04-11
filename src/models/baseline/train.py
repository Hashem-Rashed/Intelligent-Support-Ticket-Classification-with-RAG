import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure the project root is on sys.path when executing this script directly.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.baseline.tfidf_logreg import TFIDFLogReg
from src.models.evaluation import evaluate_model, get_confusion_matrix
from src.preprocessing.pipeline import run_pipeline
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_clean_ticket_data(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load and clean CRM ticket data for supervised training."""
    base_dir = Path(settings.PROJECT_ROOT)

    if raw_path is None:
        raw_path = base_dir / settings.DATA_RAW_PATH / "tickets.csv"
    else:
        raw_path = Path(raw_path)

    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"
    else:
        output_path = Path(output_path)

    if output_path.exists() and not force_rebuild:
        logger.info(f"Loading cleaned ticket data from {output_path}")
        df = pd.read_csv(output_path, low_memory=False, dtype={"clean_text": str, "Issue_Category": str})
    else:
        logger.info("Cleaning raw CRM tickets for baseline training")
        df = run_pipeline(
            input_path=str(raw_path),
            output_path=str(output_path),
            use_merged_data=False,
        )

    if "Issue_Category" not in df.columns:
        raise ValueError("Cleaned ticket data does not contain 'Issue_Category' labels")

    return df


def prepare_train_test(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "Issue_Category",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[list, list, list, list]:
    """Prepare a stratified train/test split from cleaned ticket data."""
    logger.info("Preparing stratified train/test split")

    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[text_col].astype(bool) & df[label_col].astype(bool)]

    y = df[label_col].values
    X = df[text_col].tolist()

    if len(df) == 0:
        raise ValueError("No valid ticket records were found after cleaning")

    logger.info(f"Dataset size after cleaning: {len(df)}")
    logger.info("Label distribution:\n%s", df[label_col].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train/test split complete")
    logger.info("Training set size: %d", len(X_train))
    logger.info("Validation set size: %d", len(X_test))

    return X_train, X_test, y_train, y_test


def train_baseline_model(
    X_train: list,
    y_train,
    max_features: int = 5000,
    max_df: float = 0.8,
    min_df: int = 2,
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> TFIDFLogReg:
    """Train the TF-IDF + Logistic Regression baseline model."""
    model = TFIDFLogReg(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline_model(
    model: TFIDFLogReg,
    X_test: list,
    y_test,
    labels: Optional[list] = None,
) -> Dict[str, object]:
    """Evaluate the baseline model and return metrics plus confusion matrix."""
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, labels=labels)
    conf_matrix = get_confusion_matrix(y_test, y_pred)

    logger.info("Confusion matrix shape: %s", conf_matrix.shape)
    logger.debug("Confusion matrix:\n%s", conf_matrix)

    return {
        "metrics": metrics,
        "confusion_matrix": conf_matrix,
    }


def run_baseline_training(
    raw_path: Optional[str] = None,
    cleaned_path: Optional[str] = None,
    model_output_path: Optional[str] = None,
    test_size: float = 0.2,
    class_weight: Optional[str] = None,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    """Run the full supervised baseline training pipeline."""
    df = load_clean_ticket_data(
        raw_path=raw_path,
        output_path=cleaned_path,
        force_rebuild=force_rebuild,
    )

    X_train, X_test, y_train, y_test = prepare_train_test(
        df,
        test_size=test_size,
    )

    model = train_baseline_model(
        X_train,
        y_train,
        class_weight=class_weight,
    )

    evaluation = evaluate_baseline_model(model, X_test, y_test)

    if model_output_path is None:
        model_output_path = Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / "baseline_ticket_classifier.pkl"

    model.save(str(model_output_path))
    logger.info("Saved baseline model to %s", model_output_path)

    return {
        "model": model,
        "evaluation": evaluation,
        "model_path": str(model_output_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline for ticket classification")
    parser.add_argument("--raw-path", type=str, help="Path to raw ticket CSV", default=None)
    parser.add_argument("--cleaned-path", type=str, help="Path to save cleaned ticket file", default=None)
    parser.add_argument("--model-output-path", type=str, help="Path to save trained model", default=None)
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation set size fraction")
    parser.add_argument("--class-weight", type=str, choices=["balanced", "None"], default="None", help="Optional class weighting strategy")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild cleaned data even if cached")

    args = parser.parse_args()
    class_weight_value = None if args.class_weight == "None" else args.class_weight

    run_baseline_training(
        raw_path=args.raw_path,
        cleaned_path=args.cleaned_path,
        model_output_path=args.model_output_path,
        test_size=args.test_size,
        class_weight=class_weight_value,
        force_rebuild=args.force_rebuild,
    )
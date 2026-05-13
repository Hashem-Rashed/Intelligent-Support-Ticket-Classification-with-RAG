"""
Command-line script to train transformer model with GPU memory optimizations.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[3]))
from src.models.transformer.bert_finetune import BERTFineTune
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_transformer_from_data(
    data_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    samples_per_class: int = None,   # None = no balancing
    test_size: float = 0.2,
    epochs: int = 5,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    max_length: int = 256,
    progress_callback=None,
):
    """Load data, optionally balance, split, train, and save."""
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Optional balancing
    if samples_per_class and samples_per_class > 0:
        balanced_dfs = []
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            if len(cat_df) >= samples_per_class:
                sampled = cat_df.sample(n=samples_per_class, random_state=42)
            else:
                sampled = cat_df.sample(n=samples_per_class, replace=True, random_state=42)
            balanced_dfs.append(sampled)
        df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Balanced dataset: {len(df)} samples")

    X = df["clean_text"].tolist()
    y = df["category"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = BERTFineTune(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_gpu=True,
        max_length=max_length,
        learning_rate=3e-5,
    )
    model.fit(X_train, y_train, progress_callback=progress_callback)
    model.save(output_dir)

    # Evaluate
    from src.models.evaluation import evaluate_model, plot_confusion_matrix
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    logger.info(f"Test metrics: {metrics}")
    plot_confusion_matrix(y_test, y_pred, labels=model.classes_,
                          save_path=Path(output_dir) / "confusion_matrix.png")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--samples-per-class", type=int, default=None, help="If omitted, no balancing")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    train_transformer_from_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        samples_per_class=args.samples_per_class,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length,
    )
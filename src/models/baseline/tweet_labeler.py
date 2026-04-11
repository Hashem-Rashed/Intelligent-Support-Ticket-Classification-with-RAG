import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure the project root is on sys.path when executed directly.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.models.baseline.tfidf_logreg import TFIDFLogReg
from src.preprocessing.text_processing import clean_text
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_FILENAME = "baseline_ticket_classifier.pkl"
DEFAULT_TWEET_FILENAME = "twcs.csv"
DEFAULT_OUTPUT_FILENAME = "tweets_high_confidence.csv"


def get_default_model_path() -> Path:
    return Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / DEFAULT_MODEL_FILENAME


def get_default_tweet_path() -> Path:
    return Path(settings.PROJECT_ROOT) / settings.DATA_RAW_PATH / DEFAULT_TWEET_FILENAME


def get_default_output_path() -> Path:
    return Path(settings.PROJECT_ROOT) / settings.DATA_PROCESSED_PATH / DEFAULT_OUTPUT_FILENAME


def load_tweet_chunks(
    tweets_path: Optional[str] = None,
    chunksize: int = 50000,
    usecols: Optional[list] = None,
):
    if usecols is None:
        usecols = ["tweet_id", "text"]

    path = Path(tweets_path) if tweets_path else get_default_tweet_path()
    logger.info("Loading tweets from %s", path)
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"tweet_id": str, "text": str},
        chunksize=chunksize,
        on_bad_lines="skip",
        iterator=True,
    )


def clean_tweet_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].fillna("").astype(str).apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    logger.info("Cleaned chunk: %d rows remain after text cleanup", len(df))
    return df


def batch_predict(
    model: TFIDFLogReg,
    texts: list,
    batch_size: int = 5000,
) -> Dict[str, np.ndarray]:
    predictions = []
    probabilities = []

    for offset in range(0, len(texts), batch_size):
        batch = texts[offset:offset + batch_size]
        probabilities_batch = model.predict_proba(batch)
        predictions_batch = model.predict(batch)
        predictions.extend(predictions_batch.tolist())
        probabilities.extend(probabilities_batch.tolist())

    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=object)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }


def label_tweet_chunk(
    df: pd.DataFrame,
    model: TFIDFLogReg,
    confidence_threshold: float = 0.7,
    batch_size: int = 5000,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    results = batch_predict(model, df["clean_text"].tolist(), batch_size=batch_size)
    proba = results["probabilities"]
    labels = results["predictions"]

    df = df.copy()
    df["predicted_category"] = labels
    df["confidence"] = np.max(proba, axis=1)
    df["source"] = "twitter"

    high_confidence = df[df["confidence"] >= confidence_threshold].copy()
    logger.info(
        "Chunk labeled: %d high-confidence rows out of %d",
        len(high_confidence),
        len(df),
    )
    return high_confidence


def run_tweet_labeling(
    model_path: Optional[str] = None,
    tweets_path: Optional[str] = None,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.7,
    tweet_chunksize: int = 50000,
    prediction_batch_size: int = 5000,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    model_file = Path(model_path) if model_path else get_default_model_path()
    if not model_file.exists():
        raise FileNotFoundError(f"Baseline model not found at {model_file}")

    output_file = Path(output_path) if output_path else get_default_output_path()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading baseline model from %s", model_file)
    model = TFIDFLogReg.load(str(model_file))

    tweets_iter = load_tweet_chunks(tweets_path=tweets_path, chunksize=tweet_chunksize)
    total_rows = 0
    retained_rows = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(tweets_iter, start=1):
        logger.info("Processing tweet chunk %d", chunk_idx)
        cleaned = clean_tweet_chunk(chunk)
        total_rows += len(cleaned)

        high_confidence = label_tweet_chunk(
            cleaned,
            model,
            confidence_threshold=confidence_threshold,
            batch_size=prediction_batch_size,
        )
        retained_rows += len(high_confidence)

        if not high_confidence.empty:
            high_confidence.to_csv(
                output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                encoding="utf-8",
            )
            first_chunk = False

    logger.info("Tweet labeling complete")
    logger.info("Total cleaned tweets processed: %d", total_rows)
    logger.info("High-confidence tweets retained: %d", retained_rows)
    logger.info("Saved high-confidence tweets to %s", output_file)

    return {
        "model_path": str(model_file),
        "output_path": str(output_file),
        "total_processed": total_rows,
        "high_confidence_retained": retained_rows,
        "confidence_threshold": confidence_threshold,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label tweets with a trained baseline classifier and keep only high-confidence predictions")
    parser.add_argument("--model-path", type=str, default=None, help="Trained TF-IDF model file path")
    parser.add_argument("--tweets-path", type=str, default=None, help="Raw tweets CSV path")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save high-confidence labeled tweets")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Minimum prediction probability to keep a label")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Number of tweets to read per chunk")
    parser.add_argument("--batch-size", type=int, default=5000, help="Number of texts to predict per batch")
    parser.add_argument("--force-rebuild", action="store_true", help="Overwrite output file if it exists")

    args = parser.parse_args()

    out_path = args.output_path or str(get_default_output_path())
    if args.force_rebuild and Path(out_path).exists():
        Path(out_path).unlink()

    run_tweet_labeling(
        model_path=args.model_path,
        tweets_path=args.tweets_path,
        output_path=out_path,
        confidence_threshold=args.confidence_threshold,
        tweet_chunksize=args.chunk_size,
        prediction_batch_size=args.batch_size,
        force_rebuild=args.force_rebuild,
    )
"""
Merge tickets and Twitter data into a single dataset.
Now only keeps 5 categories: Account, Billing, Fraud, General Inquiry, Technical.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_CATEGORIES = {
    'Account', 'Billing', 'Fraud', 'General Inquiry', 'Technical'
}
VALID_CATEGORIES = TARGET_CATEGORIES


def filter_ticket_categories(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    original_count = len(df)
    df = df[df[category_col].isin(TARGET_CATEGORIES)]
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} rows with non‑target categories")
    return df


def filter_tweet_categories(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    original_count = len(df)
    df = df[df[category_col].isin(VALID_CATEGORIES)]
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} rows with non‑target tweet categories")
    return df


def merge_datasets(
    tickets_path: Optional[str] = None,
    tweets_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force_reprocess_tweets: bool = False
) -> pd.DataFrame:
    base_dir = Path(settings.PROJECT_ROOT)

    if tickets_path is None:
        tickets_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"
    if tweets_path is None:
        tweets_path = base_dir / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"
    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("=" * 70)
    logger.info("MERGING TICKETS AND TWITTER DATA (5 categories)")
    logger.info("=" * 70)
    logger.info(f"Target categories: {sorted(TARGET_CATEGORIES)}")

    logger.info(f"\nLoading tickets from {tickets_path}")
    tickets = pd.read_csv(tickets_path)
    logger.info(f"  Loaded {len(tickets):,} tickets")

    logger.info(f"\nLoading tweets from {tweets_path}")
    tweets = pd.read_csv(tweets_path)
    logger.info(f"  Loaded {len(tweets):,} tweets")

    tickets_std = pd.DataFrame()
    tickets_std['clean_text'] = tickets['clean_text'].astype(str)
    tickets_std['category'] = tickets['Issue_Category'].astype(str)
    tickets_std['source'] = 'ticket'
    tickets_std['confidence'] = 1.0

    tweets_std = pd.DataFrame()
    tweets_std['clean_text'] = tweets['clean_text'].astype(str)
    tweets_std['category'] = tweets['category'].astype(str)
    tweets_std['source'] = 'twitter'
    tweets_std['confidence'] = tweets['confidence'] if 'confidence' in tweets.columns else 0.7

    tickets_std = filter_ticket_categories(tickets_std, 'category')
    tweets_std = filter_tweet_categories(tweets_std, 'category')

    logger.info("\nMerging datasets...")
    merged = pd.concat([tickets_std, tweets_std], ignore_index=True)

    before = len(merged)
    merged = merged.drop_duplicates(subset=['clean_text'], keep='first')
    if len(merged) < before:
        logger.info(f"  Removed {before - len(merged)} duplicate rows after merge")

    logger.info(f"\nMerged dataset size: {len(merged):,} rows")
    logger.info(f"  - Tickets: {len(tickets_std):,} rows")
    logger.info(f"  - Twitter: {len(tweets_std):,} rows")

    logger.info("\nCategory Distribution After Merge:")
    cat_dist = merged['category'].value_counts()
    for cat, count in cat_dist.items():
        logger.info(f"  {cat}: {count:,} ({count/len(merged)*100:.1f}%)")

    merged.to_csv(output_path, index=False)
    logger.info(f"\nSaved merged dataset to {output_path}")

    return merged


if __name__ == "__main__":
    merge_datasets()
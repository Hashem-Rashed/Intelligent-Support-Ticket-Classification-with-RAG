"""
Merge tickets and Twitter data into a single dataset.
Tickets: 9 categories
Tweets: 9 categories (kept as-is)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ticket categories (9)
TICKET_CATEGORIES = {'Account', 'Billing', 'Fraud', 'General Inquiry', 'Technical',
    'Delivery', 'Feature Request', 'Customer Support', 'Security'}

# Tweet categories (9) - all valid
TWEET_CATEGORIES = {
    'Account', 'Billing', 'Fraud', 'General Inquiry', 'Technical',
    'Delivery', 'Feature Request', 'Customer Support', 'Security'
}

# All valid categories for merged data
VALID_CATEGORIES = TICKET_CATEGORIES | TWEET_CATEGORIES


def filter_ticket_categories(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Filter tickets to only keep 9 categories."""
    original_count = len(df)
    df = df[df[category_col].isin(TICKET_CATEGORIES)]
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} rows with invalid ticket categories")
    return df


def filter_tweet_categories(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """Filter tweets to only keep valid tweet categories."""
    original_count = len(df)
    df = df[df[category_col].isin(VALID_CATEGORIES)]
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} rows with invalid tweet categories")
    return df


def resolve_fraud_security_overpredictions(text: str, predicted_class: str, confidence: float) -> str:
    """
    Post‑processing rule to override Fraud/Security predictions that are likely false positives.
    Use this after model prediction (baseline or transformer).
    """
    if predicted_class not in ('Fraud', 'Security'):
        return predicted_class
    if confidence >= 0.75:   # high confidence predictions are kept
        return predicted_class

    text_lower = text.lower()
    # Override if text contains typical non‑fraud/non‑security language
    if any(phrase in text_lower for phrase in ['how to', 'suggest', 'feature', 'why does', 'not working', 'please help']):
        return 'General Inquiry'
    if any(word in text_lower for word in ['password reset', 'login', '2fa', 'authentication', 'locked out']):
        return 'Account'
    if any(word in text_lower for word in ['refund', 'charge', 'bill', 'subscription', 'payment']):
        return 'Billing'
    if any(word in text_lower for word in ['crash', 'bug', 'freeze', 'error', 'slow']):
        return 'Technical'
    return predicted_class


def merge_datasets(
    tickets_path: Optional[str] = None,
    tweets_path: Optional[str] = None,
    output_path: Optional[str] = None,
    force_reprocess_tweets: bool = False
) -> pd.DataFrame:
    """
    Merge ticket and Twitter datasets.
    Tickets: 9 categories
    Tweets: 9 categories (preserved)
    """
    base_dir = Path(settings.PROJECT_ROOT)

    if tickets_path is None:
        tickets_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"

    if tweets_path is None:
        tweets_path = base_dir / settings.DATA_PROCESSED_PATH / "tweets_processed.csv"

    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "merged_support_data.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("=" * 70)
    logger.info("MERGING TICKETS AND TWITTER DATA")
    logger.info("=" * 70)
    logger.info(f"Ticket categories (9): {sorted(TICKET_CATEGORIES)}")
    logger.info(f"Tweet categories (9): {sorted(TWEET_CATEGORIES)}")

    # Load tickets
    logger.info(f"\nLoading tickets from {tickets_path}")
    tickets = pd.read_csv(tickets_path)
    logger.info(f"  Loaded {len(tickets):,} tickets")

    # Load tweets
    logger.info(f"\nLoading tweets from {tweets_path}")
    tweets = pd.read_csv(tweets_path)
    logger.info(f"  Loaded {len(tweets):,} tweets")

    # Standardize column names
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

    # Filter categories (tickets only 9, tweets all 9)
    tickets_std = filter_ticket_categories(tickets_std, 'category')
    tweets_std = filter_tweet_categories(tweets_std, 'category')

    # Merge
    logger.info("\nMerging datasets...")
    merged = pd.concat([tickets_std, tweets_std], ignore_index=True)

    # Remove duplicates after merge
    before = len(merged)
    merged = merged.drop_duplicates(subset=['clean_text'], keep='first')
    if len(merged) < before:
        logger.info(f"  Removed {before - len(merged)} duplicate rows after merge")

    logger.info(f"\nMerged dataset size: {len(merged):,} rows")
    logger.info(f"  - Tickets: {len(tickets_std):,} rows")
    logger.info(f"  - Twitter: {len(tweets_std):,} rows")

    # Category distribution
    logger.info("\nCategory Distribution After Merge:")
    cat_dist = merged['category'].value_counts()
    for cat, count in cat_dist.items():
        logger.info(f"  {cat}: {count:,} ({count/len(merged)*100:.1f}%)")

    # Save
    merged.to_csv(output_path, index=False)
    logger.info(f"\nSaved merged dataset to {output_path}")

    return merged


if __name__ == "__main__":
    merge_datasets()
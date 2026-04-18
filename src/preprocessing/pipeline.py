"""
Main preprocessing pipeline for tickets.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.config import settings
from src.utils.logger import get_logger
from .text_processing import clean_text

logger = get_logger(__name__)


def run_pipeline(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    use_merged_data: bool = False
) -> pd.DataFrame:
    """
    Execute the preprocessing pipeline for tickets only.

    Args:
        input_path: Path to raw tickets.csv
        output_path: Path to save cleaned data
        use_merged_data: Deprecated - kept for compatibility

    Returns:
        Cleaned DataFrame
    """
    base_dir = Path(settings.PROJECT_ROOT)

    if output_path is None:
        output_path = base_dir / settings.DATA_PROCESSED_PATH / "tickets_cleaned.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("=" * 70)
    logger.info("STARTING PREPROCESSING PIPELINE (TICKETS ONLY)")
    logger.info("=" * 70)

    try:
        # Load tickets
        if input_path is None:
            input_path = base_dir / settings.DATA_RAW_PATH / "tickets.csv"

        logger.info(f"Loading tickets from {input_path}")
        data = pd.read_csv(input_path)

        # Use Ticket_Description as text
        data['clean_text'] = data['Ticket_Description'].astype(str)
        data['category'] = data['Issue_Category'].astype(str)
        data['source'] = 'ticket'

        # Remove duplicates based on clean_text
        before = len(data)
        data = data.drop_duplicates(subset=['clean_text'], keep='first')
        if len(data) < before:
            logger.info(f"  Removed {before - len(data)} duplicate tickets")

        # Clean text
        logger.info("\nCleaning text data...")
        data['clean_text'] = data['clean_text'].apply(
            lambda x: clean_text(x, max_words=8, remove_greetings_flag=True, is_twitter=False)
        )

        # Remove empty rows
        before = len(data)
        data = data[data['clean_text'].str.len() > 0]
        if len(data) < before:
            logger.info(f"  Removed {before - len(data)} rows with empty text")

        # Remove duplicates again after cleaning (same text may become identical after cleaning)
        before = len(data)
        data = data.drop_duplicates(subset=['clean_text'], keep='first')
        if len(data) < before:
            logger.info(f"  Removed {before - len(data)} duplicate tickets after cleaning")

        # Prepare final dataset
        logger.info("\nPreparing final dataset...")
        final_data = data[['clean_text', 'category']].copy()
        final_data.rename(columns={'category': 'Issue_Category'}, inplace=True)
        final_data['source'] = data['source']

        # Save
        final_data.to_csv(output_path, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Final dataset size: {len(final_data):,} rows")

        logger.info("\nCategory distribution:")
        for cat, count in final_data['Issue_Category'].value_counts().items():
            logger.info(f"  {cat}: {count:,} ({count/len(final_data)*100:.1f}%)")

        return final_data

    except Exception as exc:
        logger.exception("Preprocessing pipeline failed")
        raise


if __name__ == "__main__":
    run_pipeline()
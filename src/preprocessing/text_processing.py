import re
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text):
    """Normalize and remove noise from a single text string."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove special characters but keep letters, numbers, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # ============================================================
    # FIX: Remove random word suffix that causes overfitting
    # The random words always appear after the first '?' or '.'
    # Keep only the meaningful part before the first punctuation
    # ============================================================
    for delimiter in ['?', '.']:
        if delimiter in text:
            parts = text.split(delimiter)
            if len(parts) >= 2:
                # Take the first part + delimiter
                text = (parts[0] + delimiter).strip()
                break
    
    # If text is still too long (no punctuation found), keep first 20 words
    words = text.split()
    if len(words) > 20:
        text = ' '.join(words[:20])
    
    logger.debug(f"Cleaned: {text[:100]}...")
    return text


def merge_subject_description(data):
    """
    Combine subject and description fields into `full_text` for tickets.
    For Twitter data, just use the text field as is.
    """
    logger.info("Merging text fields into full_text")
    
    # Check if we have ticket-specific columns
    if 'Ticket_Subject' in data.columns and 'Ticket_Description' in data.columns:
        # For CRM tickets: combine subject and description
        data["full_text"] = (
            data["Ticket_Subject"].astype(str) + " " +
            data["Ticket_Description"].astype(str)
        )
    elif 'text' in data.columns:
        # For already standardized data (like merged dataset)
        data["full_text"] = data["text"].astype(str)
    else:
        raise ValueError("No suitable text columns found for processing")
    
    return data


# Import pandas for NA check
import pandas as pd
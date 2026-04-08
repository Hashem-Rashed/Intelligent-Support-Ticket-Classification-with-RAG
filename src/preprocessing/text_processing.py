import re
from nltk.corpus import stopwords
from src.utils.logger import get_logger

logger = get_logger(__name__)

stop = set(stopwords.words("english"))

def clean_text(text):
    """Normalize and remove noise from a single text string."""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    cleaned = " ".join([w for w in text.split() if w not in stop])

    logger.debug(f"Original: {text[:100]}... | Cleaned: {cleaned[:100]}...")
    return cleaned

def merge_subject_description(data):
    """Combine subject and description fields into `full_text` for tickets.
    For Twitter data, just use the text field as is."""
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
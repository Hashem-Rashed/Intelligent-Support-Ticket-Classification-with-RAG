"""
Text cleaning utilities for tickets and tweets.
"""

import re
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

CUSTOM_STOPWORDS = {
    'hi', 'hello', 'hey', 'dear', 'support', 'team', 'please', 'thanks',
    'thank', 'would', 'could', 'get', 'make', 'want', 'need', 'ask', 'tell',
    'via', 'rt', 'amp'
}

COMMON_GREETINGS = [
    "hi support", "hello support", "dear support", "hi team", "hello team",
    "dear team", "thank you for", "thanks for", "please help"
]


def remove_greetings(text: str) -> str:
    """Remove common greeting phrases."""
    if not text:
        return text

    text = text.lower().strip()

    for greeting in COMMON_GREETINGS:
        if text.startswith(greeting):
            remainder = text[len(greeting):].strip()
            if remainder.startswith(("i ", "my ", "the ", "a ", "to ", "with ")):
                parts = remainder.split(" ", 1)
                if len(parts) > 1:
                    remainder = parts[1]
            return remainder

    return text


def truncate_at_punctuation(text: str) -> str:
    """Keep only text before first '?' or '.' to remove random suffixes."""
    if not text:
        return text

    for delimiter in ['?', '.', '!']:
        if delimiter in text:
            text = text.split(delimiter)[0] + delimiter
            break

    return text.strip()


def clean_text(text, max_words=8, remove_greetings_flag=True, is_twitter=False):
    """
    Clean text for classification.

    Args:
        text: Input text string
        max_words: Maximum words to keep
        remove_greetings_flag: Remove common greetings
        is_twitter: If True, apply Twitter-specific cleaning
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Twitter-specific cleaning
    if is_twitter:
        text = re.sub(r'@\w+', '', text)  # Remove @mentions
        text = re.sub(r'^rt\s+', '', text, flags=re.IGNORECASE)  # Remove RT

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove everything after first '?' or '.'
    text = truncate_at_punctuation(text)

    # Remove special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove greetings (only for tickets)
    if remove_greetings_flag and not is_twitter:
        text = remove_greetings(text)

    # Remove stopwords and limit length
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 2]

    if len(words) > max_words:
        words = words[:max_words]

    return ' '.join(words)


def merge_subject_description(data):
    """Use only Ticket_Description, drop Ticket_Subject."""
    logger.info("Using Ticket_Description as full_text")

    if 'Ticket_Description' in data.columns:
        data["full_text"] = data["Ticket_Description"].astype(str)
    elif 'clean_text' in data.columns:
        data["full_text"] = data["clean_text"].astype(str)
    elif 'text' in data.columns:
        data["full_text"] = data["text"].astype(str)
    else:
        raise ValueError("No suitable text columns found")

    return data
"""
Text cleaning utilities for tickets and tweets.
Improved: no word limit, keeps important punctuation, preserves negations.
"""

import re
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Stopwords – removed short negation words
CUSTOM_STOPWORDS = {
    'hi', 'hello', 'hey', 'dear', 'support', 'team', 'please', 'thanks',
    'thank', 'would', 'could', 'get', 'make', 'want', 'need', 'ask', 'tell',
    'via', 'rt', 'amp', 'just', 'like', 'well', 'also', 'even', 'still'
}

# Keep these short words (negations and important ones)
KEEP_SHORT_WORDS = {'no', 'not', 'don', 'doesn', 'didn', 'won', 'wouldn',
                    'couldn', 'shouldn', 'isn', 'aren', 'wasn', 'weren',
                    'haven', 'hasn', 'hadn', 'can', 'let', 'out', 'off'}

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


def clean_text(text, max_words=None, remove_greetings_flag=True, is_twitter=False):
    """
    Clean text for classification.

    Args:
        text: Input text string
        max_words: Maximum words to keep (None = no limit)
        remove_greetings_flag: Remove common greetings (tickets only)
        is_twitter: If True, apply Twitter-specific cleaning
    """
    if pd.isna(text):
        return ""

    text = str(text)

    if is_twitter:
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'^rt\s+', '', text, flags=re.IGNORECASE)

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Keep important punctuation (! and ?) but remove others
    text = re.sub(r"[^a-z\s!?']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if remove_greetings_flag and not is_twitter:
        text = remove_greetings(text)

    words = text.split()

    filtered_words = []
    for w in words:
        if w in KEEP_SHORT_WORDS:
            filtered_words.append(w)
        elif len(w) > 2 and w not in CUSTOM_STOPWORDS:
            filtered_words.append(w)
        elif len(w) == 2 and w not in CUSTOM_STOPWORDS and w.isalpha():
            filtered_words.append(w)

    if max_words is not None and max_words > 0 and len(filtered_words) > max_words:
        filtered_words = filtered_words[:max_words]

    return ' '.join(filtered_words)


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
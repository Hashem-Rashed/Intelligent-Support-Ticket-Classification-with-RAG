"""
Text cleaning and preprocessing utilities.
"""
import re
import string
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """Text preprocessing and cleaning utilities."""

    def __init__(self, lowercase: bool = True, remove_special_chars: bool = True):
        """
        Initialize TextCleaner.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_special_chars: Whether to remove special characters
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars

    def clean(self, text: str) -> str:
        """
        Clean text by applying various preprocessing steps.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters and digits
        if self.remove_special_chars:
            text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts in batch.

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]


def remove_stopwords(tokens: List[str], stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Remove stopwords from tokenized text.

    Args:
        tokens: List of tokens
        stopwords: List of stopwords to remove

    Returns:
        List of tokens with stopwords removed
    """
    if stopwords is None:
        # Common English stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "of", "as", "by", "from", "with", "about", "into", "through", "during",
            "before", "after", "above", "below", "up", "down", "out", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "same",
            "so", "than", "too", "very", "can", "own", "so", "just", "should", "now"
        }

    return [token for token in tokens if token.lower() not in stopwords]

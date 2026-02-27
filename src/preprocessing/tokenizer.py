"""
Tokenization utilities for text processing.
"""
import re
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimpleTokenizer:
    """Basic whitespace and punctuation-based tokenizer."""

    def __init__(self, lowercase: bool = True):
        """
        Initialize SimpleTokenizer.

        Args:
            lowercase: Whether to convert tokens to lowercase
        """
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []

        # Remove punctuation and split
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()

        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        return tokens

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts in batch.

        Args:
            texts: List of texts to tokenize

        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]


class AdvancedTokenizer:
    """Sentence and word-level tokenizer with advanced features."""

    def __init__(self, lowercase: bool = True):
        """
        Initialize AdvancedTokenizer.

        Args:
            lowercase: Whether to convert tokens to lowercase
        """
        self.lowercase = lowercase

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on common punctuation
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        tokenizer = SimpleTokenizer(lowercase=self.lowercase)
        return tokenizer.tokenize(text)

    def tokenize(self, text: str, return_sentences: bool = False) -> List[str]:
        """
        Full tokenization pipeline.

        Args:
            text: Text to tokenize
            return_sentences: If True, return sentences; otherwise return words

        Returns:
            List of tokens
        """
        if return_sentences:
            return self.tokenize_sentences(text)
        else:
            return self.tokenize_words(text)

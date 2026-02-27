"""
Unit tests for preprocessing module.
"""
import pytest
from src.preprocessing.cleaner import TextCleaner, remove_stopwords
from src.preprocessing.tokenizer import SimpleTokenizer, AdvancedTokenizer


class TestTextCleaner:
    """Test cases for TextCleaner."""

    def test_clean_removes_urls(self):
        """Test that URLs are removed."""
        cleaner = TextCleaner()
        text = "Check this https://example.com for details"
        result = cleaner.clean(text)
        assert "https" not in result

    def test_clean_lowercase(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        text = "HELLO WORLD"
        result = cleaner.clean(text)
        assert result == "hello world"

    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ["the", "quick", "brown", "fox"]
        result = remove_stopwords(tokens)
        assert "the" not in result
        assert "quick" in result


class TestSimpleTokenizer:
    """Test cases for SimpleTokenizer."""

    def test_tokenize(self):
        """Test basic tokenization."""
        tokenizer = SimpleTokenizer()
        text = "hello world"
        result = tokenizer.tokenize(text)
        assert len(result) == 2
        assert result[0] == "hello"

    def test_batch_tokenize(self):
        """Test batch tokenization."""
        tokenizer = SimpleTokenizer()
        texts = ["hello world", "foo bar"]
        result = tokenizer.batch_tokenize(texts)
        assert len(result) == 2
        assert len(result[0]) == 2

"""
Prompt engineering utilities for RAG system.
"""
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptTemplate:
    """Template for generating prompts."""

    def __init__(self, template: str):
        """
        Initialize PromptTemplate.

        Args:
            template: Prompt template with placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format prompt with variables.

        Args:
            **kwargs: Variables to fill in template

        Returns:
            Formatted prompt
        """
        return self.template.format(**kwargs)


class ClassificationPrompt(PromptTemplate):
    """Prompt template for ticket classification."""

    def __init__(self):
        """Initialize classification prompt template."""
        template = """You are an expert support ticket classifier. Based on the ticket content and relevant context, classify this ticket into one of the predefined categories.

Ticket:
{ticket_content}

Relevant Context:
{context}

Categories:
{categories}

Provide your classification and confidence level (0-1)."""

        super().__init__(template)

    def create(
        self,
        ticket_content: str,
        context: str,
        categories: List[str]
    ) -> str:
        """
        Create a classification prompt.

        Args:
            ticket_content: The ticket text
            context: Retrieved context
            categories: List of classification categories

        Returns:
            Formatted prompt
        """
        category_text = "\n".join([f"- {cat}" for cat in categories])
        return self.format(
            ticket_content=ticket_content,
            context=context,
            categories=category_text
        )


class RAGPrompt(PromptTemplate):
    """Prompt template for RAG-based responses."""

    def __init__(self):
        """Initialize RAG prompt template."""
        template = """You are a helpful support assistant. Use the provided context to answer questions.

Context:
{context}

Question:
{question}

Answer based only on the provided context."""

        super().__init__(template)

    def create(self, context: str, question: str) -> str:
        """
        Create a RAG prompt.

        Args:
            context: Retrieved context
            question: User question

        Returns:
            Formatted prompt
        """
        return self.format(context=context, question=question)


def get_prompt_template(template_type: str = "classification") -> PromptTemplate:
    """
    Get a prompt template by type.

    Args:
        template_type: Type of template ('classification', 'rag')

    Returns:
        PromptTemplate instance
    """
    if template_type == "classification":
        return ClassificationPrompt()
    elif template_type == "rag":
        return RAGPrompt()
    else:
        raise ValueError(f"Unknown template type: {template_type}")

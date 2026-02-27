"""
RAG Pipeline for intelligent ticket classification and response generation.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.rag.retriever import FAISSRetriever
from src.rag.generator import TextGenerator
from src.rag.prompt_engineering import ClassificationPrompt
from src.preprocessing.embedding_generator import EmbeddingGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for ticket classification.
    
    This pipeline combines:
    1. Document retrieval based on embeddings
    2. Prompt engineering with context
    3. Text generation for predictions
    """

    def __init__(
        self,
        retriever,
        generator: TextGenerator,
        embedding_generator: EmbeddingGenerator,
        categories: List[str],
    ):
        """
        Initialize RAGPipeline.

        Args:
            retriever: Document retriever instance
            generator: Text generator instance
            embedding_generator: Embedding generator instance
            categories: List of classification categories
        """
        self.retriever = retriever
        self.generator = generator
        self.embedding_generator = embedding_generator
        self.categories = categories
        self.prompt_template = ClassificationPrompt()

        logger.info("Initialized RAG Pipeline")

    def classify_ticket(
        self,
        ticket_text: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Classify a support ticket using RAG.

        Args:
            ticket_text: The ticket content
            top_k: Number of context documents to retrieve

        Returns:
            Dictionary with classification results
        """
        logger.info("Processing ticket with RAG pipeline")

        # Step 1: Generate embedding for the ticket
        ticket_embedding = self.embedding_generator.generate(ticket_text)

        # Step 2: Retrieve relevant documents
        context_docs = self.retriever.retrieve(ticket_embedding, top_k=top_k)
        context_text = "\n".join([doc[0] for doc in context_docs])

        # Step 3: Create prompt with context
        prompt = self.prompt_template.create(
            ticket_content=ticket_text,
            context=context_text,
            categories=self.categories
        )

        # Step 4: Generate classification response
        response = self.generator.generate(prompt)

        result = {
            "ticket": ticket_text,
            "response": response,
            "context_docs": context_docs,
            "prompt": prompt,
        }

        logger.info(f"Classification complete")
        return result

    def batch_classify(
        self,
        tickets: List[str],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Classify multiple tickets.

        Args:
            tickets: List of ticket texts
            top_k: Number of context documents per ticket

        Returns:
            List of classification results
        """
        results = []
        for ticket in tickets:
            result = self.classify_ticket(ticket, top_k=top_k)
            results.append(result)
        return results

    def add_documents(self, documents: List[str], embeddings: np.ndarray) -> None:
        """
        Add documents to the retriever's index.

        Args:
            documents: List of document texts
            embeddings: Document embeddings
        """
        self.retriever.add_documents(documents, embeddings)
        logger.info(f"Added {len(documents)} documents to RAG pipeline")

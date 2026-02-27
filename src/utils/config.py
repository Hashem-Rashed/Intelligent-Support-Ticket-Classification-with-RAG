"""
Configuration module for managing environment variables and settings.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_RAW_PATH: str = os.getenv("DATA_RAW_PATH", "data/raw")
    DATA_PROCESSED_PATH: str = os.getenv("DATA_PROCESSED_PATH", "data/processed")
    DATA_EMBEDDINGS_PATH: str = os.getenv("DATA_EMBEDDINGS_PATH", "data/embeddings")

    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "bert-base-uncased")
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    EPOCHS: int = int(os.getenv("EPOCHS", "5"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-5"))

    # RAG Configuration
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "True").lower() == "true"
    RETRIEVER_TYPE: str = os.getenv("RETRIEVER_TYPE", "faiss")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Vector Database
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ticket-embeddings")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ticket-classification")

    # Azure Configuration
    AZURE_SUBSCRIPTION_ID: Optional[str] = os.getenv("AZURE_SUBSCRIPTION_ID")
    AZURE_RESOURCE_GROUP: Optional[str] = os.getenv("AZURE_RESOURCE_GROUP")
    AZURE_MODEL_NAME: str = os.getenv("AZURE_MODEL_NAME", "ticket-classifier")
    AZURE_DEPLOYMENT_NAME: str = os.getenv("AZURE_DEPLOYMENT_NAME", "ticket-classifier-prod")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ticket_db.db")


# Create a single instance
settings = Settings()

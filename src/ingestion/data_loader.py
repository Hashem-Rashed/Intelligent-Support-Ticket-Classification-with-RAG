"""
Data loading utilities for support tickets.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_ticket_data(
    filepath: Union[str, Path],
    file_format: str = "csv",
    **kwargs
) -> pd.DataFrame:
    """
    Load ticket data from file.

    Args:
        filepath: Path to the data file
        file_format: File format ('csv', 'json', 'parquet', 'excel')
        **kwargs: Additional arguments to pass to the reader function

    Returns:
        DataFrame containing ticket data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading data from {filepath}")

    if file_format.lower() == "csv":
        df = pd.read_csv(filepath, **kwargs)
    elif file_format.lower() == "json":
        df = pd.read_json(filepath, **kwargs)
    elif file_format.lower() == "parquet":
        df = pd.read_parquet(filepath, **kwargs)
    elif file_format.lower() == "excel":
        df = pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def save_ticket_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    file_format: str = "csv",
    **kwargs
) -> None:
    """
    Save ticket data to file.

    Args:
        df: DataFrame to save
        filepath: Path to save the file
        file_format: File format ('csv', 'json', 'parquet', 'excel')
        **kwargs: Additional arguments to pass to the writer function
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving data to {filepath}")

    if file_format.lower() == "csv":
        df.to_csv(filepath, index=False, **kwargs)
    elif file_format.lower() == "json":
        df.to_json(filepath, **kwargs)
    elif file_format.lower() == "parquet":
        df.to_parquet(filepath, index=False, **kwargs)
    elif file_format.lower() == "excel":
        df.to_excel(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    logger.info(f"Saved {len(df)} records to {filepath}")

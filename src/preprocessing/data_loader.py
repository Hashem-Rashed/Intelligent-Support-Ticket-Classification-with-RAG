import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw tickets data from CSV.

    Args:
        path: filesystem path to the CSV file.

    Returns:
        A pandas DataFrame containing the ticket records.
    """
    logger.info(f"Loading data from {path}")
    data = pd.read_csv(path)
    logger.info(f"Loaded {len(data)} rows from {path}")
    return data

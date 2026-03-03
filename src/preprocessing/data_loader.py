import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw tickets data from CSV.
    """
    data = pd.read_csv(path)
    return data

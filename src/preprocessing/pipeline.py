import warnings
import pandas as pd
from src.ingestion.data_loader import load_data
from .text_processing import clean_text, merge_subject_description
from .data_merger import merge_datasets
import os
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_pipeline(
    input_path: str | None = None,
    output_path: str | None = None,
    use_merged_data: bool = True
):
    """Execute the preprocessing pipeline.
    
    The function is parameterised to support configurable paths and can be used
    programmatically by other components (e.g. training scripts or tests).
    
    Args:
        input_path: optional path to the raw data CSV. Defaults to
            ``{settings.DATA_RAW_PATH}/tickets.csv`` under project root.
            If use_merged_data is True, this parameter is ignored.
        output_path: optional destination for cleaned data. Defaults to
            ``{settings.DATA_PROCESSED_PATH}/tickets_cleaned.csv`` under project root.
        use_merged_data: If True, merge ticket and Twitter datasets first.
    
    Returns:
        The cleaned :class:`pandas.DataFrame` that was persisted to disk.
    """
    base_dir = settings.PROJECT_ROOT
    
    if output_path is None:
        output_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info("Starting preprocessing pipeline")
    try:
        # Load or merge data
        if use_merged_data:
            logger.info("Using merged ticket and Twitter dataset")
            merged_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "merged_support_data.csv")
            
            # Check if merged data already exists
            if os.path.exists(merged_path):
                logger.info(f"Loading existing merged data from {merged_path}")
                # Fix: Specify dtypes to avoid mixed type warning
                data = pd.read_csv(
                    merged_path,
                    low_memory=False,  # Read entire file to infer types correctly
                    dtype={
                        'id': str,  # Force ID column to string to handle mixed types
                        'text': str,
                        'category': str,
                        'source': str,
                        'Ticket_Subject': str,
                        'Ticket_Description': str
                    }
                )
            else:
                logger.info("Creating merged dataset from raw files")
                data = merge_datasets()
        else:
            if input_path is None:
                input_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "tickets.csv")
            logger.info(f"Loading data from {input_path}")
            data = load_data(input_path)
        
        # Drop unnecessary columns if they exist
        cols_to_drop = [
            "Customer_Name", "Customer_Email", "Assigned_Agent",
            "Submission_Date", "Ticket_ID", "Satisfaction_Score"
        ]
        # Only drop columns that exist in the dataframe
        existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
        if existing_cols_to_drop:
            data = data.drop(columns=existing_cols_to_drop)
        
        # Merge text fields
        data = merge_subject_description(data)
        
        # Clean text
        data["clean_text"] = data["full_text"].apply(clean_text)
        
        # Remove any rows with empty clean_text
        initial_count = len(data)
        data = data[data["clean_text"].str.len() > 0]
        if len(data) < initial_count:
            logger.info(f"Removed {initial_count - len(data)} rows with empty text")
        
        # Prepare final dataset
        if "category" in data.columns:
            # Use standardized category column
            final_data = data[["clean_text", "category"]].copy()
            final_data.rename(columns={"category": "Issue_Category"}, inplace=True)
        elif "Issue_Category" in data.columns:
            final_data = data[["clean_text", "Issue_Category"]]
        else:
            # If no category column exists, create a default one
            logger.warning("No category column found, creating default")
            final_data = data[["clean_text"]].copy()
            final_data["Issue_Category"] = "General"
        
        # Add source information for tracking (optional)
        if "source" in data.columns:
            final_data["source"] = data["source"]
        
        final_data.to_csv(output_path, index=False)
        logger.info(f"Pipeline executed successfully, output written to {output_path}")
        logger.info(f"Final dataset shape: {final_data.shape}")
        logger.info(f"Records by source: {final_data['source'].value_counts().to_dict() if 'source' in final_data.columns else 'N/A'}")
        
        return final_data
        
    except Exception as exc:
        logger.exception("Preprocessing pipeline failed")
        raise
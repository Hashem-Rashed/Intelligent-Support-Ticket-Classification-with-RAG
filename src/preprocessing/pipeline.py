from .data_loader import load_data
from .text_processing import clean_text, merge_subject_description
import os
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

def run_pipeline(input_path: str | None = None, output_path: str | None = None):
    """Execute the preprocessing pipeline.

    The function is parameterised to support configurable paths and can be used
    programmatically by other components (e.g. training scripts or tests).

    Args:
        input_path: optional path to the raw data CSV. Defaults to
            ``{settings.DATA_RAW_PATH}/tickets.csv`` under project root.
        output_path: optional destination for cleaned data. Defaults to
            ``{settings.DATA_PROCESSED_PATH}/tickets_cleaned.csv`` under project root.

    Returns:
        The cleaned :class:`pandas.DataFrame` that was persisted to disk.
    """
    base_dir = settings.PROJECT_ROOT

    if input_path is None:
        input_path = os.path.join(base_dir, settings.DATA_RAW_PATH, "tickets.csv")

    if output_path is None:
        output_path = os.path.join(base_dir, settings.DATA_PROCESSED_PATH, "tickets_cleaned.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Starting preprocessing pipeline")
    try:
        data = load_data(input_path)

        cols_to_drop = [
            "Customer_Name",
            "Customer_Email",
            "Assigned_Agent",
            "Submission_Date",
            "Ticket_ID",
            "Satisfaction_Score",
        ]

        data = data.drop(columns=cols_to_drop)
        data = merge_subject_description(data)
        data["clean_text"] = data["full_text"].apply(clean_text)
        final_data = data[["clean_text", "Issue_Category"]]

        final_data.to_csv(output_path, index=False)
        logger.info(f"Pipeline executed successfully, output written to {output_path}")
        return final_data
    except Exception as exc:
        logger.exception("Preprocessing pipeline failed")
        raise

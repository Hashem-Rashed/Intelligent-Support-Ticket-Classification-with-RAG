from .data_loader import load_data
from .text_processing import clean_text, merge_subject_description
import os

def run_pipeline():

    import os

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    input_path = os.path.join(BASE_DIR, "data/raw/tickets.csv")
    output_path = os.path.join(BASE_DIR, "data/interim/tickets_cleaned.csv")

    data = load_data(input_path)

    cols_to_drop = [
        "Customer_Name",
        "Customer_Email",
        "Assigned_Agent",
        "Submission_Date",
        "Ticket_ID",
        "Satisfaction_Score"
    ]

    data = data.drop(columns=cols_to_drop)

    data = merge_subject_description(data)

    data["clean_text"] = data["full_text"].apply(clean_text)

    final_data = data[["clean_text", "Issue_Category"]]

    final_data.to_csv(output_path, index=False)

    print("Pipeline executed successfully!")
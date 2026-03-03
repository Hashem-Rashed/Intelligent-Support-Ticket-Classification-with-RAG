from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.clean_data import drop_unnecessary_columns
from src.preprocessing.encode_normalize import encode_categorical, normalize_numerical
from src.preprocessing.text_processing import merge_subject_description, clean_text_column, drop_original_text_columns
from src.preprocessing.save_data import save_cleaned_data

# 1. Load data
data = load_raw_data(r"D:\Graduation_Depi\Intelligent-Support-Ticket-Classification-with-RAG\data\raw\tickets.csv")

# 2. Drop unnecessary columns
cols_to_drop = ["Customer_Name", "Customer_Email", "Assigned_Agent", "Submission_Date", "Ticket_ID"]
data = drop_unnecessary_columns(data, cols_to_drop)

# 3. Encode categorical columns
label_cols = ["Issue_Category", "Priority_Level", "Ticket_Channel"]
data, encoders = encode_categorical(data, label_cols)

# 4. Normalize numerical columns
num_cols = ["Resolution_Time_Hours"]
data, scaler = normalize_numerical(data, num_cols)

# 5. Merge and clean text
data = merge_subject_description(data)
data = clean_text_column(data)
data = drop_original_text_columns(data)

# 6. Save cleaned data
save_cleaned_data(data)

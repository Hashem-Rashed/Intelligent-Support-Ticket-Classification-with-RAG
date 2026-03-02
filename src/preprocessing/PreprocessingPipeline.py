import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

nltk.download("stopwords", quiet=True)


class DataPreprocessingPipeline:

    def __init__(self):
        self.cols_to_drop = [
            "Customer_Name",
            "Customer_Email",
            "Assigned_Agent",
            "Submission_Date",
            "Ticket_ID"
        ]

        self.categorical_cols = [
            "Issue_Category",
            "Priority_Level",
            "Ticket_Channel"
        ]

        self.scaler = MinMaxScaler()

    def clean_text(self, text):
        stop = set(stopwords.words("english"))

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = " ".join([word for word in text.split() if word not in stop])

        return text

    def fit_transform(self, df):
        df = df.copy()

        df = df.drop(columns=self.cols_to_drop, errors="ignore")

        df["full_text"] = (
            df["Ticket_Subject"].astype(str) + " " +
            df["Ticket_Description"].astype(str)
        )

        df["clean_text"] = df["full_text"].apply(self.clean_text)

        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col])

        df = df.drop(columns=self.categorical_cols, errors="ignore")

        if "Resolution_Time_Hours" in df.columns:
            df[["Resolution_Time_Hours"]] = self.scaler.fit_transform(
                df[["Resolution_Time_Hours"]]
            )

        df = df.drop(columns=["Ticket_Subject", "Ticket_Description"], errors="ignore")

        return df

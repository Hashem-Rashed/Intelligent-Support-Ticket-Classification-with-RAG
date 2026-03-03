import re
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop = set(stopwords.words("english"))

def merge_subject_description(data):
    """
    Merge Ticket_Subject and Ticket_Description into full_text.
    """
    data["full_text"] = data["Ticket_Subject"].astype(str) + " " + data["Ticket_Description"].astype(str)
    return data

def clean_text_column(data, column="full_text"):
    """
    Clean text: lowercase, remove special chars, remove stopwords.
    """
    def clean_text(txt):
        txt = txt.lower()
        txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
        txt = " ".join([w for w in txt.split() if w not in stop])
        return txt

    data["clean_text"] = data[column].apply(clean_text)
    return data

def drop_original_text_columns(data, cols=["Ticket_Subject", "Ticket_Description"]):
    """
    Drop original text columns after merging.
    """
    return data.drop(columns=cols)

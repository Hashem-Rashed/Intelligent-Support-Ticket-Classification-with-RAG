"""Unit tests for the preprocessing package."""

import sys, os
# make sure the src folder is on the import path when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import numpy as np

from preprocessing import data_loader, pipeline, embedding_generator


def test_load_data(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    loaded = data_loader.load_data(str(path))
    assert loaded.equals(df)


def test_run_pipeline(tmp_path):
    # create fake raw dataset with the required columns
    df = pd.DataFrame({
        "Ticket_Subject": ["s1", "s2"],
        "Ticket_Description": ["d1", "d2"],
        "Issue_Category": ["c1", "c2"],
        "Customer_Name": ["n", "n"],
        "Customer_Email": ["e", "e"],
        "Assigned_Agent": ["a", "a"],
        "Submission_Date": ["x", "y"],
        "Ticket_ID": [1, 2],
        "Satisfaction_Score": [5, 4],
    })
    in_path = tmp_path / "raw.csv"
    df.to_csv(in_path, index=False)
    out_path = tmp_path / "clean.csv"

    cleaned = pipeline.run_pipeline(input_path=str(in_path), output_path=str(out_path))
    assert os.path.exists(out_path)
    assert "clean_text" in cleaned.columns


def test_generate_embeddings(tmp_path, monkeypatch):
    df = pd.DataFrame({"clean_text": ["hello world", "foo bar"]})
    in_path = tmp_path / "clean.csv"
    df.to_csv(in_path, index=False)
    out_path = tmp_path / "emb.npy"

    class DummyModel:
        def encode(self, texts, show_progress_bar=False):
            return np.zeros((len(texts), 5))

    monkeypatch.setattr("preprocessing.embedding_generator.SentenceTransformer", lambda name: DummyModel())
    embeds = embedding_generator.generate_embeddings(input_path=str(in_path), output_path=str(out_path), model_name="dummy")
    assert embeds.shape == (2, 5)
    assert os.path.exists(out_path)
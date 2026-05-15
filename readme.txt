# Intelligent Support Ticket Classification with RAG

A machine learning project for intelligent support ticket classification using traditional ML, transformer-based NLP models, and Retrieval-Augmented Generation (RAG) to improve contextual understanding, accuracy, and explainability.

---

## Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Workflow](#workflow)
* [Installation](#installation)
* [Usage](#usage)
* [Model Training](#model-training)
* [RAG Pipeline](#rag-pipeline)
* [API Deployment](#api-deployment)
* [Evaluation Metrics](#evaluation-metrics)
* [MLOps & Monitoring](#mlops--monitoring)
* [Future Improvements](#future-improvements)
* [Requirements](#requirements)
* [Contributing](#contributing)
* [License](#license)

---

# Overview

Customer support systems receive thousands of tickets daily, making manual classification inefficient and time-consuming.

This project provides an end-to-end AI-powered solution for automatically classifying support tickets into predefined categories using:

* Classical Machine Learning models
* Transformer-based Deep Learning models
* Retrieval-Augmented Generation (RAG)

The RAG pipeline enhances predictions by retrieving semantically relevant historical tickets and contextual information before classification.

---

# Architecture

```text
User Ticket
     │
     ▼
Text Preprocessing
     │
     ▼
Embedding Generation
     │
     ▼
Vector Database Retrieval
     │
     ▼
Retrieved Context + User Ticket
     │
     ▼
Classification Model (BERT / Logistic Regression)
     │
     ▼
Predicted Ticket Category
```

---

# Features

* Automated support ticket classification
* Text preprocessing and normalization
* TF-IDF + Logistic Regression baseline
* BERT fine-tuning for advanced NLP understanding
* Retrieval-Augmented Generation (RAG)
* Semantic similarity search
* Embedding generation using transformer models
* FastAPI REST API
* Dockerized deployment
* MLflow experiment tracking
* MLOps monitoring pipeline
* Modular and scalable architecture

---

# Tech Stack

## Machine Learning & NLP

* Python
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* spaCy
* NLTK
* Sentence Transformers

## RAG & Vector Search

* FAISS / ChromaDB
* LangChain

## Backend & API

* FastAPI
* Uvicorn

## MLOps & Deployment

* Docker
* MLflow
* GitHub Actions

---

# Project Structure

```bash
intelligent-support-rag/
│
├── data/                    # Dataset storage
├── notebooks/               # Jupyter notebooks
│
├── src/
│   ├── ingestion/           # Data ingestion utilities
│   ├── preprocessing/       # Cleaning & preprocessing
│   ├── models/              # ML & DL models
│   ├── rag/                 # RAG pipeline
│   ├── api/                 # FastAPI application
│   ├── deployment/          # Deployment configs
│   ├── mlops/               # Monitoring & retraining
│   └── utils/               # Helper functions
│
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt
├── Dockerfile
├── README.md
└── .env.example
```

---

# Workflow

## 1. Data Ingestion

Load support ticket datasets from CSV files, databases, or APIs.

## 2. Preprocessing

* Remove noise
* Normalize text
* Tokenization
* Stopword removal
* Lemmatization

## 3. Embedding Generation

Convert ticket text into dense vector embeddings using transformer models.

## 4. Retrieval

Retrieve semantically similar historical tickets from the vector database.

## 5. Classification

Classify tickets using:

* Logistic Regression
* BERT
* RAG-enhanced models

## 6. Evaluation

Evaluate models using:

* Accuracy
* Precision
* Recall
* F1-Score

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd intelligent-support-rag
```

## Create Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

```bash
cp .env.example .env
```

---

# Usage

## Data Preparation

```python
from src.ingestion.data_loader import load_ticket_data

tickets = load_ticket_data('data/raw/tickets.csv')
```

---

# Model Training

## Baseline Models

Train classical ML models using TF-IDF features.

```bash
python src/models/train_baseline.py
```

## BERT Fine-Tuning

```bash
python src/models/train_bert.py
```

---

# RAG Pipeline

The Retrieval-Augmented Generation pipeline improves classification quality by retrieving relevant contextual tickets before prediction.

## Components

* Embedding Model
* Vector Database
* Retriever
* Context Builder
* Classification Engine

## Example Flow

```python
query_embedding = embedder.encode(ticket_text)

retrieved_docs = retriever.search(query_embedding)

prediction = classifier.predict(
    ticket_text,
    context=retrieved_docs
)
```

---

# API Deployment

Run the FastAPI server:

```bash
python src/api/main.py
```

API Documentation:

```text
http://localhost:8000/docs
```

---

# Evaluation Metrics

| Metric    | Description                        |
| --------- | ---------------------------------- |
| Accuracy  | Overall prediction correctness     |
| Precision | Positive prediction quality        |
| Recall    | Ability to detect relevant classes |
| F1-Score  | Balance between precision & recall |

---

# MLOps & Monitoring

This project includes:

* Experiment tracking with MLflow
* Model versioning
* Automatic retraining pipelines
* Performance monitoring
* Deployment-ready infrastructure

---

# Future Improvements

* Multi-label ticket classification
* Real-time streaming support
* Hybrid retrieval search
* LLM-based summarization
* Human-in-the-loop feedback
* Kubernetes deployment
* Advanced observability dashboards

---

# Requirements

* Python 3.8+
* PyTorch
* Transformers
* Scikit-learn
* FastAPI
* MLflow
* Docker

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Contributing

Contributions are welcome.

Please ensure:

* Code follows PEP 8 standards
* Unit tests pass
* Documentation is updated

Run tests:

```bash
pytest tests/
```

---

# License

This project is licensed under the MIT License.

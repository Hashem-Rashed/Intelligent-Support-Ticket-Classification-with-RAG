# Intelligent Support Ticket Classification with RAG

An enterprise-grade support ticket system that combines **ML classification** (TF-IDF, DistilBERT, Ensemble) with **Retrieval-Augmented Generation (RAG)** using Chroma vector database and Groq's Llama 3.1.

The system automatically classifies tickets, retrieves similar historical tickets, and generates human-readable explanations.

---

## Overview

This project solves the problem of manual ticket triage by:

- **Classifying** tickets into 5 categories with >99% accuracy.
- **Retrieving** similar past tickets using semantic search.
- **Explaining** predictions using a large language model (LLM).
- **Flagging** low-confidence predictions for human review.

It is designed for production with:

- FastAPI backend
- Prometheus monitoring
- Streamlit demo frontend
- Flask production frontend

---

## Categories

The system classifies tickets into **5 well-defined categories**:

| Category                  | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| **Account**         | Login, password, 2FA, and access issues                  |
| **Billing**         | Refunds, charges, subscriptions, invoices                |
| **Fraud**           | Unauthorized transactions, stolen cards, hacked accounts |
| **General Inquiry** | How-to questions, feature requests, general help         |
| **Technical**       | Crashes, bugs, performance, and errors                   |

---

## Architecture

The system follows a **two-stage pipeline**:

1. **ML Classification** – Fast, accurate category prediction.
2. **RAG (Retrieval-Augmented Generation)** – Retrieves similar tickets and uses an LLM to generate explanations.

### Data Flow

```text
User Ticket
    ↓
Preprocessing
    ↓
ML Model
    ↓
Category + Confidence
    ↓
(if confidence ≥ threshold)
    ↓
Embedding Generation
    ↓
Chroma Semantic Search
    ↓
Top-k Similar Tickets
    ↓
LLM (Groq Llama 3.1)
    ↓
Explanation Generation
    ↓
Final Result
(category + confidence + explanation + similar tickets)
```

## Tech Stack

| Component             | Technology                                         |
| --------------------- | -------------------------------------------------- |
| Backend API           | **FastAPI**                                  |
| ML Models             | TF-IDF + Logistic Regression, DistilBERT, Ensemble |
| Vector Database       | **Chroma**                                   |
| Embeddings            | Sentence-Transformer `all-MiniLM-L6-v2`          |
| LLM                   | **Groq**(Llama 3.1 8B)                       |
| Frontend (Demo)       | Streamlit                                          |
| Frontend (Production) | Flask + Bootstrap                                  |
| Monitoring            | Prometheus + structured JSON logs                  |
| Deployment            | Docker (optional)                                  |

## Project Structure

Intelligent-Support-Ticket-Classification-with-RAG/
│
├── data/
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Cleaned and merged data
│   └── embeddings/              # Chroma DB + embeddings
│
├── models/
│   ├── saved/
│   │   ├── baseline/            # TF-IDF + Logistic Regression
│   │   └── transformer/         # DistilBERT model
│   └── twitter_classifier.pkl
│
├── src/
│   ├── preprocessing/           # Cleaning and preprocessing scripts
│   ├── models/                  # Model training scripts
│   ├── rag/                     # Retriever, indexer, LLM logic
│   ├── api/                     # FastAPI routes and schemas
│   ├── ui/                      # Streamlit app
│   └── utils/                   # Config and logging
│
├── flask_app/                   # Flask production frontend
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md


## Key Features

* ✅ **3 ML model options**
* Baseline (fast)
* Transformer (accurate)
* Ensemble (best overall)
* ✅ **LLM fallback**
* Automatically uses Groq when confidence is low
* ✅ **RAG explanations**
* Retrieves similar tickets and explains predictions
* ✅ **Batch classification**
* Upload CSV and download results
* ✅ **Monitoring**
* Prometheus metrics and structured logs
* ✅ **Human-in-the-loop workflow**
* Flags uncertain predictions for review
* ✅ **Two user interfaces**
* Streamlit (demo)
* Flask (production)

## Model Performance


| Model                    | Accuracy | F1 (Macro) | Inference Time | GPU Required |
| ------------------------ | -------- | ---------- | -------------- | ------------ |
| Baseline (TF-IDF + LR)   | 97.24%   | 97.25%     | 2-5 ms         | ❌ No        |
| Transformer (DistilBERT) | 99.80%   | 99.80%     | 20-50 ms       | ✅ Yes       |
| Ensemble                 | ~99.90%  | ~99.90%    | 30-60 ms       | ✅ Yes       |

## Installation


### 1. Clone the Repository

<pre class="overflow-visible! px-0!" data-start="4231" data-end="4395" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span class="ͼ10" data--h-bstatus="0OBSERVED">git</span><span data--h-bstatus="0OBSERVED"> clone https://github.com/Hashem-Rashed/Intelligent-Support-Ticket-Classification-with-RAG.git</span><br data--h-bstatus="0OBSERVED"/><br data--h-bstatus="0OBSERVED"/><span class="ͼ10" data--h-bstatus="0OBSERVED">cd</span><span data--h-bstatus="0OBSERVED"> Intelligent-Support-Ticket-Classification-with-RAG</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### 2. Create a Virtual Environment

#### Linux / macOS

<pre class="overflow-visible! px-0!" data-start="4454" data-end="4510" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python </span><span class="ͼ12" data--h-bstatus="0OBSERVED">-m</span><span data--h-bstatus="0OBSERVED"> venv venv</span><br data--h-bstatus="0OBSERVED"/><span class="ͼ10" data--h-bstatus="0OBSERVED">source</span><span data--h-bstatus="0OBSERVED"> venv/bin/activate</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

#### Windows

<pre class="overflow-visible! px-0!" data-start="4526" data-end="4579" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python </span><span class="ͼ12" data--h-bstatus="0OBSERVED">-m</span><span data--h-bstatus="0OBSERVED"> venv venv</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">venv\Scripts\activate</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### 3. Install Dependencies

<pre class="overflow-visible! px-0!" data-start="4610" data-end="4653" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">pip install </span><span class="ͼ12" data--h-bstatus="0OBSERVED">-r</span><span data--h-bstatus="0OBSERVED"> requirements.txt</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and add your Groq API key:

<pre class="overflow-visible! px-0!" data-start="4753" data-end="4799" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">GROQ_API_KEY=your_groq_api_key_here</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### 5. Preprocess Data

<pre class="overflow-visible! px-0!" data-start="4825" data-end="4882" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python src/preprocessing/preprocessing_run.py</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

Then follow the interactive menu to:

* Clean tickets
* Process Twitter data
* Merge datasets
* Generate embeddings

### 6. Train Models

#### Baseline Model

<pre class="overflow-visible! px-0!" data-start="5043" data-end="5113" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python src/models/run_models.py </span><span class="ͼ12" data--h-bstatus="0OBSERVED">--model</span><span data--h-bstatus="0OBSERVED"> baseline </span><span class="ͼ12" data--h-bstatus="0OBSERVED">--balance</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

#### Transformer Model

<pre class="overflow-visible! px-0!" data-start="5139" data-end="5249" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python src/models/run_models.py </span><span class="ͼ12" data--h-bstatus="0OBSERVED">--model</span><span data--h-bstatus="0OBSERVED"> transformer </span><span class="ͼ12" data--h-bstatus="0OBSERVED">--balance</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼ12" data--h-bstatus="0OBSERVED">--auto-limit</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼ12" data--h-bstatus="0OBSERVED">--transformer-max</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼy" data--h-bstatus="0OBSERVED">10000</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

---

## Running the System

### Start the FastAPI Backend

<pre class="overflow-visible! px-0!" data-start="5313" data-end="5347" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">python </span><span class="ͼ12" data--h-bstatus="0OBSERVED">-m</span><span data--h-bstatus="0OBSERVED"> src.api.main</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

API available at:

* `http://localhost:8000`
* Swagger Docs: `http://localhost:8000/docs`

---

### Start the Streamlit UI

<pre class="overflow-visible! px-0!" data-start="5473" data-end="5522" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">streamlit run src/ui/streamlit_app.py</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

---

### Start the Flask Production UI

<pre class="overflow-visible! px-0!" data-start="5564" data-end="5602" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span class="ͼ10" data--h-bstatus="0OBSERVED">cd</span><span data--h-bstatus="0OBSERVED"> flask_app</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">python app.py</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

Then open:

<pre class="overflow-visible! px-0!" data-start="5616" data-end="5649" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1" data--h-bstatus="0OBSERVED"></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="pe-11 pt-3" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">http://localhost:5000</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

---

## 📡 API Endpoints

| Endpoint                   | Method | Description                      |
| -------------------------- | ------ | -------------------------------- |
| `/api/v1/classify`       | POST   | Single ticket classification     |
| `/api/v1/classify/batch` | POST   | Batch ticket classification      |
| `/api/v1/rag/explain`    | POST   | Classification + RAG explanation |
| `/api/v1/health`         | GET    | Health check                     |
| `/api/v1/metrics`        | GET    | Prometheus metrics               |

---

## Example API Request

### Classify Ticket

<pre class="overflow-visible! px-0!" data-start="6067" data-end="6284" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span class="ͼ10" data--h-bstatus="0OBSERVED">curl</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼ12" data--h-bstatus="0OBSERVED">-X</span><span data--h-bstatus="0OBSERVED"> POST </span><span class="ͼz" data--h-bstatus="0OBSERVED">"http://localhost:8000/api/v1/classify"</span><span data--h-bstatus="0OBSERVED"> \</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED"></span><span class="ͼ12" data--h-bstatus="0OBSERVED">-H</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼz" data--h-bstatus="0OBSERVED">"Content-Type: application/json"</span><span data--h-bstatus="0OBSERVED"> \</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED"></span><span class="ͼ12" data--h-bstatus="0OBSERVED">-d</span><span data--h-bstatus="0OBSERVED"></span><span class="ͼz" data--h-bstatus="0OBSERVED">'{</span><br data--h-bstatus="0OBSERVED"/><span class="ͼz" data--h-bstatus="0OBSERVED">    "text":"Someone stole my credit card",</span><br data--h-bstatus="0OBSERVED"/><span class="ͼz" data--h-bstatus="0OBSERVED">    "model_type":"ensemble",</span><br data--h-bstatus="0OBSERVED"/><span class="ͼz" data--h-bstatus="0OBSERVED">    "return_details":true</span><br data--h-bstatus="0OBSERVED"/><span class="ͼz" data--h-bstatus="0OBSERVED">  }'</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### Example Response

<pre class="overflow-visible! px-0!" data-start="6308" data-end="6421" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4" data--h-bstatus="0OBSERVED"><div class="pointer-events-none sticky z-40 shrink-0 z-1!" data--h-bstatus="0OBSERVED"><div class="sticky bg-token-border-light" data--h-bstatus="0OBSERVED"></div></div></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">{</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">  "category": </span><span class="ͼz" data--h-bstatus="0OBSERVED">"Fraud"</span><span data--h-bstatus="0OBSERVED">,</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">  "confidence": </span><span class="ͼy" data--h-bstatus="0OBSERVED">0.995</span><span data--h-bstatus="0OBSERVED">,</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">  "needs_review": </span><span class="ͼy" data--h-bstatus="0OBSERVED">false</span><span data--h-bstatus="0OBSERVED">,</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">  "model_used": </span><span class="ͼz" data--h-bstatus="0OBSERVED">"ensemble"</span><br data--h-bstatus="0OBSERVED"/><span data--h-bstatus="0OBSERVED">}</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

---

## UI Screenshots

### Streamlit – Classification Dashboard

<pre class="overflow-visible! px-0!" data-start="6493" data-end="6534" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1" data--h-bstatus="0OBSERVED"></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="pe-11 pt-3" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">images/streamlit_classify.png</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### Streamlit – RAG Explanation View

<pre class="overflow-visible! px-0!" data-start="6574" data-end="6610" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1" data--h-bstatus="0OBSERVED"></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="pe-11 pt-3" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">images/streamlit_rag.png</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### Flask – Production Interface

<pre class="overflow-visible! px-0!" data-start="6646" data-end="6677" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1" data--h-bstatus="0OBSERVED"></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="pe-11 pt-3" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">images/flask_ui.png</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

---

## Monitoring

Prometheus metrics are exposed at:

<pre class="overflow-visible! px-0!" data-start="6738" data-end="6765" data--h-bstatus="0OBSERVED"><div class="relative w-full mt-4 mb-1" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"><div class="relative" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="h-full min-h-0 min-w-0" data--h-bstatus="0OBSERVED"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl" data--h-bstatus="0OBSERVED"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback" data--h-bstatus="0OBSERVED"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1" data--h-bstatus="0OBSERVED"></div><div class="relative" data--h-bstatus="0OBSERVED"><div class="pe-11 pt-3" data--h-bstatus="0OBSERVED"><div class="relative z-0 flex max-w-full" data--h-bstatus="0OBSERVED"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16" data--h-bstatus="0OBSERVED"><div class="cm-scroller" data--h-bstatus="0OBSERVED"><pre class="cm-content q9tKkq_readonly m-0" data--h-bstatus="0OBSERVED"><code data--h-bstatus="0OBSERVED"><span data--h-bstatus="0OBSERVED">/api/v1/metrics</span></code></pre></div></div></div></div></div></div></div></div></div><div class="" data--h-bstatus="0OBSERVED"><div class="" data--h-bstatus="0OBSERVED"></div></div></div></div></div></pre>

### Structured Logs Include

* `request_id`
* `latency`
* `confidence`
* `fraud_flag`
* `llm_fallback_status`

### Health Endpoint

Confirms:

* API availability
* Model readiness
* Vector database status

---

## Limitations

* Groq LLM requires:
* Internet connection
* API key
* Very long tickets (>512 tokens) are truncated.
* Fraud detection may not handle all adversarial cases.
* Synthetic datasets may not fully reflect real-world ticket distributions.

---

## Future Improvements

*  Cloud deployment (AWS ECS / Google Cloud Run)
* Local LLM support using Ollama
* Human feedback fine-tuning loop
* Grafana dashboards
* Authentication and rate limiting

---

## Project Team

* Hasham Abdelrahman
* Enas Essam
* Hossam Ashraf
* Ahmed Magdy
* Randa Hamada

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgements

* Groq for free LLM API access
* Chroma for vector database infrastructure
* Hugging Face for transformers and embeddings
* FastAPI and Streamlit for modern web frameworks

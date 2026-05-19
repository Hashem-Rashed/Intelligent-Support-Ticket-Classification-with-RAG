# Support Ticket Classification using RAG + LLM

An AI-powered support ticket classification system that uses a Retrieval-Augmented Generation (RAG) pipeline to automatically classify customer support tickets and generate contextual solutions using Large Language Models.

---

## 📌 Overview

This project combines semantic search, vector databases, and Large Language Models to build an intelligent support ticket assistant capable of:

* Automatically classifying support tickets
* Generating short AI-powered solutions
* Handling single and bulk ticket classification
* Retrieving relevant context using RAG architecture

The system integrates:

* **Llama 3.1** via Groq API
* **FAISS Vector Database**
* **HuggingFace Embeddings**
* **LangChain RAG Pipeline**
* **Flask Web Application**

---

 Features

* ✅ Automatic support ticket classification
* ✅ AI-generated short solutions
* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Semantic similarity search using FAISS
* ✅ Bulk ticket classification support
* ✅ Simple Flask + HTML web interface
* ✅ Context-aware responses using LLMs

---

 Supported Categories

The system classifies tickets into predefined categories such as:

* Login Issues
* App Functionality
* Billing
* Account Management
* Performance Issues

---

## 📂 Project Structure

```bash
Support-Ticket-Classifier-with-RAG/
│
├── server.py                # Flask backend & RAG pipeline
├── templates/
│   └── index.html           # Web interface
│
├── Data/
│   └── knowledge_base.txt   # Knowledge base categories
│
├── images/
│   ├── single_ticket.png    # Single ticket result example
│   └── bulk_ticket.png      # Bulk ticket result example
│
├── notebook.ipynb           # Experiments & testing
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## ⚙️ System Architecture

### 1️⃣ Knowledge Base Creation

Predefined support categories and issue descriptions are stored inside a knowledge base.

### 2️⃣ Document Processing

Text is split into chunks for better semantic retrieval.

### 3️⃣ Embedding Generation

Embeddings are generated using:

```bash
all-MiniLM-L6-v2
```

### 4️⃣ Vector Store

FAISS is used for efficient similarity search and retrieval.

### 5️⃣ RAG Pipeline

The pipeline:

* Retrieves relevant context from the vector database
* Sends context + ticket to Llama 3.1
* Generates:

  * Ticket Category
  * Suggested Solution

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### AI & NLP

* LangChain
* FAISS
* HuggingFace Transformers
* Sentence Transformers
* Groq API
* Llama 3.1

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd Support-Ticket-Classifier-with-RAG
```

### 2️⃣ Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Start the Flask server:

```bash
python server.py
```

Open the application in your browser:

```bash
http://localhost:5000/
```

---

## 💡 Usage

### Single Ticket Classification

1. Enter a support ticket
2. Click **Classify**
3. Receive:

   * Predicted Category
   * AI-Generated Solution

---

### Bulk Ticket Classification

1. Enter multiple tickets (one ticket per line)
2. Click **Classify All**
3. Get structured classification results for all tickets

---

## 🖼️ Example Results

### Single Ticket Classification

```bash
images/single_ticket.png
```

### Bulk Ticket Classification

```bash
images/bulk_ticket.png
```

---

## 📈 Example Output

| Ticket                               | Predicted Category | Suggested Solution                 |
| ------------------------------------ | ------------------ | ---------------------------------- |
| Unable to login after password reset | Login Issues       | Reset cache and verify credentials |
| Payment failed during checkout       | Billing            | Verify payment method and retry    |

---

## ⚠️ Limitations

* Performance depends on the quality of the knowledge base
* Ambiguous tickets may be misclassified
* Requires predefined structured categories
* Responses depend on retrieved context quality

---

## 🔮 Future Improvements

* Improve knowledge base using real-world datasets
* Add authentication system
* Cloud deployment (AWS / Render / Railway)
* Replace HTML frontend with React or Streamlit
* Add ticket history database
* Structured JSON responses with schema validation
* Multi-language support

---

## 🤝 Contributing

Contributions are welcome.

Before submitting changes:

```bash
pytest
```



---


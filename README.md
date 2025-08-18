
````markdown
# 🤖 MoE Research Agent

This project is an advanced AI agent designed to perform Retrieval-Augmented Generation (RAG) on a specialized knowledge base of research papers about Mixture of Experts (MoE) models.

The agent uses an agentic workflow built with **LangGraph** to not only answer questions but to first critically evaluate the relevance of the retrieved information. This self-correction mechanism allows the agent to avoid hallucination and refuse to answer when the context is insufficient, making it a more robust and reliable system.

---
## ✨ Key Features

* **Agentic RAG Workflow**: Implements a multi-step "Retrieve -> Grade -> Generate" process, moving beyond simple Q&A.
* **Self-Correction Logic**: The agent intelligently decides if the retrieved documents are relevant to the question before generating an answer.
* **Built with LangGraph**: Leverages LangGraph to create a stateful, cyclical graph for sophisticated agentic behavior.
* **High-Performance LLM**: Powered by Llama 3 via the fast Groq API.
* **Local & Efficient Retrieval**: Uses a FAISS vector store for fast, local similarity searches on document embeddings.

---
## 🏗️ Architecture

The agent's decision-making process is modeled as a graph. It retrieves information, grades it for relevance, and then takes a different path depending on the outcome.

```mermaid
graph TD
    A[Start: User Question] --> B(Retrieve Documents);
    B --> C{Grade Document Relevance};
    C -- Relevant --> D(Generate Answer);
    C -- Not Relevant --> E[End: Cannot Answer];
    D --> F[End: Provide Answer];
````

-----

## 🛠️ Tech Stack

  * **Python**
  * **LangChain & LangGraph** for the core framework
  * **Groq API** for LLM inference (Llama 3)
  * **FAISS** for the vector store
  * **Hugging Face** for sentence embeddings (`BAAI/bge-small-en-v1.5`)

-----

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1\. Prerequisites

  * Python 3.9+
  * Git

### 2\. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 3\. Set Up the Environment

Create and activate a Python virtual environment:

```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 4\. Install Dependencies

Install all the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5\. Set Up API Keys

Create a `.env` file in the root of the project and add your Groq API key:

```
GROQ_API_KEY="your_groq_api_key_here"
```

-----

## ⚙️ Usage

### 1\. Ingest Your Data

Place your PDF research papers into the `/data` directory. Then, run the ingestion script to create the vector store. This only needs to be done once.

```bash
python ingest.py
```

This will create a `vectorstores/db_faiss` folder containing the indexed knowledge base.

### 2\. Run the Agent

Execute the agent script to start asking questions.

```bash
python agent.py
```

-----

## 💡 Example Output

The agent demonstrates its ability to handle both relevant and irrelevant questions correctly.

### Relevant Question

```
> python agent.py

--- RUNNING WITH RELEVANT QUESTION ---
...
FINAL GENERATION:
The role of the gating network in a Mixture of Experts (MoE) model is to produce a sparse n-dimensional vector that determines which expert networks to use for a given input.
```

### Irrelevant Question

```
--- RUNNING WITH IRRELEVANT QUESTION ---
...
FINAL RESPONSE: No relevant documents found to answer the question.
```

```
```
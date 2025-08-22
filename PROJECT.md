# Final Project: FAQ Chatbot using Simple RAG with Free Models from Hugging Face or OpenAI

## Objective
Create a basic chatbot capable of answering frequently asked questions (FAQ) within a small domain (e.g., tourism, coffee, movies), combining:

- **RAG (Retrieval-Augmented Generation):** for retrieving information from local documents.
- **Open-source model from Hugging Face** (such as `mistralai/Mistral-7B-Instruct` via *text-generation-inference* or `google/flan-t5-base`). Or an OpenAI model.
- **Simple indexing with FAISS.**
- **No paid APIs** (only local models or free Hugging Face endpoints).

---

## Project Components

### 1. Local Document Corpus
A set of `.txt` files with information about the chosen domain.
Examples:
- *Rio de Janeiro Travel Guide*
- *Coffee Facts*
- *Explanations of Famous Movies*

---

### 2. Basic RAG Pipeline

Step 1: Indexing - Use **FAISS** to index embeddings (with `sentence-transformers/all-MiniLM-L6-v2`).

Step 2: Retrieval - Retrieve the most relevant excerpts from the corpus based on the user’s question.

Step 3: Generation - Use a model such as `flan-t5-base` to answer using the retrieved excerpt.

---

### 3. Simple Interface
- A **terminal script** or **notebook**.
- Basic interface: user types a question, chatbot responds.

---

## Bonus (Optional)
- Implement using **LangChain** (works even with free models).
- Save **history of questions and answers**.

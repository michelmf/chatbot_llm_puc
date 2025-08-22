# 🤖 FAQ Chatbot with RAG (Retrieval-Augmented Generation)

An intelligent chatbot that uses RAG to answer questions based on a corpus of documents about cats. The system combines semantic search with language models to provide accurate and contextualized responses.

## 📋 About the Project

This project implements a FAQ chatbot system that uses:
- **RAG (Retrieval-Augmented Generation)**: Combines information retrieval with text generation
- **FAISS**: For efficient indexing and semantic search
- **Sentence Transformers**: For creating semantic embeddings
- **Transformers**: For response generation (T5 or causal models)
- **Rich**: For elegant command-line interface

## 🏗️ System Architecture

### Directory Structure

```
chatbot_llm_puc/
├── app.py                 # Main entry point
├── requirement.txt        # Project dependencies
├── README.md             # This file
├── data/                 # Project data
│   └── corpus/           # Text documents (.txt)
├── conversations/        # Saved conversation history
└── utils/               # Utility modules
    ├── __init__.py
    ├── cli.py           # CLI interface commands
    ├── exceptions.py    # Custom exceptions
    ├── indexing.py      # FAISS indexing logic
    ├── model.py         # Text generation pipelines
    ├── retrieval.py     # Search and retrieval system
    └── text.py          # Text processing and chunking
```

### Main Components

1. **Indexing** (`utils/indexing.py`):
   - Loads documents from corpus
   - Splits texts into overlapping chunks
   - Creates embeddings using Sentence Transformers
   - Builds FAISS index for fast search

2. **Retrieval** (`utils/retrieval.py`):
   - Semantic search using embeddings
   - Checks if indexing has been performed
   - Returns most relevant documents

3. **Generation** (`utils/model.py`):
   - Support for T5 and causal models
   - Generates responses based on retrieved context
   - Configurable through environment variables

4. **Interface** (`utils/cli.py`):
   - Commands for indexing and chat
   - Automatic prerequisite checking
   - Rich interface with colors and panels

## 🚀 How to Run the Program

### 1. Environment Setup

```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd chatbot_llm_puc

# Install dependencies
pip install -r requirement.txt
```

### 2. Environment Variables Configuration

Create a `.env` file in the project root with the following variables:

```env
# Embedding model
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Generation model
TOKENIZER_NAME=unicamp-dl/ptt5-base-portuguese-vocab
MODEL_NAME=unicamp-dl/ptt5-base-portuguese-vocab

# Data paths
DATA_PATH=data/
DATA_CORPUS_PATH=data/corpus/
DATA_INDEX_PATH=data/index.faiss
DATA_META_PATH=data/meta.jsonl
DATA_EMBEDDINGS_PATH=data/embeddings.npy

# Chunking settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Search settings
TOP_K=4
```

### 3. Running the System

#### Step 1: Indexing (Required on first run)

```bash
python app.py index
```

This command:
- Reads all `.txt` files from `data/corpus/` folder
- Splits texts into chunks
- Creates semantic embeddings
- Builds FAISS index
- Saves metadata for retrieval

#### Step 2: Start the Chatbot

```bash
# Using T5 model (default)
python app.py chat

# Or using causal model
python app.py chat --mode causal
```

### 4. Using the Chatbot

1. The system automatically checks if indexing has been done
2. Type your questions about cats
3. The system searches for relevant information and generates responses
4. To exit: type `sair`, `exit` or `quit`

## 💡 Usage Example

```
┌─ FAQ Chatbot (RAG) ─┐
└─────────────────────┘

✅ Index found! Starting chatbot...
Type your question. Type 'sair', 'exit' or 'quit' to close the chat.

You: What do cats think about their owners?

┌─ Answer ─┐
│ Cats develop affective bonds with their owners and see them │
│ as figures of security and comfort... │
└──────────┘

┌─ Sources ─┐
│ - gatos_pensam.txt (chunk 2, score 0.845) │
│ - gatos_tutores.txt (chunk 1, score 0.723) │
└───────────┘
```

## 🛠️ Features

- ✅ **Automatic indexing verification**
- ✅ **Rich interface with colors and panels**
- ✅ **Conversation history automatically saved**
- ✅ **Support for multiple generation models**
- ✅ **Efficient semantic search with FAISS**
- ✅ **Source references for answers**
- ✅ **Optimized text processing with chunks**

## 🔧 Main Dependencies

- `faiss-cpu>=1.7.4` - Vector indexing and search
- `sentence-transformers>=2.7.0` - Semantic embeddings
- `transformers>=4.42.0` - Language models
- `torch>=2.2.0` - Deep learning backend
- `rich>=13.7.0` - Rich CLI interface
- `numpy>=1.26.0` - Numerical computing

## 📝 Available Commands

```bash
# General help
python app.py --help

# Index documents
python app.py index

# Start chatbot
python app.py chat [--mode {t5,causal}]
```

## 🎯 Next Steps

- [ ] Support for more document formats (PDF, DOCX)
- [ ] Web interface with Streamlit/Gradio
- [ ] Model configuration via interface
- [ ] Response quality metrics
- [ ] Embedding cache for better performance

---

**Developed as an academic project to demonstrate RAG concepts and natural language processing.**

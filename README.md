## RAG Chatbot Project

An industry-level, modular Python architecture for production Retrieval-Augmented Generation (RAG) applications using LangChain, ChromaDB, OpenAI, and Google Gemini.

---

## 📁 Project Architecture & File Structure

```
RAG PROJECT/
├── .env.example                 # Environment configuration template
├── .gitignore                   # Ignores sensitive .env, Chroma database, and caches
├── requirements.txt             # Production dependency specifications
├── README.md                    # Project documentation
│
├── config/                      # [Task i] Configuration Subsystem
│   ├── __init__.py
│   └── config.py                # Strongly-typed configs (API keys, LLM, DB, Chunking)
│
├── data/                        # Document storage
│   └── cricket.txt              # Sample document for initial indexing
│
├── src/                         # Core RAG Application Packages
│   ├── __init__.py
│   ├── document_loader.py       # Loaders for .txt, .pdf, web URLs, and directories
│   ├── text_splitter.py         # Chunking (Recursive, Character, Semantic, Code)
│   ├── vector_store.py          # Vector DB management (ChromaDB) & Embeddings factory
│   ├── prompts.py               # [Task iii] Modular prompts (Standard, Strict, Summary)
│   └── rag_pipeline.py          # [Task ii] Core RAG pipeline with LCEL orchestration
│
├── user_queries.py              # [Task iv] User queries runner (Single, Batch, CLI)
├── main.py                      # Application entrypoint & pipeline demonstration
└── rag_project_2.py             # Original reference notebook script
```

---

## 🎯 Modular Task Breakdown

| Task Component | File | Responsibilities |
| :--- | :--- | :--- |
| **i) Configs** | [`config/config.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/config/config.py) | API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`), LLM parameters (`temperature`, `model_name`, `max_tokens`), Embedding settings, Vector DB settings (`collection_name`, `persist_dir`, `top_k`), and Text splitting parameters. |
| **ii) RAG Pipeline** | [`src/rag_pipeline.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/src/rag_pipeline.py) | End-to-end LCEL chain (`retriever \| format_docs \| prompt \| llm \| parser`), document ingestion, chunking, vector indexing, document summarization, and query execution. |
| **iii) Prompts** | [`src/prompts.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/src/prompts.py) | Centralized, reusable prompt templates: Standard QA, Strict anti-hallucination grounded QA, Document summarization, and Conversational RAG with history. |
| **iv) User Queries** | [`user_queries.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/user_queries.py) | Predefined query test suites, batch query execution, interactive terminal CLI mode, and query formatting with retrieved sources and chunks. |

---

## ⚙️ Configuration Setup

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Configure your API keys & preferences in `.env`**:
   ```env
   # API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   # OR for Google Gemini:
   GOOGLE_API_KEY=your_gemini_api_key_here

   # LLM Settings (openai or google_genai)
   LLM_PROVIDER=openai
   LLM_MODEL_NAME=gpt-4o-mini
   LLM_TEMPERATURE=0.2

   # Embedding Settings (openai or google_genai)
   EMBEDDING_PROVIDER=openai
   EMBEDDING_MODEL_NAME=text-embedding-3-small

   # Vector DB Settings
   CHROMA_PERSIST_DIR=./data/chroma_db
   CHROMA_COLLECTION_NAME=rag_collection
   RETRIEVER_TOP_K=4
   ```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Main Pipeline
Run the full demonstration (document indexing + sample queries):
```bash
python main.py
```

### 3. Ask Custom Questions Directly
Pass questions via CLI:
```bash
python main.py "How is cricket played and what are the formats?"
```
Or with [`user_queries.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/user_queries.py):
```bash
python user_queries.py "What are the ways a batter can be out?"
```

### 4. Interactive Question Answering Session
Launch an interactive question-answering CLI session:
```bash
python main.py --interactive
```

---

## 🧩 Supported Document Loaders & Splitters

- **Loaders** ([`src/document_loader.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/src/document_loader.py)):
  - Text files (`.txt`, `.md`)
  - PDF documents (`.pdf`) via `PyPDFLoader`
  - Web URLs via `WebBaseLoader`
  - Entire directories via `DirectoryLoader`
  - Raw strings / in-memory text

- **Chunkers** ([`src/text_splitter.py`](file:///Users/pushpendramishra/Downloads/RAG%20PROJECT/src/text_splitter.py)):
  - `RecursiveCharacterTextSplitter` (best practice general purpose)
  - `CharacterTextSplitter` (separator-based)
  - `SemanticChunker` (embedding breakpoint detection)
  - Code-aware splitters for Python/JS/etc.
